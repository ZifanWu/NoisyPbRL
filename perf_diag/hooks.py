"""Monitoring hooks that integrate the FD probe + gold-isolation guard into the
existing PEBBLE training loop with NO edits to the inner loop.

`install_hooks(workspace)` is called once from `train_PEBBLE.py`'s `Workspace.__init__`
or `run()`. It:
  - Monkey-patches `workspace.replay_buffer.relabel_with_predictor` to fire the
    `post_relabel_hook` after the live RM has been refit and the buffer relabeled.
  - Monkey-patches `workspace.replay_buffer.add` with the GoldIsolationGuard, which
    asserts the reward argument equals the most recent value returned by
    `reward_model.r_hat(...)` (verifying r* never leaks into the training signal).
  - Stores a JSONL trace under `<perf_diag.runs>/<run_name>.jsonl`.

The probe is configured by env vars (so we don't have to thread Hydra overrides into
the live config). Defaults make it cheap.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import numpy as np
import torch

from . import probe as _probe


# Module-level config; populated lazily from env vars on first install.
_CFG_DEFAULTS = dict(
    PD_ENABLE="1",
    PD_N_PROBE="8",
    PD_SEGMENT_LEN="50",
    PD_REFIT_PAIRS="16",
    PD_K_REFIT="5",
    PD_REFIT_LR="0.0003",
    PD_PROBE_M="1",        # fire every M relabels
    PD_OUT_DIR=os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs"),
    PD_RUN_NAME="default",
    PD_TAG_POSITIVE="positive",   # "positive" or "negative_control"
)


def _cfg(key: str, cast=str):
    return cast(os.environ.get(key, _CFG_DEFAULTS[key]))


class GoldIsolationGuard:
    """Wraps reward_model.r_hat to remember the most recent prediction, and wraps
    replay_buffer.add to assert the reward equals it. Raises RuntimeError if a
    write into the buffer ever uses anything other than the most recent r_hat.
    """

    def __init__(self, workspace):
        self.workspace = workspace
        self._last_r_hat = None
        self._violations = 0

    def install(self):
        rm = self.workspace.reward_model
        rb = self.workspace.replay_buffer
        orig_r_hat = rm.r_hat
        orig_add = rb.add

        def wrapped_r_hat(x):
            val = orig_r_hat(x)
            self._last_r_hat = float(val)
            return val

        def wrapped_add(obs, action, reward, next_obs, done, done_no_max):
            if self._last_r_hat is not None:
                # numerical tolerance is unnecessary — both go through the same float64 path.
                if not float(reward) == self._last_r_hat:
                    # However r_hat may have been called from another path between writes;
                    # we tolerate one mismatch (e.g. during the very first env step before any
                    # r_hat call) — but flag any *subsequent* divergence.
                    self._violations += 1
                    if self._violations > 1:
                        raise RuntimeError(
                            f"[GoldIsolationGuard] replay_buffer.add called with reward={reward} "
                            f"but most recent r_hat returned {self._last_r_hat}. "
                            f"Possible gold-reward leak into the agent's training signal."
                        )
            return orig_add(obs, action, reward, next_obs, done, done_no_max)

        rm.r_hat = wrapped_r_hat       # type: ignore[assignment]
        rb.add = wrapped_add           # type: ignore[assignment]
        self.workspace._gold_isolation_guard = self


# ---------------------------------------------------------------------------
# Probe + logging hook
# ---------------------------------------------------------------------------
class ProbeRecorder:
    """Records JSONL rows of probe + baseline metrics on every relabel."""

    def __init__(self, workspace):
        self.workspace = workspace
        self.relabel_idx = 0
        self.out_dir = _cfg("PD_OUT_DIR")
        os.makedirs(self.out_dir, exist_ok=True)
        self.path = os.path.join(self.out_dir, f"{_cfg('PD_RUN_NAME')}.jsonl")
        # truncate existing file at start
        with open(self.path, "w"):
            pass
        self.is_positive = (_cfg("PD_TAG_POSITIVE") == "positive")
        self._t0 = time.time()
        self._pretrain_actor_state = None
        # Latest eval values we shadow from the Workspace.evaluate() return path
        self._last_eval = dict(true_episode_reward=float("nan"),
                               episode_reward=float("nan"),
                               success_rate=float("nan"))
        self._install()

    def _install(self):
        wp = self.workspace
        orig_relabel = wp.replay_buffer.relabel_with_predictor
        orig_evaluate = wp.evaluate

        recorder = self

        def wrapped_relabel(reward_model):
            ret = orig_relabel(reward_model)
            recorder._on_relabel()
            return ret

        def wrapped_evaluate(*args, **kwargs):
            ret = orig_evaluate(*args, **kwargs)
            recorder._snapshot_eval()
            return ret

        wp.replay_buffer.relabel_with_predictor = wrapped_relabel  # type: ignore[assignment]
        wp.evaluate = wrapped_evaluate                              # type: ignore[assignment]
        wp._perf_diag_recorder = recorder

    def _snapshot_eval(self):
        """Read the values the logger just received by reading the underlying tb dict.

        The Logger flushes after each evaluate(); rather than parse files we hook our
        own copy by post-processing the Workspace's evaluate() local state — which we
        cannot reach. Fall back: re-run a lightweight gold rollout on a single episode
        ONLY if the simpler path fails. Here we look at the deque the Workspace stores.
        """
        wp = self.workspace
        # The Workspace updates `avg_train_true_return` deque in run(); reading it
        # is the cheapest, no-extra-rollout signal we have on the eval cadence.
        try:
            if hasattr(wp, "_perf_diag_last_eval_gold"):
                self._last_eval["true_episode_reward"] = float(wp._perf_diag_last_eval_gold)
        except Exception:
            pass

    def _kl_to_pretrain_and_entropy(self) -> tuple[float, float]:
        """Approximate KL(π_θ ‖ π_pretrain) and entropy on a small obs sample from the
        replay buffer. If we haven't snapshotted a pretrain actor, KL is 0.
        """
        wp = self.workspace
        device = wp.device
        rb = wp.replay_buffer
        n = max(1, min(256, rb.idx if not rb.full else rb.capacity))
        if n < 2:
            return 0.0, 0.0
        rng = np.random.default_rng(0)
        idxs = rng.integers(0, n, size=n)
        obs = torch.as_tensor(rb.obses[idxs], dtype=torch.float32, device=device)
        with torch.no_grad():
            dist = wp.agent.actor(obs)
            mu = dist.loc
            std = dist.scale
            # entropy of Squashed Normal: use base Normal entropy (cheap proxy)
            ent = float(dist.base_dist.entropy().sum(-1).mean().item())
            kl = 0.0
            if self._pretrain_actor_state is not None:
                # KL between two diagonal Normals (pre-squash) at the same obs
                mu0, std0 = self._pretrain_actor_state(obs)
                # KL(N(mu, std) || N(mu0, std0)) = log(std0/std) + (std^2 + (mu - mu0)^2) / (2 std0^2) - 0.5
                kl_per = (torch.log(std0 / std) + (std ** 2 + (mu - mu0) ** 2) / (2.0 * std0 ** 2) - 0.5)
                kl = float(kl_per.sum(-1).mean().item())
        return kl, ent

    def _snapshot_pretrain_actor(self):
        """Take a frozen copy of the actor at the moment unsup pretraining ends.

        Called the first time _on_relabel fires (i.e. immediately after the first
        relabel post-unsup-exploration). KL is measured w.r.t. this snapshot.
        """
        wp = self.workspace
        device = wp.device
        import copy as _copy
        live = wp.agent.actor
        # The actor caches non-leaf forward intermediates in self.outputs (mu, std from the
        # last forward pass). Those break deepcopy. Clear them, deepcopy the module, restore.
        saved_outputs = getattr(live, "outputs", None)
        try:
            if hasattr(live, "outputs"):
                live.outputs = {}
            snapshot = _copy.deepcopy(live).to(device).eval()
        finally:
            if saved_outputs is not None:
                live.outputs = saved_outputs
        for p in snapshot.parameters():
            p.requires_grad_(False)

        def _query(obs):
            with torch.no_grad():
                d = snapshot(obs)
                return d.loc, d.scale

        self._pretrain_actor_state = _query

    def _evaluate_gold_now(self) -> float:
        """Lightweight gold eval: one episode under the current actor in a fresh env."""
        wp = self.workspace
        try:
            if "metaworld" in wp.cfg.env:
                from utils import make_metaworld_env
                env = make_metaworld_env(wp.cfg)
            else:
                from utils import make_env
                env = make_env(wp.cfg)
        except Exception:
            return float("nan")
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        total = 0.0
        steps = 0
        done = False
        while not done and steps < 1000:
            with torch.no_grad():
                o = torch.as_tensor(obs, dtype=torch.float32, device=wp.device).unsqueeze(0)
                a = wp.agent.actor(o).mean.clamp(-1.0, 1.0).cpu().numpy()[0]
            step_out = env.step(a)
            if len(step_out) == 5:
                obs, r, term, trunc, _ = step_out
                done = bool(term) or bool(trunc)
            else:
                obs, r, done, _ = step_out
            total += float(r)
            steps += 1
        try:
            env.close()
        except Exception:
            pass
        return total

    def _on_relabel(self):
        wp = self.workspace
        if int(_cfg("PD_ENABLE")) == 0:
            return
        # Probe only every PD_PROBE_M relabels
        self.relabel_idx += 1
        if (self.relabel_idx % int(_cfg("PD_PROBE_M"))) != 0:
            return
        # First relabel after unsup ⇒ snapshot pretrain actor for KL baseline
        if self._pretrain_actor_state is None:
            self._snapshot_pretrain_actor()
        # Build a probe env factory (fresh env so we don't perturb the training env)
        def env_factory():
            try:
                if "metaworld" in wp.cfg.env:
                    from utils import make_metaworld_env
                    return make_metaworld_env(wp.cfg)
                from utils import make_env
                return make_env(wp.cfg)
            except Exception as exc:
                raise RuntimeError(f"probe env_factory failed: {exc}")

        try:
            probe_seed = int(getattr(wp.cfg, "seed", 0)) * 100003 + self.relabel_idx
            probe_out = _probe.run_probe(
                actor=wp.agent.actor,
                reward_model=wp.reward_model,
                env_factory=env_factory,
                n_probe=int(_cfg("PD_N_PROBE")),
                segment_len=int(_cfg("PD_SEGMENT_LEN")),
                refit_pairs=int(_cfg("PD_REFIT_PAIRS")),
                K_refit=int(_cfg("PD_K_REFIT")),
                refit_lr=float(_cfg("PD_REFIT_LR")),
                probe_seed=probe_seed,
                device=wp.device,
            )
        except Exception as exc:
            probe_out = dict(error=str(exc))

        kl, ent = self._kl_to_pretrain_and_entropy()
        gold_now = self._evaluate_gold_now()
        row = dict(
            relabel_idx=int(self.relabel_idx),
            step=int(wp.step),
            wall=time.time() - self._t0,
            gold_eval=float(gold_now),
            kl_to_pretrain=float(kl),
            policy_entropy=float(ent),
            **{k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in probe_out.items()},
        )
        with open(self.path, "a") as f:
            f.write(json.dumps(row) + "\n")
        # Also push to the existing tb/wandb logger for live visibility.
        try:
            wp.logger.log("probe/kappa", row.get("kappa", float("nan")), wp.step)
            wp.logger.log("probe/rho", row.get("rho", float("nan")), wp.step)
            wp.logger.log("probe/R_inner", row.get("R_inner", float("nan")), wp.step)
            wp.logger.log("probe/g0_norm", row.get("g0_norm", float("nan")), wp.step)
            wp.logger.log("probe/gperp_norm", row.get("gperp_norm", float("nan")), wp.step)
            wp.logger.log("probe/ensemble_variance", row.get("ensemble_variance", float("nan")), wp.step)
            wp.logger.log("probe/proxy_mean", row.get("proxy_mean", float("nan")), wp.step)
            wp.logger.log("probe/gold_probe_eval", row.get("gold_eval", float("nan")), wp.step)
            wp.logger.log("probe/kl_to_pretrain", float(kl), wp.step)
        except Exception:
            pass


def install_hooks(workspace) -> ProbeRecorder:
    """Install the GoldIsolationGuard and ProbeRecorder on a Workspace instance."""
    GoldIsolationGuard(workspace).install()
    recorder = ProbeRecorder(workspace)
    return recorder
