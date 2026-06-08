#!/usr/bin/env python3
"""PG-RLHF experiment.

Compares Standard SAC vs PG-RLHF SAC on the same Tier B mis-condition env.
The PG correction g_perp is computed via either finite-difference (FD) or
HVP+CG (HVP), and is added to the actor parameters *after* each standard SAC
actor step (matching perfg_gauge's apply_actor_correction convention).

g_perp is cached and only refreshed every `pg_fresh_refit_every` actor
updates to keep cost bounded.

Run:
  conda run -n bpref python train_pg_rlhf.py method=standard seed=1
  conda run -n bpref python train_pg_rlhf.py method=pg_fd seed=1
  conda run -n bpref python train_pg_rlhf.py method=pg_hvp seed=1
"""
import os
import sys
import csv


def _reexec_with_mujoco_library_path():
    if os.environ.get('NOISYPBRL_MUJOCO_LD_READY') == '1':
        return
    candidates = [
        os.path.expanduser('~/.mujoco/mujoco210/bin'),
        '/usr/lib/nvidia',
    ]
    parts = [p for p in os.environ.get('LD_LIBRARY_PATH', '').split(':') if p]
    changed = False
    for candidate in candidates:
        if os.path.isdir(candidate) and candidate not in parts:
            parts.append(candidate)
            changed = True
    os.environ['NOISYPBRL_MUJOCO_LD_READY'] = '1'
    if changed:
        os.environ['LD_LIBRARY_PATH'] = ':'.join(parts)
        os.execv(sys.executable, [sys.executable] + sys.argv)


_reexec_with_mujoco_library_path()

import numpy as np
import torch
import hydra

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_PEBBLE import Workspace as PebbleWorkspace
from reward_model import RewardModel

from axis2_dynamics_pg.perf_correction_hvp import (
    compute_g_perp_hvp,
    apply_actor_correction,
)
from axis2_dynamics_pg.perf_correction_fd import compute_g_perp_fd


_VALID_METHODS = {'standard', 'pg_fd', 'pg_hvp'}


class Workspace(PebbleWorkspace):
    def __init__(self, cfg):
        super().__init__(cfg)
        if hasattr(self.agent, 'critic'):
            self.agent.critic.wandb_dormant_logging = False

        rm_hidden_dim = getattr(cfg, 'rm_hidden_dim', 256)
        rm_num_layers = getattr(cfg, 'rm_num_layers', 3)
        rm_output_activation = getattr(cfg, 'rm_output_activation', 'tanh')

        self.reward_model = RewardModel(
            self.env.observation_space.shape[0],
            self.env.action_space.shape[0],
            ensemble_size=cfg.ensemble_size,
            size_segment=cfg.segment,
            activation=cfg.activation,
            lr=cfg.reward_lr,
            mb_size=cfg.reward_batch,
            large_batch=cfg.large_batch,
            label_margin=cfg.label_margin,
            teacher_beta=cfg.teacher_beta,
            teacher_gamma=cfg.teacher_gamma,
            teacher_eps_mistake=cfg.teacher_eps_mistake,
            teacher_eps_skip=cfg.teacher_eps_skip,
            teacher_eps_equal=cfg.teacher_eps_equal,
            dormant_log_period=cfg.dormant_log_period,
            dormant_threshold=cfg.dormant_threshold,
            use_wandb=False,
            bt_log_period=cfg.bt_log_period,
            feed_type=cfg.feed_type,
            capacity=cfg.max_feedback * cfg.large_batch,
            hidden_dim=rm_hidden_dim,
            num_layers=rm_num_layers,
            output_activation=rm_output_activation,
            buffer_window_rounds=cfg.buffer_window_rounds if cfg.buffer_window_rounds > 0 else None,
            bt_normalize_by_gap_std=getattr(cfg, 'bt_normalize_by_gap_std', False),
            bt_gap_ema_alpha=getattr(cfg, 'bt_gap_ema_alpha', 0.9),
        )

        # ---- PG-RLHF config ----
        self._method = str(getattr(cfg, 'method', 'standard'))
        if self._method not in _VALID_METHODS:
            raise ValueError(
                f"method={self._method} invalid; expected one of {_VALID_METHODS}")
        self._pg_refit_every = int(getattr(cfg, 'pg_fresh_refit_every', 1000))
        self._pg_pref_batch = int(getattr(cfg, 'pg_pref_batch_size', 128))
        self._pg_fd_refit_epochs = int(getattr(cfg, 'pg_fd_refit_epochs', 10))
        self._pg_hvp_ridge = float(getattr(cfg, 'pg_hvp_ridge', 0.01))
        self._pg_hvp_cg_iters = int(getattr(cfg, 'pg_hvp_cg_iters', 10))
        self._pg_corr_scale = float(getattr(cfg, 'pg_correction_scale', 1.0))
        # Trust-region clip on ‖g_perp‖ before applying (the score-function
        # estimator can produce gradients ~ batch × horizon × |coeff| which
        # blows up the actor; this caps the per-step actor displacement).
        self._pg_clip_norm = float(getattr(cfg, 'pg_clip_norm', 1.0))

        self._pg_rng = np.random.default_rng(cfg.seed + 1234)
        self._cached_g_perp = None
        self._actor_updates_since_refresh = 10**9   # force refresh on first call
        self._total_actor_updates = 0

        # CSV log
        self._pg_csv_path = os.path.join(self.work_dir, 'pg_rlhf_metrics.csv')
        self._pg_csv_fields = [
            'step', 'actor_updates', 'method',
            'refreshed', 'g_perp_norm', 'apply_norm',
            'u_base_norm', 'w_base_norm', 'cos_stale_fresh',
        ]
        with open(self._pg_csv_path, 'w', newline='') as f:
            csv.writer(f).writerow(self._pg_csv_fields)

        # Install actor-update wrapper.
        # We can't subclass the agent (it's already constructed by base class),
        # so we monkey-patch update_actor_and_alpha to add the correction
        # AFTER the standard SAC actor step.
        self._install_actor_wrapper()

    # ------------------------------------------------------------------
    # Actor-update wrapper
    # ------------------------------------------------------------------

    def _install_actor_wrapper(self):
        agent = self.agent
        original_update_actor = agent.update_actor_and_alpha
        workspace = self

        def wrapped_update_actor(obs, logger, step, print_flag=False):
            # Standard SAC step (computes g0, takes one actor + log_alpha step).
            original_update_actor(obs, logger, step, print_flag=print_flag)
            workspace._total_actor_updates += 1

            if workspace._method == 'standard':
                return

            workspace._actor_updates_since_refresh += 1
            refreshed = False
            diag = {}
            if workspace._actor_updates_since_refresh >= workspace._pg_refit_every:
                # Recompute g_perp on a fresh preference batch.
                try:
                    g_perp, diag = workspace._compute_g_perp()
                    if g_perp is not None:
                        workspace._cached_g_perp = g_perp
                        workspace._actor_updates_since_refresh = 0
                        refreshed = True
                except Exception as err:
                    print(f"[pg step={workspace.step}] g_perp refresh failed: {err}")

            apply_norm = 0.0
            if workspace._cached_g_perp is not None:
                scaled = workspace._scale_and_clip(workspace._cached_g_perp)
                apply_norm = apply_actor_correction(
                    workspace.agent, scaled, actor_lr=None
                )

            workspace._log_pg_row(refreshed=refreshed, diag=diag,
                                   apply_norm=apply_norm)

        agent.update_actor_and_alpha = wrapped_update_actor

    def _scale_and_clip(self, grads):
        """Apply pg_correction_scale, then clip overall L2 norm to
        pg_clip_norm if it exceeds the threshold.  Returns same-shape list."""
        import torch
        scaled = [self._pg_corr_scale * g for g in grads]
        if self._pg_clip_norm <= 0.0:
            return scaled
        flat = torch.cat([g.reshape(-1) for g in scaled])
        norm = float(torch.linalg.vector_norm(flat))
        if norm > self._pg_clip_norm and norm > 1e-12:
            f = self._pg_clip_norm / norm
            scaled = [g * f for g in scaled]
        return scaled

    def _compute_g_perp(self):
        ds = self.env.observation_space.shape[0]
        device = str(self.device)
        if self._method == 'pg_fd':
            return compute_g_perp_fd(
                actor=self.agent.actor,
                reward_model=self.reward_model,
                ds=ds,
                device=device,
                pref_batch_size=self._pg_pref_batch,
                refit_epochs=self._pg_fd_refit_epochs,
                rng=self._pg_rng,
            )
        if self._method == 'pg_hvp':
            # Build "trajectories" for u_base = ∂/∂φ [mean_τ Σ reward_φ(τ)] from
            # the preference buffer: just use the seg1 segments.  This is a
            # crude proxy for on-policy trajectories but matches what perfg_gauge
            # does in spirit (they use the preference buffer too).
            max_len = self.reward_model.capacity if self.reward_model.buffer_full else self.reward_model.buffer_index
            n_trajs = min(64, max_len)
            if n_trajs <= 0:
                return None, {"applied": 0.0, "reason_no_buffer": 1.0}
            idxs = self._pg_rng.choice(max_len, size=n_trajs, replace=False)
            trajectories = [self.reward_model.buffer_seg1[i].astype(np.float32)
                            for i in idxs]
            return compute_g_perp_hvp(
                actor=self.agent.actor,
                reward_model=self.reward_model,
                trajectories=trajectories,
                ds=ds,
                device=device,
                pref_batch_size=self._pg_pref_batch,
                cg_iters=self._pg_hvp_cg_iters,
                ridge=self._pg_hvp_ridge,
                rng=self._pg_rng,
            )
        return None, {"applied": 0.0, "reason_unknown_method": 1.0}

    def _log_pg_row(self, refreshed: bool, diag: dict, apply_norm: float):
        row = {
            'step': self.step,
            'actor_updates': self._total_actor_updates,
            'method': self._method,
            'refreshed': 1 if refreshed else 0,
            'g_perp_norm': diag.get('g_perp_norm', float('nan')),
            'apply_norm': apply_norm,
            'u_base_norm': diag.get('u_base_norm', float('nan')),
            'w_base_norm': diag.get('w_base_norm', float('nan')),
            'cos_stale_fresh': diag.get('cos_stale_fresh', float('nan')),
        }
        # Only write a CSV row on refresh OR every 1000 actor updates to keep
        # the log small.
        if not refreshed and self._total_actor_updates % 1000 != 0:
            return
        with open(self._pg_csv_path, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=self._pg_csv_fields).writerow(row)
        if refreshed:
            if getattr(self.cfg, 'use_wandb', False):
                try:
                    import wandb
                    wandb.log({f'pg/{k}': v for k, v in row.items()
                               if k not in ('step', 'method')}, step=self.step)
                except Exception:
                    pass
            print(f"[pg step={self.step} method={self._method} updates={self._total_actor_updates}] "
                  f"REFRESH g_perp_norm={row['g_perp_norm']:.4g} "
                  f"apply_norm={apply_norm:.4g}")

    # Inherit run() from PebbleWorkspace — the actor wrapper transparently
    # adds the g_perp correction inside.


@hydra.main(config_path='config/train_pg_rlhf.yaml', strict=False)
def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()


if __name__ == '__main__':
    main()
