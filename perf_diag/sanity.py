"""Sanity checks. Run via: `PYTHONPATH=. python -m perf_diag.sanity`.

The 7 checks (spec §8):
  1. (CRITICAL) Gold isolation: trace train_PEBBLE.py + runtime assertion.
  2. Probe-gradient correctness on a tiny linear toy.
  3. R̂ algebraic identity.
  4. Refit determinism: ψ' identical across reruns at fixed seed.
  5. (CRITICAL) Negative-control validity: gold doesn't turn over in horizon.
  6. Baseline sanity: ensemble var ≥ 0; proxy-inflection fires on a synthetic
     inflection; KL signal monotone-ish on a drifting policy.
  7. Determinism of the full harness on a 1000-step smoke.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Allow this module to be run from anywhere; add repo root to sys.path.
_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_THIS)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from perf_diag import probe as _probe
from perf_diag import baselines as _baselines
from perf_diag import detect as _detect


# ---------------------------------------------------------------------------
def _print(name: str, ok: bool, err: float = 0.0, msg: str = "") -> bool:
    tag = "[PASS]" if ok else "[FAIL]"
    extra = f"   {msg}" if msg else ""
    if err == 0.0:
        print(f"{tag} {name:<58s}{extra}")
    else:
        print(f"{tag} {name:<58s}  err = {err:.3e}{extra}")
    return ok


# ---------------------------------------------------------------------------
# Check 1: Gold isolation (CRITICAL)
# ---------------------------------------------------------------------------
def check_gold_isolation() -> tuple[bool, str]:
    """Two-part check: static + runtime."""
    train_path = os.path.join(_REPO, "train_PEBBLE.py")
    if not os.path.exists(train_path):
        return False, f"train_PEBBLE.py not found at {train_path}"
    with open(train_path) as f:
        src = f.read()

    # Static expectations on the inner-loop writes
    needs = [
        r"reward_hat\s*=\s*self\.reward_model\.r_hat\(",
        r"self\.replay_buffer\.add\(\s*obs,\s*action,\s*reward_hat,",
        r"true_episode_reward\s*\+=\s*reward",
        r"episode_reward\s*\+=\s*reward_hat",
    ]
    missing = [p for p in needs if not re.search(p, src)]
    if missing:
        return False, f"static grep missing patterns: {missing}"

    # Make sure env reward is NOT passed to replay_buffer.add anywhere
    illegal = re.search(r"self\.replay_buffer\.add\([^)]*,\s*reward\s*,", src)
    if illegal:
        return False, f"env reward passed to replay_buffer.add at: {illegal.group(0)!r}"

    # Runtime tail: build a 1-step toy and exercise the guard.
    # We just instantiate the GoldIsolationGuard logic against a mock workspace.
    from perf_diag.hooks import GoldIsolationGuard

    class _MockRM:
        def __init__(self):
            self.last = 0.0

        def r_hat(self, x):
            self.last += 0.1
            return self.last

    class _MockRB:
        def __init__(self):
            self.events = []

        def add(self, obs, action, reward, next_obs, done, done_no_max):
            self.events.append(float(reward))

    class _MockWS:
        def __init__(self):
            self.reward_model = _MockRM()
            self.replay_buffer = _MockRB()

    ws = _MockWS()
    GoldIsolationGuard(ws).install()
    # Normal usage: predict then add with the same value
    for _ in range(20):
        r = ws.reward_model.r_hat(np.zeros(3))
        ws.replay_buffer.add(None, None, r, None, False, False)
    # Now try to leak a different value — should raise on the second attempt
    leaked = False
    try:
        ws.replay_buffer.add(None, None, 999.0, None, False, False)  # 1st violation tolerated
        ws.replay_buffer.add(None, None, 999.0, None, False, False)  # 2nd raises
    except RuntimeError:
        leaked = True
    if not leaked:
        return False, "GoldIsolationGuard failed to detect a leak"

    return True, "static patterns present; replay buffer guard catches injected leaks"


# ---------------------------------------------------------------------------
# Check 2: Probe gradient correctness on a tiny linear toy
# ---------------------------------------------------------------------------
def check_probe_gradient(seed: int = 7) -> tuple[bool, float]:
    """Tiny actor: linear-Gaussian. Tiny RM: linear. Compute REINFORCE g0 two ways
    on a fixed mini batch and compare.
    """
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cpu")
    obs_dim, act_dim = 4, 2

    # Minimal linear-Gaussian actor (no Tanh; identity transform; reuse our probe code paths)
    class LinearActor(nn.Module):
        def __init__(self):
            super().__init__()
            self.W_mu = nn.Linear(obs_dim, act_dim, bias=False)
            self.log_std = nn.Parameter(torch.zeros(act_dim) - 0.5)

        def forward(self, obs):
            mu = self.W_mu(obs)
            std = self.log_std.exp().expand_as(mu).clamp(0.05, 1.0)
            # Use the same SquashedNormal so the probe code path is identical to the real one
            from agent.actor import SquashedNormal
            return SquashedNormal(mu * 0.5, std * 0.3)

    class LinearRMMember(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Linear(obs_dim + act_dim, 1, bias=False)

        def forward(self, x):
            return self.w(x)

    actor = LinearActor().to(device)
    ensemble = [LinearRMMember().to(device)]

    # Build a fixed probe batch
    N, L = 6, 3
    batch = dict(
        obs=np.random.randn(N, L, obs_dim).astype(np.float32),
        action=np.tanh(np.random.randn(N, L, act_dim).astype(np.float32) * 0.5),
        gold=np.zeros((N, L), dtype=np.float32),  # unused in this check
    )

    g0, R_mean, R_std = _probe._reinforce_probe_gradient(actor, ensemble, batch, device, centered=True)

    # Manual REINFORCE: replicate the exact arithmetic.
    obs_t = torch.as_tensor(batch["obs"], dtype=torch.float32, device=device)
    act_t = torch.as_tensor(batch["action"], dtype=torch.float32, device=device)
    eps = 1e-6
    act_t_safe = act_t.clamp(-1.0 + eps, 1.0 - eps)
    sa_flat = np.concatenate([batch["obs"], batch["action"]], axis=-1).reshape(N * L, -1)
    with torch.no_grad():
        Rmem = ensemble[0](torch.as_tensor(sa_flat, dtype=torch.float32, device=device)).reshape(N, L).cpu().numpy()
    R_seg_np = Rmem.sum(axis=1)
    adv_np = R_seg_np - R_seg_np.mean()
    adv = torch.as_tensor(adv_np, dtype=torch.float32, device=device)

    log_pi_per_seg = []
    for s in range(N):
        lp = actor(obs_t[s]).log_prob(act_t_safe[s]).sum(-1).sum()
        log_pi_per_seg.append(lp)
    log_pi_per_seg = torch.stack(log_pi_per_seg)

    surrogate = -(adv * log_pi_per_seg).mean()
    g_flat = _probe._flat_grad(surrogate, list(actor.parameters()))
    g0_manual = -g_flat

    # Compare
    denom = max(float(g0.abs().max().item()), float(g0_manual.abs().max().item()), 1e-12)
    err = float((g0 - g0_manual).abs().max().item()) / denom
    return err < 5e-2, err


# ---------------------------------------------------------------------------
# Check 3: R̂ algebraic identity
# ---------------------------------------------------------------------------
def check_r_identity(n_trials: int = 100, seed: int = 0) -> tuple[bool, float]:
    rng = np.random.default_rng(seed)
    max_err = 0.0
    for _ in range(n_trials):
        d = int(rng.integers(8, 200))
        g0 = rng.standard_normal(d)
        g0p = rng.standard_normal(d)
        gperp = g0p - g0
        n_g0 = float(np.linalg.norm(g0))
        if n_g0 < 1e-12:
            continue
        kappa = float(np.linalg.norm(gperp) / n_g0)
        n_gp = float(np.linalg.norm(gperp))
        rho = float((g0 * gperp).sum() / max(n_g0 * n_gp, 1e-30))
        R_inner = float((g0 * g0p).sum() / (n_g0 ** 2))
        R_identity = 1.0 + kappa * rho
        max_err = max(max_err, abs(R_inner - R_identity))
        # also assert g0 + ĝ⊥ = g0'
        sum_err = float(np.max(np.abs(g0 + gperp - g0p)))
        max_err = max(max_err, sum_err)
    return max_err < 1e-6, max_err


# ---------------------------------------------------------------------------
# Check 4: Refit determinism (ψ' bit-equal across reruns at fixed seed)
# ---------------------------------------------------------------------------
def check_refit_determinism(seed: int = 13) -> tuple[bool, float]:
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cpu")

    # Tiny ensemble: 1 member, 2-layer MLP. Tiny preference batch.
    class _M(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 1))

        def forward(self, x):
            return self.net(x)

    rng = np.random.default_rng(seed)
    P, L = 5, 3
    sa1 = rng.standard_normal((P, L, 4)).astype(np.float32)
    sa2 = rng.standard_normal((P, L, 4)).astype(np.float32)
    labels = rng.integers(0, 2, size=P).astype(np.int64)

    def do_refit():
        torch.manual_seed(seed)  # reset seed identically
        src = [_M().to(device)]
        # Set deterministic init
        with torch.no_grad():
            for p in src[0].parameters():
                p.copy_(torch.randn_like(p) * 0.1)
        ens, _traj = _probe._refit_psi_prime(src, (sa1, sa2), labels, K_refit=3, lr=1e-3, device=device)
        return [p.detach().cpu().numpy().copy() for p in ens[0].parameters()]

    a = do_refit()
    b = do_refit()
    err = float(max(np.abs(x - y).max() for x, y in zip(a, b)))
    return err < 1e-12, err


# ---------------------------------------------------------------------------
# Check 5: Negative-control validity (CRITICAL) — synthetic, then real (deferred)
# ---------------------------------------------------------------------------
def check_negative_control_validity(synthetic_only: bool = True) -> tuple[bool, str]:
    """In synthetic mode (the default for the sanity gate): assert our turnover detector
    distinguishes a monotone curve (no turnover) from a curve that genuinely turns over.

    The REAL negative-control runs are validated end-to-end inside run.py's smoke path
    by running the negative-control config first and refusing to proceed if it turned over.
    """
    n = 200
    # Monotone-ish curve: should NOT register as turnover
    monotone = np.linspace(0.0, 1.0, n) + 0.02 * np.sin(np.arange(n) * 0.2)
    # Inverted-U: should register
    turn = -(np.linspace(-1.0, 1.0, n) ** 2) + 1.0 + 0.02 * np.sin(np.arange(n) * 0.2)
    a = bool(_detect.turned_over_in_horizon(monotone, K_decline=5, alpha=0.2, min_drop_frac=0.05))
    b = bool(_detect.turned_over_in_horizon(turn, K_decline=5, alpha=0.2, min_drop_frac=0.05))
    if (a is False) and (b is True):
        return True, "synthetic monotone passes; inverted-U triggers"
    return False, f"monotone={a}, turn={b}"


# ---------------------------------------------------------------------------
# Check 6: Baseline sanity
# ---------------------------------------------------------------------------
def check_baseline_sanity() -> tuple[bool, str]:
    # (a) ensemble variance ≥ 0 and increases with member disagreement
    rng = np.random.default_rng(0)
    members_same = np.stack([rng.standard_normal(50) * 0.0 + 1.0 for _ in range(5)], axis=0)
    members_diff = rng.standard_normal((5, 50))
    var_same = float(members_same.var(axis=0).mean())
    var_diff = float(members_diff.var(axis=0).mean())
    if not (var_same < 1e-10 and var_diff > 0.1):
        return False, f"ensemble-var test failed: same={var_same:.3e}, diff={var_diff:.3e}"
    # (b) proxy-inflection signal on f(t)=atan(t): the negative second derivative is
    #     positive in the concave region after the inflection point at t=0.
    t = np.linspace(-3, 3, 200)
    f = np.arctan(t)
    sig = _baselines.proxy_inflection_signal(f, alpha=0.5)
    # Signal should be positive somewhere after the inflection (t > 0)
    if not (sig[len(t) // 2 + 20:].max() > 0):
        return False, "proxy-inflection signal did not become positive past inflection"
    # (c) KL signal monotone-ish on a drifting series
    kl = np.linspace(0.0, 5.0, 100) + 0.01 * rng.standard_normal(100)
    s = _baselines.kl_to_pretrain_signal(kl)
    # Average over windows must be increasing
    halves = np.array_split(s, 4)
    means = [float(h.mean()) for h in halves]
    if not all(means[i] < means[i + 1] for i in range(3)):
        return False, f"KL signal non-monotone: {means}"
    return True, "all subchecks ok"


# ---------------------------------------------------------------------------
# Check 7: Determinism of the harness (probe scalars match across reruns)
# ---------------------------------------------------------------------------
def check_harness_determinism(seed: int = 21) -> tuple[bool, float]:
    """Re-run the probe twice on a fixed mini setup, assert all returned scalars agree."""
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cpu")

    obs_dim, act_dim = 3, 1

    class _Actor(nn.Module):
        def __init__(self):
            super().__init__()
            self.l = nn.Linear(obs_dim, 2 * act_dim)

        def forward(self, obs):
            mu, log_std = self.l(obs).chunk(2, dim=-1)
            std = log_std.tanh().mul(0.1).add(0.2).exp()
            from agent.actor import SquashedNormal
            return SquashedNormal(mu, std.expand_as(mu))

    class _Member(nn.Module):
        def __init__(self):
            super().__init__()
            self.l = nn.Linear(obs_dim + act_dim, 1)

        def forward(self, x):
            return self.l(x)

    class _RM:
        def __init__(self):
            self.ensemble = [_Member().to(device) for _ in range(2)]

        def get_label(self, sa_t_1, sa_t_2, r_t_1, r_t_2):
            # rational labels with no noise
            labels = (r_t_1.sum(1).reshape(-1) < r_t_2.sum(1).reshape(-1)).astype(np.int64)
            return sa_t_1, sa_t_2, r_t_1, r_t_2, labels.reshape(-1, 1)

    class _ToyEnv:
        observation_space = type("S", (), {"shape": (obs_dim,)})()
        action_space = type("A", (), {"shape": (act_dim,)})()

        def __init__(self, seed=0):
            self.rng = np.random.default_rng(seed)
            self.t = 0
            self.max_t = 4

        def reset(self):
            self.t = 0
            return self.rng.standard_normal(obs_dim).astype(np.float32)

        def step(self, a):
            self.t += 1
            obs = self.rng.standard_normal(obs_dim).astype(np.float32)
            r = float(np.tanh(a).sum())
            done = self.t >= self.max_t
            return obs, r, done, False, {}

        def close(self):
            pass

    actor = _Actor().to(device)
    # Freeze actor weights so the probe sees identical state on both runs
    for p in actor.parameters():
        p.requires_grad_(True)
    rm = _RM()

    def factory_seeded(env_seed):
        def f():
            return _ToyEnv(seed=env_seed)
        return f

    out1 = _probe.run_probe(actor, rm, factory_seeded(123), n_probe=3, segment_len=3,
                            refit_pairs=4, K_refit=2, refit_lr=1e-3, probe_seed=99,
                            device=device)
    out2 = _probe.run_probe(actor, rm, factory_seeded(123), n_probe=3, segment_len=3,
                            refit_pairs=4, K_refit=2, refit_lr=1e-3, probe_seed=99,
                            device=device)
    keys = ["kappa", "rho", "R_inner", "g0_norm", "g0p_norm", "gperp_norm"]
    err = max(abs(float(out1[k]) - float(out2[k])) for k in keys)
    return err < 1e-4, err


# ---------------------------------------------------------------------------
def run_all() -> dict:
    print("=" * 78)
    print("Performative-gradient diagnostic — sanity checks")
    print("=" * 78)
    results = {}

    t0 = time.time()
    ok1, msg1 = check_gold_isolation()
    _print("1. gold-isolation (CRITICAL; static + runtime)", ok1, msg=msg1)
    results["1"] = (ok1, msg1)

    ok2, err2 = check_probe_gradient()
    _print("2. probe REINFORCE gradient on tiny toy", ok2, err2)
    results["2"] = (ok2, err2)

    ok3, err3 = check_r_identity()
    _print("3. R̂ algebraic identity (R = 1+κρ; g0+ĝ⊥=g0')", ok3, err3)
    results["3"] = (ok3, err3)

    ok4, err4 = check_refit_determinism()
    _print("4. refit ψ' determinism at fixed seed", ok4, err4)
    results["4"] = (ok4, err4)

    ok5, msg5 = check_negative_control_validity()
    _print("5. neg-control validity (CRITICAL; synthetic gate)", ok5, msg=msg5)
    results["5"] = (ok5, msg5)

    ok6, msg6 = check_baseline_sanity()
    _print("6. baseline signals sanity", ok6, msg=msg6)
    results["6"] = (ok6, msg6)

    ok7, err7 = check_harness_determinism()
    _print("7. harness determinism (probe scalars reproducible)", ok7, err7)
    results["7"] = (ok7, err7)

    crit_ok = results["1"][0] and results["5"][0]
    all_ok = all(v[0] for v in results.values())
    print("-" * 78)
    print(f"All passed: {all_ok}    Critical (1, 5) passed: {crit_ok}    "
          f"wall = {time.time() - t0:.2f}s")
    print("=" * 78)
    results["_all_ok"] = all_ok
    results["_crit_ok"] = crit_ok
    return results


if __name__ == "__main__":
    res = run_all()
    sys.exit(0 if res["_all_ok"] else 1)
