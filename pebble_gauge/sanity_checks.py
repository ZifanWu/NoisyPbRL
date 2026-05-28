import os
import sys
from typing import Dict, List, Optional
from types import SimpleNamespace

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import reward_model as reward_model_module
from agent.actor import DiagGaussianActor
from pebble_gauge.perf_correction import apply_actor_correction, compute_performative_correction
from pebble_gauge.reference_dataset import _reset_env, _step_env
from pebble_gauge.reward_model_gauge import GaugedRewardModel, make_reward_model


def _make_small_reward_model(final_activation):
    reward_model_module.device = "cpu"
    return make_reward_model(
        2,
        1,
        final_activation=final_activation,
        ensemble_size=1,
        lr=1e-3,
        mb_size=4,
        size_segment=3,
        capacity=20,
        large_batch=1,
        label_margin=0.0,
        use_wandb=False,
    )


def check_h1_algebraic_invariant():
    """H1 check runnable without a live environment.

    Verifies that for any (obs, action) trajectory the per-step difference
    between gauge outputs is exactly the stored shift:

        r_none(s,a) - r_g(s,a) == shift[g]   for every (s,a) and gauge g

    and therefore the cumulative proxy-return difference over T steps equals
    shift[g] * T to floating-point precision.  This is the core algebraic
    property that the full H1 end-of-run check also tests against a live env.
    """
    rng = np.random.RandomState(42)

    for final_activation in ("tanh", "identity"):
        base = _make_small_reward_model(final_activation)
        base_gauge = "no_tanh" if final_activation == "identity" else "none"
        gauged = GaugedRewardModel(base, base_gauge, n_shift_sample=4, seed=0)
        gauges = gauged.family_gauges()

        # Set known, non-trivial shifts on the non-base gauges.
        for g in gauges:
            gauged.shifts[g] = 0.0
        gauged.shifts[gauges[1]] = 0.314      # mean_buf / no_tanh_mean_buf
        gauged.shifts[gauges[2]] = -0.271     # mean_ref / no_tanh_mean_ref

        for T in (1, 17, 47, 1000):
            trajectory = rng.normal(size=(T, 3)).astype(np.float32)  # ds=2, da=1
            proxy = {g: 0.0 for g in gauges}
            for step in range(T):
                sa = trajectory[step]
                for g in gauges:
                    proxy[g] += float(gauged.r_hat_for_gauge(sa, g))

            for g in gauges[1:]:
                shift = gauged.get_shift(g)
                expected_diff = shift * T
                actual_diff = proxy[gauges[0]] - proxy[g]
                err = abs(actual_diff - expected_diff)
                assert err < 1e-3, (
                    "H1 algebraic invariant failed: activation={} gauge={} T={} "
                    "expected_diff={:.6f} actual_diff={:.6f} err={:.2e}".format(
                        final_activation, g, T, expected_diff, actual_diff, err
                    )
                )


def check_final_activation():
    tanh_model = _make_small_reward_model("tanh")
    identity_model = _make_small_reward_model("identity")
    assert isinstance(tanh_model.ensemble[0][-1], torch.nn.Tanh)
    assert not isinstance(identity_model.ensemble[0][-1], torch.nn.Tanh)


def check_gauge_algebra():
    base = _make_small_reward_model("identity")
    gauged = GaugedRewardModel(base, "no_tanh_mean_buf", n_shift_sample=4, seed=0)
    x = np.random.RandomState(0).normal(size=(5, 3)).astype(np.float32)
    gauged.shifts["no_tanh_mean_buf"] = 1.25
    raw = base.r_hat_batch(x)
    shifted = gauged.r_hat_batch(x)
    assert np.allclose(raw - shifted, 1.25)


def check_perf_correction_shapes():
    rng = np.random.RandomState(1)
    base = _make_small_reward_model("identity")
    n = 8
    base.buffer_seg1[:n] = rng.uniform(-0.5, 0.5, size=(n, 3, 3)).astype(np.float32)
    base.buffer_seg2[:n] = rng.uniform(-0.5, 0.5, size=(n, 3, 3)).astype(np.float32)
    base.buffer_label[:n] = rng.randint(0, 2, size=(n, 1)).astype(np.float32)
    base.buffer_index = n
    gauged = GaugedRewardModel(base, "no_tanh", seed=2)

    actor = DiagGaussianActor(obs_dim=2, action_dim=1, hidden_dim=8, hidden_depth=1, log_std_bounds=[-5, 2])
    agent = SimpleNamespace(actor=actor, actor_optimizer=torch.optim.Adam(actor.parameters(), lr=1e-4))
    trajectories = [rng.uniform(-0.5, 0.5, size=(3, 3)).astype(np.float32) for _ in range(2)]

    grads, diagnostics = compute_performative_correction(
        agent,
        gauged,
        trajectories,
        rng=rng,
        horizon=3,
        pref_batch_size=4,
        cg_iters=2,
        ridge=1e-2,
    )
    assert grads is not None
    assert diagnostics["applied"] == 1.0
    norm = apply_actor_correction(agent, grads, actor_lr=1e-5)
    assert np.isfinite(norm)


def run_rrm_bit_identity_check(
    reward_model: GaugedRewardModel,
    agent,
    env,
    n_episodes: int = 20,
    tol: float = 1e-4,
) -> Dict:
    """H1 sanity check for RRM runs.

    For each evaluation episode the same policy produces the same (obs, action) trajectory
    regardless of gauge.  This function collects episodes once while accumulating per-gauge
    proxy sums, then verifies that every pair of gauges satisfies

        proxy_return[g_base] - proxy_return[g_other] = shift[g_other] * episode_steps

    to within `tol`.  If this fails, a gauge shift is being applied inconsistently or
    the shift itself is computed incorrectly.

    Returns a dict with keys:
        passed      bool
        failures    list of dicts describing each failed episode/gauge pair
        n_episodes  int
        gauges      tuple of gauge names checked
        mean_true_return  float
    """
    gauges = reward_model.family_gauges()
    base_gauge = gauges[0]

    import utils  # deferred: pulls in dmc2gym which may not be present in unit-test environments

    proxy_sums: Dict[str, List[float]] = {g: [] for g in gauges}
    episode_steps: List[int] = []
    true_returns: List[float] = []

    for _ep in range(n_episodes):
        obs = _reset_env(env)
        agent.reset()
        done = False
        ep_true = 0.0
        ep_proxy = {g: 0.0 for g in gauges}
        steps = 0

        while not done:
            with utils.eval_mode(agent):
                action = agent.act(obs, sample=False)
            sa = np.concatenate([obs, action], axis=-1).astype(np.float32)
            next_obs, reward, done, _info = _step_env(env, action)

            for g in gauges:
                ep_proxy[g] += float(reward_model.r_hat_for_gauge(sa, g))
            ep_true += float(reward)
            steps += 1
            obs = next_obs

        true_returns.append(ep_true)
        episode_steps.append(steps)
        for g in gauges:
            proxy_sums[g].append(ep_proxy[g])

    failures = []
    for g_other in gauges[1:]:
        shift = reward_model.get_shift(g_other)
        for ep_idx in range(n_episodes):
            expected_diff = shift * episode_steps[ep_idx]
            actual_diff = proxy_sums[base_gauge][ep_idx] - proxy_sums[g_other][ep_idx]
            err = abs(actual_diff - expected_diff)
            if err > tol:
                failures.append({
                    "gauge_pair": (base_gauge, g_other),
                    "episode": ep_idx,
                    "expected_diff": expected_diff,
                    "actual_diff": actual_diff,
                    "error": err,
                    "shift": shift,
                    "episode_steps": episode_steps[ep_idx],
                })

    return {
        "passed": len(failures) == 0,
        "failures": failures,
        "n_episodes": n_episodes,
        "gauges": gauges,
        "mean_true_return": float(np.mean(true_returns)) if true_returns else 0.0,
    }


def main():
    check_final_activation()
    check_gauge_algebra()
    check_h1_algebraic_invariant()
    check_perf_correction_shapes()
    print("pebble_gauge sanity checks passed")


if __name__ == "__main__":
    main()

