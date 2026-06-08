"""Sanity checks for axis2_dynamics_pg.

Two levels:
  1. Import check: confirms perf_correction_hvp.py and perf_correction_fd.py
     load and their public functions exist.
  2. Shape check: builds a tiny RM ensemble + a tiny actor on a synthetic
     state-action space and verifies that compute_g_perp_hvp() and
     compute_g_perp_fd() return tensors aligned with actor params.

The full FD-vs-HVP cosine-agreement check (cos > 0.95) requires a real
PEBBLE workspace and is expected to be performed by running
train_pg_rlhf.py with method=pg_fd and method=pg_hvp on a short job and
comparing the logged g_perp_norm values.  We don't replicate that here
because Hydra and the metaworld env are heavy dependencies.

Run:
  conda run -n bpref python -m axis2_dynamics_pg.sanity_check
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn as nn


def _tiny_actor(ds: int, da: int):
    """Mock actor returning a SquashedNormal-like distribution from
    agent.actor.  We re-use the actual DiagGaussianActor for parity."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from agent.actor import DiagGaussianActor
    return DiagGaussianActor(
        obs_dim=ds, action_dim=da, hidden_dim=32, hidden_depth=2,
        log_std_bounds=[-5, 2],
    )


def _tiny_reward_model(ds: int, da: int):
    """Mock RewardModel-like object with the attributes used by our g_perp
    code: .ensemble (list of nn.Sequential), .de, .capacity, .buffer_seg1/2,
    .buffer_label, .buffer_full, .buffer_index."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    import reward_model as reward_model_module
    # Force CPU for sanity check (reward_model defaults to 'cuda')
    reward_model_module.device = 'cpu'
    # Hack: build a tiny RewardModel with cheap settings.
    rm = reward_model_module.RewardModel(
        ds=ds, da=da,
        ensemble_size=2,
        size_segment=20,
        activation='tanh',
        lr=3e-4,
        mb_size=10,
        large_batch=2,
        label_margin=0.0,
        teacher_beta=1,
        teacher_gamma=1,
        teacher_eps_mistake=0.0,
        teacher_eps_skip=0.0,
        teacher_eps_equal=0.0,
        dormant_log_period=10**9,
        dormant_threshold=0.1,
        use_wandb=False,
        bt_log_period=10**9,
        feed_type=0,
        capacity=100,
        hidden_dim=16,
        num_layers=1,
        output_activation='tanh',
    )
    return rm


def _populate_pref_buffer(rm, n_pairs: int = 8):
    """Populate a synthetic preference buffer (no env interaction)."""
    seg_len = rm.size_segment
    ds, da = rm.ds, rm.da
    rng = np.random.default_rng(0)
    for i in range(n_pairs):
        rm.buffer_seg1[i] = rng.standard_normal((seg_len, ds + da)).astype(np.float32)
        rm.buffer_seg2[i] = rng.standard_normal((seg_len, ds + da)).astype(np.float32)
        rm.buffer_label[i] = int(rng.integers(0, 2))
    rm.buffer_index = n_pairs


def check_imports():
    from axis2_dynamics_pg import perf_correction_hvp, perf_correction_fd, monitors_fd
    for name, fn in [
        ('perf_correction_hvp.compute_g_perp_hvp',
         getattr(perf_correction_hvp, 'compute_g_perp_hvp', None)),
        ('perf_correction_hvp.apply_actor_correction',
         getattr(perf_correction_hvp, 'apply_actor_correction', None)),
        ('perf_correction_hvp.conjugate_gradient',
         getattr(perf_correction_hvp, 'conjugate_gradient', None)),
        ('perf_correction_fd.compute_g_perp_fd',
         getattr(perf_correction_fd, 'compute_g_perp_fd', None)),
        ('perf_correction_fd.compute_rho_kappa_fd',
         getattr(perf_correction_fd, 'compute_rho_kappa_fd', None)),
        ('monitors_fd.compute_rho_kappa_fd_monitor',
         getattr(monitors_fd, 'compute_rho_kappa_fd_monitor', None)),
    ]:
        assert fn is not None, f"missing: {name}"
        print(f"  OK {name}")


def check_shapes():
    ds, da = 6, 3
    actor = _tiny_actor(ds, da)
    rm = _tiny_reward_model(ds, da)
    _populate_pref_buffer(rm, n_pairs=8)
    device = 'cpu'

    # Move models to CPU for the sanity check
    for member in rm.ensemble:
        member.to(device)
    actor.to(device)

    actor_params = [p for p in actor.parameters() if p.requires_grad]
    n_actor_params = sum(p.numel() for p in actor_params)
    print(f"  actor params: {n_actor_params}")

    from axis2_dynamics_pg.perf_correction_fd import compute_g_perp_fd
    from axis2_dynamics_pg.perf_correction_hvp import compute_g_perp_hvp

    rng = np.random.default_rng(0)

    g_fd, diag_fd = compute_g_perp_fd(
        actor=actor, reward_model=rm,
        ds=ds, device=device,
        pref_batch_size=4, refit_epochs=2, rng=rng,
    )
    assert g_fd is not None, f"FD returned None; diag={diag_fd}"
    assert len(g_fd) == len(actor_params), (
        f"FD grads count {len(g_fd)} != actor params {len(actor_params)}")
    for g, p in zip(g_fd, actor_params):
        assert g.shape == p.shape, f"shape mismatch: {g.shape} vs {p.shape}"
    print(f"  OK FD g_perp shapes; norm={diag_fd['g_perp_norm']:.4g}")

    # Build synthetic trajectories for HVP
    seg_len = rm.size_segment
    trajectories = [rm.buffer_seg1[i].astype(np.float32) for i in range(4)]
    g_hvp, diag_hvp = compute_g_perp_hvp(
        actor=actor, reward_model=rm,
        trajectories=trajectories,
        ds=ds, device=device,
        pref_batch_size=4, cg_iters=3, ridge=1e-2, rng=rng,
    )
    assert g_hvp is not None, f"HVP returned None; diag={diag_hvp}"
    assert len(g_hvp) == len(actor_params), (
        f"HVP grads count {len(g_hvp)} != actor params {len(actor_params)}")
    for g, p in zip(g_hvp, actor_params):
        assert g.shape == p.shape, f"shape mismatch: {g.shape} vs {p.shape}"
    print(f"  OK HVP g_perp shapes; norm={diag_hvp['g_perp_norm']:.4g}")

    # Cosine similarity between FD and HVP (informational only — both are
    # rough estimates on a tiny synthetic dataset; this is a smoke check, not
    # a validation of agreement).
    flat_fd = torch.cat([g.reshape(-1).detach().cpu() for g in g_fd])
    flat_hvp = torch.cat([g.reshape(-1).detach().cpu() for g in g_hvp])
    nf, nh = float(torch.linalg.vector_norm(flat_fd)), float(torch.linalg.vector_norm(flat_hvp))
    if nf > 1e-12 and nh > 1e-12:
        cos = float(torch.dot(flat_fd, flat_hvp)) / (nf * nh)
        print(f"  INFO cos(FD, HVP) on synthetic data = {cos:.3f}  "
              "(no threshold; PEBBLE-level check needed for actual agreement)")
    else:
        print(f"  INFO cos undefined (||FD||={nf:.2g}, ||HVP||={nh:.2g})")


def main():
    print("[sanity_check] imports ...")
    check_imports()
    print("[sanity_check] shapes ...")
    check_shapes()
    print("[sanity_check] PASS")


if __name__ == '__main__':
    main()
