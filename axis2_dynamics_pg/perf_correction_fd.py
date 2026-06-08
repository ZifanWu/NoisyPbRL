"""Finite-difference approximation of g_perp.

g_perp_FD = g_score(ψ_fresh) - g_score(ψ_stale)

where g_score(ψ) is the REINFORCE-style policy gradient of E_π[r_ψ(τ)],
evaluated on a fixed batch of preference segments via the score-function
estimator:
    g_score(ψ) = ∂/∂θ_actor [ Σ_i coeff_i(ψ) · (logπ(seg1_i) + logπ(seg2_i)) ]
with coeff_i(ψ) = (Σ_t r_ψ(seg1_{i,t}) + Σ_t r_ψ(seg2_{i,t})) / N.

ψ_fresh is obtained by snapshotting ψ_stale, running a few BT training epochs
on the current preference buffer, computing g_score, then restoring ψ_stale.

Shares the score-grad / pref-batch / mean-reward primitives with
perf_correction_hvp.py so the two outputs are sign-compatible (both vectors
live in actor-parameter space and add to the standard SAC actor step).
"""
from typing import List, Optional, Tuple

import copy
import numpy as np
import torch

from axis2_dynamics_pg.perf_correction_hvp import (
    _mean_reward_output,
    _sample_preference_batch,
    _actor_score_grads,
    _params_to_list,
)


def _coeff_from_rewards(reward_model, seg1, seg2, device) -> torch.Tensor:
    """coeff[i] = (Σ_t r_ψ(seg1_i,t) + Σ_t r_ψ(seg2_i,t)) / N."""
    with torch.no_grad():
        r1 = _mean_reward_output(reward_model, seg1, device).sum(dim=1).squeeze(-1)
        r2 = _mean_reward_output(reward_model, seg2, device).sum(dim=1).squeeze(-1)
    n = max(1, r1.shape[0])
    return (r1 + r2) / float(n)


def compute_g_score(actor, reward_model, seg1, seg2, ds: int,
                     device) -> List[torch.Tensor]:
    """REINFORCE-style policy gradient on a fixed preference batch."""
    coeff = _coeff_from_rewards(reward_model, seg1, seg2, device)
    return _actor_score_grads(actor, seg1, seg2, ds, coeff)


def _diff_grads(grads_a: List[torch.Tensor],
                 grads_b: List[torch.Tensor]) -> List[torch.Tensor]:
    return [a - b for a, b in zip(grads_a, grads_b)]


def _grad_norm(grads: List[torch.Tensor]) -> float:
    if not grads:
        return 0.0
    flat = torch.cat([g.reshape(-1).detach().cpu() for g in grads])
    return float(torch.linalg.vector_norm(flat))


def _grad_cosine(grads_a: List[torch.Tensor],
                  grads_b: List[torch.Tensor]) -> float:
    if not grads_a or not grads_b:
        return float('nan')
    fa = torch.cat([g.reshape(-1).detach().cpu() for g in grads_a])
    fb = torch.cat([g.reshape(-1).detach().cpu() for g in grads_b])
    na = float(torch.linalg.vector_norm(fa))
    nb = float(torch.linalg.vector_norm(fb))
    if na < 1e-12 or nb < 1e-12:
        return float('nan')
    return float(torch.dot(fa, fb)) / (na * nb)


def snapshot_ensemble_state(reward_model) -> List[dict]:
    """Deep-copied state_dicts for each ensemble member, on CPU to save GPU
    memory (we move back when restoring)."""
    return [
        {k: v.detach().clone() for k, v in m.state_dict().items()}
        for m in reward_model.ensemble
    ]


def restore_ensemble_state(reward_model, snapshot: List[dict]) -> None:
    for member, sd in zip(reward_model.ensemble, snapshot):
        member.load_state_dict(sd)


def refit_ensemble_in_place(reward_model, num_epochs: int) -> None:
    """Run `num_epochs` calls of train_reward() on the current preference
    buffer.  Modifies reward_model in place — caller must snapshot first.

    NOTE: train_reward() iterates over the buffer once per call (using
    self.train_batch_size).  So num_epochs controls total epoch count.
    """
    for _ in range(int(num_epochs)):
        reward_model.train_reward()


def compute_g_perp_fd(
    actor,
    reward_model,
    ds: int,
    device,
    pref_batch_size: int = 256,
    refit_epochs: int = 10,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[Optional[List[torch.Tensor]], dict]:
    """Returns (g_perp_FD list-of-tensors aligned with actor params, diagnostics).

    Internally:
      1. snapshot stale ensemble state
      2. sample preference batch (seg1, seg2, _)
      3. g_score_stale = score_grad(ψ_stale on this batch)
      4. refit ensemble in place for `refit_epochs` epochs → ψ_fresh
      5. g_score_fresh = score_grad(ψ_fresh on same batch)
      6. restore stale ensemble state
      7. g_perp_FD = g_score_fresh - g_score_stale

    Caller should ADD the returned grads to actor.grad (matching HVP sign).
    """
    if rng is None:
        rng = np.random.default_rng(0)

    seg1, seg2, _ = _sample_preference_batch(reward_model, pref_batch_size, rng)
    if seg1 is None:
        return None, {"applied": 0.0, "reason_no_pref_batch": 1.0}

    # ---- stale g_score ---------------------------------------------------
    g_score_stale = compute_g_score(actor, reward_model, seg1, seg2, ds, device)

    # ---- snapshot, refit fresh, fresh g_score, restore -------------------
    snapshot = snapshot_ensemble_state(reward_model)
    try:
        refit_ensemble_in_place(reward_model, refit_epochs)
        g_score_fresh = compute_g_score(actor, reward_model, seg1, seg2, ds, device)
    finally:
        restore_ensemble_state(reward_model, snapshot)

    g_perp_fd = _diff_grads(g_score_fresh, g_score_stale)

    diagnostics = {
        "applied": 1.0,
        "pref_batch": float(seg1.shape[0]),
        "g_score_stale_norm": _grad_norm(g_score_stale),
        "g_score_fresh_norm": _grad_norm(g_score_fresh),
        "g_perp_norm": _grad_norm(g_perp_fd),
        "cos_stale_fresh": _grad_cosine(g_score_stale, g_score_fresh),
    }
    return g_perp_fd, diagnostics


def compute_rho_kappa_fd(
    actor,
    reward_model,
    ds: int,
    device,
    pref_batch_size: int = 256,
    refit_epochs: int = 10,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """ρ_FD = cos(g0, g_perp_FD), κ_FD = ‖g_perp_FD‖ / ‖g0‖,
    where g0 := g_score_stale (REINFORCE estimator under current RM).

    Returns dict with rho_fd, kappa_fd, g0_norm, g_perp_norm,
    cos_stale_fresh, plus diagnostic fields.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    seg1, seg2, _ = _sample_preference_batch(reward_model, pref_batch_size, rng)
    if seg1 is None:
        return {"rho_fd": float('nan'), "kappa_fd": float('nan'),
                "applied": 0.0, "reason_no_pref_batch": 1.0}

    g0 = compute_g_score(actor, reward_model, seg1, seg2, ds, device)

    snapshot = snapshot_ensemble_state(reward_model)
    try:
        refit_ensemble_in_place(reward_model, refit_epochs)
        g_fresh = compute_g_score(actor, reward_model, seg1, seg2, ds, device)
    finally:
        restore_ensemble_state(reward_model, snapshot)

    g_perp = _diff_grads(g_fresh, g0)

    g0_norm = _grad_norm(g0)
    gperp_norm = _grad_norm(g_perp)
    rho_fd = _grad_cosine(g0, g_perp)
    kappa_fd = (gperp_norm / g0_norm) if g0_norm > 1e-12 else float('nan')
    cos_sf = _grad_cosine(g0, g_fresh)

    return {
        "rho_fd": rho_fd,
        "kappa_fd": kappa_fd,
        "g0_norm": g0_norm,
        "g_perp_norm": gperp_norm,
        "cos_stale_fresh": cos_sf,
        "pref_batch": float(seg1.shape[0]),
        "applied": 1.0,
    }
