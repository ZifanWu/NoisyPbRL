from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple  # noqa: F401

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Pylance does not re-export nn.Parameter from the stub, but it is a public
# runtime class.  Import it directly so type annotations work without lint noise.
_Parameter = nn.parameter.Parameter


@dataclass
class PerfCorrectionConfig:
    pref_subsample: int = 32
    replay_subsample: int = 256
    shift_subsample: int = 2048
    cg_iter: int = 5
    eps_ridge: float = 1e-3
    horizon: float = 1000.0
    max_grad_norm: float = 100.0


# ── Utility helpers ──────────────────────────────────────────────────────────


def _flatten_tensors(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    if not tensors:
        return torch.zeros(1)
    return torch.cat([t.reshape(-1) for t in tensors])



def _zeros_like_actor_params(actor) -> Dict[_Parameter, torch.Tensor]:
    return {p: torch.zeros_like(p) for p in actor.parameters()}


def grad_l2_norm(grad_map: Dict[_Parameter, torch.Tensor]) -> float:
    total = 0.0
    for g in grad_map.values():
        total += float(torch.sum(g * g).item())
    return total ** 0.5


def combine_actor_grads(
    base: Dict[_Parameter, torch.Tensor],
    shift: Dict[_Parameter, torch.Tensor],
) -> Dict[_Parameter, torch.Tensor]:
    return {p: base[p] + shift[p] for p in base}


# ── Preference buffer helpers ────────────────────────────────────────────────


def _sample_preference_indices(reward_model, n: int) -> np.ndarray:
    max_len = reward_model.capacity if reward_model.buffer_full else reward_model.buffer_index
    if max_len <= 0:
        return np.array([], dtype=np.int64)
    size = min(int(n), int(max_len))
    return np.random.choice(max_len, size=size, replace=False)


def _bt_loss_subset(reward_model, params: Sequence[_Parameter], idxs: np.ndarray) -> torch.Tensor:
    """BT cross-entropy loss averaged over ensemble members for the indexed preference pairs."""
    if len(idxs) == 0:
        return torch.zeros((), device=params[0].device if params else "cpu")
    device = params[0].device
    loss = torch.zeros((), device=device)
    labels_np = reward_model.buffer_label[idxs].flatten().astype(np.int64)
    labels = torch.from_numpy(np.clip(labels_np, 0, 1)).long().to(device)
    for member in range(reward_model.de):
        seg1 = reward_model.buffer_seg1[idxs]
        seg2 = reward_model.buffer_seg2[idxs]
        r1 = reward_model.r_hat_member(seg1, member=member).sum(axis=1)
        r2 = reward_model.r_hat_member(seg2, member=member).sum(axis=1)
        logits = torch.cat([r1, r2], dim=-1)
        loss = loss + F.cross_entropy(logits, labels)
    return loss / max(1, reward_model.de)


def _bt_grad_flat_detached(
    reward_model,
    params: Sequence[_Parameter],
    idxs: np.ndarray,
) -> torch.Tensor:
    """Flattened gradient of the BT loss w.r.t. RM params for the given indices, detached."""
    loss = _bt_loss_subset(reward_model, params, idxs)
    if not loss.requires_grad:
        return torch.zeros(sum(p.numel() for p in params),
                           device=params[0].device if params else "cpu")
    grads = torch.autograd.grad(loss, params, retain_graph=False, allow_unused=True)
    grads = [torch.zeros_like(p) if g is None else g for g, p in zip(grads, params)]
    return _flatten_tensors(grads).detach()


def _bt_grad_per_sample_dots(
    reward_model,
    params: Sequence[_Parameter],
    pref_idxs: np.ndarray,
    directions: Sequence[torch.Tensor],
) -> List[torch.Tensor]:
    """For each preference pair i compute dot(d, ∇_φ ℓ_BT^{(i)}) for every direction d.

    Returns a list of len(directions) tensors of shape (N,).  Requires N backward
    passes through the RM (one per pair); all dot products per pass are O(|φ|) and
    share the same gradient vector.
    """
    N = len(pref_idxs)
    if N == 0 or not directions:
        return [torch.zeros(0) for _ in directions]
    device = directions[0].device
    results = [torch.zeros(N, device=device) for _ in directions]
    for i, idx in enumerate(pref_idxs):
        g_i = _bt_grad_flat_detached(reward_model, params, np.array([idx]))
        for j, d in enumerate(directions):
            results[j][i] = torch.dot(d, g_i)
    return results


# ── HVP and conjugate gradient ───────────────────────────────────────────────


def _hvp(
    reward_model,
    params: Sequence[_Parameter],
    idxs: np.ndarray,
    vec: torch.Tensor,
    eps_ridge: float,
) -> torch.Tensor:
    """Hessian-vector product (BT loss Hessian) + ridge regularisation."""
    loss = _bt_loss_subset(reward_model, params, idxs)
    grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
    flat_grads = _flatten_tensors(grads)
    dot = torch.dot(flat_grads, vec)
    hvp_parts = torch.autograd.grad(dot, params, retain_graph=False, allow_unused=True)
    hvp_parts = [torch.zeros_like(p) if g is None else g for g, p in zip(hvp_parts, params)]
    return _flatten_tensors(hvp_parts) + eps_ridge * vec


def _conjugate_gradient(Avp, b: torch.Tensor, n_iter: int = 10, tol: float = 1e-10) -> torch.Tensor:
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rdotr = torch.dot(r, r)
    for _ in range(int(n_iter)):
        Ap = Avp(p)
        if not torch.isfinite(Ap).all():
            return torch.zeros_like(b)
        denom = torch.dot(p, Ap) + 1e-12
        alpha = rdotr / denom
        if not torch.isfinite(alpha):
            return torch.zeros_like(b)
        x = x + alpha * p
        r = r - alpha * Ap
        if not torch.isfinite(r).all():
            return torch.zeros_like(b)
        new_rdotr = torch.dot(r, r)
        if new_rdotr.item() < tol:
            break
        beta = new_rdotr / (rdotr + 1e-12)
        p = r + beta * p
        rdotr = new_rdotr
    return x


# ── Gradient helpers ─────────────────────────────────────────────────────────


def _forward_member_reward_objective(
    reward_model,
    sa_batch: np.ndarray,
    member: int,
    use_no_tanh: bool = False,
) -> torch.Tensor:
    """Differentiable RM reward used by the policy objective correction.

    Preference training still uses RewardModel.r_hat_member() and therefore the
    learned BT likelihood with the final tanh.  The no_tanh gauges change the
    policy objective reward, so only the objective-side u and shift derivatives
    bypass the final tanh here.
    """
    if not use_no_tanh:
        return reward_model.r_hat_member(sa_batch, member=member)

    params = reward_model.get_rm_parameters()
    device = params[0].device if params else "cpu"
    x = torch.from_numpy(np.asarray(sa_batch, dtype=np.float32)).float().to(device)
    model = reward_model.ensemble[member]
    if len(model) > 0 and isinstance(model[-1], nn.Tanh):
        out = x
        for layer in model[:-1]:
            out = layer(out)
        return out
    return model(x)


def _grad_phi_mean_nn(
    reward_model,
    sa_batch: np.ndarray,
    use_no_tanh: bool = False,
) -> torch.Tensor:
    """grad_φ E_{(s,a)~batch}[r_φ(s,a)] — the 'u' vector for the CG right-hand side."""
    params = reward_model.get_rm_parameters()
    if len(sa_batch) == 0:
        return torch.zeros(sum(p.numel() for p in params), device=params[0].device)
    mean_val = torch.zeros((), device=params[0].device)
    for member in range(reward_model.de):
        pred = _forward_member_reward_objective(
            reward_model, sa_batch, member=member, use_no_tanh=use_no_tanh
        )
        mean_val = mean_val + pred.mean()
    mean_val = mean_val / max(1, reward_model.de)
    grads = torch.autograd.grad(mean_val, params, retain_graph=False, allow_unused=True)
    grads = [torch.zeros_like(p) if g is None else g for g, p in zip(grads, params)]
    return _flatten_tensors(grads).detach()


def _grad_phi_mean_traj_return(
    reward_model,
    trajectories: Sequence[np.ndarray],
    use_no_tanh: bool = False,
) -> torch.Tensor:
    """grad_phi E_{tau~pi}[sum_t r_phi(s_t,a_t)] for sampled on-policy trajectories."""
    params = reward_model.get_rm_parameters()
    valid = [np.asarray(traj, dtype=np.float32) for traj in trajectories if len(traj) > 0]
    if not valid:
        return torch.zeros(sum(p.numel() for p in params), device=params[0].device)

    mean_return = torch.zeros((), device=params[0].device)
    for member in range(reward_model.de):
        member_total = torch.zeros((), device=params[0].device)
        for traj in valid:
            pred = _forward_member_reward_objective(
                reward_model, traj, member=member, use_no_tanh=use_no_tanh
            )
            member_total = member_total + pred.sum()
        mean_return = mean_return + member_total / float(len(valid))
    mean_return = mean_return / max(1, reward_model.de)
    grads = torch.autograd.grad(mean_return, params, retain_graph=False, allow_unused=True)
    grads = [torch.zeros_like(p) if g is None else g for g, p in zip(grads, params)]
    return _flatten_tensors(grads).detach()


def _policy_score_grad(actor, obs_batch: torch.Tensor, act_batch: torch.Tensor) -> Dict:
    """grad_θ log π_θ(a|s) averaged over (obs, act) batch."""
    actor_device = next(actor.parameters()).device
    obs_batch = obs_batch.to(actor_device)
    act_batch = act_batch.to(actor_device)
    dist = actor(obs_batch)
    log_prob = dist.log_prob(act_batch).sum(-1).mean()
    grads = torch.autograd.grad(log_prob, tuple(actor.parameters()), retain_graph=False, allow_unused=True)
    return {p: (torch.zeros_like(p) if g is None else g.detach())
            for p, g in zip(actor.parameters(), grads)}


def _policy_score_grad_from_segs(
    actor,
    seg1: np.ndarray,
    seg2: np.ndarray,
    obs_dim: int,
) -> Dict:
    """Score gradient using (obs, action) pairs stored in preference buffer segments.

    seg1, seg2 have shape (N_pairs, T, ds+da).  We flatten over pairs and
    timesteps and evaluate the CURRENT policy's log π(a|s) at the stored
    (obs, action) pairs — this is the correct distribution for the performative
    correction (the preference data was collected under past policies).
    Both segments contribute equally.
    """
    device = next(actor.parameters()).device
    sa1 = seg1.reshape(-1, seg1.shape[-1])   # (N*T, ds+da)
    sa2 = seg2.reshape(-1, seg2.shape[-1])
    all_sa = np.concatenate([sa1, sa2], axis=0)
    obs = torch.as_tensor(all_sa[:, :obs_dim], device=device).float()
    act = torch.as_tensor(all_sa[:, obs_dim:], device=device).float()
    return _policy_score_grad(actor, obs, act)


def _weighted_score_grad(
    actor,
    seg1: np.ndarray,         # (N, T, ds+da)
    seg2: np.ndarray,         # (N, T, ds+da)
    obs_dim: int,
    weights: torch.Tensor,    # (N,) per-pair scalar weights
) -> Dict:
    """Compute (1/N) * sum_i weights[i] * score_grad_i via one actor backward pass.

    score_grad_i is the gradient of the summed trajectory-pair score,
    sum_t log pi(a_t^0|s_t^0) + sum_t log pi(a_t^1|s_t^1). The average is
    only over preference pairs, not over timesteps.

    The layout of the concatenated array mirrors _policy_score_grad_from_segs:
      rows  0 .. N*T-1     -> seg1 pairs 0..N-1
      rows  N*T .. 2*N*T-1 -> seg2 pairs 0..N-1
    Each pair's 2T timesteps receive the same per-step weight = weights[i] / N,
    so grad scalar = (1/N) * sum_i weights[i] * score_grad_i.
    """
    N = len(weights)
    if N == 0:
        return _zeros_like_actor_params(actor)
    T = seg1.shape[1]
    device = next(actor.parameters()).device

    sa1 = seg1.reshape(-1, seg1.shape[-1])    # (N*T, ds+da)
    sa2 = seg2.reshape(-1, seg2.shape[-1])
    all_sa = np.concatenate([sa1, sa2], axis=0)  # (2*N*T, ds+da)

    obs = torch.as_tensor(all_sa[:, :obs_dim], device=device).float()
    act = torch.as_tensor(all_sa[:, obs_dim:], device=device).float()

    dist = actor(obs)
    log_prob = dist.log_prob(act).sum(-1)   # (2*N*T,)

    # Build per-timestep weight tensor. Do not divide by 2T: the trajectory
    # score is a sum over timesteps; only the outer Monte Carlo average divides by N.
    denom = float(N)
    w_step = torch.zeros(2 * N * T, device=device)
    for i in range(N):
        w_i = float(weights[i].item()) / denom
        w_step[i * T : (i + 1) * T] = w_i
        w_step[N * T + i * T : N * T + (i + 1) * T] = w_i

    scalar = (log_prob * w_step).sum()
    if not torch.isfinite(scalar):
        return _zeros_like_actor_params(actor)

    grads = torch.autograd.grad(
        scalar, tuple(actor.parameters()), retain_graph=False, allow_unused=True
    )
    return {p: (torch.zeros_like(p) if g is None else g.detach())
            for p, g in zip(actor.parameters(), grads)}


def _sample_replay_obs_action(replay_buffer, n: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
    total = len(replay_buffer)
    if total <= 0:
        return (
            torch.zeros((0, replay_buffer.obses.shape[-1]), device=device),
            torch.zeros((0, replay_buffer.actions.shape[-1]), device=device),
        )
    size = min(int(total), int(n))
    idxs = np.random.choice(total, size=size, replace=False)
    obs = torch.as_tensor(replay_buffer.obses[idxs], device=device).float()
    act = torch.as_tensor(replay_buffer.actions[idxs], device=device).float()
    return obs, act


def _sample_sa_from_replay(replay_buffer, n: int) -> np.ndarray:
    total = len(replay_buffer)
    if total <= 0:
        return np.zeros(
            (0, replay_buffer.obses.shape[-1] + replay_buffer.actions.shape[-1]),
            dtype=np.float32,
        )
    size = min(int(total), int(n))
    idxs = np.random.choice(total, size=size, replace=False)
    return np.concatenate(
        [replay_buffer.obses[idxs], replay_buffer.actions[idxs]], axis=-1
    ).astype(np.float32)


def _sample_on_policy_sa(actor, obs_np: np.ndarray) -> np.ndarray:
    """Sample (obs, action) pairs from the CURRENT policy at the given observations.

    Actions are drawn with rsample() under torch.no_grad() so the RM's forward
    pass (which needs gradients w.r.t. RM params) is unaffected.
    """
    device = next(actor.parameters()).device
    obs_t = torch.as_tensor(obs_np, device=device).float()
    with torch.no_grad():
        dist = actor(obs_t)
        act_t = dist.rsample()
    return np.concatenate(
        [obs_np, act_t.detach().cpu().numpy()], axis=-1
    ).astype(np.float32)


# ── Main entry point ─────────────────────────────────────────────────────────


def compute_total_perf_correction(
    reward_model,
    actor,
    replay_buffer,
    gauge: str,
    reference_sa: Optional[np.ndarray],
    cfg: PerfCorrectionConfig,
    on_policy_trajs: Optional[Sequence[np.ndarray]] = None,
    on_policy_obs: Optional[np.ndarray] = None,
    component: str = "total",
) -> Tuple[Dict, Dict]:
    """Compute G_perf_base + G_perf_shift with correct per-sample IFT weighting.

    Fixes vs. the previous implementation
    --------------------------------------
    1. Per-sample weighting (§4.8 algorithm box):
       The correction is now (1/N) Σ_i <w, ∇_φ ℓ_BT^(i)> · score_i instead of
       the biased product-of-means <w, mean_i ∇_φ ℓ_BT^(i)> · mean_i score_i.
       Requires N backward passes through the RM (one per preference pair); all
       direction dot products per pass are shared.

    2. On-policy u vector (§4.8 step 5a):
       When on_policy_obs is provided, actions are sampled from the current actor
       rather than using stored replay-buffer actions, making u closer to the
       true on-policy gradient E_{π_θ}[∇_φ r_φ(τ)].

    Parameters
    ----------
    on_policy_obs : numpy array of shape (M, obs_dim), optional
        Observations at which to sample current-policy actions for computing u.
        When None, falls back to stored (obs, action) pairs from the replay buffer.

    Returns
    -------
    actor_grads : dict mapping actor parameter → gradient tensor
    metrics     : dict of scalar diagnostic values
    """
    zero_grads = _zeros_like_actor_params(actor)
    use_no_tanh_objective = gauge.startswith("no_tanh")
    metrics: Dict = {
        "rm/d_shift_norm": 0.0,
        "actor/perf_correction_base_norm": 0.0,
        "actor/perf_correction_shift_norm": 0.0,
        "actor/perf_correction_total_norm": 0.0,
        "rm/base_cg_norm": 0.0,
        "rm/perf_uses_no_tanh_objective": float(use_no_tanh_objective),
    }

    params = reward_model.get_rm_parameters()
    if len(params) == 0:
        return zero_grads, metrics

    # ── Sample preference pairs (shared for Hessian, per-sample dots, score) ──
    pref_idxs = _sample_preference_indices(reward_model, cfg.pref_subsample)
    if len(pref_idxs) == 0:
        return zero_grads, metrics

    seg1 = reward_model.buffer_seg1[pref_idxs]   # (N, T, ds+da)
    seg2 = reward_model.buffer_seg2[pref_idxs]   # (N, T, ds+da)
    obs_dim: int = reward_model.ds

    def Avp(v: torch.Tensor) -> torch.Tensor:
        return _hvp(reward_model, params, pref_idxs, v, cfg.eps_ridge)

    # ── u = grad_phi E_{tau~pi_theta}[sum_t r_phi(s_t,a_t)] ─────────────────
    # Prefer actual current-policy trajectories. Fallback paths keep the function
    # usable in tests/legacy callers, scaling per-step samples by the horizon to
    # estimate a trajectory-return gradient.
    if on_policy_trajs is not None and len(on_policy_trajs) > 0:
        u = _grad_phi_mean_traj_return(reward_model, on_policy_trajs, use_no_tanh=use_no_tanh_objective)
    elif on_policy_obs is not None and len(on_policy_obs) > 0:
        on_obs = on_policy_obs
        if len(on_obs) > cfg.replay_subsample:
            on_obs = on_obs[np.random.choice(len(on_obs), size=cfg.replay_subsample, replace=False)]
        sa_np = _sample_on_policy_sa(actor, on_obs)
        u = float(cfg.horizon) * _grad_phi_mean_nn(reward_model, sa_np, use_no_tanh=use_no_tanh_objective)
    else:
        obs_t, act_t = _sample_replay_obs_action(replay_buffer, cfg.replay_subsample, params[0].device)
        if obs_t.numel() == 0:
            return zero_grads, metrics
        sa_np = torch.cat([obs_t, act_t], dim=-1).detach().cpu().numpy().astype(np.float32)
        u = float(cfg.horizon) * _grad_phi_mean_nn(reward_model, sa_np, use_no_tanh=use_no_tanh_objective)
    w = _conjugate_gradient(Avp, u, n_iter=cfg.cg_iter)
    if not torch.isfinite(w).all():
        return zero_grads, metrics

    # ── Shift CG solution (only when gauge has a θ-dependent shift) ───────────
    w_shift: Optional[torch.Tensor] = None
    if gauge not in {"none", "no_tanh"}:
        if (
            gauge in {"mean_ref", "no_tanh_mean_ref"}
            and reference_sa is not None
            and len(reference_sa) > 0
        ):
            sa_shift = reference_sa
            if len(sa_shift) > cfg.shift_subsample:
                rng = np.random.default_rng(0)
                idxs = rng.choice(len(sa_shift), size=cfg.shift_subsample, replace=False)
                sa_shift = sa_shift[idxs]
        else:
            sa_shift = _sample_sa_from_replay(replay_buffer, cfg.shift_subsample)
        if len(sa_shift) > 0:
            u_shift = _grad_phi_mean_nn(reward_model, sa_shift, use_no_tanh=use_no_tanh_objective)
            w_shift_cand = _conjugate_gradient(Avp, u_shift, n_iter=cfg.cg_iter)
            if torch.isfinite(w_shift_cand).all():
                w_shift = w_shift_cand

    # ── Per-sample BT directional derivatives ─────────────────────────────────
    # c_i = <w, ∇_φ ℓ_BT^{(i)}> for each preference pair i.
    # Both directions are processed in a single loop over pref pairs (N RM backward
    # passes total regardless of how many directions).
    directions: List[torch.Tensor] = [w]
    if w_shift is not None:
        directions.append(w_shift)

    dot_results = _bt_grad_per_sample_dots(reward_model, params, pref_idxs, directions)
    c_base = dot_results[0]                                        # (N,)
    c_shift = dot_results[1] if len(dot_results) > 1 else None    # (N,) or None

    # Zero out any NaN/Inf entries to avoid propagating numerical failures.
    c_base = torch.where(torch.isfinite(c_base), c_base, torch.zeros_like(c_base))
    if c_shift is not None:
        c_shift = torch.where(torch.isfinite(c_shift), c_shift, torch.zeros_like(c_shift))

    # ── Base correction: G_base = -(1/N) Σ_i c_i · score_i ──────────────────
    g_base = _zeros_like_actor_params(actor)
    if c_base.abs().sum().item() > 0:
        g_base_raw = _weighted_score_grad(actor, seg1, seg2, obs_dim, c_base)
        for p in g_base:
            g_base[p] = -g_base_raw[p]
        metrics["rm/base_cg_norm"] = float(torch.norm(w).item())
    metrics["actor/perf_correction_base_norm"] = grad_l2_norm(g_base)

    # ── Shift correction: G_shift = +(horizon/N) Σ_i c_shift_i · score_i ─────
    # Positive sign: subtracting the mean-shift from J reduces by horizon * d(shift)/dθ.
    g_shift = _zeros_like_actor_params(actor)
    if c_shift is not None and c_shift.abs().sum().item() > 0:
        g_shift_raw = _weighted_score_grad(actor, seg1, seg2, obs_dim, c_shift)
        for p in g_shift:
            g_shift[p] = cfg.horizon * g_shift_raw[p]
        metrics["rm/d_shift_norm"] = float(torch.norm(w_shift).item())
    metrics["actor/perf_correction_shift_norm"] = grad_l2_norm(g_shift)

    # ── Combine / select component ─────────────────────────────────────────────
    total_grads = {p: g_base[p] + g_shift[p] for p in g_base}
    metrics["actor/perf_correction_total_norm"] = grad_l2_norm(total_grads)
    if component == "total":
        actor_grads = total_grads
    elif component == "base":
        actor_grads = g_base
    elif component == "shift":
        actor_grads = g_shift
    else:
        raise ValueError(f"Unknown performative correction component: {component!r}")
    return actor_grads, metrics


# ── Legacy two-function API (kept for any external callers) ──────────────────
# These now delegate to compute_total_perf_correction and split the result.


def compute_perf_correction_base(
    reward_model,
    actor,
    replay_buffer,
    cfg: PerfCorrectionConfig,
):
    g_total, metrics = compute_total_perf_correction(
        reward_model, actor, replay_buffer,
        gauge="none",          # no shift correction
        reference_sa=None,
        cfg=cfg,
        component="base",
    )
    return g_total, {"rm/d_shift_norm": metrics["rm/d_shift_norm"]}


def compute_perf_correction_shift(
    reward_model,
    actor,
    replay_buffer,
    gauge: str,
    reference_sa: Optional[np.ndarray],
    cfg: PerfCorrectionConfig,
):
    # Return only the shift component by running a gauge="none" base and
    # subtracting — simpler: just run the full combined and return the shift part.
    # Since the old API is only used in tests/legacy code, return zeros for base.
    if gauge in {"none", "no_tanh"}:
        return _zeros_like_actor_params(actor), {"rm/d_shift_norm": 0.0}
    g_total, metrics = compute_total_perf_correction(
        reward_model, actor, replay_buffer,
        gauge=gauge,
        reference_sa=reference_sa,
        cfg=cfg,
        component="shift",
    )
    return g_total, {"rm/d_shift_norm": metrics["rm/d_shift_norm"]}
