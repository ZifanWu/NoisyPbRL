"""HVP+CG implementation of the IFT performative correction g_perp.

Stripped from perfg_gauge branch's pebble_gauge/perf_correction.py:
  - Removed gauge-shift / merge_cg / fd_weights paths (not needed here).
  - Removed dependency on GaugedRewardModel; works directly on the standard
    NoisyPbRL RewardModel.
  - Single entrypoint compute_g_perp_hvp() returns the actor-param-shaped
    gradient correction.

g_perp = score_grad(per_pair_weights), where per_pair_weights[i] = <w, ∇_φ ℓ_i>
and w = -(H + ridge·I)^{-1} · u_base, with H = ∂²L_BT/∂φ² and
u_base = ∂[mean_τ Σ_t r_φ(τ_t)] / ∂φ.

The sign convention matches perfg_gauge's `apply_actor_correction` step
direction:  θ ← θ + actor_lr * g_perp  (so g_perp is *added* to whatever
the standard SAC actor optimizer would do).
"""
from typing import Iterable, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F


def _params_to_list(params: Iterable[torch.nn.Parameter]) -> List[torch.nn.Parameter]:
    return [p for p in params if p.requires_grad]


def _flat_grad(output: torch.Tensor, params: Sequence[torch.nn.Parameter],
               retain_graph: bool = False, create_graph: bool = False) -> torch.Tensor:
    grads = torch.autograd.grad(
        output, params,
        retain_graph=retain_graph, create_graph=create_graph, allow_unused=True,
    )
    flat = []
    for param, grad in zip(params, grads):
        if grad is None:
            flat.append(torch.zeros_like(param).reshape(-1))
        else:
            flat.append(grad.reshape(-1))
    if not flat:
        return torch.zeros(0, device=output.device)
    return torch.cat(flat)


def _zero_like_params(params: Sequence[torch.nn.Parameter], device) -> torch.Tensor:
    if not params:
        return torch.zeros(0, device=device)
    return torch.cat([torch.zeros_like(p).reshape(-1) for p in params])


def iter_reward_parameters(reward_model) -> List[torch.nn.Parameter]:
    """Flat list of all parameters across the RM ensemble."""
    out: List[torch.nn.Parameter] = []
    for member in reward_model.ensemble:
        out.extend(p for p in member.parameters() if p.requires_grad)
    return out


def _mean_reward_output(reward_model, inputs, device) -> torch.Tensor:
    """Ensemble-mean reward output, shape (..., 1) matching member output."""
    x = torch.as_tensor(inputs, dtype=torch.float32, device=device)
    outputs = []
    for m in range(reward_model.de):
        outputs.append(reward_model.ensemble[m](x))
    return torch.stack(outputs, dim=0).mean(dim=0)


def _bt_losses(reward_model, seg1, seg2, labels, device,
               label_margin: float = 0.0) -> torch.Tensor:
    """Per-pair BT cross-entropy loss (preserves graph w.r.t. RM params)."""
    labels_t = torch.as_tensor(labels, dtype=torch.long, device=device).flatten()
    r1 = _mean_reward_output(reward_model, seg1, device).sum(dim=1)
    r2 = _mean_reward_output(reward_model, seg2, device).sum(dim=1)
    logits = torch.cat([r1, r2], dim=-1)

    uniform = labels_t < 0
    if bool(uniform.any()) or label_margin > 0:
        safe_labels = labels_t.clone()
        safe_labels[uniform] = 0
        target = torch.zeros_like(logits)
        target.scatter_(1, safe_labels.unsqueeze(1), 1.0 - 2.0 * label_margin)
        target += label_margin
        if bool(uniform.any()):
            target[uniform] = 0.5
        return -(target * F.log_softmax(logits, dim=1)).sum(dim=1)

    return F.cross_entropy(logits, labels_t, reduction="none")


def _sample_preference_batch(reward_model, batch_size: int, rng: np.random.Generator):
    max_len = reward_model.capacity if reward_model.buffer_full else reward_model.buffer_index
    if max_len <= 0:
        return None, None, None
    n = min(int(batch_size), max_len)
    idxs = rng.choice(max_len, size=n, replace=False)
    return (
        reward_model.buffer_seg1[idxs].astype(np.float32),
        reward_model.buffer_seg2[idxs].astype(np.float32),
        reward_model.buffer_label[idxs].flatten(),
    )


def _hvp(loss_fn, params: Sequence[torch.nn.Parameter],
         vector: torch.Tensor, ridge: float) -> torch.Tensor:
    """(H + ridge·I) · vector via double-backward through loss_fn()."""
    loss = loss_fn()
    grad = _flat_grad(loss, params, retain_graph=True, create_graph=True)
    dot = torch.dot(grad, vector)
    hv = _flat_grad(dot, params, retain_graph=False, create_graph=False)
    return hv.detach() + ridge * vector


def conjugate_gradient(matvec, b: torch.Tensor,
                       iters: int = 10, tol: float = 1e-10) -> torch.Tensor:
    """Solve A·w = b for w; matvec(v) returns A·v."""
    x = torch.zeros_like(b)
    r = b.detach().clone()
    p = r.clone()
    rs_old = torch.dot(r, r)
    if float(rs_old.cpu()) <= tol:
        return x
    for _ in range(int(iters)):
        Ap = matvec(p)
        denom = torch.dot(p, Ap)
        if abs(float(denom.cpu())) <= 1e-12:
            break
        alpha = rs_old / denom
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = torch.dot(r, r)
        if float(rs_new.cpu()) <= tol:
            break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    return x.detach()


def _trajectory_reward_grad(reward_model, params: Sequence[torch.nn.Parameter],
                             trajectories, device) -> torch.Tensor:
    """u^T = ∂/∂φ [mean_τ Σ_t r_φ(τ_t)].

    `trajectories` is an iterable of segments (each shape [horizon, ds+da])
    representing recent on-policy data; we use the mean across them as the
    objective.
    """
    if not trajectories:
        return _zero_like_params(params, device)
    objective = None
    n_valid = 0
    for traj in trajectories:
        if traj is None or len(traj) == 0:
            continue
        reward_sum = _mean_reward_output(reward_model, traj, device).sum()
        objective = reward_sum if objective is None else objective + reward_sum
        n_valid += 1
    if objective is None or n_valid == 0:
        return _zero_like_params(params, device)
    objective = objective / float(n_valid)
    return _flat_grad(objective, params,
                       retain_graph=False, create_graph=False).detach()


def _per_example_weight_dots(reward_model, params, seg1, seg2, labels,
                              vector: torch.Tensor, device,
                              label_margin: float = 0.0) -> torch.Tensor:
    """For each preference pair i, compute <vector, ∇_φ ℓ_i>."""
    losses = _bt_losses(reward_model, seg1, seg2, labels, device,
                        label_margin=label_margin)
    n = losses.shape[0]
    out = []
    for idx in range(n):
        retain = idx < n - 1
        grad_i = _flat_grad(losses[idx], params,
                            retain_graph=retain, create_graph=False).detach()
        out.append(torch.dot(vector, grad_i))
    if not out:
        return torch.zeros(0, device=device)
    return torch.stack(out).detach()


def _actor_score_grads(actor, seg1, seg2, ds: int,
                       coeff: torch.Tensor) -> List[torch.Tensor]:
    """∂/∂θ_actor [Σ_i coeff_i · (Σ_t log π(a1_{i,t}|s1_{i,t}) +
                                  Σ_t log π(a2_{i,t}|s2_{i,t}))]."""
    actor_params = _params_to_list(actor.parameters())
    if coeff.numel() == 0:
        return [torch.zeros_like(p) for p in actor_params]

    device = next(actor.parameters()).device
    seg1_t = torch.as_tensor(seg1, dtype=torch.float32, device=device)
    seg2_t = torch.as_tensor(seg2, dtype=torch.float32, device=device)
    obs1, act1 = seg1_t[..., :ds], seg1_t[..., ds:]
    obs2, act2 = seg2_t[..., :ds], seg2_t[..., ds:]
    act1 = act1.clamp(-0.999999, 0.999999)
    act2 = act2.clamp(-0.999999, 0.999999)

    bsz, horizon = obs1.shape[0], obs1.shape[1]
    dist1 = actor(obs1.reshape(bsz * horizon, ds))
    dist2 = actor(obs2.reshape(bsz * horizon, ds))
    logp1 = dist1.log_prob(act1.reshape(bsz * horizon, -1)).sum(dim=-1).reshape(bsz, horizon)
    logp2 = dist2.log_prob(act2.reshape(bsz * horizon, -1)).sum(dim=-1).reshape(bsz, horizon)
    pair_scores = logp1.sum(dim=1) + logp2.sum(dim=1)

    coeff = coeff.to(device=device, dtype=pair_scores.dtype)
    objective = torch.dot(coeff, pair_scores)
    grads = torch.autograd.grad(
        objective, actor_params, retain_graph=False, allow_unused=True
    )
    out = []
    for param, grad in zip(actor_params, grads):
        out.append(torch.zeros_like(param) if grad is None else grad.detach().clone())
    return out


def compute_g_perp_hvp(
    actor,
    reward_model,
    trajectories,
    ds: int,
    device,
    pref_batch_size: int = 256,
    cg_iters: int = 10,
    ridge: float = 1e-2,
    label_margin: float = 0.0,
    rng: np.random.Generator = None,
):
    """Compute g_perp via HVP+CG. Returns (List[Tensor] one per actor param,
    diagnostics dict). Caller should *add* the returned grads to actor.grad
    before optimizer.step() (matching sign convention of perfg_gauge).

    `trajectories` is a list of recent on-policy segments; ds is the
    observation dim (so that the seg layout is [..., :ds] = obs, [..., ds:] = act).
    """
    if rng is None:
        rng = np.random.default_rng(0)

    seg1, seg2, labels = _sample_preference_batch(reward_model, pref_batch_size, rng)
    if seg1 is None:
        return None, {"applied": 0.0, "reason_no_pref_batch": 1.0}

    params = iter_reward_parameters(reward_model)
    if not params:
        return None, {"applied": 0.0, "reason_no_reward_params": 1.0}

    def loss_fn():
        return _bt_losses(reward_model, seg1, seg2, labels, device,
                          label_margin=label_margin).mean()

    def matvec(v):
        return _hvp(loss_fn, params, v, ridge=ridge)

    u_base = _trajectory_reward_grad(reward_model, params, trajectories, device)
    u_base_norm = float(torch.linalg.vector_norm(u_base).cpu()) if u_base.numel() else 0.0
    if u_base_norm == 0.0:
        return None, {"applied": 0.0, "reason_zero_u_base": 1.0}

    # w = -(H + ridge·I)^{-1} · u_base
    w_base = -conjugate_gradient(matvec, u_base, iters=int(cg_iters))

    weights = _per_example_weight_dots(
        reward_model, params, seg1, seg2, labels, w_base, device,
        label_margin=label_margin,
    )
    n = max(1, int(weights.numel()))
    coeff = weights / float(n)
    actor_grads = _actor_score_grads(actor, seg1, seg2, ds, coeff)

    flat_g = torch.cat([g.reshape(-1).detach().cpu() for g in actor_grads])
    diagnostics = {
        "applied": 1.0,
        "pref_batch": float(n),
        "rm_param_count": float(sum(p.numel() for p in params)),
        "u_base_norm": u_base_norm,
        "w_base_norm": float(torch.linalg.vector_norm(w_base).cpu()),
        "weight_mean": float(weights.mean().cpu()) if weights.numel() else 0.0,
        "weight_abs_mean": float(weights.abs().mean().cpu()) if weights.numel() else 0.0,
        "g_perp_norm": float(torch.linalg.vector_norm(flat_g)),
    }
    return actor_grads, diagnostics


def apply_actor_correction(agent, actor_grads, actor_lr=None) -> float:
    """θ ← θ + actor_lr * g_perp.  Used after standard SAC actor step."""
    actor_params = _params_to_list(agent.actor.parameters())
    if actor_lr is None:
        actor_lr = float(agent.actor_optimizer.param_groups[0]["lr"])
    with torch.no_grad():
        for param, grad in zip(actor_params, actor_grads):
            param.add_(actor_lr * grad.to(param.device))
    if not actor_grads:
        return 0.0
    flat = torch.cat([g.reshape(-1).detach().cpu() for g in actor_grads])
    return float(torch.linalg.vector_norm(flat))
