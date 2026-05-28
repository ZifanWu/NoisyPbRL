from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import reward_model as reward_model_module
from pebble_gauge.reward_model_gauge import GaugedRewardModel, iter_reward_parameters


def _params_to_list(params: Iterable[torch.nn.Parameter]) -> List[torch.nn.Parameter]:
    return [p for p in params if p.requires_grad]


def _flat_grad(
    output: torch.Tensor,
    params: Sequence[torch.nn.Parameter],
    retain_graph: bool = False,
    create_graph: bool = False,
) -> torch.Tensor:
    grads = torch.autograd.grad(
        output,
        params,
        retain_graph=retain_graph,
        create_graph=create_graph,
        allow_unused=True,
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


def _zero_like_params(params: Sequence[torch.nn.Parameter]) -> torch.Tensor:
    if not params:
        return torch.zeros(0, device=reward_model_module.device)
    return torch.cat([torch.zeros_like(p).reshape(-1) for p in params])


def _vector_norm(x: Optional[torch.Tensor]) -> float:
    if x is None or x.numel() == 0:
        return 0.0
    return float(torch.linalg.vector_norm(x.detach()).cpu())


def _actor_grad_norm(grads: Sequence[torch.Tensor]) -> float:
    if not grads:
        return 0.0
    flat = torch.cat([g.reshape(-1).detach().cpu() for g in grads])
    return float(torch.linalg.vector_norm(flat)) if flat.numel() else 0.0


def _zero_actor_grads(actor) -> List[torch.Tensor]:
    return [torch.zeros_like(p) for p in _params_to_list(actor.parameters())]


def reward_parameters_for_scope(base_model, scope: str) -> List[torch.nn.Parameter]:
    scope = str(scope)
    if scope == "all":
        return _params_to_list(iter_reward_parameters(base_model))

    params: List[torch.nn.Parameter] = []
    for member in base_model.ensemble:
        final_linear = None
        for module in reversed(list(member.modules())):
            if isinstance(module, torch.nn.Linear):
                final_linear = module
                break
        if final_linear is None:
            continue
        if scope == "final_layer":
            params.append(final_linear.weight)
            if final_linear.bias is not None:
                params.append(final_linear.bias)
        elif scope == "final_bias":
            if final_linear.bias is not None:
                params.append(final_linear.bias)
        else:
            raise ValueError("unknown rm_param_scope '{}'; expected all, final_layer, or final_bias".format(scope))
    return _params_to_list(params)


def _mean_reward_output(base_model, inputs) -> torch.Tensor:
    x = torch.as_tensor(inputs, dtype=torch.float32, device=reward_model_module.device)
    outputs = []
    for member in range(base_model.de):
        outputs.append(base_model.ensemble[member](x))
    return torch.stack(outputs, dim=0).mean(dim=0)


def _bt_losses(base_model, seg1, seg2, labels, label_margin: float = 0.0) -> torch.Tensor:
    labels_t = torch.as_tensor(labels, dtype=torch.long, device=reward_model_module.device).flatten()
    r1 = _mean_reward_output(base_model, seg1).sum(dim=1)
    r2 = _mean_reward_output(base_model, seg2).sum(dim=1)
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


def _sample_preference_batch(base_model, batch_size: int, rng: np.random.RandomState):
    max_len = base_model.capacity if base_model.buffer_full else base_model.buffer_index
    if max_len <= 0:
        return None, None, None
    n = min(int(batch_size), max_len)
    idxs = rng.choice(max_len, size=n, replace=False)
    return (
        base_model.buffer_seg1[idxs].astype(np.float32),
        base_model.buffer_seg2[idxs].astype(np.float32),
        base_model.buffer_label[idxs].flatten(),
    )


def _hvp(loss_fn, params: Sequence[torch.nn.Parameter], vector: torch.Tensor, ridge: float) -> torch.Tensor:
    loss = loss_fn()
    grad = _flat_grad(loss, params, retain_graph=True, create_graph=True)
    dot = torch.dot(grad, vector)
    hv = _flat_grad(dot, params, retain_graph=False, create_graph=False)
    return hv.detach() + ridge * vector


def conjugate_gradient(matvec, b: torch.Tensor, iters: int = 10, tol: float = 1e-10) -> torch.Tensor:
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


def _trajectory_reward_grad(base_model, params: Sequence[torch.nn.Parameter], trajectories) -> torch.Tensor:
    if not trajectories:
        return _zero_like_params(params)
    objective = None
    for traj in trajectories:
        if traj is None or len(traj) == 0:
            continue
        reward_sum = _mean_reward_output(base_model, traj).sum()
        objective = reward_sum if objective is None else objective + reward_sum
    if objective is None:
        return _zero_like_params(params)
    objective = objective / float(len(trajectories))
    return _flat_grad(objective, params, retain_graph=False, create_graph=False).detach()


def _shift_reward_grad(base_model, params: Sequence[torch.nn.Parameter], shift_inputs) -> torch.Tensor:
    if shift_inputs is None or len(shift_inputs) == 0:
        return _zero_like_params(params)
    objective = _mean_reward_output(base_model, shift_inputs).mean()
    return _flat_grad(objective, params, retain_graph=False, create_graph=False).detach()


def _per_example_weight_dots_multi(
    base_model,
    params: Sequence[torch.nn.Parameter],
    seg1,
    seg2,
    labels,
    vectors: Sequence[torch.Tensor],
    label_margin: float,
) -> List[torch.Tensor]:
    """Compute ⟨vector_j, ∇_φ ℓ_i⟩ for every vector j and preference pair i.

    Computes ∇_φ ℓ_i once per pair and projects onto all vectors, avoiding a
    separate backward pass per vector.  The graph is released after the final
    pair to avoid holding it in memory for the full batch.
    """
    losses = _bt_losses(base_model, seg1, seg2, labels, label_margin=label_margin)
    n = losses.shape[0]
    dots_per_vec: List[List[torch.Tensor]] = [[] for _ in vectors]
    for idx in range(n):
        retain = idx < n - 1
        grad_i = _flat_grad(losses[idx], params, retain_graph=retain, create_graph=False).detach()
        for j, vec in enumerate(vectors):
            dots_per_vec[j].append(torch.dot(vec, grad_i))
    return [
        torch.stack(d).detach() if d else torch.zeros(0, device=reward_model_module.device)
        for d in dots_per_vec
    ]


def _add_flat_to_params(
    params: Sequence[torch.nn.Parameter],
    flat_direction: torch.Tensor,
    scale: float,
) -> None:
    offset = 0
    for param in params:
        n = param.numel()
        delta = flat_direction[offset : offset + n].view_as(param).to(param.device)
        param.add_(float(scale) * delta)
        offset += n


def _per_example_weight_dots_fd(
    base_model,
    params: Sequence[torch.nn.Parameter],
    seg1,
    seg2,
    labels,
    vectors: Sequence[torch.Tensor],
    label_margin: float,
    fd_eps: float,
) -> List[torch.Tensor]:
    """Approximate ⟨vector_j, ∇_φ ℓ_i⟩ with two RM forward passes per vector."""
    if not vectors:
        return []

    eps = max(float(fd_eps), 1e-12)
    dots = []
    with torch.no_grad():
        for vec in vectors:
            norm = torch.linalg.vector_norm(vec.detach())
            if vec.numel() == 0 or float(norm.cpu()) <= 0.0:
                labels_t = torch.as_tensor(labels, dtype=torch.long, device=reward_model_module.device).flatten()
                dots.append(torch.zeros(labels_t.shape[0], device=reward_model_module.device))
                continue

            direction = (vec.detach() / norm).to(device=reward_model_module.device)
            _add_flat_to_params(params, direction, eps)
            try:
                loss_plus = _bt_losses(base_model, seg1, seg2, labels, label_margin=label_margin).detach()
                _add_flat_to_params(params, direction, -2.0 * eps)
                loss_minus = _bt_losses(base_model, seg1, seg2, labels, label_margin=label_margin).detach()
            finally:
                _add_flat_to_params(params, direction, eps)

            dots.append((loss_plus - loss_minus) * (norm / (2.0 * eps)))
    return dots


def _per_example_weight_dots(
    base_model,
    params: Sequence[torch.nn.Parameter],
    seg1,
    seg2,
    labels,
    vectors: Sequence[torch.Tensor],
    label_margin: float,
    fd_weights: bool,
    fd_eps: float,
) -> List[torch.Tensor]:
    if fd_weights:
        return _per_example_weight_dots_fd(
            base_model, params, seg1, seg2, labels, vectors, label_margin=label_margin, fd_eps=fd_eps
        )
    return _per_example_weight_dots_multi(
        base_model, params, seg1, seg2, labels, vectors, label_margin=label_margin
    )


def _actor_score_grads(
    actor,
    seg1,
    seg2,
    ds: int,
    coeff: torch.Tensor,
) -> List[torch.Tensor]:
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
    grads = torch.autograd.grad(objective, actor_params, retain_graph=False, allow_unused=True)
    out = []
    for param, grad in zip(actor_params, grads):
        out.append(torch.zeros_like(param) if grad is None else grad.detach().clone())
    return out


def apply_actor_correction(agent, actor_grads: Sequence[torch.Tensor], actor_lr: Optional[float] = None) -> float:
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


def compute_performative_correction(
    agent,
    reward_model: GaugedRewardModel,
    trajectories,
    rng: np.random.RandomState,
    horizon: float,
    pref_batch_size: int = 256,
    cg_iters: int = 10,
    ridge: float = 1e-3,
    label_margin: float = 0.0,
    perf_grad: bool = True,
    gauge_corr: bool = True,
    shift_inputs=None,
    merge_cg: bool = True,
    fd_weights: bool = False,
    fd_eps: float = 1e-3,
    rm_param_scope: str = "all",
    solver: str = "cg",
) -> Tuple[Optional[List[torch.Tensor]], Dict[str, float]]:
    perf_grad = bool(perf_grad)
    gauge_corr = bool(gauge_corr)
    merge_cg = bool(merge_cg)
    fd_weights = bool(fd_weights)
    solver = str(solver)
    if solver not in ("cg", "ridge"):
        raise ValueError("unknown gauge.pg.solver '{}'; expected cg or ridge".format(solver))
    if not perf_grad:
        return None, {
            "applied": 0.0,
            "perf_grad_enabled": 0.0,
            "gauge_corr_enabled": float(gauge_corr),
            "gauge_corr_active": 0.0,
            "merge_cg_enabled": float(merge_cg),
            "fd_weights_enabled": float(fd_weights),
            "rm_param_count": 0.0,
            "solver_cg": 1.0 if solver == "cg" else 0.0,
            "solver_ridge": 1.0 if solver == "ridge" else 0.0,
            "reason_perf_grad_disabled": 1.0,
        }

    base_model = reward_model.base
    seg1, seg2, labels = _sample_preference_batch(base_model, pref_batch_size, rng)
    if seg1 is None:
        return None, {"applied": 0.0, "reason_no_pref_batch": 1.0}

    params = reward_parameters_for_scope(base_model, rm_param_scope)
    if not params:
        return None, {"applied": 0.0, "reason_no_reward_params": 1.0}

    def loss_fn():
        return _bt_losses(base_model, seg1, seg2, labels, label_margin=label_margin).mean()

    def matvec(v):
        return _hvp(loss_fn, params, v, ridge=ridge)

    u_base = _trajectory_reward_grad(base_model, params, trajectories)

    shift_kind = reward_model.shift_kind()
    shift_active = bool(gauge_corr and shift_kind != "none")
    if shift_active:
        if shift_inputs is None:
            shift_inputs = reward_model.inputs_for_shift_kind(shift_kind)
        u_shift = _shift_reward_grad(base_model, params, shift_inputs)
    else:
        u_shift = _zero_like_params(params)

    ridge_scale = max(float(ridge), 1e-12)

    if not merge_cg:
        if solver == "cg":
            w_base = conjugate_gradient(matvec, u_base, iters=cg_iters)
            w_shift = conjugate_gradient(matvec, u_shift, iters=cg_iters) if shift_active else _zero_like_params(params)
        else:
            w_base = u_base / ridge_scale
            w_shift = u_shift / ridge_scale if shift_active else _zero_like_params(params)

        if shift_active:
            weights_base, weights_shift = _per_example_weight_dots(
                base_model, params, seg1, seg2, labels, [w_base, w_shift],
                label_margin=label_margin, fd_weights=fd_weights, fd_eps=fd_eps
            )
        else:
            [weights_base] = _per_example_weight_dots(
                base_model, params, seg1, seg2, labels, [w_base],
                label_margin=label_margin, fd_weights=fd_weights, fd_eps=fd_eps
            )
            weights_shift = torch.zeros_like(weights_base)

        n = max(1, int(weights_base.numel()))
        coeff_base = -weights_base / float(n)
        coeff_shift = float(horizon) * weights_shift / float(n)
        base_actor_grads = _actor_score_grads(agent.actor, seg1, seg2, reward_model.ds, coeff_base)
        shift_actor_grads = _actor_score_grads(agent.actor, seg1, seg2, reward_model.ds, coeff_shift) if shift_active else _zero_actor_grads(agent.actor)
        actor_grads = [base_grad + shift_grad for base_grad, shift_grad in zip(base_actor_grads, shift_actor_grads)]

        diagnostics = {
            "applied": 1.0,
            "perf_grad_enabled": 1.0,
            "gauge_corr_enabled": float(gauge_corr),
            "gauge_corr_active": float(shift_active),
            "merge_cg_enabled": 0.0,
            "fd_weights_enabled": float(fd_weights),
            "solver_cg": 1.0 if solver == "cg" else 0.0,
            "solver_ridge": 1.0 if solver == "ridge" else 0.0,
            "fd_eps": float(fd_eps),
            "pref_batch": float(n),
            "rm_param_count": float(sum(p.numel() for p in params)),
            "rm_param_scope_all": 1.0 if str(rm_param_scope) == "all" else 0.0,
            "rm_param_scope_final_layer": 1.0 if str(rm_param_scope) == "final_layer" else 0.0,
            "rm_param_scope_final_bias": 1.0 if str(rm_param_scope) == "final_bias" else 0.0,
            "u_base_norm": _vector_norm(u_base),
            "w_base_norm": _vector_norm(w_base),
            "u_shift_norm": _vector_norm(u_shift),
            "w_shift_norm": _vector_norm(w_shift),
            "d_shift_norm": _vector_norm(w_shift),
            "shift_horizon": float(horizon),
            "weight_base_mean": float(weights_base.mean().detach().cpu()) if weights_base.numel() else 0.0,
            "weight_shift_mean": float(weights_shift.mean().detach().cpu()) if weights_shift.numel() else 0.0,
            "perf_correction_base_norm": _actor_grad_norm(base_actor_grads),
            "perf_correction_shift_norm": _actor_grad_norm(shift_actor_grads),
            "perf_correction_total_norm": _actor_grad_norm(actor_grads),
            "actor_grad_norm": _actor_grad_norm(actor_grads),
        }
        return actor_grads, diagnostics

    # Linearity gives the exact same target correction with one inverse solve:
    # -H^{-1}u_base + H_horizon H^{-1}u_shift = H^{-1}(-u_base + H_horizon u_shift).
    # With truncated CG this changes the approximation path, so it is configurable.
    u_total = -u_base + float(horizon) * u_shift
    if solver == "cg":
        w_total = conjugate_gradient(matvec, u_total, iters=cg_iters)
    else:
        w_total = u_total / ridge_scale

    [weights_total] = _per_example_weight_dots(
        base_model, params, seg1, seg2, labels, [w_total],
        label_margin=label_margin, fd_weights=fd_weights, fd_eps=fd_eps
    )
    n = max(1, int(weights_total.numel()))
    coeff_total = weights_total / float(n)
    actor_grads = _actor_score_grads(agent.actor, seg1, seg2, reward_model.ds, coeff_total)

    base_component = -u_base
    shift_component = float(horizon) * u_shift if shift_active else _zero_like_params(params)

    diagnostics = {
        "applied": 1.0,
        "perf_grad_enabled": 1.0,
        "gauge_corr_enabled": float(gauge_corr),
        "gauge_corr_active": float(shift_active),
        "merge_cg_enabled": 1.0,
        "fd_weights_enabled": float(fd_weights),
        "solver_cg": 1.0 if solver == "cg" else 0.0,
        "solver_ridge": 1.0 if solver == "ridge" else 0.0,
        "fd_eps": float(fd_eps),
        "pref_batch": float(n),
        "rm_param_count": float(sum(p.numel() for p in params)),
        "rm_param_scope_all": 1.0 if str(rm_param_scope) == "all" else 0.0,
        "rm_param_scope_final_layer": 1.0 if str(rm_param_scope) == "final_layer" else 0.0,
        "rm_param_scope_final_bias": 1.0 if str(rm_param_scope) == "final_bias" else 0.0,
        "u_base_norm": _vector_norm(u_base),
        "u_shift_norm": _vector_norm(u_shift),
        "u_total_norm": _vector_norm(u_total),
        "w_total_norm": _vector_norm(w_total),
        "d_shift_norm": _vector_norm(u_shift),
        "shift_horizon": float(horizon),
        "weight_total_mean": float(weights_total.mean().detach().cpu()) if weights_total.numel() else 0.0,
        "perf_correction_base_norm": _vector_norm(base_component),
        "perf_correction_shift_norm": _vector_norm(shift_component),
        "perf_correction_total_norm": _actor_grad_norm(actor_grads),
        "actor_grad_norm": _actor_grad_norm(actor_grads),
    }
    return actor_grads, diagnostics
