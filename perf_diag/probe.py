"""Finite-difference performative-gradient probe.

Single monitoring step:
  1. Fresh on-policy probe batch (N_probe segments, length L) under current actor.
  2. g0 = REINFORCE probe gradient under current RM ψ_t  (autograd w.r.t. actor params).
  3. ψ' = brief refit of a DEEP COPY of the RM ensemble on K_refit SGD steps over a
         fresh preference batch labeled by the scripted teacher at the current π_θ.
  4. g0' = REINFORCE on the SAME probe batch with ψ'.
  5. Diagnostics: ĝ⊥ = g0' − g0, κ̂, ρ̂, R̂.

The live reward model and actor are never modified. The probe is read-only on the
training state.
"""

from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def _flat_grad(loss: torch.Tensor, params) -> torch.Tensor:
    """Compute ∇_params loss as a single flat 1-D tensor (zeros for params with no grad path)."""
    grads = torch.autograd.grad(
        loss, list(params), retain_graph=False, allow_unused=True
    )
    flat = []
    for g, p in zip(grads, params):
        if g is None:
            flat.append(torch.zeros_like(p).reshape(-1))
        else:
            flat.append(g.reshape(-1))
    return torch.cat(flat)


def _rollout_probe_segments(env_factory, actor, n_segments: int, segment_len: int,
                            device: torch.device) -> dict:
    """Roll out the current actor for n_segments segments of length segment_len.

    Uses a fresh env (env_factory()) to avoid touching the training env state.
    Returns a dict of np.arrays:
        obs:    (n_segments, segment_len, obs_dim)
        action: (n_segments, segment_len, action_dim)
        gold:   (n_segments, segment_len)     — env (gold) reward, ONLY for the teacher
    Actions are sampled stochastically from the actor (sample=True).
    """
    env = env_factory()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    obs_buf = np.zeros((n_segments, segment_len, obs_dim), dtype=np.float32)
    act_buf = np.zeros((n_segments, segment_len, act_dim), dtype=np.float32)
    gold_buf = np.zeros((n_segments, segment_len), dtype=np.float32)

    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    for s in range(n_segments):
        for t in range(segment_len):
            with torch.no_grad():
                o_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                dist = actor(o_t)
                a = dist.sample().clamp(-1.0, 1.0).squeeze(0).cpu().numpy()
            obs_buf[s, t] = obs
            act_buf[s, t] = a
            step_out = env.step(a)
            if len(step_out) == 5:
                next_obs, r, terminated, truncated, _ = step_out
                done = bool(terminated) or bool(truncated)
            else:
                next_obs, r, done, _ = step_out
            gold_buf[s, t] = float(r)
            obs = next_obs
            if done:
                reset_out = env.reset()
                obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    try:
        env.close()
    except Exception:
        pass
    return dict(obs=obs_buf, action=act_buf, gold=gold_buf)


def _reinforce_probe_gradient(actor, ensemble, batch: dict, device: torch.device,
                              centered: bool = True) -> tuple[torch.Tensor, float, float]:
    """REINFORCE probe gradient as a flat tensor over actor.parameters().

    L0 = -(1/N) Σ_seg (R_seg - b) · Σ_t log π_θ(a_t | s_t)
    g0 = ∇_θ L0     (then negated since L is the negative of the surrogate score-function loss)

    Returns: (flat_grad, mean_R_seg, var_R_seg).
    """
    obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=device)
    act = torch.as_tensor(batch["action"], dtype=torch.float32, device=device)
    N, L, _ = obs.shape

    # actor log-prob path
    flat_obs = obs.reshape(N * L, -1)
    flat_act = act.reshape(N * L, -1)
    # Numerical safety: SquashedNormal log_prob diverges if |a|=1 exactly.
    eps = 1e-6
    flat_act_safe = flat_act.clamp(-1.0 + eps, 1.0 - eps)
    dist = actor(flat_obs)
    log_pi = dist.log_prob(flat_act_safe).sum(-1)                  # (N*L,)
    log_pi_per_seg = log_pi.reshape(N, L).sum(dim=1)               # (N,) — Σ_t log π over each segment

    # Segment reward under the ensemble mean — detached, no grad
    sa = np.concatenate([batch["obs"], batch["action"]], axis=-1)  # (N, L, ds+da)
    sa_flat = sa.reshape(N * L, -1)
    with torch.no_grad():
        member_r = []
        for member_net in ensemble:
            r_mem = member_net(torch.as_tensor(sa_flat, dtype=torch.float32, device=device))
            member_r.append(r_mem.reshape(N, L).cpu().numpy())
        member_r = np.stack(member_r, axis=0)                       # (E, N, L)
        # Per-segment reward = sum over t of ensemble-mean per-step reward
        R_seg_np = member_r.mean(axis=0).sum(axis=1)                # (N,)
    R_seg = torch.as_tensor(R_seg_np, dtype=torch.float32, device=device)
    if centered:
        b = R_seg.mean()
        adv = R_seg - b
    else:
        adv = R_seg

    # surrogate loss: minimize  -(adv · Σ log π) so grad = -adv · ∇Σ log π → g0
    surrogate = -(adv * log_pi_per_seg).mean()
    g_flat = _flat_grad(surrogate, list(actor.parameters()))
    # the gradient of the surrogate already equals -E[(R-b)·∇ Σ log π], so we flip sign:
    g0 = -g_flat
    return g0, float(R_seg_np.mean()), float(R_seg_np.std())


def _flatten_ensemble_params(ensemble) -> torch.Tensor:
    """Concatenate all params of all ensemble members into a single 1-D tensor (detached)."""
    return torch.cat([p.detach().reshape(-1) for m in ensemble for p in m.parameters()])


def _restore_ensemble_params(ensemble, flat: torch.Tensor) -> None:
    """In-place: copy a flat param vector back into the ensemble's parameters."""
    offset = 0
    with torch.no_grad():
        for m in ensemble:
            for p in m.parameters():
                n = p.numel()
                p.copy_(flat[offset:offset + n].view_as(p))
                offset += n


def _refit_psi_prime(ensemble_src, refit_pairs_inputs, refit_pairs_labels, K_refit: int,
                     lr: float, device: torch.device):
    """Deep-copy ensemble; take K_refit Adam SGD steps on the BT loss over the given
    preference pairs; return the refitted ensemble + a per-step trajectory of
    (param_step_norm, loss, flat_param_snapshot) so the caller can compute κ̂_k at each k
    and interpolate across capacities at matched param_step_norm.

    Snapshots are flat parameter vectors (cheap) — the caller can stamp them into the
    ensemble with _restore_ensemble_params(...) to compute g0' at any k.

    Returns: (final_ensemble, trajectory_dict) where trajectory_dict has:
        param_step_norms: list[float], length K_refit  (cumulative ‖ψ_k − ψ_0‖)
        losses:           list[float], length K_refit+1  (BT loss at k=0..K_refit)
        param_snapshots:  list[torch.Tensor 1-D], length K_refit+1  (flat ψ_k vectors)
    """
    ensemble_p = []
    params = []
    for m in ensemble_src:
        m_copy = copy.deepcopy(m).to(device)
        ensemble_p.append(m_copy)
        params.extend(m_copy.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    sa1 = torch.as_tensor(refit_pairs_inputs[0], dtype=torch.float32, device=device)
    sa2 = torch.as_tensor(refit_pairs_inputs[1], dtype=torch.float32, device=device)
    labels = torch.as_tensor(refit_pairs_labels.reshape(-1), dtype=torch.long, device=device)
    # Drop ambiguous labels (-1) — match repo behavior on label_margin
    valid = labels >= 0
    sa1 = sa1[valid]; sa2 = sa2[valid]; labels = labels[valid]

    psi_0 = _flatten_ensemble_params(ensemble_p).clone()
    param_snapshots = [psi_0.clone()]
    param_step_norms = []
    losses = []

    if labels.numel() == 0:
        return ensemble_p, dict(param_step_norms=[], losses=[], param_snapshots=param_snapshots)

    ce = torch.nn.CrossEntropyLoss()

    def _compute_loss() -> float:
        with torch.no_grad():
            loss = 0.0
            for member in ensemble_p:
                r1 = member(sa1).reshape(sa1.shape[0], sa1.shape[1]).sum(dim=1, keepdim=True)
                r2 = member(sa2).reshape(sa2.shape[0], sa2.shape[1]).sum(dim=1, keepdim=True)
                logits = torch.cat([r1, r2], dim=-1)
                loss = loss + ce(logits, labels)
            return float(loss.item())

    losses.append(_compute_loss())
    for _ in range(K_refit):
        opt.zero_grad()
        loss = 0.0
        for member in ensemble_p:
            r1 = member(sa1).reshape(sa1.shape[0], sa1.shape[1]).sum(dim=1, keepdim=True)
            r2 = member(sa2).reshape(sa2.shape[0], sa2.shape[1]).sum(dim=1, keepdim=True)
            logits = torch.cat([r1, r2], dim=-1)
            loss = loss + ce(logits, labels)
        loss.backward()
        opt.step()
        with torch.no_grad():
            psi_k = _flatten_ensemble_params(ensemble_p)
            param_step_norms.append(float((psi_k - psi_0).norm().item()))
            param_snapshots.append(psi_k.clone())
        losses.append(_compute_loss())
    return ensemble_p, dict(param_step_norms=param_step_norms, losses=losses,
                            param_snapshots=param_snapshots)


def _build_refit_pairs(probe_batch: dict, reward_model, refit_pairs: int,
                       rng: np.random.Generator,
                       n_holdout: int = 0):
    """Build a fresh preference batch from the on-policy probe segments.

    Pairs are sampled with replacement from the N_probe segments. The scripted teacher
    in reward_model.get_label is used for labels (the same teacher that drives training).
    We additionally compute gold-derived labels (from the segments' gold reward sums),
    so the caller can: (a) report held-out accuracy vs GOLD (not vs noisy teacher),
    and (b) measure the teacher's empirical disagreement with gold — used to match
    noise level across teacher configurations (cf. teacher_capacity_protocol §1).

    Returns: tuple
        refit_inputs:        (sa_t_1, sa_t_2)
        refit_labels:        teacher labels for refit, may include -1 (skipped)
        fallback_used:       bool — teacher skipped ALL pairs, fell back to gold labels
        holdout_inputs:      (sa_t_1, sa_t_2) or (None, None) if n_holdout==0
        holdout_labels_t:    teacher labels for holdout, or None
        holdout_labels_gold: gold labels for holdout, or None
        teacher_gold_disagreement: fraction of valid (label>=0) teacher labels that
                             disagree with gold-derived label; NaN if no valid labels.
    """
    N = probe_batch["obs"].shape[0]
    L = probe_batch["obs"].shape[1]
    total = refit_pairs + int(n_holdout)
    idx1 = rng.integers(0, N, size=total)
    idx2 = rng.integers(0, N, size=total)
    sa_t_1 = np.concatenate([probe_batch["obs"][idx1], probe_batch["action"][idx1]], axis=-1).astype(np.float32)
    sa_t_2 = np.concatenate([probe_batch["obs"][idx2], probe_batch["action"][idx2]], axis=-1).astype(np.float32)
    r_t_1 = probe_batch["gold"][idx1].reshape(total, L, 1).astype(np.float32)
    r_t_2 = probe_batch["gold"][idx2].reshape(total, L, 1).astype(np.float32)
    # Reuse the live teacher; it consumes gold returns and emits noisy BT labels
    sa_t_1, sa_t_2, r_t_1, r_t_2, labels = reward_model.get_label(sa_t_1, sa_t_2, r_t_1, r_t_2)
    fallback_used = False
    if labels is None or len(labels) == 0:
        # fall back to gold-derived labels on the same pairs if the teacher skipped all.
        fallback_used = True
        labels = (r_t_1.sum(1).reshape(-1) < r_t_2.sum(1).reshape(-1)).astype(np.int64)
        sa_t_1 = np.concatenate([probe_batch["obs"][idx1], probe_batch["action"][idx1]], axis=-1).astype(np.float32)
        sa_t_2 = np.concatenate([probe_batch["obs"][idx2], probe_batch["action"][idx2]], axis=-1).astype(np.float32)
    labels = np.asarray(labels).astype(np.int64).reshape(-1)

    # Gold labels — aligned with the (post-filter) teacher labels by construction
    # (r_t_1/r_t_2 are returned from get_label after the same filtering).
    gold_labels = (r_t_1.sum(axis=(1, 2)) < r_t_2.sum(axis=(1, 2))).astype(np.int64)

    # Teacher's empirical disagreement-with-gold over the pairs where teacher gave a valid label.
    valid = labels >= 0
    if bool(valid.any()):
        teacher_gold_disagreement = float((labels[valid] != gold_labels[valid]).mean())
    else:
        teacher_gold_disagreement = float("nan")

    # Split: refit half (first refit_pairs), holdout half (remaining).
    refit_inputs = (sa_t_1[:refit_pairs], sa_t_2[:refit_pairs])
    refit_labels = labels[:refit_pairs]
    if n_holdout > 0 and labels.size > refit_pairs:
        holdout_inputs = (sa_t_1[refit_pairs:], sa_t_2[refit_pairs:])
        holdout_labels_t = labels[refit_pairs:]
        holdout_labels_gold = gold_labels[refit_pairs:]
    else:
        holdout_inputs = (None, None)
        holdout_labels_t = None
        holdout_labels_gold = None
    return (refit_inputs, refit_labels, fallback_used,
            holdout_inputs, holdout_labels_t, holdout_labels_gold,
            teacher_gold_disagreement)


def _heldout_acc(ensemble, holdout_inputs, holdout_labels_teacher,
                 holdout_labels_gold, device) -> tuple[float, float, int, int]:
    """Phase 2 (corrected per protocol §6): held-out RM accuracy vs BOTH labels.

    Returns: (acc_vs_gold, acc_vs_teacher, n_gold, n_teacher).
        acc_vs_gold:   primary metric — distance from the Bayes-optimal predictor.
                       The "gold" label is the gold-reward-sum sign over the segment pair.
                       Comparable across teachers (no noisy-ceiling confound).
        acc_vs_teacher: legacy / control metric — distance from what the agent was trained on.
                       Per protocol, this is uninterpretable across teachers because the
                       noisy-label ceiling differs by teacher.
    Gap (acc_vs_teacher_train − acc_vs_gold_heldout) = fitting-noise signature.
    """
    sa1, sa2 = holdout_inputs
    if sa1 is None or len(sa1) == 0:
        return float("nan"), float("nan"), 0, 0
    sa1_t = torch.as_tensor(sa1, dtype=torch.float32, device=device)
    sa2_t = torch.as_tensor(sa2, dtype=torch.float32, device=device)
    with torch.no_grad():
        member_diffs = []
        for member in ensemble:
            r1 = member(sa1_t).reshape(sa1_t.shape[0], sa1_t.shape[1]).sum(dim=1)
            r2 = member(sa2_t).reshape(sa2_t.shape[0], sa2_t.shape[1]).sum(dim=1)
            member_diffs.append((r2 - r1).cpu().numpy())
        mean_diff = np.mean(member_diffs, axis=0)
    pred = (mean_diff > 0).astype(np.int64)

    # Accuracy vs gold (primary)
    if holdout_labels_gold is None or len(holdout_labels_gold) == 0:
        acc_gold, n_gold = float("nan"), 0
    else:
        acc_gold = float((pred == holdout_labels_gold).mean())
        n_gold = int(len(holdout_labels_gold))

    # Accuracy vs teacher's noisy labels (legacy)
    if holdout_labels_teacher is None or len(holdout_labels_teacher) == 0:
        acc_teacher, n_teacher = float("nan"), 0
    else:
        valid = holdout_labels_teacher >= 0
        if not bool(valid.any()):
            acc_teacher, n_teacher = float("nan"), 0
        else:
            acc_teacher = float((pred[valid] == holdout_labels_teacher[valid]).mean())
            n_teacher = int(valid.sum())

    return acc_gold, acc_teacher, n_gold, n_teacher


def _kresample_decomposition(actor, reward_model, batch, g0, refit_pairs: int,
                             K_refit: int, refit_lr: float, K_resamples: int,
                             base_seed: int, device: torch.device) -> dict:
    """Phase 3: K independent refits at FIXED (policy, probe batch, ψ_t). Only the
    refit-batch sampling seed varies across k. Returns the systematic/scatter
    decomposition + orthogonality assertion.

    Decomposition (assumes near-orthogonality):
        ‖ĝ⊥‖² ≈ ‖systematic‖² + ‖scatter‖²
            systematic = mean_k ĝ⊥^(k)            (direction-stable misspecification-shift)
            scatter_k  = ĝ⊥^(k) − systematic      (variance / epistemic)
        pairwise_cos = mean over i≠j of cos(ĝ⊥^(i), ĝ⊥^(j))
            ~1 → systematic; ~0 → pure variance

    Orthogonality check (review request — decomposition is biased if violated):
        cos(systematic, scatter_k) per k → max and mean. Should be ≈0.
    """
    gperps = []
    g0p_norms = []
    for k in range(int(K_resamples)):
        # Vary only the refit-batch seed across resamples.
        k_seed = (int(base_seed) + (k + 1) * 9973) & 0x7FFFFFFF
        rng_k = np.random.default_rng(k_seed)
        np.random.seed(k_seed)
        refit_inputs_k, refit_labels_k, _, _, _, _, _ = _build_refit_pairs(
            batch, reward_model, refit_pairs, rng_k, n_holdout=0
        )
        ensemble_k, _ = _refit_psi_prime(
            reward_model.ensemble, refit_inputs_k, refit_labels_k,
            K_refit, refit_lr, device
        )
        g0p_k, _, _ = _reinforce_probe_gradient(
            actor, ensemble_k, batch, device, centered=True
        )
        with torch.no_grad():
            gperp_k = (g0p_k - g0).detach().clone()
            gperps.append(gperp_k)
            g0p_norms.append(float(g0p_k.norm().item()))
        for m in ensemble_k:
            for p in m.parameters():
                p.grad = None
        del ensemble_k

    with torch.no_grad():
        Gp = torch.stack(gperps, dim=0)         # (K, P)
        gperp_mean = Gp.mean(dim=0)              # (P,) systematic direction
        residuals = Gp - gperp_mean.unsqueeze(0) # (K, P)

        systematic_norm = float(gperp_mean.norm().item())
        scatter_norms = residuals.norm(dim=1)
        scatter_norm_mean = float(scatter_norms.mean().item())

        norms_per_k = Gp.norm(dim=1, keepdim=True)
        Gp_unit = Gp / (norms_per_k + 1e-30)
        cos_matrix = Gp_unit @ Gp_unit.t()
        K = int(K_resamples)
        mask = ~torch.eye(K, dtype=torch.bool, device=cos_matrix.device)
        pairwise_cos = float(cos_matrix[mask].mean().item()) if K > 1 else float("nan")

        # Orthogonality assertion data: cos between systematic mean and each residual.
        mean_norm = gperp_mean.norm() + 1e-30
        ortho_cos_per_k = (residuals @ gperp_mean) / (residuals.norm(dim=1) * mean_norm + 1e-30)
        ortho_cos_max = float(ortho_cos_per_k.abs().max().item())
        ortho_cos_mean = float(ortho_cos_per_k.mean().item())

        total_sq_mean = float((Gp.norm(dim=1) ** 2).mean().item())
        decomp_sq_sum = systematic_norm ** 2 + scatter_norm_mean ** 2

    return dict(
        kresample_K=int(K_resamples),
        kresample_systematic_norm=systematic_norm,
        kresample_scatter_norm=scatter_norm_mean,
        kresample_pairwise_cos=pairwise_cos,
        kresample_ortho_cos_max=ortho_cos_max,
        kresample_ortho_cos_mean=ortho_cos_mean,
        kresample_total_sq_mean=total_sq_mean,
        kresample_decomp_sq_sum=decomp_sq_sum,
        kresample_g0p_norms=[float(x) for x in g0p_norms],
    )


def run_probe(actor, reward_model, env_factory,
              n_probe: int = 8, segment_len: int = 50,
              refit_pairs: int = 16, K_refit: int = 5, refit_lr: float = 3e-4,
              probe_seed: int = 0, device: torch.device | str = "cpu",
              kresample_K: int = 0, n_holdout: int = 0) -> dict:
    """Top-level probe call. Returns a dict of scalars + the raw vectors.

    Does NOT modify the live actor or reward_model in any way.

    kresample_K > 0 enables Phase 3 (K-resample decomposition at this checkpoint).
    n_holdout > 0 enables Phase 2 (held-out on-policy accuracy of the live RM).
    """
    if isinstance(device, str):
        device = torch.device(device)
    rng = np.random.default_rng(probe_seed)
    # Seed both torch and legacy numpy RNGs so the rollout, Adam init, ψ' SGD, and
    # get_label() (which uses np.random.rand) are all reproducible at a given probe_seed.
    torch.manual_seed(int(probe_seed) & 0x7FFFFFFF)
    np.random.seed(int(probe_seed) & 0x7FFFFFFF)

    # ---- 1. fresh probe batch ------------------------------------------------
    batch = _rollout_probe_segments(env_factory, actor, n_probe, segment_len, device)

    # ---- 2. g0 under current RM ---------------------------------------------
    actor_was_training = actor.training
    actor.train()
    g0, proxy_mean_t, proxy_std_t = _reinforce_probe_gradient(
        actor, reward_model.ensemble, batch, device, centered=True
    )

    # ---- 3. ψ' refit (with per-step trajectory for matched-step-norm analysis) ----
    (refit_inputs, refit_labels, refit_label_fallback,
     holdout_inputs, holdout_labels_teacher, holdout_labels_gold,
     teacher_gold_disagreement) = _build_refit_pairs(
        batch, reward_model, refit_pairs, rng, n_holdout=n_holdout
    )
    ensemble_p, refit_traj = _refit_psi_prime(
        reward_model.ensemble, refit_inputs, refit_labels, K_refit, refit_lr, device
    )

    # ---- 3b. Phase 2 (corrected): held-out on-policy accuracy of the LIVE RM,
    #          reported vs BOTH gold (primary, comparable across teachers) and
    #          vs teacher's noisy label (legacy, kept for the fitting-noise gap).
    if n_holdout > 0:
        heldout_acc_gold, heldout_acc_teacher, heldout_n_gold, heldout_n_teacher = _heldout_acc(
            reward_model.ensemble, holdout_inputs,
            holdout_labels_teacher, holdout_labels_gold, device
        )
    else:
        heldout_acc_gold = float("nan")
        heldout_acc_teacher = float("nan")
        heldout_n_gold = 0
        heldout_n_teacher = 0

    # ---- 4. g0' on the same probe batch under ψ' (final) -------------------------
    g0p, proxy_mean_p, proxy_std_p = _reinforce_probe_gradient(
        actor, ensemble_p, batch, device, centered=True
    )

    # ---- 4b. Per-refit-step κ̂_k trajectory ---------------------------------------
    # Compute g0' at each intermediate ψ_k by stamping the snapshot into ensemble_p,
    # then computing the REINFORCE gradient. κ̂_k = ‖g0'_k − g0‖ / ‖g0‖.
    # Note: ψ_0 is the live RM (κ̂_0 ≡ 0 trivially) — start from k=1.
    kappa_traj = []
    g0p_norm_traj = []
    snapshots = refit_traj.get("param_snapshots", [])
    for k in range(1, len(snapshots)):
        _restore_ensemble_params(ensemble_p, snapshots[k])
        g0p_k, _, _ = _reinforce_probe_gradient(
            actor, ensemble_p, batch, device, centered=True
        )
        with torch.no_grad():
            n_perp_k = float((g0p_k - g0).norm().item())
            kappa_traj.append(n_perp_k / max(float(g0.norm().item()), 1e-30))
            g0p_norm_traj.append(float(g0p_k.norm().item()))
    # Restore final ψ' so g0p / proxy_mean_p computed above remains the "endpoint" value
    if snapshots:
        _restore_ensemble_params(ensemble_p, snapshots[-1])

    # ---- 5. diagnostics -----------------------------------------------------
    with torch.no_grad():
        g_perp = g0p - g0
        n_g0 = float(g0.norm().item())
        n_perp = float(g_perp.norm().item())
        n_g0p = float(g0p.norm().item())
        kappa = n_perp / max(n_g0, 1e-30)
        rho = float((g0 * g_perp).sum().item() / max(n_g0 * n_perp, 1e-30))
        # R̂ = ⟨g0', g0⟩ / ‖g0‖²
        R_inner = float((g0p * g0).sum().item() / max(n_g0 ** 2, 1e-30))
        R_identity = 1.0 + kappa * rho

    # also compute ensemble predictive variance on the same probe batch — used as baseline.
    obs = batch["obs"]; act = batch["action"]
    sa_flat = np.concatenate([obs, act], axis=-1).reshape(-1, obs.shape[-1] + act.shape[-1])
    with torch.no_grad():
        member_r = []
        for member_net in reward_model.ensemble:
            r_mem = member_net(torch.as_tensor(sa_flat, dtype=torch.float32, device=device)).cpu().numpy()
            member_r.append(r_mem.reshape(-1))
        member_r = np.stack(member_r, axis=0)
        ens_var = float(member_r.var(axis=0).mean())

    # cleanup ψ' (free GPU mem) — done BEFORE K-resample so the next refit allocates fresh.
    for m in ensemble_p:
        for p in m.parameters():
            p.grad = None
    del ensemble_p

    # ---- 6. Phase 3: K-resample decomposition (optional, expensive) --------------
    if kresample_K and kresample_K > 1:
        kresample_out = _kresample_decomposition(
            actor=actor, reward_model=reward_model, batch=batch, g0=g0,
            refit_pairs=refit_pairs, K_refit=K_refit, refit_lr=refit_lr,
            K_resamples=int(kresample_K), base_seed=int(probe_seed),
            device=device,
        )
    else:
        kresample_out = {}

    # Restore actor training mode so the live training loop is unaffected
    actor.train(actor_was_training)

    # Final-step refit diagnostics (endpoint)
    refit_param_step_norm_final = float(refit_traj["param_step_norms"][-1]) if refit_traj["param_step_norms"] else float("nan")
    refit_loss_decrement_final = (
        float(refit_traj["losses"][0] - refit_traj["losses"][-1])
        if len(refit_traj["losses"]) >= 2 else float("nan")
    )

    return dict(
        kappa=kappa,
        rho=rho,
        R_inner=R_inner,
        R_identity=R_identity,
        g0_norm=n_g0,
        g0p_norm=n_g0p,
        gperp_norm=n_perp,
        ensemble_variance=ens_var,
        proxy_mean=proxy_mean_t,
        proxy_std=proxy_std_t,
        probe_batch_gold_return=float(batch["gold"].sum(axis=1).mean()),
        n_probe=int(n_probe),
        segment_len=int(segment_len),
        refit_pairs=int(refit_pairs),
        K_refit=int(K_refit),
        refit_label_fallback=bool(refit_label_fallback),
        # Per-refit-step trajectory (analysis interpolates κ̂ at matched param_step_norm
        # across capacities so the κ̂ ordering isn't a refit-distance artifact).
        refit_param_step_norms=[float(x) for x in refit_traj["param_step_norms"]],
        refit_losses=[float(x) for x in refit_traj["losses"]],
        refit_kappa_trajectory=[float(x) for x in kappa_traj],
        refit_g0p_norm_trajectory=[float(x) for x in g0p_norm_traj],
        refit_param_step_norm_final=refit_param_step_norm_final,
        refit_loss_decrement_final=refit_loss_decrement_final,
        # Phase 2 (corrected per protocol §6): held-out accuracy of the LIVE RM.
        # Primary metric: accuracy vs GOLD label (comparable across teachers).
        # Legacy metric: accuracy vs TEACHER label (kept; gap_to_gold = fitting noise).
        heldout_acc_gold=float(heldout_acc_gold),
        heldout_acc_teacher=float(heldout_acc_teacher),
        heldout_n_gold=int(heldout_n_gold),
        heldout_n_teacher=int(heldout_n_teacher),
        # Backward-compat shim: older analysis code reads heldout_acc_onpolicy.
        heldout_acc_onpolicy=float(heldout_acc_gold),
        heldout_n=int(heldout_n_gold),
        # Teacher's empirical disagreement with gold on this probe's refit batch.
        # Used to match noise level across teacher configurations (protocol §1).
        teacher_gold_disagreement=float(teacher_gold_disagreement),
        # Phase 3 (optional, present only when kresample_K > 1):
        **kresample_out,
    )
