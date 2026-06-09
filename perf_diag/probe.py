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


def _refit_psi_prime(ensemble_src, refit_pairs_inputs, refit_pairs_labels, K_refit: int,
                     lr: float, device: torch.device):
    """Deep-copy ensemble; take K_refit Adam SGD steps on the BT loss over the given
    preference pairs; return the refitted ensemble (still on device).

    refit_pairs_inputs:  (sa_t_1, sa_t_2) each (P, L, ds+da) np.float32
    refit_pairs_labels:  (P,) int64 in {0, 1}
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
    if labels.numel() == 0:
        return ensemble_p  # nothing to refit; same as warm start

    ce = torch.nn.CrossEntropyLoss()
    for _ in range(K_refit):
        opt.zero_grad()
        loss = 0.0
        for member in ensemble_p:
            r1 = member(sa1).reshape(sa1.shape[0], sa1.shape[1]).sum(dim=1, keepdim=True)
            r2 = member(sa2).reshape(sa2.shape[0], sa2.shape[1]).sum(dim=1, keepdim=True)
            logits = torch.cat([r1, r2], dim=-1)                  # (P, 2)
            loss = loss + ce(logits, labels)
        loss.backward()
        opt.step()
    return ensemble_p


def _build_refit_pairs(probe_batch: dict, reward_model, refit_pairs: int,
                       rng: np.random.Generator) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray]:
    """Build a fresh preference batch from the on-policy probe segments.

    Pairs are sampled with replacement from the N_probe segments. The scripted teacher
    in reward_model.get_label is used for labels (the same teacher that drives training).
    """
    N = probe_batch["obs"].shape[0]
    L = probe_batch["obs"].shape[1]
    idx1 = rng.integers(0, N, size=refit_pairs)
    idx2 = rng.integers(0, N, size=refit_pairs)
    sa_t_1 = np.concatenate([probe_batch["obs"][idx1], probe_batch["action"][idx1]], axis=-1).astype(np.float32)
    sa_t_2 = np.concatenate([probe_batch["obs"][idx2], probe_batch["action"][idx2]], axis=-1).astype(np.float32)
    r_t_1 = probe_batch["gold"][idx1].reshape(refit_pairs, L, 1).astype(np.float32)
    r_t_2 = probe_batch["gold"][idx2].reshape(refit_pairs, L, 1).astype(np.float32)
    # Reuse the live teacher; it consumes gold returns and emits noisy BT labels
    sa_t_1, sa_t_2, r_t_1, r_t_2, labels = reward_model.get_label(sa_t_1, sa_t_2, r_t_1, r_t_2)
    if labels is None or len(labels) == 0:
        # fall back to perfectly-rational labels on the same pairs if the teacher skipped all
        labels = (r_t_1.sum(1).reshape(-1) < r_t_2.sum(1).reshape(-1)).astype(np.int64)
        sa_t_1 = np.concatenate([probe_batch["obs"][idx1], probe_batch["action"][idx1]], axis=-1).astype(np.float32)
        sa_t_2 = np.concatenate([probe_batch["obs"][idx2], probe_batch["action"][idx2]], axis=-1).astype(np.float32)
    return (sa_t_1, sa_t_2), np.asarray(labels).astype(np.int64)


def run_probe(actor, reward_model, env_factory,
              n_probe: int = 8, segment_len: int = 50,
              refit_pairs: int = 16, K_refit: int = 5, refit_lr: float = 3e-4,
              probe_seed: int = 0, device: torch.device | str = "cpu") -> dict:
    """Top-level probe call. Returns a dict of scalars + the raw vectors.

    Does NOT modify the live actor or reward_model in any way.
    """
    if isinstance(device, str):
        device = torch.device(device)
    rng = np.random.default_rng(probe_seed)
    # Seed torch RNG so dist.sample() in the rollout, deep-copies' Adam init, and
    # ψ' SGD trajectory are all reproducible across reruns at the same probe_seed.
    torch.manual_seed(int(probe_seed) & 0x7FFFFFFF)

    # ---- 1. fresh probe batch ------------------------------------------------
    batch = _rollout_probe_segments(env_factory, actor, n_probe, segment_len, device)

    # ---- 2. g0 under current RM ---------------------------------------------
    actor.train()
    g0, proxy_mean_t, proxy_std_t = _reinforce_probe_gradient(
        actor, reward_model.ensemble, batch, device, centered=True
    )

    # ---- 3. ψ' refit --------------------------------------------------------
    refit_inputs, refit_labels = _build_refit_pairs(batch, reward_model, refit_pairs, rng)
    ensemble_p = _refit_psi_prime(
        reward_model.ensemble, refit_inputs, refit_labels, K_refit, refit_lr, device
    )

    # ---- 4. g0' on the same probe batch under ψ' ----------------------------
    g0p, proxy_mean_p, proxy_std_p = _reinforce_probe_gradient(
        actor, ensemble_p, batch, device, centered=True
    )

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

    # cleanup ψ' (free GPU mem)
    for m in ensemble_p:
        for p in m.parameters():
            p.grad = None
    del ensemble_p

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
    )
