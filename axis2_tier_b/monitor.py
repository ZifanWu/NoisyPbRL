"""
Axis 2 Tier B monitoring module.

All five public functions are called at fixed-step monitoring events with the agent and RM in
eval mode.  They co-measure three gold-free instruments (kappa_hat, ensemble_spread, gof) and
three error components (e_shift, e_epi, e_mis) at the same frozen theta.

Key design choices (see plan for rationale):
- probe_obs is refreshed every call to stay on current on-policy support (not frozen).
- n_fin is read from reward_model.total_feedback (set by the workspace) to match the deployed
  RM's actual consumed label budget, keeping e_shift purely about staleness.
- R_star comes from recent_targets (true env rewards, not preference labels).
- GoF floor is max(H(eps), floor_ref) where floor_ref is psi_inf's held-out CE loss.
- GoF is computed on both the deployed RM (gof_deployed) and psi_inf (gof_inf = pure misspec).
"""

import copy
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _concat_sa(obs: np.ndarray, acts: np.ndarray) -> np.ndarray:
    """Concatenate (N, obs_dim) and (N, act_dim) -> (N, obs_dim+act_dim)."""
    return np.concatenate([obs, acts], axis=-1)


def _rm_rewards(rm, sa: np.ndarray) -> np.ndarray:
    """Return per-step rewards from rm for sa array (N, d). numpy output."""
    with torch.no_grad():
        x = torch.from_numpy(sa).float().to(next(rm.ensemble[0].parameters()).device)
        r_list = [m(x).squeeze(-1).cpu().numpy() for m in rm.ensemble]
    return np.mean(r_list, axis=0)  # (N,)


def _rm_rewards_per_member(rm, sa: np.ndarray) -> np.ndarray:
    """Return (n_members, N) rewards from each ensemble member."""
    with torch.no_grad():
        x = torch.from_numpy(sa).float().to(next(rm.ensemble[0].parameters()).device)
        return np.stack([m(x).squeeze(-1).cpu().numpy() for m in rm.ensemble], axis=0)


def _gauge_fix(R_psi: np.ndarray, R_ref: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Per-segment center R_psi and R_ref, then scale-align R_psi to R_ref.

    Both arrays: (N_segs, T).
    Returns scale-aligned R_psi_aligned and the global scale s*.
    """
    # per-segment center
    R_psi_c = R_psi - R_psi.mean(axis=1, keepdims=True)
    R_ref_c = R_ref - R_ref.mean(axis=1, keepdims=True)
    denom = np.sum(R_psi_c ** 2)
    s_star = np.sum(R_psi_c * R_ref_c) / (denom + 1e-12)
    return s_star * R_psi_c, s_star


def _segment_l2(A: np.ndarray, B: np.ndarray) -> float:
    """Mean per-element L2 distance over (N_segs, T) arrays."""
    return float(np.mean((A - B) ** 2) ** 0.5)


def _build_clean_pref_pairs(
    inputs: List[np.ndarray],
    targets: List[np.ndarray],
    seg_len: int,
    n_pairs: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample n_pairs preference pairs from recent on-policy data with clean (eps=0) labels.

    inputs:  list of trajectory arrays (T_ep, obs_dim+act_dim)
    targets: list of true per-step env rewards (T_ep,)
    Returns: (seg1, seg2, labels) arrays of shape (n_pairs, seg_len, d), (n_pairs,).
    """
    # collect all valid segments (length >= seg_len) from recent episodes
    segs_sa = []
    segs_ret = []
    for sa, r in zip(inputs, targets):
        if len(sa) < seg_len:
            continue
        n_full = len(sa) // seg_len
        for i in range(n_full):
            segs_sa.append(sa[i * seg_len: (i + 1) * seg_len])
            chunk = r[i * seg_len: (i + 1) * seg_len]
            segs_ret.append(float(np.sum(chunk)))

    if len(segs_sa) < 2:
        return None, None, None

    segs_sa = np.array(segs_sa)   # (N_segs, seg_len, d)
    segs_ret = np.array(segs_ret)  # (N_segs,)

    idx1 = rng.integers(0, len(segs_sa), size=n_pairs)
    idx2 = rng.integers(0, len(segs_sa), size=n_pairs)
    # avoid same-segment pairs
    same = idx1 == idx2
    idx2[same] = (idx2[same] + 1) % len(segs_sa)

    # clean BT label: prefer seg with higher true return
    labels = (segs_ret[idx2] > segs_ret[idx1]).astype(np.int64)
    # mark as equal (-1) when returns are identical
    labels[segs_ret[idx1] == segs_ret[idx2]] = -1

    return segs_sa[idx1], segs_sa[idx2], labels


def _build_teacher_pref_pairs(
    inputs: List[np.ndarray],
    targets: List[np.ndarray],
    seg_len: int,
    n_pairs: int,
    rng: np.random.Generator,
    rm_template=None,
    teacher_eps_mistake: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build preference pairs labeled by the SAME teacher used to train rm_template.

    If rm_template is provided and rm_template.teacher_beta > 0, applies the BT teacher
    (with gap-std normalization if rm_template.bt_normalize_by_gap_std=True), using the
    template's running gap_std_ema so the noise level matches the deployed pipeline exactly.

    teacher_eps_mistake (i.i.d. flip on top) is taken from rm_template if provided, else from
    the explicit argument.  Default behaviour with no rm_template falls back to clean labels +
    optional eps_mistake flip, matching the previous oracle-style implementation.
    """
    # collect all valid segments
    segs_sa = []
    segs_ret = []
    for sa, r in zip(inputs, targets):
        if len(sa) < seg_len:
            continue
        r_flat = np.asarray(r).squeeze(-1) if np.asarray(r).ndim == 2 else np.asarray(r)
        n_full = len(sa) // seg_len
        for i in range(n_full):
            segs_sa.append(sa[i * seg_len: (i + 1) * seg_len])
            segs_ret.append(float(np.sum(r_flat[i * seg_len: (i + 1) * seg_len])))

    if len(segs_sa) < 2:
        return None, None, None

    segs_sa = np.array(segs_sa)
    segs_ret = np.array(segs_ret)

    idx1 = rng.integers(0, len(segs_sa), size=n_pairs)
    idx2 = rng.integers(0, len(segs_sa), size=n_pairs)
    same = idx1 == idx2
    idx2[same] = (idx2[same] + 1) % len(segs_sa)

    r1 = segs_ret[idx1]
    r2 = segs_ret[idx2]

    # Pull teacher params from rm_template if provided; otherwise use oracle + explicit eps.
    if rm_template is not None:
        t_beta = rm_template.teacher_beta
        t_eps = rm_template.teacher_eps_mistake
        bt_norm = getattr(rm_template, 'bt_normalize_by_gap_std', False)
        gap_std_ema = getattr(rm_template, 'gap_std_ema', None)
    else:
        t_beta = -1
        t_eps = teacher_eps_mistake
        bt_norm = False
        gap_std_ema = None

    if t_beta > 0:
        # BT teacher: P(label=1) = sigmoid(effective_beta * (r2 - r1))
        effective_beta = t_beta
        if bt_norm and gap_std_ema is not None and gap_std_ema > 1e-8:
            effective_beta = t_beta / gap_std_ema
        # numerically stable sigmoid
        delta = effective_beta * (r2 - r1)
        prob_label_1 = 1.0 / (1.0 + np.exp(-np.clip(delta, -50.0, 50.0)))
        labels = (rng.random(size=n_pairs) < prob_label_1).astype(np.int64)
    else:
        # Oracle: deterministic by higher cumulative return
        labels = (r2 > r1).astype(np.int64)

    # eps_mistake flip on top (covers the mis confirmatory run that intentionally uses eps>0)
    if t_eps > 0:
        flip = rng.random(size=n_pairs) < t_eps
        labels[flip] = 1 - labels[flip]

    # equal-preference marker
    labels[r1 == r2] = -1

    return segs_sa[idx1], segs_sa[idx2], labels


def _train_fresh_rm(
    rm_template,          # RewardModel used for ds/da/ensemble_size (and arch defaults)
    seg1: np.ndarray,
    seg2: np.ndarray,
    labels: np.ndarray,
    lr: float = 3e-4,
    max_epochs: int = 500,
    conv_tol: float = 1e-4,
    device: str = 'cuda',
    hidden_dim: Optional[int] = None,
    num_layers: Optional[int] = None,
    output_activation: Optional[str] = None,
) -> object:
    """
    Train a brand-new RewardModel on (seg1, seg2, labels).

    Architecture defaults to rm_template's, but can be overridden — critical for the mis
    condition, where ψ_fin/ψ_inf must use the full-capacity yardstick architecture rather
    than the deployed RM's shrunk width.  Without this override, e_mis collapses to zero by
    construction (yardstick can't outperform a model with identical capacity).

    Uses early stopping: stops when loss improvement < conv_tol for 10 consecutive epochs.
    Returns the trained RewardModel.
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from reward_model import RewardModel

    ds = rm_template.ds
    da = rm_template.da
    seg_len = seg1.shape[1]
    fresh_rm = RewardModel(
        ds=ds, da=da,
        ensemble_size=rm_template.de,
        lr=lr,
        size_segment=seg_len,
        hidden_dim=hidden_dim if hidden_dim is not None else rm_template.hidden_dim,
        num_layers=num_layers if num_layers is not None else rm_template.num_layers,
        output_activation=output_activation if output_activation is not None
                          else rm_template.output_activation,
    )

    # filter out equal-preference pairs
    valid = labels != -1
    s1 = seg1[valid].astype(np.float32)   # (N, T, d)
    s2 = seg2[valid].astype(np.float32)
    lbl = labels[valid].astype(np.int64)
    if len(lbl) == 0:
        return fresh_rm

    s1_t = torch.from_numpy(s1).to(device)
    s2_t = torch.from_numpy(s2).to(device)
    lbl_t = torch.from_numpy(lbl).to(device)

    ce = nn.CrossEntropyLoss()
    prev_loss = float('inf')
    no_improve = 0

    for epoch in range(max_epochs):
        fresh_rm.opt.zero_grad()
        loss = 0.0
        for m in range(fresh_rm.de):
            r1 = fresh_rm.ensemble[m](s1_t.view(-1, ds + da)).view(len(s1), seg_len).sum(dim=1)
            r2 = fresh_rm.ensemble[m](s2_t.view(-1, ds + da)).view(len(s2), seg_len).sum(dim=1)
            logits = torch.stack([r1, r2], dim=1)
            loss += ce(logits, lbl_t)
        loss.backward()
        fresh_rm.opt.step()

        loss_val = loss.item() / fresh_rm.de
        if prev_loss - loss_val < conv_tol:
            no_improve += 1
            if no_improve >= 10:
                break
        else:
            no_improve = 0
        prev_loss = loss_val

    return fresh_rm


def _bt_ce_loss(rm, seg1: np.ndarray, seg2: np.ndarray, labels: np.ndarray, device: str) -> float:
    """
    Compute mean cross-entropy BT loss of rm on (seg1, seg2, labels).
    Skips label=-1 pairs.  Returns loss value.
    """
    valid = labels != -1
    if valid.sum() == 0:
        return 0.0
    s1 = torch.from_numpy(seg1[valid].astype(np.float32)).to(device)
    s2 = torch.from_numpy(seg2[valid].astype(np.float32)).to(device)
    lbl = torch.from_numpy(labels[valid].astype(np.int64)).to(device)
    seg_len = s1.shape[1]
    ds_da = s1.shape[2]
    ce = nn.CrossEntropyLoss()
    with torch.no_grad():
        r1_all = []
        r2_all = []
        for m in rm.ensemble:
            r1_all.append(m(s1.view(-1, ds_da)).view(len(s1), seg_len).sum(dim=1))
            r2_all.append(m(s2.view(-1, ds_da)).view(len(s2), seg_len).sum(dim=1))
        r1 = torch.stack(r1_all).mean(0)
        r2 = torch.stack(r2_all).mean(0)
        logits = torch.stack([r1, r2], dim=1)
        return ce(logits, lbl).item()


# ---------------------------------------------------------------------------
# MonitorState
# ---------------------------------------------------------------------------

@dataclass
class MonitorState:
    """Mutable per-run state for the monitoring pipeline."""
    seg_len: int = 50
    probe_seg_size: int = 200
    probe_size: int = 1024
    kappa_warmup_steps: int = 10
    kappa_fresh_prefs: int = 64
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_kappa_hat(
    agent,
    reward_model,
    recent_inputs: List[np.ndarray],
    recent_targets: List[np.ndarray],
    state: MonitorState,
    device: str,
) -> Tuple[float, float]:
    """
    Estimate kappa_hat and sigma_hat via REINFORCE finite-difference probe.

    probe_obs is refreshed from current on-policy data every call (not frozen) so that
    the measurement is on the same support as the error components.

    Returns (kappa_hat, sigma_hat). Both NaN if g0 is near-zero.
    """
    # --- build probe obs from recent on-policy data ---
    all_sa = np.concatenate([ep for ep in recent_inputs if len(ep) > 0], axis=0)
    n = min(state.probe_size, len(all_sa))
    idx = state.rng.choice(len(all_sa), size=n, replace=False)
    probe_sa = all_sa[idx]   # (n, obs_dim + act_dim)

    ds = reward_model.ds
    da = reward_model.da
    probe_obs_np = probe_sa[:, :ds]
    probe_obs = torch.from_numpy(probe_obs_np.astype(np.float32)).to(device)

    # --- sample probe actions from current policy ---
    agent.actor.eval()
    with torch.no_grad():
        dist = agent.actor(probe_obs)
        probe_acts = dist.sample().clamp(*agent.action_range)  # (n, act_dim)

    probe_sa_np = _concat_sa(probe_obs_np, probe_acts.cpu().numpy())

    # --- r_cur: current RM reward, detached ---
    r_cur = torch.from_numpy(_rm_rewards(reward_model, probe_sa_np).astype(np.float32)).to(device)

    # --- g0: REINFORCE gradient w.r.t. actor params ---
    agent.actor.zero_grad()
    dist = agent.actor(probe_obs)
    log_prob = dist.log_prob(probe_acts).sum(-1)  # (n,)
    loss_g0 = -(r_cur.detach() * log_prob).mean()
    loss_g0.backward()
    g0_vec = torch.cat([p.grad.detach().flatten() for p in agent.actor.parameters()
                        if p.grad is not None])
    agent.actor.zero_grad()

    g0_norm = g0_vec.norm().item()
    if g0_norm < 1e-12:
        return float('nan'), float('nan')

    # --- clone RM and warm-start on fresh on-policy prefs labeled by the SAME teacher
    #     (BT-with-normalization if reward_model.teacher_beta > 0) so the clone reflects the
    #     same noise model as the deployed RM, not a deterministic+flip approximation.
    rm_clone = copy.deepcopy(reward_model)
    seg1_f, seg2_f, lbl_f = _build_teacher_pref_pairs(
        recent_inputs, recent_targets,
        seg_len=state.seg_len,
        n_pairs=state.kappa_fresh_prefs,
        rng=state.rng,
        rm_template=reward_model,
    )

    if seg1_f is not None:
        valid = lbl_f != -1
        if valid.sum() > 0:
            s1_t = torch.from_numpy(seg1_f[valid].astype(np.float32)).to(device)
            s2_t = torch.from_numpy(seg2_f[valid].astype(np.float32)).to(device)
            lbl_t = torch.from_numpy(lbl_f[valid].astype(np.int64)).to(device)
            ce = nn.CrossEntropyLoss()
            opt_clone = torch.optim.Adam(
                list(p for m in rm_clone.ensemble for p in m.parameters()), lr=3e-4)
            for _ in range(state.kappa_warmup_steps):
                opt_clone.zero_grad()
                loss_c = sum(
                    ce(torch.stack([
                        rm_clone.ensemble[m](s1_t.view(-1, ds + da))
                                   .view(len(s1_t), state.seg_len).sum(1),
                        rm_clone.ensemble[m](s2_t.view(-1, ds + da))
                                   .view(len(s2_t), state.seg_len).sum(1),
                    ], dim=1), lbl_t)
                    for m in range(rm_clone.de)
                )
                loss_c.backward()
                opt_clone.step()

    # --- r_prime: cloned RM reward on same probe ---
    r_prime = torch.from_numpy(_rm_rewards(rm_clone, probe_sa_np).astype(np.float32)).to(device)
    delta_r = (r_prime - r_cur).detach()

    # --- g_delta: REINFORCE gradient weighted by delta_r ---
    dist2 = agent.actor(probe_obs)
    log_prob2 = dist2.log_prob(probe_acts).sum(-1)
    loss_gd = -(delta_r * log_prob2).mean()
    loss_gd.backward()
    gd_vec = torch.cat([p.grad.detach().flatten() for p in agent.actor.parameters()
                        if p.grad is not None])
    agent.actor.zero_grad()

    kappa_hat = gd_vec.norm().item() / (g0_norm + 1e-12)
    sigma_hat = float(torch.dot(g0_vec, gd_vec).item() /
                      (g0_norm * gd_vec.norm().item() + 1e-12))
    return kappa_hat, sigma_hat


def compute_ensemble_spread(
    reward_model,
    probe_segs: np.ndarray,
) -> float:
    """
    Per-segment-centered variance across ensemble members on probe_segs.

    probe_segs: (N_segs, T, obs_dim+act_dim)
    Returns scalar spread value.
    """
    N, T, d = probe_segs.shape
    flat = probe_segs.reshape(-1, d)
    # (n_members, N*T)
    per_member = _rm_rewards_per_member(reward_model, flat).reshape(reward_model.de, N, T)
    # per-segment center each member
    centered = per_member - per_member.mean(axis=2, keepdims=True)  # (M, N, T)
    # variance across members at each (n, t)
    var = centered.var(axis=0)  # (N, T)
    return float(var.mean())


def compute_gof(
    reward_model,
    seg1: np.ndarray,
    seg2: np.ndarray,
    labels: np.ndarray,
    teacher_eps_mistake: float,
    device: str,
    reference_rm=None,
) -> Tuple[float, float]:
    """
    GoF residual = held-out BT CE loss - floor.

    Floor = max(H(eps), floor_ref) where floor_ref is reference_rm's CE loss on same pairs
    (reference_rm is psi_inf from compute_error_components).

    Returns (gof, floor_used).
    """
    if seg1 is None or len(seg1) == 0:
        return float('nan'), float('nan')

    loss = _bt_ce_loss(reward_model, seg1, seg2, labels, device)

    # analytical floor: binary entropy of mistake rate
    eps = float(teacher_eps_mistake)
    if eps > 0 and eps < 1:
        floor_he = -eps * math.log(eps) - (1.0 - eps) * math.log(1.0 - eps)
    else:
        floor_he = 0.0

    floor_ref = 0.0
    if reference_rm is not None:
        floor_ref = _bt_ce_loss(reference_rm, seg1, seg2, labels, device)

    floor = max(floor_he, floor_ref)
    gof = max(0.0, loss - floor)
    return gof, floor


def compute_error_components(
    reward_model_cur,
    n_fin: int,
    recent_inputs: List[np.ndarray],
    recent_targets: List[np.ndarray],
    state: MonitorState,
    device: str,
    lr: float = 3e-4,
    max_epochs: int = 500,
    conv_tol: float = 1e-4,
    # When None (default), ψ_fin/ψ_inf are refits of the SAME RM class as reward_model_cur.
    # This is the spec: ψ_inf measures the irreducible floor *of the diagnosed class*.
    # Override to a fixed full-capacity arch only when you explicitly want a cross-class
    # yardstick (e.g., small/fast smoke tests in sanity_checks.py).
    yardstick_hidden_dim: Optional[int] = None,
    yardstick_num_layers: Optional[int] = None,
    yardstick_output_activation: Optional[str] = None,
):
    """
    Compute e_shift, e_epi, e_mis via three nested refits at current theta.

    n_fin = reward_model_cur.total_feedback (passed from workspace).
    n_inf = 10 * n_fin.
    recent_targets contains true per-step env rewards (NOT preference labels).

    Returns dict with keys:
      e_shift, e_epi, e_mis, psi_fin, psi_inf
    """
    n_inf = max(10 * n_fin, 200)  # ensure at least 200 pairs for psi_inf

    rng = state.rng
    seg_len = state.seg_len

    # --- build probe segments (carry R_star from recent_targets) ---
    probe_sa_list = []
    probe_rstar_list = []
    for sa, r in zip(recent_inputs, recent_targets):
        if len(sa) < seg_len:
            continue
        # r may be (T, 1) from add_data's reshape(1,1); squeeze to (T,)
        r_flat = np.asarray(r).squeeze(-1) if np.asarray(r).ndim == 2 else np.asarray(r)
        n_full = len(sa) // seg_len
        for i in range(n_full):
            probe_sa_list.append(sa[i * seg_len: (i + 1) * seg_len])
            probe_rstar_list.append(r_flat[i * seg_len: (i + 1) * seg_len])

    if len(probe_sa_list) == 0:
        nan = float('nan')
        return dict(e_shift=nan, e_epi=nan, e_mis=nan, psi_fin=None, psi_inf=None)

    # cap number of probe segments
    n_probe = min(len(probe_sa_list), state.probe_seg_size)
    sel = rng.choice(len(probe_sa_list), size=n_probe, replace=False)
    probe_sa = np.array([probe_sa_list[i] for i in sel])    # (n_probe, T, d)
    probe_rstar = np.array([probe_rstar_list[i] for i in sel])  # (n_probe, T) — true env reward

    # --- build preference pairs labeled by the SAME teacher as reward_model_cur.
    # Under BT consistency, ψ_inf → β·R* up to gauge regardless of noise → asymptotically clean.
    # Using the deployed teacher (not clean) ensures ψ_fin captures the deployed RM's finite-sample
    # noise model, so e_epi reflects the actual reducible variance ψ_cur is subject to.
    seg1_fin, seg2_fin, lbl_fin = _build_teacher_pref_pairs(
        recent_inputs, recent_targets, seg_len, n_fin, rng,
        rm_template=reward_model_cur)
    seg1_inf, seg2_inf, lbl_inf = _build_teacher_pref_pairs(
        recent_inputs, recent_targets, seg_len, n_inf, rng,
        rm_template=reward_model_cur)

    if seg1_fin is None:
        nan = float('nan')
        return dict(e_shift=nan, e_epi=nan, e_mis=nan, psi_fin=None, psi_inf=None)

    # --- train psi_fin and psi_inf with the YARDSTICK arch (full capacity, NOT shrunk) ---
    # In the mis condition reward_model_cur has shrunk width; if we copied that here,
    # the yardstick could not outperform it and e_mis would collapse to zero by construction.
    psi_fin = _train_fresh_rm(
        reward_model_cur, seg1_fin, seg2_fin, lbl_fin,
        lr=lr, max_epochs=max_epochs, conv_tol=conv_tol, device=device,
        hidden_dim=yardstick_hidden_dim,
        num_layers=yardstick_num_layers,
        output_activation=yardstick_output_activation,
    )
    psi_inf = _train_fresh_rm(
        reward_model_cur, seg1_inf, seg2_inf, lbl_inf,
        lr=lr, max_epochs=max_epochs, conv_tol=conv_tol, device=device,
        hidden_dim=yardstick_hidden_dim,
        num_layers=yardstick_num_layers,
        output_activation=yardstick_output_activation,
    )

    # --- evaluate all three RMs on probe segments ---
    flat_probe = probe_sa.reshape(-1, probe_sa.shape[-1])
    R_cur = _rm_rewards(reward_model_cur, flat_probe).reshape(n_probe, seg_len)
    R_fin = _rm_rewards(psi_fin, flat_probe).reshape(n_probe, seg_len)
    R_inf = _rm_rewards(psi_inf, flat_probe).reshape(n_probe, seg_len)
    R_star = probe_rstar   # (n_probe, seg_len) — true env rewards

    # --- gauge fix: per-segment center + global scale ---
    g_cur, _ = _gauge_fix(R_cur, R_fin)
    g_fin, _ = _gauge_fix(R_fin, R_inf)
    g_inf, _ = _gauge_fix(R_inf, R_star)

    # normalise by ||R_star||
    rstar_norm = float(np.sqrt(np.mean(R_star ** 2))) + 1e-12

    e_shift = _segment_l2(g_cur, R_fin - R_fin.mean(axis=1, keepdims=True)) / rstar_norm
    e_epi   = _segment_l2(g_fin, R_inf - R_inf.mean(axis=1, keepdims=True)) / rstar_norm
    e_mis   = _segment_l2(g_inf, R_star - R_star.mean(axis=1, keepdims=True)) / rstar_norm

    return dict(e_shift=e_shift, e_epi=e_epi, e_mis=e_mis, psi_fin=psi_fin, psi_inf=psi_inf)
