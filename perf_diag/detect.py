"""Firing rules, turnover detection, lead-time vs FAR curves, bootstrap CIs.

A signal is a 1-D array of real values indexed by monitoring step. A firing
rule converts (signal, threshold, K_persist) → first-fire-index or None.

Conventions:
- All signals are oriented "larger ⇒ more alarming".
- `t_gold` is the argmax of the smoothed gold-return curve (alternative: first
  sustained-decline-below-running-max for K_decline consecutive logging steps).
- `lead = t_gold − t_fire` if t_fire ≤ t_gold, else negative (a "late" detection).
- A `false alarm` is firing on a negative-control run (where gold never turns over).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from . import baselines


# ---------------------------------------------------------------------------
# Firing rule
# ---------------------------------------------------------------------------
def cusum_fire(signal: np.ndarray, threshold: float, K_persist: int = 3) -> Optional[int]:
    """Return the first index where signal stays ≥ threshold for K_persist consecutive
    steps. None if never. K_persist=1 ⇒ first crossing.
    """
    s = np.asarray(signal, dtype=np.float64)
    if s.size < K_persist:
        return None
    run = 0
    for i, v in enumerate(s):
        if v >= threshold:
            run += 1
            if run >= K_persist:
                return i - K_persist + 1
        else:
            run = 0
    return None


# ---------------------------------------------------------------------------
# Turnover detection on the gold curve
# ---------------------------------------------------------------------------
def find_t_gold_argmax(gold_series: np.ndarray, alpha: float = 0.1) -> Optional[int]:
    g = baselines.ema_smooth(np.asarray(gold_series, dtype=np.float64), alpha=alpha)
    if g.size == 0:
        return None
    return int(np.argmax(g))


def find_t_gold_sustained_decline(gold_series: np.ndarray, K_decline: int = 5,
                                   alpha: float = 0.1) -> Optional[int]:
    """First index where the smoothed gold has stayed below running max for K_decline steps."""
    g = baselines.ema_smooth(np.asarray(gold_series, dtype=np.float64), alpha=alpha)
    if g.size < K_decline + 1:
        return None
    running_max = -np.inf
    run = 0
    for i, v in enumerate(g):
        if v > running_max:
            running_max = v
            run = 0
        elif v < running_max:
            run += 1
            if run >= K_decline:
                return i - K_decline + 1
        else:
            run = 0
    return None


def confirmed_turnover_info(
    gold_series: np.ndarray,
    proxy_series: np.ndarray | None = None,
    K_decline: int = 5,
    alpha: float = 0.2,
    min_drop_frac: float = 0.2,
    min_post_points: int = 8,
    max_peak_frac: float = 0.8,
    late_window: int = 5,
    require_proxy_noncollapse: bool = False,
) -> dict:
    """Strict reward-over-optimization event detector.

    A confirmed turnover is not just "some drop after a running max". We require:
      1. enough samples after the peak;
      2. the peak is not in the last tail of the horizon;
      3. the post-peak late-window mean is clearly below the peak;
      4. the post-peak curve stays below the drop threshold for K_decline points.

    If proxy_series is supplied and require_proxy_noncollapse=True, we also require
    the late proxy not to collapse along with gold. That gate is off by default
    because proxy traces can be noisy; Table 2 reports gold-confirmed turnover.
    """
    g_raw = np.asarray(gold_series, dtype=np.float64)
    g_raw = g_raw[np.isfinite(g_raw)]
    if g_raw.size == 0:
        return dict(turned_over=False, t_gold=None, reason="empty_gold")
    g = baselines.ema_smooth(g_raw, alpha=alpha)
    n = int(g.size)

    # Adapt absolute thresholds to trace length so short traces (~50 probes for
    # rare_relabel) are not penalized vs long traces (~170 probes for frequent_relabel).
    effective_min_post = min(min_post_points, max(3, int(0.07 * n)))
    effective_max_peak_frac = max(max_peak_frac, 1.0 - float(effective_min_post) / max(n - 1, 1))
    effective_K_decline = min(K_decline, effective_min_post)

    min_needed = max(effective_K_decline + 2, effective_min_post + 2, late_window + 2)
    if n < min_needed:
        return dict(turned_over=False, t_gold=None, reason="too_few_points", n=n)

    t_peak = int(np.argmax(g))
    n_post = n - t_peak - 1
    if n_post < effective_min_post:
        return dict(turned_over=False, t_gold=t_peak, reason="too_few_post_peak_points",
                    n=n, n_post=n_post)
    if t_peak > int(effective_max_peak_frac * (n - 1)):
        return dict(turned_over=False, t_gold=t_peak, reason="peak_too_late",
                    n=n, n_post=n_post)

    peak = float(g[t_peak])
    denom = max(abs(peak), 1e-8)
    threshold = peak - min_drop_frac * denom
    post = g[t_peak + 1:]
    late_n = min(late_window, post.size)
    late_mean = float(np.mean(post[-late_n:]))
    drop_frac = float((peak - late_mean) / denom)
    if drop_frac < min_drop_frac:
        return dict(turned_over=False, t_gold=t_peak, reason="late_drop_too_small",
                    n=n, n_post=n_post, peak=peak, late_mean=late_mean,
                    drop_frac=drop_frac)

    run = 0
    first_sustained = None
    for i, v in enumerate(post, start=t_peak + 1):
        if v <= threshold:
            run += 1
            if run >= effective_K_decline:
                first_sustained = i - effective_K_decline + 1
                break
        else:
            run = 0
    if first_sustained is None:
        return dict(turned_over=False, t_gold=t_peak, reason="no_sustained_post_peak_drop",
                    n=n, n_post=n_post, peak=peak, late_mean=late_mean,
                    drop_frac=drop_frac)

    # Always compute proxy trend when available — used to distinguish Goodhart
    # (proxy↑ while gold↓) from training instability (both collapse together).
    goodhart_confirmed = None   # None = proxy data unavailable or length mismatch
    proxy_at_peak = float("nan")
    proxy_late_mean = float("nan")
    if proxy_series is not None:
        p_raw = np.asarray(proxy_series, dtype=np.float64)
        p_raw = p_raw[np.isfinite(p_raw)]
        if p_raw.size == n:
            p = baselines.ema_smooth(p_raw, alpha=alpha)
            proxy_at_peak = float(p[t_peak])
            proxy_late_mean = float(np.mean(p[t_peak + 1:][-late_n:]))
            proxy_tol = 0.05 * max(abs(proxy_at_peak), 1e-8)
            # Goodhart: proxy did NOT collapse alongside gold (stayed ≥ peak - 5%)
            goodhart_confirmed = bool(proxy_late_mean >= proxy_at_peak - proxy_tol)
            if require_proxy_noncollapse and not goodhart_confirmed:
                return dict(turned_over=False, t_gold=t_peak, reason="proxy_collapsed_too",
                            n=n, n_post=n_post, peak=peak, late_mean=late_mean,
                            drop_frac=drop_frac, proxy_at_peak=proxy_at_peak,
                            proxy_late_mean=proxy_late_mean, goodhart_confirmed=False)

    return dict(turned_over=True, t_gold=t_peak, reason="confirmed",
                n=n, n_post=n_post, peak=peak, late_mean=late_mean,
                drop_frac=drop_frac, first_sustained=first_sustained,
                goodhart_confirmed=goodhart_confirmed,
                proxy_at_peak=proxy_at_peak, proxy_late_mean=proxy_late_mean)


def find_t_gold_confirmed(gold_series: np.ndarray, proxy_series: np.ndarray | None = None,
                          **kwargs) -> Optional[int]:
    info = confirmed_turnover_info(gold_series, proxy_series=proxy_series, **kwargs)
    return int(info["t_gold"]) if info.get("turned_over") else None


def turned_over_in_horizon(gold_series: np.ndarray, K_decline: int = 5,
                            alpha: float = 0.2, min_drop_frac: float = 0.2,
                            **kwargs) -> bool:
    """Strict confirmed turnover test used for negative-control validity."""
    return bool(confirmed_turnover_info(
        gold_series, K_decline=K_decline, alpha=alpha,
        min_drop_frac=min_drop_frac, **kwargs
    ).get("turned_over"))


# ---------------------------------------------------------------------------
# (FAR, lead-time) curves
# ---------------------------------------------------------------------------
@dataclass
class SignalRun:
    label: str            # task / config / seed identifier
    is_positive: bool     # true if this run is a positive (turns over)
    signal: np.ndarray
    t_gold: Optional[int] # argmax of smoothed gold (None on negatives)


def far_lead_curve(runs: Iterable[SignalRun], thresholds: np.ndarray,
                   K_persist: int = 3) -> dict:
    """For each threshold in `thresholds`, compute:
       - detect_rate over positive runs
       - mean_lead and median_lead over the positive runs that detected
       - false_alarm_rate over negative-control runs (fraction that fired at all)

    Returns dict of arrays, all aligned with `thresholds`.
    """
    runs = list(runs)
    pos = [r for r in runs if r.is_positive]
    neg = [r for r in runs if not r.is_positive]
    detect_rate = np.zeros_like(thresholds, dtype=np.float64)
    mean_lead = np.full_like(thresholds, np.nan, dtype=np.float64)
    median_lead = np.full_like(thresholds, np.nan, dtype=np.float64)
    far = np.zeros_like(thresholds, dtype=np.float64)

    for ti, thr in enumerate(thresholds):
        leads = []
        n_detect = 0
        for r in pos:
            t_f = cusum_fire(r.signal, thr, K_persist=K_persist)
            if t_f is not None and r.t_gold is not None and t_f <= r.t_gold:
                leads.append(r.t_gold - t_f)
                n_detect += 1
        n_fa = 0
        for r in neg:
            t_f = cusum_fire(r.signal, thr, K_persist=K_persist)
            if t_f is not None:
                n_fa += 1
        detect_rate[ti] = n_detect / max(len(pos), 1)
        if leads:
            mean_lead[ti] = float(np.mean(leads))
            median_lead[ti] = float(np.median(leads))
        far[ti] = n_fa / max(len(neg), 1)
    return dict(thresholds=np.asarray(thresholds, dtype=np.float64),
                detect_rate=detect_rate,
                mean_lead=mean_lead,
                median_lead=median_lead,
                far=far,
                n_pos=len(pos), n_neg=len(neg))


def lead_at_far(curve: dict, target_far: float = 0.1) -> tuple[float, float]:
    """Best mean_lead achievable at false-alarm rate ≤ target_far.

    Returns (lead, threshold). If no threshold achieves FAR ≤ target_far with a detection,
    returns (nan, nan).
    """
    far = curve["far"]; mean_lead = curve["mean_lead"]; thr = curve["thresholds"]
    valid = (far <= target_far) & np.isfinite(mean_lead)
    if not np.any(valid):
        return float("nan"), float("nan")
    idx = np.argmax(np.where(valid, mean_lead, -np.inf))
    return float(mean_lead[idx]), float(thr[idx])


# ---------------------------------------------------------------------------
# Bootstrap CI for lead@FAR=0.1 over (task, seed) pairs
# ---------------------------------------------------------------------------
def bootstrap_lead_at_far(runs: list[SignalRun], thresholds: np.ndarray,
                          target_far: float = 0.1, K_persist: int = 3,
                          n_boot: int = 500, alpha: float = 0.05,
                          seed: int = 0) -> tuple[float, float, float]:
    """Bootstrap (lead@FAR) over the supplied runs.

    Resamples positive runs and negative runs *separately* with replacement, recomputes
    the curve, takes lead@FAR. Returns (mean, lo, hi).
    """
    rng = np.random.default_rng(seed)
    pos = [r for r in runs if r.is_positive]
    neg = [r for r in runs if not r.is_positive]
    leads = []
    for _ in range(n_boot):
        idx_p = rng.integers(0, len(pos), size=len(pos)) if pos else np.array([], dtype=int)
        idx_n = rng.integers(0, len(neg), size=len(neg)) if neg else np.array([], dtype=int)
        sub = [pos[i] for i in idx_p] + [neg[i] for i in idx_n]
        curve = far_lead_curve(sub, thresholds, K_persist=K_persist)
        ld, _ = lead_at_far(curve, target_far=target_far)
        leads.append(ld)
    leads = np.asarray(leads, dtype=np.float64)
    leads = leads[np.isfinite(leads)]
    if leads.size == 0:
        return float("nan"), float("nan"), float("nan")
    return (float(leads.mean()),
            float(np.quantile(leads, alpha / 2)),
            float(np.quantile(leads, 1 - alpha / 2)))


# ---------------------------------------------------------------------------
# Top-level convenience
# ---------------------------------------------------------------------------
def build_signal_runs(traces: list[dict], signal_fn, signal_key_for_run: str = "is_positive") -> list[SignalRun]:
    """Convert a list of per-run traces (dicts with keys 'gold', 'is_positive', 'label',
    and the input keys signal_fn needs) into a list of SignalRun objects ready for
    far_lead_curve.
    """
    runs = []
    for t in traces:
        sig = signal_fn(t)
        t_gold = find_t_gold_confirmed(t["gold"]) if t.get(signal_key_for_run, False) else None
        runs.append(SignalRun(label=t["label"], is_positive=bool(t.get(signal_key_for_run, False)),
                              signal=np.asarray(sig, dtype=np.float64),
                              t_gold=t_gold))
    return runs
