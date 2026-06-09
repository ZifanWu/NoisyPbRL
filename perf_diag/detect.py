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


def turned_over_in_horizon(gold_series: np.ndarray, K_decline: int = 5,
                            alpha: float = 0.1, min_drop_frac: float = 0.05) -> bool:
    """Conservative test: gold turned over if a sustained decline of ≥ min_drop_frac of
    its running max actually happened in-horizon. Used for negative-control validity.
    """
    g = baselines.ema_smooth(np.asarray(gold_series, dtype=np.float64), alpha=alpha)
    if g.size < K_decline + 1:
        return False
    running_max = -np.inf
    drop = 0.0
    for v in g:
        if v > running_max:
            running_max = v
        else:
            drop = max(drop, running_max - v)
    if running_max <= 0:
        return drop > 0
    return drop > min_drop_frac * abs(running_max)


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
        t_gold = find_t_gold_argmax(t["gold"]) if t.get(signal_key_for_run, False) else None
        runs.append(SignalRun(label=t["label"], is_positive=bool(t.get(signal_key_for_run, False)),
                              signal=np.asarray(sig, dtype=np.float64),
                              t_gold=t_gold))
    return runs
