"""Gold-free baseline detectors.

All baselines consume a per-step trace (a list of dicts produced by the monitoring hook)
and produce a per-step real-valued signal. The firing rule (CUSUM threshold) lives in
detect.py — the baselines themselves are signals, not classifiers.
"""

from __future__ import annotations

import numpy as np


def ema_smooth(x: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """One-sided EMA with alpha = (1 − decay). Larger alpha → less smoothing."""
    x = np.asarray(x, dtype=np.float64)
    y = np.zeros_like(x)
    if x.size == 0:
        return y
    y[0] = x[0]
    for i in range(1, x.size):
        y[i] = alpha * x[i] + (1 - alpha) * y[i - 1]
    return y


# ---------------------------------------------------------------------------
# Proxy-inflection signal
# ---------------------------------------------------------------------------
def proxy_inflection_signal(proxy_series: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """Negative-second-derivative of the smoothed proxy series.

    Fires (in detect.py) when the signal stays above a threshold for K_persist steps,
    which means the proxy curve has flattened / turned down. Returns a same-length
    np.float64 array with NaNs at the boundaries replaced by 0.
    """
    p = ema_smooth(np.asarray(proxy_series, dtype=np.float64), alpha=alpha)
    # second derivative by central differences; pad edges
    n = p.size
    d2 = np.zeros(n, dtype=np.float64)
    if n >= 3:
        d2[1:-1] = p[2:] - 2.0 * p[1:-1] + p[:-2]
    # Positive signal == proxy curving downward
    return -d2


# ---------------------------------------------------------------------------
# KL-to-pretrain / entropy signals
# ---------------------------------------------------------------------------
def kl_to_pretrain_signal(kl_series: np.ndarray) -> np.ndarray:
    """Pass-through: bigger KL → fires more. detect.py thresholds it directly."""
    return np.asarray(kl_series, dtype=np.float64)


def entropy_drop_signal(entropy_series: np.ndarray) -> np.ndarray:
    """Negative-entropy signal: more saturated policy → larger signal."""
    return -np.asarray(entropy_series, dtype=np.float64)


# ---------------------------------------------------------------------------
# Ensemble-variance signal
# ---------------------------------------------------------------------------
def ensemble_variance_signal(ens_var_series: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """Lightly smoothed ensemble predictive variance."""
    return ema_smooth(np.asarray(ens_var_series, dtype=np.float64), alpha=alpha)


# ---------------------------------------------------------------------------
# R̂ degradation signal (used by detect.py for the headline)
# ---------------------------------------------------------------------------
def R_degradation_signal(R_series: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """(1 − R̂_smoothed). Big when R̂ has dropped from 1; sustained-downturn rule fires.

    Per the spec: R̂ ≈ 1 healthy, R̂ ≈ 0 vanishing, R̂ < 0 would-be-reversed by relabel.
    """
    R = ema_smooth(np.asarray(R_series, dtype=np.float64), alpha=alpha)
    return 1.0 - R
