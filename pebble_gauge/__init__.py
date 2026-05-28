"""Utilities for the post-hoc gauge experiment."""

from .reward_model_gauge import GaugedRewardModel
from .reference_dataset import build_or_load_reference_dataset
from .perf_correction import (
    PerfCorrectionConfig,
    compute_perf_correction_base,
    compute_perf_correction_shift,
    combine_actor_grads,
    grad_l2_norm,
)

__all__ = [
    "GaugedRewardModel",
    "build_or_load_reference_dataset",
    "PerfCorrectionConfig",
    "compute_perf_correction_base",
    "compute_perf_correction_shift",
    "combine_actor_grads",
    "grad_l2_norm",
]
