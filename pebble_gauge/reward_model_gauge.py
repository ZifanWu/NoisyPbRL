from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn


STANDARD_GAUGES = ("none", "mean_buf", "mean_ref")
NO_TANH_GAUGES = ("no_tanh", "no_tanh_mean_buf", "no_tanh_mean_ref")
VALID_GAUGES = STANDARD_GAUGES + NO_TANH_GAUGES


@dataclass
class GaugeCheckResult:
    max_abs_err_mean_buf: float
    max_abs_err_mean_ref: float
    max_abs_err_no_tanh_mean_buf: float
    max_abs_err_no_tanh_mean_ref: float
    passed: bool


class GaugedRewardModel:
    """Post-hoc additive-gauge wrapper for a trained reward model.

    The wrapped reward model stays unchanged. This class only stores additive
    constants and applies them at reward query time.
    """

    def __init__(
        self,
        base_reward_model,
        gauge: str = "none",
        n_shift_sample: int = 2048,
        reference_sa: Optional[np.ndarray] = None,
    ) -> None:
        if gauge not in VALID_GAUGES:
            raise ValueError(f"Unknown gauge: {gauge!r}")
        self.base_reward_model = base_reward_model
        self.active_gauge = gauge
        self.n_shift_sample = int(n_shift_sample)
        self.reference_sa = reference_sa
        self.shift_values: Dict[str, float] = {
            "none": 0.0,
            "mean_buf": 0.0,
            "mean_ref": 0.0,
            "no_tanh": 0.0,
            "no_tanh_mean_buf": 0.0,
            "no_tanh_mean_ref": 0.0,
        }

    def __getattr__(self, name):
        # Keep drop-in compatibility with RewardModel usages in train loop.
        return getattr(self.base_reward_model, name)

    def set_active_gauge(self, gauge: str) -> None:
        if gauge not in VALID_GAUGES:
            raise ValueError(f"Unknown gauge: {gauge!r}")
        self.active_gauge = gauge

    def set_reference_sa(self, reference_sa: np.ndarray) -> None:
        self.reference_sa = reference_sa

    def _sample_replay_sa(self, replay_buffer, n: int) -> Optional[np.ndarray]:
        total = len(replay_buffer)
        if total <= 0:
            return None
        size = min(int(n), int(total))
        idxs = np.random.choice(total, size=size, replace=False)
        obs = replay_buffer.obses[idxs]
        actions = replay_buffer.actions[idxs]
        return np.concatenate([obs, actions], axis=-1).astype(np.float32)

    @staticmethod
    def _is_no_tanh_gauge(gauge: str) -> bool:
        return gauge.startswith("no_tanh")

    def _forward_member(self, member_model: nn.Sequential, x_tensor: torch.Tensor, use_no_tanh: bool) -> torch.Tensor:
        if not use_no_tanh:
            return member_model(x_tensor)
        # no_tanh gauges remove the output tanh while keeping all hidden nonlinearity.
        if len(member_model) > 0 and isinstance(member_model[-1], nn.Tanh):
            out = x_tensor
            for layer in member_model[:-1]:
                out = layer(out)
            return out
        return member_model(x_tensor)

    def _base_r_hat_batch(self, sa_batch: np.ndarray, use_no_tanh: bool) -> np.ndarray:
        sample_param = next(self.base_reward_model.ensemble[0].parameters())
        x = torch.from_numpy(sa_batch).float().to(sample_param.device)
        preds = []
        with torch.no_grad():
            for member in self.base_reward_model.ensemble:
                out = self._forward_member(member, x, use_no_tanh=use_no_tanh)
                preds.append(out.detach().cpu().numpy())
        return np.asarray(preds, dtype=np.float32).mean(axis=0)

    def recompute_shifts(
        self,
        replay_buffer=None,
        reference_sa: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        if reference_sa is not None:
            self.reference_sa = reference_sa

        self.shift_values["none"] = 0.0
        self.shift_values["no_tanh"] = 0.0

        buf_sa = None
        if replay_buffer is not None:
            buf_sa = self._sample_replay_sa(replay_buffer, self.n_shift_sample)
        if buf_sa is not None and len(buf_sa) > 0:
            self.shift_values["mean_buf"] = float(
                np.mean(self.base_reward_model.r_hat_batch(buf_sa))
            )
            self.shift_values["no_tanh_mean_buf"] = float(
                np.mean(self._base_r_hat_batch(buf_sa, use_no_tanh=True))
            )

        if self.reference_sa is not None and len(self.reference_sa) > 0:
            ref_sa = self.reference_sa
            if len(ref_sa) > self.n_shift_sample:
                # Fixed random subset for reproducibility.
                rng = np.random.default_rng(0)
                idxs = rng.choice(len(ref_sa), size=self.n_shift_sample, replace=False)
                ref_sa = ref_sa[idxs]
            self.shift_values["mean_ref"] = float(
                np.mean(self.base_reward_model.r_hat_batch(ref_sa))
            )
            self.shift_values["no_tanh_mean_ref"] = float(
                np.mean(self._base_r_hat_batch(ref_sa, use_no_tanh=True))
            )

        return dict(self.shift_values)

    def get_shift(self, gauge: Optional[str] = None) -> float:
        gauge_name = gauge or self.active_gauge
        if gauge_name not in VALID_GAUGES:
            raise ValueError(f"Unknown gauge: {gauge_name!r}")
        return float(self.shift_values[gauge_name])

    def compute(self, sa: np.ndarray, gauge: Optional[str] = None) -> float:
        gauge_name = gauge or self.active_gauge
        use_no_tanh = self._is_no_tanh_gauge(gauge_name)
        raw = float(np.asarray(self._base_r_hat_batch(np.asarray(sa, dtype=np.float32).reshape(1, -1), use_no_tanh=use_no_tanh)).reshape(-1)[0])
        return raw - self.get_shift(gauge_name)

    def compute_batch(self, sa_batch: np.ndarray, gauge: Optional[str] = None) -> np.ndarray:
        gauge_name = gauge or self.active_gauge
        use_no_tanh = self._is_no_tanh_gauge(gauge_name)
        if use_no_tanh:
            raw = np.asarray(self._base_r_hat_batch(sa_batch, use_no_tanh=True), dtype=np.float32)
        else:
            raw = np.asarray(self.base_reward_model.r_hat_batch(sa_batch), dtype=np.float32)
        return raw - self.get_shift(gauge_name)

    def proxy_return(self, sa_batch: np.ndarray, gauge: Optional[str] = None) -> float:
        return float(self.compute_batch(sa_batch, gauge=gauge).sum())

    def proxy_returns_all_gauges(self, sa_batch: np.ndarray, gauges: Optional[Sequence[str]] = None) -> Dict[str, float]:
        eval_gauges = tuple(gauges) if gauges is not None else VALID_GAUGES
        return {g: self.proxy_return(sa_batch, gauge=g) for g in eval_gauges}

    def r_hat(self, sa: np.ndarray) -> float:
        return self.compute(sa, gauge=self.active_gauge)

    def r_hat_batch(self, sa_batch: np.ndarray) -> np.ndarray:
        return self.compute_batch(sa_batch, gauge=self.active_gauge)

    def gauge_consistency_check(
        self,
        sa_batch: np.ndarray,
        atol: float = 1e-5,
        include_no_tanh: bool = False,
    ) -> GaugeCheckResult:
        """Check that enabled gauges differ from ``none`` only by stored constants."""
        if len(sa_batch) == 0:
            return GaugeCheckResult(0.0, 0.0, 0.0, 0.0, True)

        r_none = self.compute_batch(sa_batch, gauge="none")
        r_buf = self.compute_batch(sa_batch, gauge="mean_buf")
        r_ref = self.compute_batch(sa_batch, gauge="mean_ref")

        err_buf = np.max(np.abs((r_none - r_buf) - self.get_shift("mean_buf")))
        err_ref = np.max(np.abs((r_none - r_ref) - self.get_shift("mean_ref")))

        err_no_tanh_buf = 0.0
        err_no_tanh_ref = 0.0
        if include_no_tanh:
            r_no_tanh = self.compute_batch(sa_batch, gauge="no_tanh")
            r_no_tanh_buf = self.compute_batch(sa_batch, gauge="no_tanh_mean_buf")
            r_no_tanh_ref = self.compute_batch(sa_batch, gauge="no_tanh_mean_ref")
            err_no_tanh_buf = np.max(
                np.abs((r_no_tanh - r_no_tanh_buf) - self.get_shift("no_tanh_mean_buf"))
            )
            err_no_tanh_ref = np.max(
                np.abs((r_no_tanh - r_no_tanh_ref) - self.get_shift("no_tanh_mean_ref"))
            )

        passed = bool(err_buf <= atol and err_ref <= atol)
        if include_no_tanh:
            passed = bool(passed and err_no_tanh_buf <= atol and err_no_tanh_ref <= atol)

        return GaugeCheckResult(
            max_abs_err_mean_buf=float(err_buf),
            max_abs_err_mean_ref=float(err_ref),
            max_abs_err_no_tanh_mean_buf=float(err_no_tanh_buf),
            max_abs_err_no_tanh_mean_ref=float(err_no_tanh_ref),
            passed=passed,
        )

    def shift_log_metrics(self) -> Dict[str, float]:
        return {"train/shift_value_active": self.get_shift(self.active_gauge)}
