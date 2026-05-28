import itertools
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

import reward_model as reward_model_module
from reward_model import RewardModel


TANH_GAUGES = ("none", "mean_buf", "mean_ref")
NO_TANH_GAUGES = ("no_tanh", "no_tanh_mean_buf", "no_tanh_mean_ref")
ALL_GAUGES = TANH_GAUGES + NO_TANH_GAUGES


def validate_gauge(gauge: str) -> None:
    if gauge not in ALL_GAUGES:
        raise ValueError("unknown gauge '{}'; expected one of {}".format(gauge, ALL_GAUGES))


def final_activation_for_gauge(gauge: str) -> str:
    validate_gauge(gauge)
    return "identity" if gauge.startswith("no_tanh") else "tanh"


def shift_kind_for_gauge(gauge: str) -> str:
    validate_gauge(gauge)
    if gauge.endswith("mean_buf"):
        return "mean_buf"
    if gauge.endswith("mean_ref"):
        return "mean_ref"
    return "none"


def family_gauges(final_activation: str) -> Tuple[str, str, str]:
    if final_activation == "identity":
        return NO_TANH_GAUGES
    if final_activation == "tanh":
        return TANH_GAUGES
    raise ValueError("unknown final activation '{}'".format(final_activation))


def base_gauge_for_activation(final_activation: str) -> str:
    return "no_tanh" if final_activation == "identity" else "none"


class IdentityFinalRewardModel(RewardModel):
    """RewardModel variant with the final tanh/sigmoid/relu replaced by identity.

    The upstream RewardModel hard-codes hidden layers in gen_net and appends a final
    activation. For the no_tanh gauges we need the same hidden architecture and
    optimizer behavior, but no activation after the final linear layer.
    """

    def construct_ensemble(self):
        for _ in range(self.de):
            layers = []
            in_size = self.ds + self.da
            for _layer in range(3):
                layers.append(nn.Linear(in_size, 256))
                layers.append(nn.LeakyReLU())
                in_size = 256
            layers.append(nn.Linear(in_size, 1))
            model = nn.Sequential(*layers).float().to(reward_model_module.device)
            self.ensemble.append(model)
            self.paramlst.extend(model.parameters())

        self.opt = torch.optim.Adam(self.paramlst, lr=self.lr)


def make_reward_model(
    ds: int,
    da: int,
    final_activation: str,
    **kwargs,
) -> RewardModel:
    if final_activation == "identity":
        kwargs = dict(kwargs)
        kwargs["activation"] = "identity"
        return IdentityFinalRewardModel(ds, da, **kwargs)
    if final_activation == "tanh":
        kwargs = dict(kwargs)
        kwargs["activation"] = "tanh"
        return RewardModel(ds, da, **kwargs)
    raise ValueError("unknown final activation '{}'".format(final_activation))


class GaugedRewardModel:
    """Thin wrapper that delegates training to a base RM and gauges prediction."""

    def __init__(
        self,
        base_model: RewardModel,
        active_gauge: str,
        n_shift_sample: int = 2048,
        pred_batch_size: int = 512,
        seed: int = 0,
        log_reward_diagnostics: bool = False,
    ):
        validate_gauge(active_gauge)
        self.base = base_model
        self.active_gauge = active_gauge
        self.final_activation = final_activation_for_gauge(active_gauge)
        self.n_shift_sample = int(n_shift_sample)
        self.pred_batch_size = int(pred_batch_size)
        self.log_reward_diagnostics = bool(log_reward_diagnostics)
        if not self.log_reward_diagnostics:
            self.base.use_wandb = False
        self.rng = np.random.RandomState(seed)
        self.shifts: Dict[str, float] = {g: 0.0 for g in family_gauges(self.final_activation)}
        self.shift_inputs: Dict[str, Optional[np.ndarray]] = {
            "none": None,
            "mean_buf": None,
            "mean_ref": None,
        }

    def __getattr__(self, name):
        return getattr(self.base, name)

    def pre_relabel_logging(self, step):
        if self.log_reward_diagnostics:
            return self.base.pre_relabel_logging(step)
        return None

    @property
    def de(self):
        return self.base.de

    @property
    def ds(self):
        return self.base.ds

    @property
    def da(self):
        return self.base.da

    @property
    def size_segment(self):
        return self.base.size_segment

    def family_gauges(self) -> Tuple[str, str, str]:
        return family_gauges(self.final_activation)

    def shift_kind(self, gauge: Optional[str] = None) -> str:
        return shift_kind_for_gauge(gauge or self.active_gauge)

    def set_active_gauge(self, gauge: str) -> None:
        validate_gauge(gauge)
        if final_activation_for_gauge(gauge) != self.final_activation:
            raise ValueError(
                "cannot switch activation family from {} to {}".format(
                    self.final_activation, final_activation_for_gauge(gauge)
                )
            )
        self.active_gauge = gauge

    def get_shift(self, gauge: Optional[str] = None) -> float:
        gauge = gauge or self.active_gauge
        validate_gauge(gauge)
        if final_activation_for_gauge(gauge) != self.final_activation:
            raise ValueError("gauge '{}' is not in activation family '{}'".format(gauge, self.final_activation))
        return float(self.shifts.get(gauge, 0.0))

    def r_hat(self, x, gauge: Optional[str] = None):
        raw = self.base.r_hat(x)
        return raw - self.get_shift(gauge)

    def r_hat_batch(self, x, gauge: Optional[str] = None):
        raw = self.base.r_hat_batch(x)
        return raw - self.get_shift(gauge)

    def r_hat_for_gauge(self, x, gauge: str):
        return self.r_hat(x, gauge=gauge)

    def r_hat_batch_for_gauge(self, x, gauge: str):
        return self.r_hat_batch(x, gauge=gauge)

    def raw_mean_reward(self, inputs: np.ndarray) -> float:
        inputs = np.asarray(inputs, dtype=np.float32)
        if inputs.size == 0:
            return 0.0
        if inputs.ndim == 1:
            inputs = inputs[None, :]
        flat_inputs = inputs.reshape(-1, inputs.shape[-1])
        total = 0.0
        count = 0
        for start in range(0, len(flat_inputs), self.pred_batch_size):
            batch = flat_inputs[start : start + self.pred_batch_size]
            pred = self.base.r_hat_batch(batch)
            total += float(np.asarray(pred).sum())
            count += int(np.asarray(pred).size)
        return total / max(count, 1)

    def update_shifts(self, replay_buffer=None, reference_data=None, logger=None, step: Optional[int] = None):
        needed = set(shift_kind_for_gauge(g) for g in self.family_gauges())

        if "mean_buf" in needed:
            buf_inputs = self._sample_replay_inputs(replay_buffer)
            self.shift_inputs["mean_buf"] = buf_inputs
            mean_buf = self.raw_mean_reward(buf_inputs) if buf_inputs is not None else 0.0
        else:
            mean_buf = 0.0

        if "mean_ref" in needed:
            ref_inputs = self._reference_inputs(reference_data)
            self.shift_inputs["mean_ref"] = ref_inputs
            mean_ref = self.raw_mean_reward(ref_inputs) if ref_inputs is not None else 0.0
        else:
            mean_ref = 0.0

        for gauge in self.family_gauges():
            kind = shift_kind_for_gauge(gauge)
            if kind == "mean_buf":
                self.shifts[gauge] = float(mean_buf)
            elif kind == "mean_ref":
                self.shifts[gauge] = float(mean_ref)
            else:
                self.shifts[gauge] = 0.0

        if logger is not None and step is not None:
            for gauge in self.family_gauges():
                logger.log("train/gauge_shift_{}".format(gauge), self.shifts[gauge], step)
            delta = self.shifts[self.family_gauges()[1]] - self.shifts[self.family_gauges()[2]]
            logger.log("train/gauge_shift_buf_ref_gap", delta, step)

        return dict(self.shifts)

    def inputs_for_shift_kind(self, kind: str) -> Optional[np.ndarray]:
        if kind == "none":
            return None
        return self.shift_inputs.get(kind)

    def _sample_replay_inputs(self, replay_buffer) -> Optional[np.ndarray]:
        if replay_buffer is None or len(replay_buffer) == 0:
            return None
        n = len(replay_buffer)
        sample_n = min(self.n_shift_sample, n)
        idxs = self.rng.choice(n, size=sample_n, replace=False)
        obses = replay_buffer.obses[idxs]
        actions = replay_buffer.actions[idxs]
        return np.concatenate([obses, actions], axis=-1).astype(np.float32)

    def _reference_inputs(self, reference_data) -> Optional[np.ndarray]:
        if reference_data is None:
            return None
        if isinstance(reference_data, dict):
            obs = reference_data.get("obs")
            action = reference_data.get("action")
        elif isinstance(reference_data, (tuple, list)) and len(reference_data) >= 2:
            obs, action = reference_data[0], reference_data[1]
        else:
            raise ValueError("reference_data must be dict or (obs, action) tuple")
        if torch.is_tensor(obs):
            obs = obs.detach().cpu().numpy()
        if torch.is_tensor(action):
            action = action.detach().cpu().numpy()
        return np.concatenate([obs, action], axis=-1).astype(np.float32)


def reward_model_kwargs_from_cfg(cfg) -> Dict:
    log_reward_diagnostics = bool(getattr(getattr(cfg, "gauge", None), "log_reward_diagnostics", False))
    return dict(
        ensemble_size=cfg.ensemble_size,
        size_segment=cfg.segment,
        activation=cfg.activation,
        lr=cfg.reward_lr,
        mb_size=cfg.reward_batch,
        large_batch=cfg.large_batch,
        label_margin=cfg.label_margin,
        teacher_beta=cfg.teacher_beta,
        teacher_gamma=cfg.teacher_gamma,
        teacher_eps_mistake=cfg.teacher_eps_mistake,
        teacher_eps_skip=cfg.teacher_eps_skip,
        teacher_eps_equal=cfg.teacher_eps_equal,
        dormant_log_period=cfg.dormant_log_period,
        dormant_threshold=cfg.dormant_threshold,
        use_wandb=bool(cfg.use_wandb and log_reward_diagnostics),
        bt_log_period=cfg.bt_log_period,
        feed_type=cfg.feed_type,
        capacity=cfg.max_feedback * cfg.large_batch,
    )


def iter_reward_parameters(base_model: RewardModel) -> Iterable[torch.nn.Parameter]:
    return itertools.chain.from_iterable(member.parameters() for member in base_model.ensemble)
