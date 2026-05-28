#!/usr/bin/env python3
import csv
import os
import sys
from collections import deque

import hydra
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import utils
import reward_model as reward_model_module
from pebble_gauge.perf_correction import apply_actor_correction, compute_performative_correction
from pebble_gauge.reference_dataset import _reset_env, _step_env, load_or_create_reference_dataset
from pebble_gauge.reward_model_gauge import (
    GaugedRewardModel,
    final_activation_for_gauge,
    make_reward_model,
    reward_model_kwargs_from_cfg,
)
from train_PEBBLE import Workspace


def _cfg_get(cfg, name, default=None):
    return getattr(cfg, name, default)


def _absolute_repo_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(REPO_ROOT, path)


def max_episode_steps(env, default=1000):
    return int(
        getattr(
            env,
            "_max_episode_steps",
            getattr(getattr(env, "spec", None), "max_episode_steps", default),
        )
    )


def make_env_like(cfg, seed_offset=0):
    cfg_like = type("GaugeEnvCfg", (), {})()
    cfg_like.env = cfg.env
    cfg_like.seed = int(cfg.seed) + int(seed_offset)
    if "metaworld" in cfg.env:
        return utils.make_metaworld_env(cfg_like)
    return utils.make_env(cfg_like)


def apply_alpha_mode_to_cfg(cfg):
    mode = _cfg_get(cfg.gauge, "alpha_mode", "auto")
    if mode == "auto":
        return
    if mode == "low":
        alpha = 0.05
    elif mode == "high":
        alpha = 0.5
    else:
        raise ValueError("unknown alpha_mode '{}'".format(mode))
    cfg.agent.params.learnable_temperature = False
    cfg.agent.params.init_temperature = alpha


class GaugeWorkspace(Workspace):
    def __init__(self, cfg):
        apply_alpha_mode_to_cfg(cfg)
        reward_model_module.device = str(cfg.device)
        if str(cfg.agent.name) == "sac_metaworld":
            cfg.agent.name = "sac"
        super().__init__(cfg)
        self.method = _cfg_get(cfg.gauge, "method", "rrm")
        self._actor_update_count = 0
        self._in_reset_update = False
        self._perf_rng = np.random.RandomState(int(cfg.seed) + int(cfg.gauge.pg.rng_offset))
        self._perf_env = None
        self._reference_data = None
        self._recent_sa = deque(maxlen=int(cfg.gauge.pg.recent_sa_capacity))
        self._replace_reward_model()
        self._install_recent_sa_hook()
        self._init_gauge_metrics()
        self._apply_runtime_alpha_mode()
        if self.method == "perfg":
            if str(self.cfg.gauge.pg.u_source) == "rollout":
                self._perf_env = make_env_like(cfg, seed_offset=cfg.gauge.pg.env_seed_offset)
            self._install_perf_hook()

    def _init_gauge_metrics(self):
        path = os.path.join(self.work_dir, "gauge_metrics.csv")
        self._gauge_metrics_file = open(path, "w", newline="")
        self._gauge_metrics_writer = csv.DictWriter(
            self._gauge_metrics_file,
            fieldnames=["step", "split", "key", "value"],
        )
        self._gauge_metrics_writer.writeheader()
        self._gauge_metrics_file.flush()

    def _write_gauge_metric(self, split, key, value, step):
        if isinstance(value, torch.Tensor):
            value = value.item()
        if not isinstance(value, (int, float, np.floating)):
            return
        self._gauge_metrics_writer.writerow(
            {
                "step": int(step),
                "split": split,
                "key": key,
                "value": float(value),
            }
        )
        self._gauge_metrics_file.flush()

    def run(self):
        try:
            result = super().run()
            if self.method == "rrm":
                self._run_rrm_bit_identity_check()
            return result
        finally:
            self._gauge_metrics_file.close()

    def _run_rrm_bit_identity_check(self):
        from pebble_gauge.sanity_checks import run_rrm_bit_identity_check

        result = run_rrm_bit_identity_check(
            self.reward_model,
            self.agent,
            self.env,
            n_episodes=20,
        )
        passed = 1.0 if result["passed"] else 0.0
        self.logger.log("eval/rrm_bit_identity_passed", passed, self.step)
        self._write_gauge_metric("eval", "rrm_bit_identity_passed", passed, self.step)
        self._write_gauge_metric("eval", "rrm_bit_identity_mean_true_return", result["mean_true_return"], self.step)
        if result["failures"]:
            first = result["failures"][0]
            self._write_gauge_metric("eval", "rrm_bit_identity_first_error", first["error"], self.step)
            raise AssertionError("RRM gauge bit-identity check failed: {}".format(first))

    def _replace_reward_model(self):
        gauge = self.cfg.gauge.active
        final_activation = final_activation_for_gauge(gauge)
        base = make_reward_model(
            self.env.observation_space.shape[0],
            self.env.action_space.shape[0],
            final_activation=final_activation,
            **reward_model_kwargs_from_cfg(self.cfg)
        )
        self.reward_model = GaugedRewardModel(
            base,
            active_gauge=gauge,
            n_shift_sample=self.cfg.gauge.n_shift_sample,
            pred_batch_size=self.cfg.gauge.pred_batch_size,
            seed=int(self.cfg.seed) + int(self.cfg.gauge.shift_seed_offset),
            log_reward_diagnostics=self.cfg.gauge.log_reward_diagnostics,
        )

    def _install_recent_sa_hook(self):
        original_add_data = self.reward_model.add_data

        def wrapped_add_data(obs, act, rew, done):
            sa = np.concatenate([obs, act], axis=-1).astype(np.float32)
            self._recent_sa.append(sa)
            return original_add_data(obs, act, rew, done)

        self.reward_model.add_data = wrapped_add_data

    def _apply_runtime_alpha_mode(self):
        mode = _cfg_get(self.cfg.gauge, "alpha_mode", "auto")
        if mode == "auto":
            return
        alpha = 0.05 if mode == "low" else 0.5
        self.agent.learnable_temperature = False
        self.agent.init_temperature = alpha
        with torch.no_grad():
            self.agent.log_alpha.data.fill_(np.log(alpha))

    def _ensure_reference_data(self):
        if self._reference_data is None:
            root = _absolute_repo_path(self.cfg.gauge.reference_root)
            self._reference_data = load_or_create_reference_dataset(
                self.cfg.env,
                root=root,
                n_ref=self.cfg.gauge.n_ref,
                seed=self.cfg.gauge.reference_seed,
                auto_create=self.cfg.gauge.auto_create_reference,
            )
        return self._reference_data

    def _refresh_gauge_shifts(self):
        reference_data = self._ensure_reference_data()
        shifts = self.reward_model.update_shifts(
            replay_buffer=self.replay_buffer,
            reference_data=reference_data,
            logger=self.logger,
            step=self.step,
        )
        for gauge, shift in shifts.items():
            self._write_gauge_metric("train", "gauge_shift_{}".format(gauge), shift, self.step)
        family = self.reward_model.family_gauges()
        gap = shifts[family[1]] - shifts[family[2]]
        self._write_gauge_metric("train", "gauge_shift_buf_ref_gap", gap, self.step)

    def learn_reward(self, first_flag=0):
        super().learn_reward(first_flag=first_flag)
        self._refresh_gauge_shifts()

    def evaluate(self):
        proxy_sums = {gauge: 0.0 for gauge in self.reward_model.family_gauges()}
        average_true_episode_reward = 0.0
        average_episode_length = 0.0
        success_rate = 0.0

        for _episode in range(self.cfg.num_eval_episodes):
            obs = _reset_env(self.env)
            self.agent.reset()
            done = False
            episode_true_reward = 0.0
            episode_proxy = {gauge: 0.0 for gauge in self.reward_model.family_gauges()}
            episode_success = 0.0
            episode_length = 0

            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                next_obs, reward, done, extra = _step_env(self.env, action)
                sa = np.concatenate([obs, action], axis=-1)
                for gauge in self.reward_model.family_gauges():
                    episode_proxy[gauge] += float(self.reward_model.r_hat_for_gauge(sa, gauge))
                episode_true_reward += float(reward)
                if self.log_success:
                    episode_success = max(episode_success, float(extra.get("success", 0.0)))
                obs = next_obs
                episode_length += 1

            average_true_episode_reward += episode_true_reward
            average_episode_length += episode_length
            for gauge in self.reward_model.family_gauges():
                proxy_sums[gauge] += episode_proxy[gauge]
            if self.log_success:
                success_rate += episode_success

        n_eval = float(self.cfg.num_eval_episodes)
        average_true_episode_reward /= n_eval
        average_episode_length /= n_eval
        proxy_means = {gauge: value / n_eval for gauge, value in proxy_sums.items()}
        active_proxy = proxy_means[self.reward_model.active_gauge]

        self.logger.log("eval/episode_reward", active_proxy, self.step)
        self.logger.log("eval/true_episode_reward", average_true_episode_reward, self.step)
        self.logger.log("eval/episode_length", average_episode_length, self.step)
        self._write_gauge_metric("eval", "episode_reward", active_proxy, self.step)
        self._write_gauge_metric("eval", "true_episode_reward", average_true_episode_reward, self.step)
        self._write_gauge_metric("eval", "episode_length", average_episode_length, self.step)
        for gauge, value in proxy_means.items():
            self.logger.log("eval/proxy_return_{}".format(gauge), value, self.step)
            self.logger.log("eval/proxy_gap_{}".format(gauge), value - average_true_episode_reward, self.step)
            shift = self.reward_model.get_shift(gauge)
            self.logger.log("eval/gauge_shift_{}".format(gauge), shift, self.step)
            self._write_gauge_metric("eval", "proxy_return_{}".format(gauge), value, self.step)
            self._write_gauge_metric("eval", "proxy_gap_{}".format(gauge), value - average_true_episode_reward, self.step)
            self._write_gauge_metric("eval", "gauge_shift_{}".format(gauge), shift, self.step)

        alpha_val = float(self.agent.log_alpha.exp().detach().cpu())
        self.logger.log("train/alpha", alpha_val, self.step)
        self._write_gauge_metric("eval", "alpha", alpha_val, self.step)

        if self.log_success:
            success_rate = 100.0 * success_rate / n_eval
            self.logger.log("eval/success_rate", success_rate, self.step)
            self.logger.log("train/true_episode_success", success_rate, self.step)
        self.logger.dump(self.step)

    def _install_perf_hook(self):
        original_update_actor_and_alpha = self.agent.update_actor_and_alpha
        original_update_after_reset = self.agent.update_after_reset

        def wrapped_update_actor_and_alpha(obs, logger, step, print_flag=False):
            result = original_update_actor_and_alpha(obs, logger, step, print_flag)
            self._actor_update_count += 1
            if self._actor_update_count % int(self.cfg.gauge.pg.k_perf) == 0:
                self._maybe_apply_perf_correction(logger, step)
            return result

        def wrapped_update_after_reset(*args, **kwargs):
            self._in_reset_update = True
            try:
                return original_update_after_reset(*args, **kwargs)
            finally:
                self._in_reset_update = False

        self.agent.update_actor_and_alpha = wrapped_update_actor_and_alpha
        self.agent.update_after_reset = wrapped_update_after_reset

    def _perf_horizon_default(self):
        configured = int(self.cfg.gauge.pg.rollout_horizon)
        if configured > 0:
            return configured
        if str(self.cfg.gauge.pg.u_source) == "recent":
            return int(self.cfg.gauge.pg.recent_u_horizon)
        self._ensure_perf_env()
        return max_episode_steps(self._perf_env)

    def _ensure_perf_env(self):
        if self._perf_env is None:
            self._perf_env = make_env_like(self.cfg, seed_offset=self.cfg.gauge.pg.env_seed_offset)
        return self._perf_env

    def _collect_rollout_perf_trajectories(self):
        trajectories = []
        horizon = self._perf_horizon_default()
        perf_env = self._ensure_perf_env()
        for _idx in range(int(self.cfg.gauge.pg.n_traj)):
            obs = _reset_env(perf_env)
            self.agent.reset()
            traj = []
            for _step in range(horizon):
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)
                traj.append(np.concatenate([obs, action], axis=-1).astype(np.float32))
                next_obs, _reward, done, _extra = _step_env(perf_env, action)
                obs = next_obs
                if done:
                    break
            trajectories.append(np.asarray(traj, dtype=np.float32))
        return trajectories

    def _collect_recent_perf_trajectories(self):
        if len(self._recent_sa) <= 0:
            return []
        arr = np.asarray(list(self._recent_sa), dtype=np.float32)
        horizon = max(1, min(self._perf_horizon_default(), len(arr)))
        if len(arr) < int(self.cfg.gauge.pg.recent_min_samples):
            return []
        trajectories = []
        max_start = len(arr) - horizon
        for _idx in range(int(self.cfg.gauge.pg.n_traj)):
            start = 0 if max_start <= 0 else int(self._perf_rng.randint(0, max_start + 1))
            trajectories.append(arr[start : start + horizon].copy())
        return trajectories

    def _collect_perf_trajectories(self):
        source = str(self.cfg.gauge.pg.u_source)
        if source == "recent":
            trajectories = self._collect_recent_perf_trajectories()
            if trajectories or not bool(self.cfg.gauge.pg.recent_fallback_rollout):
                return trajectories
            return self._collect_rollout_perf_trajectories()
        if source == "rollout":
            return self._collect_rollout_perf_trajectories()
        raise ValueError("unknown gauge.pg.u_source '{}'; expected recent or rollout".format(source))

    def _shift_inputs_for_perf_correction(self):
        kind = self.reward_model.shift_kind()
        if kind == "mean_buf":
            return self.reward_model._sample_replay_inputs(self.replay_buffer)
        if kind == "mean_ref":
            return self.reward_model._reference_inputs(self._ensure_reference_data())
        return None

    def _maybe_apply_perf_correction(self, logger, step):
        if self.method != "perfg":
            return
        if self._in_reset_update and not bool(self.cfg.gauge.pg.apply_during_reset):
            logger.log("train/perf_skipped_reset_update", 1.0, step)
            self._write_gauge_metric("train", "perf_skipped_reset_update", 1.0, step)
            return
        if self.step < self.cfg.num_seed_steps + self.cfg.num_unsup_steps:
            return
        if self.reward_model.base.buffer_index <= 0 and not self.reward_model.base.buffer_full:
            return

        perf_grad = bool(self.cfg.gauge.pg.perf_grad)
        gauge_corr = bool(self.cfg.gauge.pg.gauge_corr)
        if not perf_grad and not gauge_corr:
            logger.log("train/perf_skipped_disabled", 1.0, step)
            self._write_gauge_metric("train", "perf_skipped_disabled", 1.0, step)
            return

        trajectories = self._collect_perf_trajectories() if perf_grad else []
        if trajectories:
            shift_horizon = float(np.mean([len(traj) for traj in trajectories]))
        else:
            shift_horizon = float(self._perf_horizon_default())
        shift_inputs = self._shift_inputs_for_perf_correction() if gauge_corr else None
        actor_grads, diagnostics = compute_performative_correction(
            self.agent,
            self.reward_model,
            trajectories,
            rng=self._perf_rng,
            horizon=shift_horizon,
            pref_batch_size=self.cfg.gauge.pg.pref_batch_size,
            cg_iters=self.cfg.gauge.pg.cg_iters,
            ridge=self.cfg.gauge.pg.ridge,
            label_margin=self.cfg.label_margin,
            perf_grad=perf_grad,
            gauge_corr=gauge_corr,
            shift_inputs=shift_inputs,
            merge_cg=self.cfg.gauge.pg.merge_cg,
            fd_weights=self.cfg.gauge.pg.fd_weights,
            fd_eps=self.cfg.gauge.pg.fd_eps,
            rm_param_scope=self.cfg.gauge.pg.rm_param_scope,
            solver=self.cfg.gauge.pg.solver,
        )
        if actor_grads is not None:
            actor_grads, grad_diag = self._stabilize_actor_grads(actor_grads)
            diagnostics.update(grad_diag)
            if actor_grads is None:
                diagnostics["applied"] = 0.0
        if actor_grads is not None:
            applied_norm = apply_actor_correction(self.agent, actor_grads)
            diagnostics["applied_actor_grad_norm"] = applied_norm
        for key, value in diagnostics.items():
            if isinstance(value, (int, float, np.floating)):
                logger.log("train/perf_{}".format(key), float(value), step)
                self._write_gauge_metric("train", "perf_{}".format(key), value, step)

    def _stabilize_actor_grads(self, actor_grads):
        if not actor_grads:
            return actor_grads, {"skipped_empty_actor_grad": 1.0}
        flat = torch.cat([grad.reshape(-1).detach().cpu() for grad in actor_grads])
        if not bool(torch.isfinite(flat).all()):
            return None, {"skipped_nonfinite_actor_grad": 1.0}
        norm = float(torch.linalg.vector_norm(flat))
        max_norm = float(self.cfg.gauge.pg.max_actor_grad_norm)
        diagnostics = {"actor_grad_preclip_norm": norm}
        if max_norm > 0.0 and norm > max_norm:
            scale = max_norm / (norm + 1e-12)
            actor_grads = [grad * scale for grad in actor_grads]
            diagnostics["actor_grad_clip_coef"] = float(scale)
            diagnostics["actor_grad_postclip_norm"] = float(max_norm)
        else:
            diagnostics["actor_grad_clip_coef"] = 1.0
            diagnostics["actor_grad_postclip_norm"] = norm
        return actor_grads, diagnostics


@hydra.main(config_path="../config/train_gauge.yaml", strict=True)
def main(cfg):
    workspace = GaugeWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
