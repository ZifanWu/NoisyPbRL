#!/usr/bin/env python3
"""One-shot PEBBLE trainer — Signal 2 (one-shot control) for the gauge-ambiguity experiment.

The RM is trained exactly once on data from the unsup-pretrained policy (pi_ref),
then frozen for the entire policy-training phase.  Comparing the variance in final
J(pi) across gauge modes between this script and the iterative train_PEBBLE.py gives
the sigma_iter / sigma_oneshot ratio that distinguishes iteration-amplified gauge
effects from static gauge sensitivity.

Usage (single run)::

    python train_PEBBLE_oneshot.py \\
        env=walker_walk seed=1 gauge_mode=none \\
        exp_dir=/your/path

Full experiment matrix — same CLI as train_PEBBLE.py, just swap the script.
"""

import os
import time

import numpy as np
import torch

import hydra

import utils
from train_PEBBLE import (
    Workspace,
    _ACTIVE_RM_DATA,
    _PASSIVE_POLICY,
)


class OneShotWorkspace(Workspace):
    """One-shot variant: train RM once from pi_ref data, then freeze it."""

    def run(self):
        cfg       = self.cfg
        mode      = cfg.tandem_mode
        warmup_end = cfg.num_seed_steps + cfg.num_unsup_steps

        # ── Phase 1: identical unsup pretrain ────────────────────────────────
        # Exits with self.step == warmup_end, self._loop_obs / _loop_done set.
        self._unsup_pretrain_loop()

        # ── Phase 1.5: collect D_ref from pi_ref for gauge diagnostics ───────
        # learn_reward checks self._ref_segments_collected internally, but we
        # collect it here first so the one-shot RM training can skip the internal
        # branch and we have full control over timing.
        if self._gauge_eval_env is not None and not self._ref_segments_collected:
            _D_ref = self._collect_on_policy_segments(256)
            self.reward_model.set_ref_segments(_D_ref)
            self._ref_segments_collected = True

        # ── Phase 2: one-shot RM training ─────────────────────────────────────
        # Use the full preference budget in a single event.
        frac = self._compute_frac()
        self.reward_model.change_batch(frac)
        if mode == 'baseline':
            self.tandem_logger.log_schedule(self.step, frac, self.reward_model.mb_size)

        new_margin = (np.mean(self._avg_train_true_return)
                      * (cfg.segment / self.env._max_episode_steps))
        self.reward_model.set_teacher_thres_skip(new_margin)
        self.reward_model.set_teacher_thres_equal(new_margin)

        # Spend the entire feedback budget at once.
        self.reward_model.set_batch(cfg.max_feedback)

        self.reward_model.env_step = self.step
        self.learn_reward(first_flag=1)

        # Log the one-shot preference count for the analysis script.
        self.logger.log('train/oneshot_n_preferences', self.total_feedback, self.step)
        self.logger.log('train/rm_update_count', self._rm_update_idx, self.step)

        # Freeze RM — no retraining will occur during policy training.
        # (total_feedback >= max_feedback already blocks further learn_reward calls;
        # set_frozen is belt-and-suspenders.)
        self.reward_model.set_frozen(True)

        # ── Warmup-end env step (mirrors _policy_training_loop at step == warmup_end) ──
        obs  = self._loop_obs
        done = self._loop_done

        # Episode boundary handling
        if done:
            self.logger.log('train/episode_reward',      0, self.step)
            self.logger.log('train/true_episode_reward', 0, self.step)
            if mode not in _PASSIVE_POLICY:
                obs = self.env.reset()
            done = False

        # Transition
        with utils.eval_mode(self.agent):
            action = self.agent.act(obs, sample=True)
        next_obs, env_reward, terminated, truncated, extra = self.env.step(action)
        done        = terminated or truncated
        done_no_max = 0.0 if (truncated and not terminated) else float(done)
        env_reward  = float(env_reward)

        # RM segment pool
        if mode in _ACTIVE_RM_DATA:
            self.reward_model.add_data(obs, action, env_reward, float(done))

        # Reward hat + replay buffer
        reward_hat = self.reward_model.r_hat(
            np.concatenate([obs, action], axis=-1))
        self.replay_buffer.add(obs, action, reward_hat,
                               next_obs, float(done), done_no_max)

        # Baseline transition log
        if mode == 'baseline':
            self.tandem_logger.log_transition(
                self.step, obs, action, env_reward,
                next_obs, float(done), done_no_max)

        # Relabel + critic reset (identical to iterative warmup_end block)
        self.reward_model.pre_relabel_logging(self.step)
        self.replay_buffer.relabel_with_predictor(self.reward_model)
        self.agent.reset_critic()
        self.agent.update_after_reset(
            self.replay_buffer, self.logger, self.step,
            gradient_update=cfg.reset_update,
            policy_update=True)

        # Hand off to _policy_training_loop starting at warmup_end + 1.
        self._loop_obs  = next_obs
        self._loop_done = done
        self.step      += 1

        # ── Phase 3: policy training with frozen RM + periodic gauge diagnostics ──
        self._policy_training_loop(
            diag_callback=self._oneshot_gauge_diag,
            diag_freq=cfg.gauge_diag_frequency)

    # ── Gauge diagnostic callback (called every gauge_diag_frequency steps) ──

    def _oneshot_gauge_diag(self, step: int) -> None:
        """Log gauge diagnostics and policy-drift metric during policy training."""
        if self._gauge_eval_env is None:
            return

        D_cur = self._collect_on_policy_segments(64)
        gauge_metrics = self.reward_model.get_gauge_diagnostics(D_cur)
        for key, value in gauge_metrics.items():
            self.logger.log(key, value, step)

        # Policy-drift: how far the current policy has moved from pi_ref in
        # RM-return units.  Positive → policy is earning more predicted reward
        # than pi_ref did at the time the RM was trained.
        if (self.reward_model.ref_segments is not None
                and 'train/rm_mean_on_policy' in gauge_metrics
                and 'train/rm_mean_on_ref' in gauge_metrics):
            drift = (gauge_metrics['train/rm_mean_on_policy']
                     - gauge_metrics['train/rm_mean_on_ref'])
            self.logger.log('train/oneshot_policy_drift', drift, step)

            ref_std = gauge_metrics.get('train/rm_std_on_ref', 1.0) or 1.0
            self.logger.log('train/oneshot_policy_drift_normalized',
                            drift / ref_std, step)


@hydra.main(config_path='config/train_PEBBLE_oneshot.yaml', strict=True)
def main(cfg):
    workspace = OneShotWorkspace(cfg)
    workspace.run()


if __name__ == '__main__':
    main()
