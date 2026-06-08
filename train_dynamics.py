#!/usr/bin/env python3
"""Dynamics over-optimization experiment.

Subclasses train_PEBBLE_axis2.Workspace and adds:
  - Phase 1 freeze: skip learn_reward() and relabel_with_predictor() while
    phase1_freeze_step <= step < phase2_resume_step.
  - Phase 2 resume: normal RM updates after phase2_resume_step.
  - FD ρ/κ monitor at every monitor_frequency step (in addition to Tier B's
    monitors), logged to dynamics_fd.csv and W&B (if enabled).

Run:
  conda run -n bpref python train_dynamics.py env=metaworld_door-open-v2 seed=1

Phase 1/2 boundaries default to 200k/500k in config/train_dynamics.yaml;
override with CLI: phase1_freeze_step=30000 phase2_resume_step=70000.
"""
import os
import sys
import csv
import time
from collections import deque


def _reexec_with_mujoco_library_path():
    if os.environ.get('NOISYPBRL_MUJOCO_LD_READY') == '1':
        return
    candidates = [
        os.path.expanduser('~/.mujoco/mujoco210/bin'),
        '/usr/lib/nvidia',
    ]
    parts = [p for p in os.environ.get('LD_LIBRARY_PATH', '').split(':') if p]
    changed = False
    for candidate in candidates:
        if os.path.isdir(candidate) and candidate not in parts:
            parts.append(candidate)
            changed = True
    os.environ['NOISYPBRL_MUJOCO_LD_READY'] = '1'
    if changed:
        os.environ['LD_LIBRARY_PATH'] = ':'.join(parts)
        os.execv(sys.executable, [sys.executable] + sys.argv)


_reexec_with_mujoco_library_path()

import numpy as np
import hydra

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# The base class is the same as train_PEBBLE_axis2.Workspace.  We import it
# without going through train_PEBBLE_axis2's argv-hijacking @hydra.main.
from train_PEBBLE import Workspace as PebbleWorkspace
from reward_model import RewardModel
from axis2_tier_b.monitor import MonitorState
from axis2_dynamics_pg.monitors_fd import compute_rho_kappa_fd_monitor

import utils


class Workspace(PebbleWorkspace):
    def __init__(self, cfg):
        super().__init__(cfg)
        if hasattr(self.agent, 'critic'):
            self.agent.critic.wandb_dormant_logging = False

        rm_hidden_dim = getattr(cfg, 'rm_hidden_dim', 256)
        rm_num_layers = getattr(cfg, 'rm_num_layers', 3)
        rm_output_activation = getattr(cfg, 'rm_output_activation', 'tanh')

        self.reward_model = RewardModel(
            self.env.observation_space.shape[0],
            self.env.action_space.shape[0],
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
            use_wandb=False,
            bt_log_period=cfg.bt_log_period,
            feed_type=cfg.feed_type,
            capacity=cfg.max_feedback * cfg.large_batch,
            hidden_dim=rm_hidden_dim,
            num_layers=rm_num_layers,
            output_activation=rm_output_activation,
            buffer_window_rounds=cfg.buffer_window_rounds if cfg.buffer_window_rounds > 0 else None,
            bt_normalize_by_gap_std=getattr(cfg, 'bt_normalize_by_gap_std', False),
            bt_gap_ema_alpha=getattr(cfg, 'bt_gap_ema_alpha', 0.9),
        )

        # ---- Dynamics-specific config ----
        self._phase1_freeze_step = int(getattr(cfg, 'phase1_freeze_step', 200000))
        self._phase2_resume_step = int(getattr(cfg, 'phase2_resume_step', 500000))
        self._fd_pref_batch = int(getattr(cfg, 'fd_pref_batch_size', 128))
        self._fd_refit_epochs = int(getattr(cfg, 'fd_refit_epochs', 10))
        self._monitor_frequency = int(getattr(cfg, 'monitor_frequency', 20000))

        self._fd_rng = np.random.default_rng(cfg.seed + 7777)

        # Persistent eval-mode flag for FD monitor sampling buffer (Tier B's
        # MonitorState isn't strictly needed here, but we keep one for parity).
        self._monitor_state = MonitorState(
            seg_len=cfg.segment,
            probe_seg_size=getattr(cfg, 'probe_seg_size', 200),
            probe_size=getattr(cfg, 'probe_size', 1024),
            kappa_warmup_steps=getattr(cfg, 'kappa_warmup_steps', 10),
            kappa_fresh_prefs=getattr(cfg, 'kappa_fresh_prefs', 64),
            rng=np.random.default_rng(cfg.seed),
        )

        self._fd_csv_path = os.path.join(self.work_dir, 'dynamics_fd.csv')
        self._fd_csv_fields = [
            'step', 'phase',
            'rho_fd', 'kappa_fd', 'g0_norm', 'g_perp_norm',
            'cos_stale_fresh', 'pref_batch',
        ]
        with open(self._fd_csv_path, 'w', newline='') as f:
            csv.writer(f).writerow(self._fd_csv_fields)

        # ---- Phase log file (which steps were frozen vs updated) ----
        self._phase_log_path = os.path.join(self.work_dir, 'phase_log.csv')
        with open(self._phase_log_path, 'w', newline='') as f:
            csv.writer(f).writerow(['step', 'event', 'phase'])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _current_phase(self):
        if self.step < self._phase1_freeze_step:
            return 'warmup_or_p0'
        if self.step < self._phase2_resume_step:
            return 'p1_frozen'
        return 'p2_resumed'

    def _rm_update_allowed(self):
        return not (self._phase1_freeze_step <= self.step < self._phase2_resume_step)

    def _log_phase_event(self, event: str):
        with open(self._phase_log_path, 'a', newline='') as f:
            csv.writer(f).writerow([self.step, event, self._current_phase()])

    def _run_fd_monitor(self):
        try:
            out = compute_rho_kappa_fd_monitor(
                agent=self.agent,
                reward_model=self.reward_model,
                ds=self.env.observation_space.shape[0],
                device=str(self.device),
                pref_batch_size=self._fd_pref_batch,
                refit_epochs=self._fd_refit_epochs,
                rng=self._fd_rng,
            )
        except Exception as err:
            print(f"[fd-monitor step={self.step}] failed: {err}")
            return

        row = {
            'step': self.step,
            'phase': self._current_phase(),
            'rho_fd': out.get('rho_fd', float('nan')),
            'kappa_fd': out.get('kappa_fd', float('nan')),
            'g0_norm': out.get('g0_norm', float('nan')),
            'g_perp_norm': out.get('g_perp_norm', float('nan')),
            'cos_stale_fresh': out.get('cos_stale_fresh', float('nan')),
            'pref_batch': out.get('pref_batch', 0.0),
        }
        with open(self._fd_csv_path, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=self._fd_csv_fields).writerow(row)

        if getattr(self.cfg, 'use_wandb', False):
            try:
                import wandb
                wandb.log({f'fd/{k}': v for k, v in row.items()
                           if k not in ('step', 'phase')}, step=self.step)
            except Exception as err:
                print(f"[fd-monitor step={self.step}] wandb skipped: {err}")

        print(f"[fd-monitor step={self.step} phase={row['phase']}] "
              f"rho_fd={row['rho_fd']:.3f} kappa_fd={row['kappa_fd']:.3f} "
              f"g0={row['g0_norm']:.4g} g_perp={row['g_perp_norm']:.4g}")

    # ------------------------------------------------------------------
    # run() — copied from train_PEBBLE_axis2 with Phase 1/2 gating added
    # ------------------------------------------------------------------

    def run(self):
        episode, episode_reward, done = 0, 0, True
        if self.log_success:
            episode_success = 0
        true_episode_reward = 0

        avg_train_true_return = deque([], maxlen=10)
        start_time = time.time()
        interact_count = 0

        while self.step < self.cfg.num_train_steps:
            if done:
                if self.step > 0:
                    self.logger.log('train/duration', time.time() - start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(self.step, save=(self.step > self.cfg.num_seed_steps))

                if self.step > 0 and self.step % self.cfg.eval_frequency == 0:
                    self.logger.log('eval/episode', episode, self.step)
                    self.evaluate()

                self.logger.log('train/episode_reward', episode_reward, self.step)
                self.logger.log('train/true_episode_reward', true_episode_reward, self.step)

                if self.log_success:
                    self.logger.log('train/episode_success', episode_success, self.step)
                    self.logger.log('train/true_episode_success', episode_success, self.step)

                obs = self.env.reset()
                self.agent.reset()
                done = False
                episode_reward = 0
                avg_train_true_return.append(true_episode_reward)
                true_episode_reward = 0
                if self.log_success:
                    episode_success = 0
                episode_step = 0
                episode += 1
                self.logger.log('train/episode', episode, self.step)

            # sample action
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            # ---- end-of-unsup-phase RM training (always allowed; it's the
            # initial RM training before any potential freeze begins) ----
            if self.step == (self.cfg.num_seed_steps + self.cfg.num_unsup_steps):
                if self.cfg.reward_schedule == 1:
                    frac = (self.cfg.num_train_steps - self.step) / self.cfg.num_train_steps
                    frac = max(frac, 0.01)
                elif self.cfg.reward_schedule == 2:
                    frac = self.cfg.num_train_steps / (self.cfg.num_train_steps - self.step + 1)
                else:
                    frac = 1
                self.reward_model.change_batch(frac)

                new_margin = (np.mean(list(avg_train_true_return))
                              * (self.cfg.segment / self.env._max_episode_steps))
                self.reward_model.set_teacher_thres_skip(new_margin)
                self.reward_model.set_teacher_thres_equal(new_margin)

                self.reward_model.env_step = self.step
                self.learn_reward(first_flag=1)
                self.reward_model.pre_relabel_logging(self.step)
                self.replay_buffer.relabel_with_predictor(self.reward_model)
                self.agent.reset_critic()
                self.agent.update_after_reset(
                    self.replay_buffer, self.logger, self.step,
                    gradient_update=self.cfg.reset_update, policy_update=True)
                interact_count = 0
                self._log_phase_event('rm_train_unsup_end')

            elif self.step > self.cfg.num_seed_steps + self.cfg.num_unsup_steps:
                if self.total_feedback < self.cfg.max_feedback:
                    if interact_count == self.cfg.num_interact:
                        # ----- PHASE GATE -----
                        if self._rm_update_allowed():
                            if self.cfg.reward_schedule == 1:
                                frac = (self.cfg.num_train_steps - self.step) / self.cfg.num_train_steps
                                frac = max(frac, 0.01)
                            elif self.cfg.reward_schedule == 2:
                                frac = self.cfg.num_train_steps / (self.cfg.num_train_steps - self.step + 1)
                            else:
                                frac = 1
                            self.reward_model.change_batch(frac)

                            new_margin = (np.mean(list(avg_train_true_return))
                                          * (self.cfg.segment / self.env._max_episode_steps))
                            self.reward_model.set_teacher_thres_skip(
                                new_margin * self.cfg.teacher_eps_skip)
                            self.reward_model.set_teacher_thres_equal(
                                new_margin * self.cfg.teacher_eps_equal)

                            if self.reward_model.mb_size + self.total_feedback > self.cfg.max_feedback:
                                self.reward_model.set_batch(self.cfg.max_feedback - self.total_feedback)

                            self.reward_model.env_step = self.step
                            self.learn_reward()
                            self.reward_model.pre_relabel_logging(self.step)
                            self.replay_buffer.relabel_with_predictor(self.reward_model)
                            self._log_phase_event('rm_update_allowed')
                        else:
                            self._log_phase_event('rm_update_skipped_frozen')
                        interact_count = 0

                self.agent.update(self.replay_buffer, self.logger, self.step, 1)

            elif self.step > self.cfg.num_seed_steps:
                self.agent.update_state_ent(self.replay_buffer, self.logger, self.step,
                                            gradient_update=1, K=self.cfg.topK)

            # environment step
            next_obs, reward, terminated, truncated, extra = self.env.step(action)
            reward_hat = self.reward_model.r_hat(np.concatenate([obs, action], axis=-1))

            done = terminated or truncated
            done_float = float(done)
            done_no_max = 0 if truncated and not terminated else done_float
            episode_reward += reward_hat
            true_episode_reward += reward

            if self.log_success:
                episode_success = max(episode_success, extra['success'])

            self.reward_model.add_data(obs, action, reward, done_float)
            self.replay_buffer.add(obs, action, reward_hat, next_obs, done_float, done_no_max)

            obs = next_obs
            episode_step += 1
            self.step += 1
            interact_count += 1

            # ---- FD monitor hook ----
            past_warmup = self.step > (self.cfg.num_seed_steps + self.cfg.num_unsup_steps)
            if past_warmup and self.step % self._monitor_frequency == 0:
                self._run_fd_monitor()

        self.agent.save(self.work_dir, self.step)
        self.reward_model.save(self.work_dir, self.step)


@hydra.main(config_path='config/train_dynamics.yaml', strict=False)
def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()


if __name__ == '__main__':
    main()
