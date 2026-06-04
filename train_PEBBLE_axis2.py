#!/usr/bin/env python3
"""
Axis 2 Tier B training script.

Subclasses train_PEBBLE.Workspace to add:
  - RM capacity knob (rm_hidden_dim, rm_num_layers, rm_output_activation)
  - Fixed-step monitoring: co-measures kappa_hat, ensemble_spread, gof, and
    e_shift/e_epi/e_mis at the same frozen theta every monitor_frequency steps.
  - Checkpoint saving at checkpoint_step for H2.5 payoff test.
"""

import os
import sys
import csv
import json
import pickle
import time
import math
from collections import deque


def _reexec_with_mujoco_library_path():
    """Restart once so mujoco_py sees LD_LIBRARY_PATH at process startup."""
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
import hydra  # installed in the project conda env (hydra 0.11); LSP may not resolve it

# make project root importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_PEBBLE import Workspace as BaseWorkspace
from reward_model import RewardModel
from axis2_tier_b.monitor import (
    MonitorState,
    compute_kappa_hat,
    compute_ensemble_spread,
    compute_gof,
    compute_error_components,
    _build_teacher_pref_pairs,
    _build_clean_pref_pairs,
    _bt_ce_loss,
)

import utils


_AXIS2_CONFIGS = {
    'train_PEBBLE_axis2',
    'train_PEBBLE_axis2_shift',
    'train_PEBBLE_axis2_epi',
    'train_PEBBLE_axis2_mis',
}


def _extract_config_name(argv):
    """Hydra 0.11 lacks --config-name; support it for Axis 2 run scripts."""
    config_name = 'train_PEBBLE_axis2'
    cleaned = [argv[0]]
    skip_next = False
    for i, arg in enumerate(argv[1:], start=1):
        if skip_next:
            skip_next = False
            continue
        if arg == '--config-name':
            if i + 1 >= len(argv):
                raise SystemExit('--config-name requires a value')
            config_name = argv[i + 1]
            skip_next = True
        elif arg.startswith('--config-name='):
            config_name = arg.split('=', 1)[1]
        else:
            cleaned.append(arg)
    if config_name not in _AXIS2_CONFIGS:
        allowed = ', '.join(sorted(_AXIS2_CONFIGS))
        raise SystemExit(f'Unknown Axis 2 config {config_name!r}; expected one of: {allowed}')
    return config_name, cleaned


_AXIS2_CONFIG_NAME, sys.argv = _extract_config_name(sys.argv)


class Workspace(BaseWorkspace):
    def __init__(self, cfg):
        super().__init__(cfg)

        # Read capacity knob params with defaults (getattr is safe on Hydra DictConfig).
        rm_hidden_dim = getattr(cfg, 'rm_hidden_dim', 256)
        rm_num_layers = getattr(cfg, 'rm_num_layers', 3)
        rm_output_activation = getattr(cfg, 'rm_output_activation', 'tanh')

        # Rebuild reward model with capacity knobs.  super().__init__ already built one
        # with default 256/3/tanh, so we replace it here.
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
            use_wandb=cfg.use_wandb,
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

        # Monitor config (with defaults)
        self._monitor_frequency = getattr(cfg, 'monitor_frequency', 20000)
        self._probe_seg_size = getattr(cfg, 'probe_seg_size', 200)
        self._probe_size = getattr(cfg, 'probe_size', 1024)
        self._kappa_warmup = getattr(cfg, 'kappa_warmup_steps', 10)
        self._kappa_fresh = getattr(cfg, 'kappa_fresh_prefs', 64)
        self._checkpoint_step = getattr(cfg, 'checkpoint_step', 500000)

        self._monitor_state = MonitorState(
            seg_len=cfg.segment,
            probe_seg_size=self._probe_seg_size,
            probe_size=self._probe_size,
            kappa_warmup_steps=self._kappa_warmup,
            kappa_fresh_prefs=self._kappa_fresh,
            rng=np.random.default_rng(cfg.seed),
        )

        if getattr(cfg, 'use_wandb', False):
            try:
                import wandb
                if wandb.run is not None:
                    wandb.run.name = f"{cfg.env}__{cfg.experiment}__seed{cfg.seed}"
            except Exception as err:
                print(f"[axis2 init] wandb run naming skipped: {err}")

        # CSV log file
        self._csv_path = os.path.join(self.work_dir, 'axis2_metrics.csv')
        self._csv_fields = [
            'step', 'kappa_hat', 'sigma_hat', 'ensemble_spread',
            'gof_deployed', 'gof_inf', 'gof_epi', 'floor_used',
            'e_shift', 'e_epi', 'e_mis',
            'floor_he', 'floor_ref',
        ]
        with open(self._csv_path, 'w', newline='') as f:
            csv.writer(f).writerow(self._csv_fields)

    # ------------------------------------------------------------------
    # Monitoring
    # ------------------------------------------------------------------

    def _get_recent_data(self, n_recent=50):
        """Return (inputs, targets) from the reward model's n_recent most recent episodes."""
        inputs = self.reward_model.inputs[-n_recent:] if self.reward_model.inputs else []
        targets = self.reward_model.targets[-n_recent:] if self.reward_model.targets else []
        return inputs, targets

    def _active_label_budget(self):
        """Finite refit budget matching the labels ψ_cur can actually train on."""
        if getattr(self.reward_model, 'buffer_window_rounds', None) is not None:
            round_sizes = getattr(self.reward_model, 'round_sizes', [])
            if round_sizes:
                return max(int(sum(round_sizes)), 10)
            return max(int(getattr(self.reward_model, 'buffer_index', 0)), 10)
        return max(int(self.labeled_feedback), 10)

    def _run_monitor(self):
        """Co-measure all instruments and error components at current frozen theta."""
        device = str(self.device)
        eps = self.cfg.teacher_eps_mistake

        # switch to eval
        self.agent.actor.eval()
        for m in self.reward_model.ensemble:
            m.eval()

        recent_inputs, recent_targets = self._get_recent_data()
        if len(recent_inputs) < 2:
            self._restore_train_mode()
            return

        # --- error components (produces psi_fin, psi_inf as side outputs) ---
        n_fin = self._active_label_budget()
        # ψ_fin/ψ_inf are refits of the SAME RM class as reward_model_cur (deployed arch).
        # In the mis condition that means 16×1; ψ_inf with abundant clean labels is the
        # irreducible floor of THAT class, which is what e_mis is supposed to measure.
        # Yardstick override is reserved for sanity-check smoke tests only.
        ec = compute_error_components(
            reward_model_cur=self.reward_model,
            n_fin=n_fin,
            recent_inputs=recent_inputs,
            recent_targets=recent_targets,
            state=self._monitor_state,
            device=device,
        )
        psi_inf = ec.get('psi_inf')

        # --- kappa_hat ---
        kappa_hat, sigma_hat = compute_kappa_hat(
            agent=self.agent,
            reward_model=self.reward_model,
            recent_inputs=recent_inputs,
            recent_targets=recent_targets,
            state=self._monitor_state,
            device=device,
        )

        # --- ensemble spread ---
        probe_segs = self._sample_probe_segs(recent_inputs)
        ens_spread = float('nan')
        if probe_segs is not None:
            ens_spread = compute_ensemble_spread(self.reward_model, probe_segs)

        n_gof_pairs = min(200, max(n_fin // 5, 20))

        # --- fresh held-out preference pairs for gof_deployed, labeled by the SAME teacher
        #     used to train ψ_cur (BT-with-normalization, or eps_mistake in the confirmatory
        #     mis run).  Routing rm_template through ensures the held-out distribution exactly
        #     matches the deployed training distribution.
        seg1_gof, seg2_gof, lbl_gof = _build_teacher_pref_pairs(
            recent_inputs, recent_targets,
            seg_len=self.cfg.segment,
            n_pairs=n_gof_pairs,
            rng=self._monitor_state.rng,
            rm_template=self.reward_model,
        )

        # --- separate clean pairs for gof_inf: isolates pure misspecification ---
        # Using teacher-labeled pairs here would conflate BT noise / finite-label effects with
        # capacity limits, invalidating the GoF × e_mis decomposition.
        seg1_clean, seg2_clean, lbl_clean = _build_clean_pref_pairs(
            recent_inputs, recent_targets,
            seg_len=self.cfg.segment,
            n_pairs=n_gof_pairs,
            rng=self._monitor_state.rng,
        )

        # --- GoF of deployed RM (epistemic + misspec), evaluated on teacher-labeled pairs ---
        gof_deployed, floor_used = compute_gof(
            reward_model=self.reward_model,
            seg1=seg1_gof, seg2=seg2_gof, labels=lbl_gof,
            teacher_eps_mistake=eps,
            device=device,
            reference_rm=psi_inf,
        )

        # --- GoF of psi_inf (pure misspecification), evaluated on clean pairs ---
        # eps=0 floor because we gave it clean labels — any remaining CE loss is misspec.
        gof_inf = float('nan')
        if psi_inf is not None and seg1_clean is not None:
            gof_inf, _ = compute_gof(
                reward_model=psi_inf,
                seg1=seg1_clean, seg2=seg2_clean, labels=lbl_clean,
                teacher_eps_mistake=0.0,
                device=device,
                reference_rm=None,
            )

        # --- compute floor_he and floor_ref separately for calibration check ---
        floor_he = (-eps * math.log(eps) - (1.0 - eps) * math.log(1.0 - eps)
                    if 0.0 < eps < 1.0 else 0.0)
        floor_ref = 0.0
        if psi_inf is not None and seg1_clean is not None:
            floor_ref = _bt_ce_loss(psi_inf, seg1_clean, seg2_clean, lbl_clean, device)

        gof_epi = (float(gof_deployed) - float(gof_inf)
                   if not any(v != v for v in [gof_deployed, gof_inf]) else float('nan'))

        # --- write to CSV ---
        row = {
            'step': self.step,
            'kappa_hat': kappa_hat,
            'sigma_hat': sigma_hat,
            'ensemble_spread': ens_spread,
            'gof_deployed': gof_deployed,
            'gof_inf': gof_inf,
            'gof_epi': gof_epi,
            'floor_used': floor_used,
            'e_shift': ec.get('e_shift', float('nan')),
            'e_epi': ec.get('e_epi', float('nan')),
            'e_mis': ec.get('e_mis', float('nan')),
            'floor_he': floor_he,
            'floor_ref': floor_ref,
        }
        with open(self._csv_path, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=self._csv_fields)
            w.writerow(row)

        if getattr(self.cfg, 'use_wandb', False):
            try:
                import wandb
                wandb.log({f'axis2/{k}': v for k, v in row.items() if k != 'step'},
                          step=self.step)
            except Exception as err:
                print(f"[monitor step={self.step}] wandb logging skipped: {err}")

        print(f"[monitor step={self.step}] "
              f"kappa={kappa_hat:.3f} spread={ens_spread:.4f} "
              f"gof_dep={gof_deployed:.4f} gof_inf={gof_inf:.4f} "
              f"e_shift={ec.get('e_shift', float('nan')):.4f} "
              f"e_epi={ec.get('e_epi', float('nan')):.4f} "
              f"e_mis={ec.get('e_mis', float('nan')):.4f}")

        self._restore_train_mode()

    def _sample_probe_segs(self, recent_inputs):
        """Sample probe segments from recent on-policy data."""
        all_segs = []
        seg_len = self.cfg.segment
        for ep in recent_inputs:
            if len(ep) < seg_len:
                continue
            n_full = len(ep) // seg_len
            for i in range(n_full):
                all_segs.append(ep[i * seg_len: (i + 1) * seg_len])
        if not all_segs:
            return None
        all_segs = np.array(all_segs)
        n = min(self._probe_seg_size, len(all_segs))
        idx = self._monitor_state.rng.choice(len(all_segs), size=n, replace=False)
        return all_segs[idx]

    def _restore_train_mode(self):
        self.agent.actor.train()
        for m in self.reward_model.ensemble:
            m.train()

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def _save_checkpoint(self):
        ckpt_dir = os.path.join(self.work_dir, f'ckpt_{self.step}')
        os.makedirs(ckpt_dir, exist_ok=True)

        self.agent.save(ckpt_dir, self.step)
        self.reward_model.save(ckpt_dir, self.step)

        with open(os.path.join(ckpt_dir, 'replay_buffer.pkl'), 'wb') as f:
            pickle.dump(self.replay_buffer, f)

        # record current metric CSV row if available
        meta = {
            'step': self.step,
            'total_feedback': self.total_feedback,
            'labeled_feedback': self.labeled_feedback,
            'cfg': {
                'rm_hidden_dim': getattr(self.cfg, 'rm_hidden_dim', 256),
                'rm_num_layers': getattr(self.cfg, 'rm_num_layers', 3),
                'rm_output_activation': getattr(self.cfg, 'rm_output_activation', 'tanh'),
                'teacher_beta': self.cfg.teacher_beta,
                'teacher_eps_mistake': self.cfg.teacher_eps_mistake,
                'bt_normalize_by_gap_std': getattr(self.cfg, 'bt_normalize_by_gap_std', False),
                'bt_gap_ema_alpha': getattr(self.cfg, 'bt_gap_ema_alpha', 0.9),
                'buffer_window_rounds': getattr(self.cfg, 'buffer_window_rounds', 0),
                'agent_name': self.cfg.agent.name,
                'actor_hidden_dim': self.cfg.diag_gaussian_actor.params.hidden_dim,
                'actor_hidden_depth': self.cfg.diag_gaussian_actor.params.hidden_depth,
                'critic_hidden_dim': self.cfg.double_q_critic.params.hidden_dim,
                'critic_hidden_depth': self.cfg.double_q_critic.params.hidden_depth,
                'agent_batch_size': self.cfg.agent.params.batch_size,
                'num_interact': self.cfg.num_interact,
                'max_feedback': self.cfg.max_feedback,
                'segment': self.cfg.segment,
                'env': self.cfg.env,
                'seed': self.cfg.seed,
            },
        }
        with open(os.path.join(ckpt_dir, 'meta.json'), 'w') as f:
            json.dump(meta, f, indent=2)

        print(f"[checkpoint] saved to {ckpt_dir}")

    # ------------------------------------------------------------------
    # run() override — adds monitoring and checkpoint hooks
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

            # training updates
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

            elif self.step > self.cfg.num_seed_steps + self.cfg.num_unsup_steps:
                if self.total_feedback < self.cfg.max_feedback:
                    if interact_count == self.cfg.num_interact:
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

            # --- monitoring hook ---
            past_warmup = self.step > (self.cfg.num_seed_steps + self.cfg.num_unsup_steps)
            if past_warmup and self.step % self._monitor_frequency == 0:
                self._run_monitor()

            # --- checkpoint hook ---
            if self.step == self._checkpoint_step:
                self._save_checkpoint()

        self.agent.save(self.work_dir, self.step)
        self.reward_model.save(self.work_dir, self.step)


@hydra.main(config_path=f'config/{_AXIS2_CONFIG_NAME}.yaml', strict=False)
def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()


if __name__ == '__main__':
    main()
