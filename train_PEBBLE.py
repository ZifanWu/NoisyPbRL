#!/usr/bin/env python3
"""PEBBLE trainer with tandem-experiment support.

tandem_mode values
------------------
baseline              — vanilla PEBBLE; logs everything for tandem replay.
all_passive           — policy and RM both passive (consume baseline data stream).
passive_pol_active_rm — policy learns on baseline transitions; RM collects its own segments.
active_pol_passive_rm — policy acts in env; RM learns on baseline preference pairs.
active_pol_passive_query — policy active; RM uses own segments but scored by baseline RM.
active_pol_passive_dist  — policy active; RM draws candidates from baseline segment pool.
"""

import os
import time
from collections import deque

import numpy as np
import torch

import hydra

import utils
import sanity_checks
from logger import Logger
from replay_buffer import ReplayBuffer
from reward_model import RewardModel
from tandem_logger import TandemLogger
from tandem_reader import TandemReader


# Conditions where the policy never steps the training env
_PASSIVE_POLICY = {"all_passive", "passive_pol_active_rm"}

# Conditions where the RM trains on baseline preference pairs (no querying from own pool)
_PASSIVE_RM_PAIRS = {"all_passive", "active_pol_passive_rm"}

# Conditions where add_data() should be called with tandem env observations
_ACTIVE_RM_DATA = {"baseline", "passive_pol_active_rm", "active_pol_passive_query"}


class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')
        self.cfg = cfg
        mode = cfg.tandem_mode

        if cfg.use_wandb:
            import socket, wandb
            wandb.init(
                project='pbrl_gauge_ambiguity',
                name=f'{cfg.env}__{cfg.agent.name}__seed{cfg.seed}__{mode}',
                config=dict(cfg),
                notes=socket.gethostname(),
            )

        self.logger = Logger(
            self.work_dir,
            save_tb=cfg.log_save_tb,
            use_wandb=cfg.use_wandb,
            log_frequency=cfg.log_frequency,
            agent=cfg.agent.name)

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.log_success = False

        # ── Environment ─────────────────────────────────────────────────────
        if 'metaworld' in cfg.env:
            self.env = utils.make_metaworld_env(cfg)
            self.log_success = True
        else:
            self.env = utils.make_env(cfg)

        # Condition (ii): needs a separate env for RM data collection while the
        # policy's gradient updates come from the baseline replay stream.
        self._tandem_env = None
        if mode == 'passive_pol_active_rm':
            if 'metaworld' in cfg.env:
                self._tandem_env = utils.make_metaworld_env(cfg)
            else:
                self._tandem_env = utils.make_env(cfg)

        obs_dim    = self.env.observation_space.shape[0]
        act_dim    = self.env.action_space.shape[0]
        action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max()),
        ]
        cfg.agent.params.obs_dim       = obs_dim
        cfg.agent.params.action_dim    = act_dim
        cfg.agent.params.action_range  = action_range
        self.agent = hydra.utils.instantiate(cfg.agent)

        self.replay_buffer = ReplayBuffer(
            self.env.observation_space.shape,
            self.env.action_space.shape,
            int(cfg.replay_buffer_capacity),
            self.device)

        # ── Reward model ────────────────────────────────────────────────────
        self.reward_model = RewardModel(
            obs_dim, act_dim,
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
            log_extra_metrics=cfg.log_extra_metrics,
            feed_type=cfg.feed_type,
            capacity=cfg.max_feedback * cfg.large_batch,
            gauge_mode=cfg.gauge_mode)

        # For condition iv-a: a scratch RewardModel that receives baseline RM
        # weights loaded from the log, used only to score candidate pairs.
        self._baseline_rm_helper = None
        if mode == 'active_pol_passive_query':
            self._baseline_rm_helper = RewardModel(
                obs_dim, act_dim,
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
                feed_type=cfg.feed_type,
                capacity=cfg.max_feedback * cfg.large_batch)

        # ── Tandem infrastructure ────────────────────────────────────────────
        self.tandem_logger = None
        self.tandem_reader = None

        # Canonical filename: same convention on every machine
        _default_log_path = os.path.join(
            cfg.tandem_log_dir,
            f'tandem_baseline_{cfg.env}_seed{cfg.seed}.h5')
        _log_path = cfg.tandem_log_path if cfg.tandem_log_path is not None \
            else _default_log_path

        if mode == 'baseline':
            os.makedirs(cfg.tandem_log_dir, exist_ok=True)
            self.tandem_logger = TandemLogger(
                _default_log_path, obs_dim, act_dim,
                max_steps=int(cfg.num_train_steps))
            print(f'[tandem] baseline logger → {_default_log_path}')
        else:
            if not os.path.exists(_log_path):
                raise FileNotFoundError(
                    f"Baseline log not found: {_log_path}\n"
                    f"Run tandem_mode=baseline first, or pass tandem_log_path=/explicit/path.h5")
            self.tandem_reader = TandemReader(_log_path)
            print(f'[tandem] reader ← {_log_path}  ({self.tandem_reader.num_query_events} events)')

        # ── Bookkeeping ──────────────────────────────────────────────────────
        self.total_feedback  = 0
        self.labeled_feedback = 0
        self.step = 0
        self._rm_update_idx   = 0        # index into query events

        # Episode tracking for baseline logging
        self._ep_id          = 0
        self._ep_start_step  = 0

        # ── Gauge experiment: on-policy eval env (baseline mode only) ────────
        # [Gauge experiment, Step 3] A separate env is used to collect fresh
        # on-policy segments without perturbing the training env's RNG state.
        self._gauge_eval_env = None
        self._gauge_obs      = None
        self._ref_segments_collected = False
        if cfg.tandem_mode == 'baseline':
            if 'metaworld' in cfg.env:
                self._gauge_eval_env = utils.make_metaworld_env(cfg)
            else:
                self._gauge_eval_env = utils.make_env(cfg)
            self._gauge_obs = self._gauge_eval_env.reset()

        # ── Shared loop state (set by _unsup_pretrain_loop, read by _policy_training_loop) ──
        self._avg_train_true_return = deque([], maxlen=10)
        self._loop_obs  = None   # observation at loop hand-off point
        self._loop_done = True   # done flag at loop hand-off point

    # ────────────────────────────────────────────────────────────────────────
    # Gauge experiment helpers
    # ────────────────────────────────────────────────────────────────────────

    def _collect_on_policy_segments(self, n_segs: int) -> np.ndarray:
        """Collect n_segs contiguous segments from the gauge eval env.

        [Gauge experiment, Step 3] Uses deterministic (sample=False) policy actions
        and the persistent _gauge_eval_env so RNG state in the training env is not
        disturbed.  Episode boundaries are handled by resetting the env; a segment
        may span a boundary, which is acceptable since the RM treats each timestep
        independently.

        Args:
            n_segs: number of segments of length cfg.segment to collect.

        Returns:
            numpy array of shape (n_segs, cfg.segment, obs_dim + act_dim).
        """
        seg_len   = self.cfg.segment
        obs_dim   = self.env.observation_space.shape[0]
        act_dim   = self.env.action_space.shape[0]
        total     = n_segs * seg_len
        flat_buf  = np.zeros((total, obs_dim + act_dim), dtype=np.float32)

        obs = self._gauge_obs
        with utils.eval_mode(self.agent):
            for t in range(total):
                action = self.agent.act(obs, sample=False)
                flat_buf[t] = np.concatenate([obs, action])
                next_obs, _, terminated, truncated, _ = self._gauge_eval_env.step(action)
                done = terminated or truncated
                obs  = self._gauge_eval_env.reset() if done else next_obs
        self._gauge_obs = obs
        return flat_buf.reshape(n_segs, seg_len, obs_dim + act_dim)

    # ────────────────────────────────────────────────────────────────────────
    # Evaluation
    # ────────────────────────────────────────────────────────────────────────

    def evaluate(self):
        average_episode_reward      = 0
        average_true_episode_reward = 0
        # [Gauge experiment, Step 3+5] Proxy return: RM-predicted episode return.
        # NOTE: proxy and true returns are on different scales (RM is not normalized)
        # and are NOT directly comparable in magnitude — compare them only as
        # trends over training, not as absolute values.
        average_proxy_episode_reward = 0
        success_rate                 = 0

        for episode in range(self.cfg.num_eval_episodes):
            obs  = self.env.reset()
            self.agent.reset()
            done = False
            episode_reward       = 0
            true_episode_reward  = 0
            proxy_episode_reward = 0
            # collect per-step (obs, action) for batch proxy-reward evaluation
            _traj_sa: list = []
            if self.log_success:
                episode_success = 0

            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                _traj_sa.append(np.concatenate([obs, action]).astype(np.float32))
                obs, reward, terminated, truncated, extra = self.env.step(action)
                done = terminated or truncated
                episode_reward      += reward
                true_episode_reward += reward
                if self.log_success:
                    episode_success = max(episode_success, extra['success'])

            # [Gauge experiment, Step 5] Batch proxy return computation.
            if _traj_sa:
                sa_batch = np.stack(_traj_sa)                # (T, ds+da)
                # r_hat_batch returns (T, 1) or (T,) averaged over ensemble members
                proxy_rewards = self.reward_model.r_hat_batch(sa_batch)
                proxy_episode_reward = float(proxy_rewards.sum())

            average_episode_reward       += episode_reward
            average_true_episode_reward  += true_episode_reward
            average_proxy_episode_reward += proxy_episode_reward
            if self.log_success:
                success_rate += episode_success

        average_episode_reward       /= self.cfg.num_eval_episodes
        average_true_episode_reward  /= self.cfg.num_eval_episodes
        average_proxy_episode_reward /= self.cfg.num_eval_episodes
        self.logger.log('eval/episode_reward',      average_episode_reward,      self.step)
        self.logger.log('eval/true_episode_reward', average_true_episode_reward, self.step)
        self.logger.log('eval/proxy_return',        average_proxy_episode_reward, self.step)
        self.logger.log('eval/return_gap',
                        average_proxy_episode_reward - average_true_episode_reward, self.step)
        if self.log_success:
            success_rate = success_rate / self.cfg.num_eval_episodes * 100.0
            self.logger.log('eval/success_rate',        success_rate, self.step)
            self.logger.log('train/true_episode_success', success_rate, self.step)
        self.logger.dump(self.step)

    # ────────────────────────────────────────────────────────────────────────
    # learn_reward — central dispatch for all tandem conditions
    # ────────────────────────────────────────────────────────────────────────

    def learn_reward(self, first_flag=0):
        mode = self.cfg.tandem_mode
        idx  = self._rm_update_idx

        # ── (A) BASELINE: query normally + log everything ──────────────────
        if mode == 'baseline':
            # Save RM weights + pool IDs before querying (scoring RM for iv-a)
            self.tandem_logger.log_query_event_start(
                idx, self.step,
                self.reward_model.input_episode_ids,
                self.reward_model)

            if first_flag == 1:
                (labeled_queries,
                 sa_t_1, sa_t_2, r_t_1, r_t_2,
                 labels, scores) = self.reward_model.uniform_sampling_with_details()
            else:
                (labeled_queries,
                 sa_t_1, sa_t_2, r_t_1, r_t_2,
                 labels, scores) = self.reward_model.disagreement_sampling_with_details()

            if labeled_queries > 0:
                self.tandem_logger.log_query_event_pairs(
                    idx, sa_t_1, sa_t_2, r_t_1, r_t_2, labels, scores)

        # ── (B) PASSIVE RM (conditions i and iii): inject baseline pairs ────
        elif mode in _PASSIVE_RM_PAIRS:
            event = self.tandem_reader.get_query_event(idx)
            labeled_queries = len(event['labels'])
            if labeled_queries > 0:
                self.reward_model.put_queries(
                    event['sa_t_1'], event['sa_t_2'], event['labels'])

        # ── (C) condition ii: passive policy, active RM ────────────────────
        #   First query uses baseline pairs (RM pool is empty at that point).
        #   Subsequent queries use the tandem's own segment pool.
        elif mode == 'passive_pol_active_rm':
            if first_flag == 1:
                event = self.tandem_reader.get_query_event(idx)
                labeled_queries = len(event['labels'])
                if labeled_queries > 0:
                    self.reward_model.put_queries(
                        event['sa_t_1'], event['sa_t_2'], event['labels'])
            else:
                labeled_queries = self.reward_model.disagreement_sampling()

        # ── (D) condition iv-a: active data dist, passive queries ──────────
        #   Own segment pool; scored by baseline RM checkpoint.
        elif mode == 'active_pol_passive_query':
            if first_flag == 1:
                event = self.tandem_reader.get_query_event(idx)
                labeled_queries = len(event['labels'])
                if labeled_queries > 0:
                    self.reward_model.put_queries(
                        event['sa_t_1'], event['sa_t_2'], event['labels'])
            else:
                self.tandem_reader.load_baseline_rm(idx, self._baseline_rm_helper)
                labeled_queries = self.reward_model.disagreement_sampling_external_scorer(
                    self._baseline_rm_helper)

        # ── (E) condition iv-b: passive data dist, active queries ──────────
        #   Baseline segment pool; scored by tandem RM (self).
        elif mode == 'active_pol_passive_dist':
            if first_flag == 1:
                event = self.tandem_reader.get_query_event(idx)
                labeled_queries = len(event['labels'])
                if labeled_queries > 0:
                    self.reward_model.put_queries(
                        event['sa_t_1'], event['sa_t_2'], event['labels'])
            else:
                event = self.tandem_reader.get_query_event(idx)
                bl_inputs, bl_targets = self.tandem_reader.reconstruct_episode_pool(
                    event['pool_episode_ids'])
                labeled_queries = self.reward_model.disagreement_sampling_external_pool(
                    bl_inputs, bl_targets)

        else:
            raise ValueError(f"Unknown tandem_mode: {mode!r}")

        # [Gauge experiment, Step 3] Collect on-policy segments for diagnostics.
        # Done in baseline mode only, before RM training so the collection policy
        # matches the current policy state (RM training doesn't change the policy).
        _D_current = None
        if mode == 'baseline' and self._gauge_eval_env is not None:
            if first_flag == 1 and not self._ref_segments_collected:
                # Freeze reference set from the initial (unsup-pretrained) policy.
                _D_ref = self._collect_on_policy_segments(256)
                self.reward_model.set_ref_segments(_D_ref)
                self._ref_segments_collected = True
            _D_current = self._collect_on_policy_segments(64)

        self.total_feedback   += self.reward_model.mb_size
        self.labeled_feedback += labeled_queries

        if self.cfg.rm_reset and first_flag != 1:
            self.reward_model.reset_ensemble()

        total_acc = 0.0
        if self.labeled_feedback > 0:
            for epoch in range(self.cfg.reward_update):
                if self.cfg.label_margin > 0 or self.cfg.teacher_eps_equal > 0:
                    train_acc = self.reward_model.train_soft_reward()
                else:
                    train_acc = self.reward_model.train_reward()
                total_acc = np.mean(train_acc)
                if total_acc > 0.97:
                    break

        print(f"Reward function updated (event {idx})  ACC: {total_acc:.4f}")
        self.logger.log('train/reward_model_acc', total_acc, self.step)

        # [Gauge experiment, Step 1] Update zero-mean offsets after full RM training.
        # Must happen before get_gauge_diagnostics so C2 is measured on corrected outputs.
        self.reward_model._update_zero_mean_offsets()

        # [Gauge experiment, Step 2] Log gauge diagnostics after RM training.
        # Includes rm/bt_loss_final (C3), rm/param_norm (C1), rm/gauge_gap (C2),
        # and rm/mean/std on ref and on-policy sets.
        if _D_current is not None:
            gauge_metrics = self.reward_model.get_gauge_diagnostics(_D_current)
            for key, value in gauge_metrics.items():
                self.logger.log(key, value, self.step)
        else:
            # Always log bt_loss_final even when diagnostic env is unavailable.
            self.logger.log('train/rm_bt_loss_final', self.reward_model.last_bt_loss, self.step)

        if mode == 'baseline':
            self.tandem_logger.flush()

        self._rm_update_idx += 1
        # [Gauge experiment, C7] Log cumulative RM update count for sanity check.
        self.logger.log('train/rm_update_count', self._rm_update_idx, self.step)

    # ────────────────────────────────────────────────────────────────────────
    # Training loop helpers
    # ────────────────────────────────────────────────────────────────────────

    def _unsup_pretrain_loop(self):
        """Run seed phase + unsupervised exploration up to (but not including) warmup_end.

        On return: self.step == warmup_end, self._loop_obs and self._loop_done hold
        the observation and done flag that _policy_training_loop should start from.
        self._avg_train_true_return is populated with recent true episode returns.
        """
        cfg  = self.cfg
        mode = cfg.tandem_mode
        warmup_end = cfg.num_seed_steps + cfg.num_unsup_steps

        episode             = 0
        episode_reward      = 0
        true_episode_reward = 0
        done                = True
        if self.log_success:
            episode_success = 0
        start_time = time.time()

        tandem_obs  = None
        tandem_done = True
        if self._tandem_env is not None:
            tandem_obs  = self._tandem_env.reset()
            tandem_done = False

        obs = None  # set by first episode-boundary reset below

        while self.step < warmup_end:

            # ── Episode boundary ─────────────────────────────────────────
            if done:
                if self.step > 0:
                    self.logger.log('train/duration', time.time() - start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(self.step, save=(self.step > cfg.num_seed_steps))

                if self.step > 0 and self.step % cfg.eval_frequency == 0:
                    self.logger.log('eval/episode', episode, self.step)
                    self.evaluate()

                self.logger.log('train/episode_reward',      episode_reward,      self.step)
                self.logger.log('train/true_episode_reward', true_episode_reward, self.step)
                if self.log_success:
                    self.logger.log('train/episode_success',      episode_success, self.step)
                    self.logger.log('train/true_episode_success', episode_success, self.step)

                if mode not in _PASSIVE_POLICY:
                    obs = self.env.reset()

                done            = False
                episode_reward  = 0
                self._avg_train_true_return.append(true_episode_reward)
                true_episode_reward = 0
                if self.log_success:
                    episode_success = 0
                episode += 1
                self.logger.log('train/episode', episode, self.step)

            # ── Transition ───────────────────────────────────────────────
            if mode in _PASSIVE_POLICY:
                (obs, action, env_reward,
                 next_obs, _done_f, done_no_max) = self.tandem_reader.get_transition(self.step)
                done        = bool(_done_f)
                done_no_max = float(done_no_max)
            else:
                if self.step < cfg.num_seed_steps:
                    action = self.env.action_space.sample()
                else:
                    with utils.eval_mode(self.agent):
                        action = self.agent.act(obs, sample=True)
                next_obs, env_reward, terminated, truncated, extra = self.env.step(action)
                done        = terminated or truncated
                done_no_max = 0.0 if (truncated and not terminated) else float(done)
                env_reward  = float(env_reward)

            # ── RM segment pool ───────────────────────────────────────────
            if mode in _ACTIVE_RM_DATA:
                if mode == 'passive_pol_active_rm':
                    if not tandem_done:
                        with utils.eval_mode(self.agent):
                            t_action = self.agent.act(tandem_obs, sample=True)
                    else:
                        t_action = self._tandem_env.action_space.sample()
                    t_next_obs, t_reward, t_term, t_trunc, _ = self._tandem_env.step(t_action)
                    t_done = t_term or t_trunc
                    self.reward_model.add_data(tandem_obs, t_action, t_reward, float(t_done))
                    tandem_obs  = self._tandem_env.reset() if t_done else t_next_obs
                    tandem_done = False
                else:
                    self.reward_model.add_data(obs, action, env_reward, float(done))

            # ── Reward hat + replay buffer ────────────────────────────────
            reward_hat = self.reward_model.r_hat(
                np.concatenate([obs, action], axis=-1))
            episode_reward      += reward_hat
            true_episode_reward += env_reward
            if self.log_success and mode not in _PASSIVE_POLICY:
                episode_success = max(episode_success, extra['success'])
            self.replay_buffer.add(obs, action, reward_hat,
                                   next_obs, float(done), done_no_max)

            # ── Baseline logging ──────────────────────────────────────────
            if mode == 'baseline':
                self.tandem_logger.log_transition(
                    self.step, obs, action, env_reward,
                    next_obs, float(done), done_no_max)
                if done:
                    ep_len = self.step - self._ep_start_step + 1
                    self.tandem_logger.log_episode_end(
                        self._ep_id, self._ep_start_step, ep_len)
                    self._ep_id         += 1
                    self._ep_start_step  = self.step + 1

            # ── Unsupervised update (state-entropy bonus) ─────────────────
            if self.step > cfg.num_seed_steps:
                self.agent.update_state_ent(
                    self.replay_buffer, self.logger, self.step,
                    gradient_update=1, K=cfg.topK)

            # ── Sanity checks ─────────────────────────────────────────────
            if cfg.sanity_mode and mode != 'baseline':
                sanity_checks.run_checks(
                    mode, self.tandem_reader, self.reward_model,
                    self.replay_buffer, self.step, self._rm_update_idx,
                    check_interval=cfg.sanity_check_interval)

            # ── Advance ───────────────────────────────────────────────────
            if mode not in _PASSIVE_POLICY:
                obs = next_obs
            self.step += 1

        # Hand off loop state to the next phase
        self._loop_obs  = obs
        self._loop_done = done

    def _policy_training_loop(self, diag_callback=None, diag_freq=None):
        """Run RL phase from self.step to cfg.num_train_steps.

        For the iterative case, call with self.step == warmup_end; this method
        handles the warmup_end RM event internally.  For the one-shot case,
        call with self.step == warmup_end + 1 (the one-shot run() already
        processed the warmup_end step and fired its single RM event).

        diag_callback(step) is called every diag_freq steps when provided.
        """
        cfg  = self.cfg
        mode = cfg.tandem_mode
        warmup_end = cfg.num_seed_steps + cfg.num_unsup_steps

        obs  = self._loop_obs
        done = self._loop_done

        episode             = 0
        episode_reward      = 0
        true_episode_reward = 0
        if self.log_success:
            episode_success = 0
        start_time  = time.time()
        interact_count = 0

        tandem_obs  = None
        tandem_done = True
        if self._tandem_env is not None:
            # Re-initialise tandem env state for this phase
            tandem_obs  = self._tandem_env.reset()
            tandem_done = False

        while self.step < cfg.num_train_steps:

            # ── Episode boundary ─────────────────────────────────────────
            if done:
                if self.step > 0:
                    self.logger.log('train/duration', time.time() - start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(self.step, save=True)

                if self.step > 0 and self.step % cfg.eval_frequency == 0:
                    self.logger.log('eval/episode', episode, self.step)
                    self.evaluate()

                self.logger.log('train/episode_reward',      episode_reward,      self.step)
                self.logger.log('train/true_episode_reward', true_episode_reward, self.step)
                if self.log_success:
                    self.logger.log('train/episode_success',      episode_success, self.step)
                    self.logger.log('train/true_episode_success', episode_success, self.step)

                if mode not in _PASSIVE_POLICY:
                    obs = self.env.reset()

                done            = False
                episode_reward  = 0
                self._avg_train_true_return.append(true_episode_reward)
                true_episode_reward = 0
                if self.log_success:
                    episode_success = 0
                episode += 1
                self.logger.log('train/episode', episode, self.step)

            # ── Transition ───────────────────────────────────────────────
            if mode in _PASSIVE_POLICY:
                (obs, action, env_reward,
                 next_obs, _done_f, done_no_max) = self.tandem_reader.get_transition(self.step)
                done        = bool(_done_f)
                done_no_max = float(done_no_max)
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)
                next_obs, env_reward, terminated, truncated, extra = self.env.step(action)
                done        = terminated or truncated
                done_no_max = 0.0 if (truncated and not terminated) else float(done)
                env_reward  = float(env_reward)

            # ── RM segment pool ───────────────────────────────────────────
            if mode in _ACTIVE_RM_DATA:
                if mode == 'passive_pol_active_rm':
                    if not tandem_done:
                        with utils.eval_mode(self.agent):
                            t_action = self.agent.act(tandem_obs, sample=True)
                    else:
                        t_action = self._tandem_env.action_space.sample()
                    t_next_obs, t_reward, t_term, t_trunc, _ = self._tandem_env.step(t_action)
                    t_done = t_term or t_trunc
                    self.reward_model.add_data(tandem_obs, t_action, t_reward, float(t_done))
                    tandem_obs  = self._tandem_env.reset() if t_done else t_next_obs
                    tandem_done = False
                else:
                    self.reward_model.add_data(obs, action, env_reward, float(done))

            # ── Reward hat + replay buffer ────────────────────────────────
            reward_hat = self.reward_model.r_hat(
                np.concatenate([obs, action], axis=-1))
            episode_reward      += reward_hat
            true_episode_reward += env_reward
            if self.log_success and mode not in _PASSIVE_POLICY:
                episode_success = max(episode_success, extra['success'])
            self.replay_buffer.add(obs, action, reward_hat,
                                   next_obs, float(done), done_no_max)

            # ── Baseline logging ──────────────────────────────────────────
            if mode == 'baseline':
                self.tandem_logger.log_transition(
                    self.step, obs, action, env_reward,
                    next_obs, float(done), done_no_max)
                if done:
                    ep_len = self.step - self._ep_start_step + 1
                    self.tandem_logger.log_episode_end(
                        self._ep_id, self._ep_start_step, ep_len)
                    self._ep_id         += 1
                    self._ep_start_step  = self.step + 1

            # ── Optional gauge-diagnostic callback ────────────────────────
            if (diag_callback is not None and diag_freq is not None
                    and self.step > 0 and self.step % diag_freq == 0):
                diag_callback(self.step)

            # ── Training updates ──────────────────────────────────────────
            if self.step == warmup_end:
                frac = self._compute_frac()
                self.reward_model.change_batch(frac)
                if mode == 'baseline':
                    self.tandem_logger.log_schedule(self.step, frac, self.reward_model.mb_size)

                new_margin = (np.mean(self._avg_train_true_return)
                              * (cfg.segment / self.env._max_episode_steps))
                self.reward_model.set_teacher_thres_skip(new_margin)
                self.reward_model.set_teacher_thres_equal(new_margin)

                self.reward_model.env_step = self.step
                self.learn_reward(first_flag=1)
                self.reward_model.pre_relabel_logging(self.step)
                self.replay_buffer.relabel_with_predictor(self.reward_model)
                self.agent.reset_critic()
                self.agent.update_after_reset(
                    self.replay_buffer, self.logger, self.step,
                    gradient_update=cfg.reset_update,
                    policy_update=True)
                interact_count = 0

            elif self.step > warmup_end:
                if self.total_feedback < cfg.max_feedback:
                    if interact_count == cfg.num_interact:
                        frac = self._compute_frac()
                        self.reward_model.change_batch(frac)
                        if mode == 'baseline':
                            self.tandem_logger.log_schedule(
                                self.step, frac, self.reward_model.mb_size)

                        new_margin = (np.mean(self._avg_train_true_return)
                                      * (cfg.segment / self.env._max_episode_steps))
                        self.reward_model.set_teacher_thres_skip(
                            new_margin * cfg.teacher_eps_skip)
                        self.reward_model.set_teacher_thres_equal(
                            new_margin * cfg.teacher_eps_equal)

                        if (self.reward_model.mb_size + self.total_feedback
                                > cfg.max_feedback):
                            self.reward_model.set_batch(
                                cfg.max_feedback - self.total_feedback)

                        self.reward_model.env_step = self.step
                        self.learn_reward()
                        self.reward_model.pre_relabel_logging(self.step)
                        self.replay_buffer.relabel_with_predictor(self.reward_model)
                        interact_count = 0

                self.agent.update(self.replay_buffer, self.logger, self.step, 1)

            # ── Sanity checks ─────────────────────────────────────────────
            if cfg.sanity_mode and mode != 'baseline':
                sanity_checks.run_checks(
                    mode, self.tandem_reader, self.reward_model,
                    self.replay_buffer, self.step, self._rm_update_idx,
                    check_interval=cfg.sanity_check_interval)

            # ── Advance ───────────────────────────────────────────────────
            if mode not in _PASSIVE_POLICY:
                obs = next_obs
            self.step      += 1
            interact_count += 1

        # ── End of training ───────────────────────────────────────────────
        self.agent.save(self.work_dir, self.step)
        self.reward_model.save(self.work_dir, self.step)
        if self.tandem_logger is not None:
            self.tandem_logger.close()
        if self.tandem_reader is not None:
            self.tandem_reader.close()

    # ────────────────────────────────────────────────────────────────────────
    # Main training entry point
    # ────────────────────────────────────────────────────────────────────────

    def run(self):
        """Iterative PEBBLE training: unsup pretrain → periodic RM updates → SAC."""
        self._unsup_pretrain_loop()       # steps 0 .. warmup_end-1
        self._policy_training_loop()      # steps warmup_end .. num_train_steps-1

    # ────────────────────────────────────────────────────────────────────────
    # Helpers
    # ────────────────────────────────────────────────────────────────────────

    def _compute_frac(self):
        cfg = self.cfg
        if cfg.reward_schedule == 1:
            frac = (cfg.num_train_steps - self.step) / cfg.num_train_steps
            return max(frac, 0.01)
        elif cfg.reward_schedule == 2:
            return cfg.num_train_steps / (cfg.num_train_steps - self.step + 1)
        else:
            return 1.0


@hydra.main(config_path='config/train_PEBBLE.yaml', strict=True)
def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()


if __name__ == '__main__':
    main()
