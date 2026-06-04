#!/usr/bin/env python3
"""
train_static_sac.py — three-phase training pipeline.

Phase 1  collect_or_load_trajectories
         Run SAC with true rewards for num_train_steps steps.
         Save every completed episode as (s,a) / r arrays (same format as
         reward_model.inputs / targets, but without the 100-episode cap).
         Cache is keyed by (env, seed) only — teacher params do not affect
         data collection, so different teacher configs reuse the same cache.

Phase 2  train_rm_offline
         Load the trajectory pool into the reward model, generate
         max_feedback preference pairs with uniform sampling (teacher type
         is fully determined by teacher_beta / teacher_gamma /
         teacher_eps_* in the config, exactly as in train_PEBBLE.py), then
         train the RM once.  Stopping criterion: up to reward_update epochs,
         early-stop when acc > 0.97.  Metrics (dormant rate, feature rank,
         BT weights) are logged every rm_log_interval gradient steps.

Phase 3  train_policy_online
         Reinitialise a fresh SAC agent and an empty replay buffer.
         Interact with the environment using the fixed RM for rewards —
         no further RM updates.

Teacher type is specified exactly as in train_PEBBLE.py, via hydra overrides:
    teacher_beta=1 teacher_gamma=1 teacher_eps_mistake=0.3   # noisy / BT
    teacher_beta=-1 teacher_gamma=0.9                         # myopic
    teacher_beta=-1 teacher_eps_skip=0.1                      # skip
"""
import numpy as np
import torch
import os
import time
import pickle as pkl
from collections import Counter, deque

import hydra

from logger import Logger
from replay_buffer import ReplayBuffer
from reward_model import RewardModel
import utils


class Workspace:
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')
        self.cfg = cfg

        self.logger = Logger(
            self.work_dir,
            save_tb=cfg.log_save_tb,
            use_wandb=cfg.use_wandb,
            log_frequency=cfg.log_frequency,
            agent=cfg.agent.name,
        )

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.log_success = False

        if 'metaworld' in cfg.env:
            self.env = utils.make_metaworld_env(cfg)
            self.log_success = True
        else:
            self.env = utils.make_env(cfg)

        cfg.agent.params.obs_dim = self.env.observation_space.shape[0]
        cfg.agent.params.action_dim = self.env.action_space.shape[0]
        cfg.agent.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max()),
        ]
        self.agent = hydra.utils.instantiate(cfg.agent)

        # Replay buffer used exclusively in Phase 3 (policy online training).
        self.policy_replay_buffer = ReplayBuffer(
            self.env.observation_space.shape,
            self.env.action_space.shape,
            int(cfg.replay_buffer_capacity),
            self.device,
        )

        # dormant_log_period / bt_log_period are wired to rm_log_interval so
        # that train_reward() fires all metric callbacks at the same cadence.
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
            dormant_log_period=cfg.rm_log_interval,
            dormant_threshold=cfg.dormant_threshold,
            use_wandb=cfg.use_wandb,
            bt_log_period=cfg.rm_log_interval,
            feed_type=0,  # always uniform for offline RM training
            capacity=cfg.max_feedback * cfg.large_batch,
        )

    # ------------------------------------------------------------------
    # Phase 1 helpers
    # ------------------------------------------------------------------

    def _traj_cache_path(self):
        orig_dir = hydra.utils.get_original_cwd()
        return os.path.join(
            orig_dir, self.cfg.traj_cache_dir,
            self.cfg.env, f'seed{self.cfg.seed}',
            'trajectories.pkl',
        )

    def collect_or_load_trajectories(self):
        """Return {'inputs': [...], 'targets': [...]} from cache or by running SAC."""
        path = self._traj_cache_path()
        if os.path.exists(path):
            print(f'[Phase 1] Loading cached trajectories from {path}')
            with open(path, 'rb') as f:
                data = pkl.load(f)
            print(f'[Phase 1] Loaded {len(data["inputs"])} episodes')
            return data

        print('[Phase 1] No cache found — running SAC to collect trajectories')
        data = self._run_sac_collect()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pkl.dump(data, f)
        print(f'[Phase 1] Saved {len(data["inputs"])} episodes to {path}')
        return data

    def _run_sac_collect(self):
        """
        Run SAC with true rewards for num_train_steps steps.

        Returns
        -------
        dict
            'inputs'  : list of float32 arrays shape (T, ds+da), one per episode
            'targets' : list of float32 arrays shape (T, 1),     one per episode
        Only episodes with at least size_segment steps are kept.
        """
        cfg = self.cfg

        # Dedicated replay buffer for the collection agent (true rewards).
        sac_buffer = ReplayBuffer(
            self.env.observation_space.shape,
            self.env.action_space.shape,
            int(cfg.replay_buffer_capacity),
            self.device,
        )

        inputs, targets = [], []
        cur_sa: list = []
        cur_r: list = []

        episode, episode_reward, done = 0, 0.0, True
        episode_step = 0
        episode_success = 0.0
        obs = None
        start_time = time.time()

        for step in range(int(cfg.num_train_steps)):
            if done:
                # Persist the just-finished episode.
                if step > 0 and len(cur_sa) >= cfg.segment:
                    inputs.append(np.array(cur_sa, dtype=np.float32))
                    targets.append(
                        np.array(cur_r, dtype=np.float32).reshape(-1, 1)
                    )

                if step > 0 and episode % 10 == 0:
                    print(
                        f'[Phase 1] step {step} | episode {episode} '
                        f'| reward {episode_reward:.2f}'
                        + (f' | success {episode_success:.2f}'
                           if self.log_success else '')
                    )
                    start_time = time.time()

                obs = self.env.reset()
                self.agent.reset()
                done = False
                episode_reward = 0.0
                episode_success = 0.0
                cur_sa, cur_r = [], []
                episode_step = 0
                episode += 1

            # Action selection
            if step < cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            # Agent updates — mirrors train_SAC.py schedule
            if (step == cfg.num_seed_steps + cfg.num_unsup_steps
                    and cfg.num_unsup_steps > 0):
                self.agent.reset_critic()
                self.agent.update_after_reset(
                    sac_buffer, self.logger, step,
                    gradient_update=cfg.reset_update,
                    policy_update=True,
                )
            elif step > cfg.num_seed_steps + cfg.num_unsup_steps:
                self.agent.update(sac_buffer, self.logger, step)
            elif step > cfg.num_seed_steps:
                self.agent.update_state_ent(
                    sac_buffer, self.logger, step,
                    gradient_update=1, K=cfg.topK,
                )

            next_obs, reward, terminated, truncated, extra = \
                self.env.step(action)
            done = terminated or truncated
            done_f = float(done)
            done_no_max = 0.0 if (truncated and not terminated) else done_f

            episode_reward += reward
            if self.log_success:
                episode_success = max(
                    episode_success, extra.get('success', 0.0)
                )

            cur_sa.append(
                np.concatenate([obs, action]).astype(np.float32)
            )
            cur_r.append(float(reward))

            sac_buffer.add(obs, action, reward, next_obs, done_f, done_no_max)

            obs = next_obs
            episode_step += 1

        # Persist whatever remains of the last (possibly incomplete) episode.
        if len(cur_sa) >= cfg.segment:
            inputs.append(np.array(cur_sa, dtype=np.float32))
            targets.append(
                np.array(cur_r, dtype=np.float32).reshape(-1, 1)
            )

        return {'inputs': inputs, 'targets': targets}

    # ------------------------------------------------------------------
    # Phase 2
    # ------------------------------------------------------------------

    def train_rm_offline(self, inputs, targets):
        """
        One-shot offline RM training.

        1. Filter to a uniform episode length (mode) — required by
           reward_model.get_queries() which calls np.array(self.inputs).
        2. Generate max_feedback preference pairs via uniform sampling.
           Teacher behaviour (rational / noisy / myopic / skip) is
           determined by the teacher_* config values, identically to
           train_PEBBLE.py.
        3. Train for up to reward_update epochs; stop early if acc > 0.97.
        4. Log dormant rate / feature rank / BT weights every
           rm_log_interval gradient steps (and once at the end).
        """
        cfg = self.cfg
        print(f'[Phase 2] Offline RM training — {len(inputs)} episodes available')

        # get_queries() requires all episodes to have the same length.
        ep_lens = [len(ep) for ep in inputs]
        mode_len = Counter(ep_lens).most_common(1)[0][0]
        valid = [
            (inp, tgt)
            for inp, tgt in zip(inputs, targets)
            if len(inp) == mode_len
        ]
        if not valid:
            raise RuntimeError(
                f'No episodes of mode length {mode_len} found. '
                'Cannot build preference pairs.'
            )
        print(f'[Phase 2] Using {len(valid)} episodes of length {mode_len}')

        self.reward_model.inputs = [v[0] for v in valid]
        self.reward_model.targets = [v[1] for v in valid]

        # Same teacher thresholds as PEBBLE's first learn_reward call.
        self.reward_model.set_teacher_thres_skip(0)
        self.reward_model.set_teacher_thres_equal(0)

        # Generate all max_feedback preference pairs in one shot.
        self.reward_model.set_batch(cfg.max_feedback)
        labeled = self.reward_model.uniform_sampling()
        print(f'[Phase 2] Generated {labeled} labeled preference pairs')
        if labeled == 0:
            raise RuntimeError(
                'No preference pairs were generated. '
                'Check episode length vs. segment size, or teacher skip threshold.'
            )

        # Initialise wandb here (Phase 2 start) so Phase 1 produces no run.
        # A single run covers both Phase 2 (RM training) and Phase 3 (policy).
        if cfg.use_wandb:
            import wandb
            import socket
            wandb.init(
                project='roo_error_dissection',
                name=f'{cfg.env}__static_sac__seed{cfg.seed}',
                config=dict(cfg),
                notes=socket.gethostname(),
            )

        # Redirect Phase 2 RM metric logging to a custom wandb x-axis
        # (rm_grad_step) so that RM grad steps never advance the global wandb
        # step counter, which Phase 3 uses for env steps starting from 0.
        _orig_wandb_log = None
        if cfg.use_wandb:
            wandb.define_metric('reward_model/*', step_metric='rm_grad_step')

            _orig_wandb_log = wandb.log

            def _rm_phase_log(data, step=None, **kwargs):
                if step is not None:
                    data = {**data, 'rm_grad_step': step}
                _orig_wandb_log(data, **kwargs)

            wandb.log = _rm_phase_log

        def _log_rm_snapshot(grad_step):
            self.reward_model.env_step = grad_step
            self.reward_model.log_buffer_bt_metrics(grad_step)
            self.reward_model.log_dormant_neurons()

        # Pre-training snapshot.
        if cfg.use_wandb:
            _log_rm_snapshot(0)

        # Train RM — same criterion as each PEBBLE learn_reward round.
        total_acc = 0.0
        try:
            for epoch in range(cfg.reward_update):
                gs = self.reward_model.reward_grad_steps
                # log_buffer_bt_metrics is not called inside train_reward();
                # call it manually at the same cadence as the other metrics.
                if cfg.use_wandb and gs > 0 and gs % cfg.rm_log_interval == 0:
                    _log_rm_snapshot(gs)

                if cfg.label_margin > 0 or cfg.teacher_eps_equal > 0:
                    train_acc = self.reward_model.train_soft_reward()
                else:
                    train_acc = self.reward_model.train_reward()

                total_acc = np.mean(train_acc)
                if total_acc > 0.97:
                    print(
                        f'[Phase 2] RM converged at epoch {epoch}, '
                        f'acc={total_acc:.4f}'
                    )
                    break

            # Final snapshot after training completes.
            if cfg.use_wandb:
                _log_rm_snapshot(self.reward_model.reward_grad_steps)

        finally:
            # Always restore wandb.log so Phase 3 uses normal global step.
            if _orig_wandb_log is not None:
                import wandb
                wandb.log = _orig_wandb_log

        self.logger.log('train/reward_model_acc', total_acc, 0)
        self.logger.dump(0)
        print(
            f'[Phase 2] Done. acc={total_acc:.4f}, '
            f'grad_steps={self.reward_model.reward_grad_steps}'
        )

    # ------------------------------------------------------------------
    # Phase 3
    # ------------------------------------------------------------------

    def _evaluate(self, step):
        avg_reward, success_rate = 0.0, 0.0
        for _ in range(self.cfg.num_eval_episodes):
            obs = self.env.reset()
            self.agent.reset()
            done = False
            ep_reward = 0.0
            ep_success = 0.0
            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                obs, reward, terminated, truncated, extra = \
                    self.env.step(action)
                done = terminated or truncated
                ep_reward += reward
                if self.log_success:
                    ep_success = max(ep_success, extra.get('success', 0.0))
            avg_reward += ep_reward
            success_rate += ep_success

        avg_reward /= self.cfg.num_eval_episodes
        self.logger.log('eval/episode_reward', avg_reward, step)
        if self.log_success:
            success_rate = success_rate / self.cfg.num_eval_episodes * 100.0
            self.logger.log('eval/success_rate', success_rate, step)
        self.logger.dump(step)

    def train_policy_online(self):
        """
        Online SAC training driven by the fixed RM reward.
        Uses a freshly initialised agent and an empty replay buffer.
        """
        cfg = self.cfg
        print('[Phase 3] Online policy training with fixed RM')

        # Reinitialise: Phase 3 must start from random weights, not from the
        # true-reward-trained agent produced in Phase 1.
        self.agent = hydra.utils.instantiate(cfg.agent)

        episode, episode_reward, done = 0, 0.0, True
        true_episode_reward = 0.0
        episode_success = 0.0
        episode_step = 0
        avg_train_true_return = deque([], maxlen=10)
        obs = None
        start_time = time.time()

        for step in range(int(cfg.num_train_steps)):
            if done:
                if step > 0:
                    self.logger.log('train/duration',
                                    time.time() - start_time, step)
                    start_time = time.time()
                    self.logger.dump(
                        step, save=(step > cfg.num_seed_steps)
                    )

                if step > 0 and step % cfg.eval_frequency == 0:
                    self.logger.log('eval/episode', episode, step)
                    self._evaluate(step)

                self.logger.log('train/episode_reward',
                                episode_reward, step)
                self.logger.log('train/true_episode_reward',
                                true_episode_reward, step)
                if self.log_success:
                    self.logger.log('train/episode_success',
                                    episode_success, step)

                avg_train_true_return.append(true_episode_reward)

                obs = self.env.reset()
                self.agent.reset()
                done = False
                episode_reward = 0.0
                true_episode_reward = 0.0
                episode_success = 0.0
                episode_step = 0
                episode += 1
                self.logger.log('train/episode', episode, step)

            # Action selection
            if step < cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            # SAC update — no unsupervised phase, no RM updates
            if step >= cfg.num_seed_steps:
                self.agent.update(
                    self.policy_replay_buffer, self.logger, step
                )

            next_obs, reward, terminated, truncated, extra = \
                self.env.step(action)
            reward_hat = self.reward_model.r_hat(
                np.concatenate([obs, action], axis=-1)
            )

            done = terminated or truncated
            done_f = float(done)
            done_no_max = 0.0 if (truncated and not terminated) else done_f

            episode_reward += reward_hat
            true_episode_reward += reward
            if self.log_success:
                episode_success = max(
                    episode_success, extra.get('success', 0.0)
                )

            self.policy_replay_buffer.add(
                obs, action, reward_hat, next_obs, done_f, done_no_max
            )

            obs = next_obs
            episode_step += 1

        self.agent.save(self.work_dir, int(cfg.num_train_steps))
        self.reward_model.save(self.work_dir, int(cfg.num_train_steps))

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self):
        # Phase 1
        data = self.collect_or_load_trajectories()

        # Phase 2
        self.train_rm_offline(data['inputs'], data['targets'])

        # Free trajectory pool before policy training to reduce peak memory.
        del data

        # Phase 3
        self.train_policy_online()


@hydra.main(config_path='config/train_static_sac.yaml', strict=True)
def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()


if __name__ == '__main__':
    main()
