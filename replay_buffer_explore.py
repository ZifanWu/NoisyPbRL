import numpy as np
import torch
import utils

class ReplayBuffer(object):
    """Buffer to store environment transitions. Stores separate extrinsic and
    intrinsic rewards for RUNE-style exploration bonuses."""
    def __init__(self, obs_shape, action_shape, capacity, device, window=1):
        self.capacity = capacity
        self.device = device

        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.extrinsic_rewards = np.empty((capacity, 1), dtype=np.float32)
        self.intrinsic_rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones_no_max = np.empty((capacity, 1), dtype=np.float32)
        self.window = window

        self.idx = 0
        self.last_save = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, ext_reward, int_reward, next_obs, done, done_no_max):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.extrinsic_rewards[self.idx], ext_reward)
        np.copyto(self.intrinsic_rewards[self.idx], int_reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.not_dones_no_max[self.idx], not done_no_max)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def relabel_with_predictor(self, predictor):
        batch_size = 200
        total_iter = int(self.idx/batch_size)

        if self.idx > batch_size*total_iter:
            total_iter += 1

        for index in range(total_iter):
            last_index = (index+1)*batch_size
            if (index+1)*batch_size > self.idx:
                last_index = self.idx

            obses = self.obses[index*batch_size:last_index]
            actions = self.actions[index*batch_size:last_index]
            inputs = np.concatenate([obses, actions], axis=-1)

            pred_reward, explore_bonus = predictor.r_hat_std_batch(inputs)
            self.extrinsic_rewards[index*batch_size:last_index] = pred_reward
            self.intrinsic_rewards[index*batch_size:last_index] = explore_bonus

    def sample(self, batch_size):
        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=batch_size)

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        ext_rewards = torch.as_tensor(self.extrinsic_rewards[idxs], device=self.device)
        int_rewards = torch.as_tensor(self.intrinsic_rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(self.next_obses[idxs], device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs], device=self.device)

        return obses, actions, ext_rewards, int_rewards, next_obses, not_dones, not_dones_no_max

    def sample_state_ent(self, batch_size):
        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=batch_size)

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        ext_rewards = torch.as_tensor(self.extrinsic_rewards[idxs], device=self.device)
        int_rewards = torch.as_tensor(self.intrinsic_rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(self.next_obses[idxs], device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs], device=self.device)

        if self.full:
            full_obs = self.obses
        else:
            full_obs = self.obses[: self.idx]
        full_obs = torch.as_tensor(full_obs, device=self.device)

        return obses, full_obs, actions, ext_rewards, int_rewards, next_obses, not_dones, not_dones_no_max
