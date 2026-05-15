import numpy as np
import torch
import torch.nn.functional as F
import utils

from torch import nn

class DoubleQCritic(nn.Module):
    """Critic network, employes double Q-learning."""
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth,
                 dormant_log_period=5000, dormant_threshold=0.1):
        super().__init__()

        self.Q1 = utils.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)
        self.Q2 = utils.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)

        self.outputs = dict()
        self.apply(utils.weight_init)

        self.dormant_log_period = dormant_log_period
        self.dormant_threshold = dormant_threshold
        self._grad_steps = 0
        self._prev_dormant_set_q1 = None

    @staticmethod
    def _get_penultimate_activations(net, x):
        last_hidden_act = None
        for layer in net:
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                last_hidden_act = x
        return last_hidden_act

    def log_dormant_stats(self, obs, action, env_step=0):
        self._grad_steps += 1
        if self._grad_steps % self.dormant_log_period != 0:
            return
        try:
            import wandb
        except ImportError:
            return
        if wandb.run is None:
            return

        with torch.no_grad():
            obs_action = torch.cat([obs, action], dim=-1)
            n = obs_action.size(0)
            if n > 512:
                idx = torch.randperm(n, device=obs_action.device)[:512]
                sample = obs_action[idx]
            else:
                sample = obs_action
            acts = self._get_penultimate_activations(self.Q1, sample)
        if acts is None:
            return

        mean_abs_acts = acts.abs().mean(dim=0)
        avg_mean_act = mean_abs_acts.mean()
        if avg_mean_act == 0:
            dormant_set = set(range(acts.shape[1]))
        else:
            dormant_set = set(
                (mean_abs_acts < self.dormant_threshold * avg_mean_act)
                .nonzero(as_tuple=False).squeeze(1).tolist()
            )

        n_neurons = acts.shape[1]
        dormant_rate = len(dormant_set) / n_neurons

        if self._prev_dormant_set_q1 is None or len(dormant_set) == 0:
            overlap_rate = 0.0
        else:
            overlap_rate = len(dormant_set & self._prev_dormant_set_q1) / len(dormant_set)
        self._prev_dormant_set_q1 = dormant_set

        n_batch = acts.size(0)
        F = acts / (n_batch ** 0.5)
        S = torch.linalg.svdvals(F)
        feature_rank = (S > 0.01).sum().item()

        wandb.log({
            'critic/q1_dormant_rate': dormant_rate,
            'critic/q1_dormant_overlap_rate': overlap_rate,
            'critic/q1_feature_rank': feature_rank,
        }, step=env_step)

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=-1)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_critic/{k}_hist', v, step)

        assert len(self.Q1) == len(self.Q2)
        for i, (m1, m2) in enumerate(zip(self.Q1, self.Q2)):
            assert type(m1) == type(m2)
            if type(m1) is nn.Linear:
                logger.log_param(f'train_critic/q1_fc{i}', m1, step)
                logger.log_param(f'train_critic/q2_fc{i}', m2, step)