import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time

from scipy.stats import norm
from reward_model import KCenterGreedy, compute_smallest_dist

device = 'cuda'


# ---------------------------------------------------------------------------
# Transformer building blocks (GPT-2 style, PyTorch)
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, embd_dim, n_head, dropout, max_seq_len):
        super().__init__()
        assert embd_dim % n_head == 0
        self.n_head = n_head
        self.head_dim = embd_dim // n_head
        self.embd_dim = embd_dim
        self.qkv = nn.Linear(embd_dim, 3 * embd_dim)
        self.proj = nn.Linear(embd_dim, embd_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer('mask', mask.view(1, 1, max_seq_len, max_seq_len))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(self.embd_dim, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class GPT2Block(nn.Module):
    def __init__(self, embd_dim, n_head, dropout, max_seq_len):
        super().__init__()
        self.ln1 = nn.LayerNorm(embd_dim)
        self.attn = CausalSelfAttention(embd_dim, n_head, dropout, max_seq_len)
        self.ln2 = nn.LayerNorm(embd_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embd_dim, 4 * embd_dim),
            nn.GELU(),
            nn.Linear(4 * embd_dim, embd_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class SingleTransRewardModel(nn.Module):
    """
    GPT-2 style causal transformer reward model for one ensemble member.

    Processes a trajectory segment as interleaved [action_t, state_t] tokens,
    then predicts a per-timestep reward scalar from the state tokens.
    """
    def __init__(self, ds, da, embd_dim=256, n_layer=1, n_head=1, dropout=0.1,
                 max_episode_steps=1000, activation='tanh'):
        super().__init__()
        self.ds = ds
        self.da = da
        self.embd_dim = embd_dim
        # sequence has 2*T tokens (action + state per step)
        max_seq_len = 2 * (max_episode_steps + 1)

        self.state_emb = nn.Linear(ds, embd_dim)
        self.action_emb = nn.Linear(da, embd_dim)
        self.timestep_emb = nn.Embedding(max_episode_steps + 1, embd_dim)
        self.input_ln = nn.LayerNorm(embd_dim)

        self.blocks = nn.Sequential(*[
            GPT2Block(embd_dim, n_head, dropout, max_seq_len)
            for _ in range(n_layer)
        ])

        inner_dim = embd_dim // 2
        if activation == 'tanh':
            act_fn = nn.Tanh()
        elif activation == 'relu':
            act_fn = nn.ReLU()
        else:
            act_fn = nn.LeakyReLU()

        self.reward_head = nn.Sequential(
            nn.Linear(embd_dim, inner_dim),
            act_fn,
            nn.Linear(inner_dim, 1),
        )

    def forward(self, states, actions, timesteps):
        """
        states:    (B, T, ds)
        actions:   (B, T, da)
        timesteps: (B, T)  long
        returns:   (B, T, 1)  per-timestep reward
        """
        B, T, _ = states.shape
        ts_emb = self.timestep_emb(timesteps)           # (B, T, embd_dim)
        s_emb = self.state_emb(states) + ts_emb         # (B, T, embd_dim)
        a_emb = self.action_emb(actions) + ts_emb       # (B, T, embd_dim)

        # interleave: [a_0, s_0, a_1, s_1, ...] → (B, 2T, embd_dim)
        stacked = torch.stack([a_emb, s_emb], dim=2).view(B, 2 * T, self.embd_dim)
        stacked = self.input_ln(stacked)
        x = self.blocks(stacked)                        # (B, 2T, embd_dim)

        # take state tokens (position 1 in each pair)
        state_tokens = x.view(B, T, 2, self.embd_dim)[:, :, 1, :]  # (B, T, embd_dim)
        return self.reward_head(state_tokens)            # (B, T, 1)


# ---------------------------------------------------------------------------
# Ensemble wrapper — mirrors the RewardModel API from reward_model.py
# ---------------------------------------------------------------------------

class TransformerRewardModel:
    def __init__(self, ds, da,
                 ensemble_size=3, lr=3e-4, mb_size=128, size_segment=50,
                 env_maker=None, max_size=100, activation='tanh', capacity=5e5,
                 large_batch=1, label_margin=0.0,
                 teacher_beta=-1, teacher_gamma=1,
                 teacher_eps_mistake=0, teacher_eps_skip=0, teacher_eps_equal=0,
                 # transformer-specific
                 embd_dim=256, n_layer=1, n_head=1, dropout=0.1,
                 max_episode_steps=1000,
                 feed_type=0,
                 use_wandb=False,
                 # ignored compat params
                 dormant_log_period=5000, dormant_threshold=0.1, bt_log_period=5000):

        self.ds = ds
        self.da = da
        self.de = ensemble_size
        self.lr = lr
        self.mb_size = mb_size
        self.origin_mb_size = mb_size
        self.size_segment = size_segment
        self.max_size = max_size
        self.activation = activation
        self.large_batch = large_batch
        self.label_margin = label_margin
        self.label_target = 1 - 2 * label_margin
        self.teacher_beta = teacher_beta
        self.teacher_gamma = teacher_gamma
        self.teacher_eps_mistake = teacher_eps_mistake
        self.teacher_eps_equal = teacher_eps_equal
        self.teacher_eps_skip = teacher_eps_skip
        self.teacher_thres_skip = 0
        self.teacher_thres_equal = 0
        self.feed_type = feed_type
        self.use_wandb = use_wandb
        self.env_step = 0
        self.train_batch_size = 128
        self.CEloss = nn.CrossEntropyLoss()

        # transformer hyperparams
        self.embd_dim = embd_dim
        self.n_layer = n_layer
        self.n_head = n_head
        self.dropout = dropout
        self.max_episode_steps = max_episode_steps

        # preference buffer
        self.capacity = int(capacity)
        self.buffer_seg1 = np.empty((self.capacity, size_segment, ds + da), dtype=np.float32)
        self.buffer_seg2 = np.empty((self.capacity, size_segment, ds + da), dtype=np.float32)
        self.buffer_label = np.empty((self.capacity, 1), dtype=np.float32)
        self.buffer_index = 0
        self.buffer_full = False

        # trajectory storage for query sampling
        self.inputs = []
        self.targets = []

        self.ensemble = []
        self.opt = None
        self.construct_ensemble()

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def construct_ensemble(self):
        self.ensemble = []
        for _ in range(self.de):
            model = SingleTransRewardModel(
                ds=self.ds, da=self.da,
                embd_dim=self.embd_dim,
                n_layer=self.n_layer,
                n_head=self.n_head,
                dropout=self.dropout,
                max_episode_steps=self.max_episode_steps,
                activation=self.activation,
            ).float().to(device)
            self.ensemble.append(model)
        self.opt = torch.optim.Adam(
            [p for m in self.ensemble for p in m.parameters()],
            lr=self.lr,
        )

    def reset_ensemble(self):
        self.construct_ensemble()

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    def _prepare_seg(self, x):
        """
        Accept numpy x with ndim in {1, 2, 3} and return
        (states, actions, timesteps) as CUDA tensors.

        1-D (D,)          → treat as single step: (1, 1, D)
        2-D (B, D)         → treat as B single steps: (B, 1, D)
        3-D (B, T, D)      → normal segment batch
        """
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, None, :]
        elif x.ndim == 2:
            x = x[:, None, :]
        B, T, _ = x.shape
        xt = torch.from_numpy(x).float().to(device)
        states = xt[:, :, :self.ds]
        actions = xt[:, :, self.ds:]
        ts = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        return states, actions, ts

    def r_hat_member(self, x, member=-1):
        """
        x: numpy (B, T, D) | (B, D) | (D,)
        returns: torch tensor (B, T, 1)  — callers decide no_grad / train mode
        """
        states, actions, ts = self._prepare_seg(x)
        return self.ensemble[member](states, actions, ts)

    def r_hat(self, x):
        """Single (obs+act) vector → scalar reward."""
        r_hats = []
        for member in range(self.de):
            with torch.no_grad():
                self.ensemble[member].eval()
                r = self.r_hat_member(x, member=member).detach().cpu().numpy()
                self.ensemble[member].train()
            r_hats.append(r)
        return np.mean(r_hats)

    def r_hat_batch(self, x):
        """
        x: numpy (B, D) — individual steps (no segment dim)
        returns: numpy (B, 1)
        """
        r_hats = []
        for member in range(self.de):
            with torch.no_grad():
                self.ensemble[member].eval()
                r = self.r_hat_member(x, member=member)   # (B, 1, 1)
                self.ensemble[member].train()
            r_hats.append(r.squeeze(1).detach().cpu().numpy())  # (B, 1)
        return np.mean(np.array(r_hats), axis=0)

    def r_hat_std(self, x):
        r_hats = []
        for member in range(self.de):
            with torch.no_grad():
                self.ensemble[member].eval()
                r = self.r_hat_member(x, member=member).detach().cpu().numpy()
                self.ensemble[member].train()
            r_hats.append(r)
        r_hats = np.array(r_hats)
        return np.mean(r_hats), np.std(r_hats)

    def r_hat_std_batch(self, x):
        r_hats = []
        for member in range(self.de):
            with torch.no_grad():
                self.ensemble[member].eval()
                r = self.r_hat_member(x, member=member)  # (B, 1, 1)
                self.ensemble[member].train()
            r_hats.append(r.squeeze(1).detach().cpu().numpy())  # (B, 1)
        r_hats = np.array(r_hats)
        return np.mean(r_hats, axis=0), np.std(r_hats, axis=0)

    # ------------------------------------------------------------------
    # Preference probability helpers
    # ------------------------------------------------------------------

    def get_rank_probability(self, x_1, x_2):
        probs = []
        for member in range(self.de):
            probs.append(self.p_hat_member(x_1, x_2, member=member).cpu().numpy())
        probs = np.array(probs)
        return np.mean(probs, axis=0), np.std(probs, axis=0)

    def get_entropy(self, x_1, x_2):
        probs = []
        for member in range(self.de):
            probs.append(self.p_hat_entropy(x_1, x_2, member=member).cpu().numpy())
        probs = np.array(probs)
        return np.mean(probs, axis=0), np.std(probs, axis=0)

    def p_hat_member(self, x_1, x_2, member=-1):
        with torch.no_grad():
            self.ensemble[member].eval()
            r_hat1 = self.r_hat_member(x_1, member=member).sum(axis=1)  # (B, 1)
            r_hat2 = self.r_hat_member(x_2, member=member).sum(axis=1)
            r_hat = torch.cat([r_hat1, r_hat2], axis=-1)                # (B, 2)
            self.ensemble[member].train()
        return F.softmax(r_hat, dim=-1)[:, 0]

    def p_hat_entropy(self, x_1, x_2, member=-1):
        with torch.no_grad():
            self.ensemble[member].eval()
            r_hat1 = self.r_hat_member(x_1, member=member).sum(axis=1)
            r_hat2 = self.r_hat_member(x_2, member=member).sum(axis=1)
            r_hat = torch.cat([r_hat1, r_hat2], axis=-1)
            self.ensemble[member].train()
        ent = F.softmax(r_hat, dim=-1) * F.log_softmax(r_hat, dim=-1)
        return ent.sum(axis=-1).abs()

    # ------------------------------------------------------------------
    # Schedule / batch helpers
    # ------------------------------------------------------------------

    def softXEnt_loss(self, input, target):
        logprobs = F.log_softmax(input, dim=1)
        return -(target * logprobs).sum() / input.shape[0]

    def change_batch(self, new_frac):
        self.mb_size = int(self.origin_mb_size * new_frac)

    def set_batch(self, new_batch):
        self.mb_size = int(new_batch)

    def set_teacher_thres_skip(self, new_margin):
        self.teacher_thres_skip = new_margin * self.teacher_eps_skip

    def set_teacher_thres_equal(self, new_margin):
        self.teacher_thres_equal = new_margin * self.teacher_eps_equal

    # ------------------------------------------------------------------
    # Trajectory / query buffer
    # ------------------------------------------------------------------

    def add_data(self, obs, act, rew, done):
        sa_t = np.concatenate([obs, act], axis=-1)
        flat_input = sa_t.reshape(1, self.da + self.ds)
        flat_target = np.array(rew).reshape(1, 1)

        init_data = len(self.inputs) == 0
        if init_data:
            self.inputs.append(flat_input)
            self.targets.append(flat_target)
        elif done:
            self.inputs[-1] = np.concatenate([self.inputs[-1], flat_input])
            self.targets[-1] = np.concatenate([self.targets[-1], flat_target])
            if len(self.inputs) > self.max_size:
                self.inputs = self.inputs[1:]
                self.targets = self.targets[1:]
            self.inputs.append([])
            self.targets.append([])
        else:
            if len(self.inputs[-1]) == 0:
                self.inputs[-1] = flat_input
                self.targets[-1] = flat_target
            else:
                self.inputs[-1] = np.concatenate([self.inputs[-1], flat_input])
                self.targets[-1] = np.concatenate([self.targets[-1], flat_target])

    def get_queries(self, mb_size=20):
        len_traj = len(self.inputs[0])
        max_len = len(self.inputs)
        if len(self.inputs[-1]) < len_traj:
            max_len = max_len - 1

        train_inputs = np.array(self.inputs[:max_len])
        train_targets = np.array(self.targets[:max_len])

        batch_index_2 = np.random.choice(max_len, size=mb_size, replace=True)
        sa_t_2 = train_inputs[batch_index_2]
        r_t_2 = train_targets[batch_index_2]

        batch_index_1 = np.random.choice(max_len, size=mb_size, replace=True)
        sa_t_1 = train_inputs[batch_index_1]
        r_t_1 = train_targets[batch_index_1]

        sa_t_1 = sa_t_1.reshape(-1, sa_t_1.shape[-1])
        r_t_1 = r_t_1.reshape(-1, r_t_1.shape[-1])
        sa_t_2 = sa_t_2.reshape(-1, sa_t_2.shape[-1])
        r_t_2 = r_t_2.reshape(-1, r_t_2.shape[-1])

        time_index = np.array([list(range(i * len_traj, i * len_traj + self.size_segment))
                                for i in range(mb_size)])
        time_index_2 = time_index + np.random.choice(
            len_traj - self.size_segment, size=mb_size, replace=True).reshape(-1, 1)
        time_index_1 = time_index + np.random.choice(
            len_traj - self.size_segment, size=mb_size, replace=True).reshape(-1, 1)

        sa_t_1 = np.take(sa_t_1, time_index_1, axis=0)
        r_t_1 = np.take(r_t_1, time_index_1, axis=0)
        sa_t_2 = np.take(sa_t_2, time_index_2, axis=0)
        r_t_2 = np.take(r_t_2, time_index_2, axis=0)

        return sa_t_1, sa_t_2, r_t_1, r_t_2

    def put_queries(self, sa_t_1, sa_t_2, labels):
        total_sample = sa_t_1.shape[0]
        next_index = self.buffer_index + total_sample
        if next_index >= self.capacity:
            self.buffer_full = True
            maximum_index = self.capacity - self.buffer_index
            np.copyto(self.buffer_seg1[self.buffer_index:self.capacity], sa_t_1[:maximum_index])
            np.copyto(self.buffer_seg2[self.buffer_index:self.capacity], sa_t_2[:maximum_index])
            np.copyto(self.buffer_label[self.buffer_index:self.capacity], labels[:maximum_index])
            remain = total_sample - maximum_index
            if remain > 0:
                np.copyto(self.buffer_seg1[0:remain], sa_t_1[maximum_index:])
                np.copyto(self.buffer_seg2[0:remain], sa_t_2[maximum_index:])
                np.copyto(self.buffer_label[0:remain], labels[maximum_index:])
            self.buffer_index = remain
        else:
            np.copyto(self.buffer_seg1[self.buffer_index:next_index], sa_t_1)
            np.copyto(self.buffer_seg2[self.buffer_index:next_index], sa_t_2)
            np.copyto(self.buffer_label[self.buffer_index:next_index], labels)
            self.buffer_index = next_index

    def get_label(self, sa_t_1, sa_t_2, r_t_1, r_t_2):
        sum_r_t_1 = np.sum(r_t_1, axis=1)
        sum_r_t_2 = np.sum(r_t_2, axis=1)

        if self.teacher_thres_skip > 0:
            max_r_t = np.maximum(sum_r_t_1, sum_r_t_2)
            max_index = (max_r_t > self.teacher_thres_skip).reshape(-1)
            if sum(max_index) == 0:
                return None, None, None, None, []
            sa_t_1, sa_t_2 = sa_t_1[max_index], sa_t_2[max_index]
            r_t_1, r_t_2 = r_t_1[max_index], r_t_2[max_index]
            sum_r_t_1 = np.sum(r_t_1, axis=1)
            sum_r_t_2 = np.sum(r_t_2, axis=1)

        margin_index = (np.abs(sum_r_t_1 - sum_r_t_2) < self.teacher_thres_equal).reshape(-1)

        seg_size = r_t_1.shape[1]
        temp_r_t_1 = r_t_1.copy()
        temp_r_t_2 = r_t_2.copy()
        for index in range(seg_size - 1):
            temp_r_t_1[:, :index + 1] *= self.teacher_gamma
            temp_r_t_2[:, :index + 1] *= self.teacher_gamma
        sum_r_t_1 = np.sum(temp_r_t_1, axis=1)
        sum_r_t_2 = np.sum(temp_r_t_2, axis=1)

        rational_labels = 1 * (sum_r_t_1 < sum_r_t_2)
        if self.teacher_beta > 0:
            r_hat = torch.cat([torch.Tensor(sum_r_t_1),
                               torch.Tensor(sum_r_t_2)], axis=-1)
            r_hat = r_hat * self.teacher_beta
            ent = F.softmax(r_hat, dim=-1)[:, 1]
            labels = torch.bernoulli(ent).int().numpy().reshape(-1, 1)
        else:
            labels = rational_labels
            if self.teacher_eps_mistake > 0:
                err_margin = self.teacher_eps_mistake
                labels[np.random.uniform(size=labels.size) < err_margin] = 1 - labels[
                    np.random.uniform(size=labels.size) < err_margin]
            labels = labels.reshape(-1, 1)

        labels[margin_index] = -1
        return sa_t_1, sa_t_2, r_t_1, r_t_2, labels

    # ------------------------------------------------------------------
    # Active learning query selection
    # ------------------------------------------------------------------

    def uniform_sampling(self):
        sa_t_1, sa_t_2, r_t_1, r_t_2 = self.get_queries(mb_size=self.mb_size)
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(sa_t_1, sa_t_2, r_t_1, r_t_2)
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)
        return len(labels)

    def disagreement_sampling(self):
        sa_t_1, sa_t_2, r_t_1, r_t_2 = self.get_queries(mb_size=self.mb_size * self.large_batch)
        _, disagree = self.get_rank_probability(sa_t_1, sa_t_2)
        top_k_index = (-disagree).argsort()[:self.mb_size]
        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(sa_t_1, sa_t_2, r_t_1, r_t_2)
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)
        return len(labels)

    def entropy_sampling(self):
        sa_t_1, sa_t_2, r_t_1, r_t_2 = self.get_queries(mb_size=self.mb_size * self.large_batch)
        entropy, _ = self.get_entropy(sa_t_1, sa_t_2)
        top_k_index = (-entropy).argsort()[:self.mb_size]
        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(sa_t_1, sa_t_2, r_t_1, r_t_2)
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)
        return len(labels)

    def kcenter_sampling(self):
        sa_t_1, sa_t_2, r_t_1, r_t_2 = self.get_queries(mb_size=self.mb_size * self.large_batch)
        temp_sa_1 = sa_t_1[:, :, :self.ds]
        temp_sa_2 = sa_t_2[:, :, :self.ds]
        temp_sa = np.concatenate([
            temp_sa_1.reshape(temp_sa_1.shape[0], -1),
            temp_sa_2.reshape(temp_sa_2.shape[0], -1),
        ], axis=1)

        max_len = self.capacity if self.buffer_full else self.buffer_index
        tot_sa_1 = self.buffer_seg1[:max_len, :, :self.ds]
        tot_sa_2 = self.buffer_seg2[:max_len, :, :self.ds]
        tot_sa = np.concatenate([
            tot_sa_1.reshape(max_len, -1),
            tot_sa_2.reshape(max_len, -1),
        ], axis=1)

        selected_index = KCenterGreedy(temp_sa, tot_sa, self.mb_size)
        r_t_1, sa_t_1 = r_t_1[selected_index], sa_t_1[selected_index]
        r_t_2, sa_t_2 = r_t_2[selected_index], sa_t_2[selected_index]
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(sa_t_1, sa_t_2, r_t_1, r_t_2)
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)
        return len(labels)

    def kcenter_disagree_sampling(self):
        sa_t_1, sa_t_2, r_t_1, r_t_2 = self.get_queries(mb_size=self.mb_size * self.large_batch)
        _, disagree = self.get_rank_probability(sa_t_1, sa_t_2)
        top_k_index = (-disagree).argsort()[:self.mb_size * self.large_batch]
        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]

        temp_sa_1 = sa_t_1[:, :, :self.ds]
        temp_sa_2 = sa_t_2[:, :, :self.ds]
        temp_sa = np.concatenate([
            temp_sa_1.reshape(temp_sa_1.shape[0], -1),
            temp_sa_2.reshape(temp_sa_2.shape[0], -1),
        ], axis=1)

        max_len = self.capacity if self.buffer_full else self.buffer_index
        tot_sa_1 = self.buffer_seg1[:max_len, :, :self.ds]
        tot_sa_2 = self.buffer_seg2[:max_len, :, :self.ds]
        tot_sa = np.concatenate([
            tot_sa_1.reshape(max_len, -1),
            tot_sa_2.reshape(max_len, -1),
        ], axis=1)

        selected_index = KCenterGreedy(temp_sa, tot_sa, self.mb_size)
        r_t_1, sa_t_1 = r_t_1[selected_index], sa_t_1[selected_index]
        r_t_2, sa_t_2 = r_t_2[selected_index], sa_t_2[selected_index]
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(sa_t_1, sa_t_2, r_t_1, r_t_2)
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)
        return len(labels)

    def kcenter_entropy_sampling(self):
        sa_t_1, sa_t_2, r_t_1, r_t_2 = self.get_queries(mb_size=self.mb_size * self.large_batch)
        entropy, _ = self.get_entropy(sa_t_1, sa_t_2)
        top_k_index = (-entropy).argsort()[:self.mb_size * self.large_batch]
        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]

        temp_sa_1 = sa_t_1[:, :, :self.ds]
        temp_sa_2 = sa_t_2[:, :, :self.ds]
        temp_sa = np.concatenate([
            temp_sa_1.reshape(temp_sa_1.shape[0], -1),
            temp_sa_2.reshape(temp_sa_2.shape[0], -1),
        ], axis=1)

        max_len = self.capacity if self.buffer_full else self.buffer_index
        tot_sa_1 = self.buffer_seg1[:max_len, :, :self.ds]
        tot_sa_2 = self.buffer_seg2[:max_len, :, :self.ds]
        tot_sa = np.concatenate([
            tot_sa_1.reshape(max_len, -1),
            tot_sa_2.reshape(max_len, -1),
        ], axis=1)

        selected_index = KCenterGreedy(temp_sa, tot_sa, self.mb_size)
        r_t_1, sa_t_1 = r_t_1[selected_index], sa_t_1[selected_index]
        r_t_2, sa_t_2 = r_t_2[selected_index], sa_t_2[selected_index]
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(sa_t_1, sa_t_2, r_t_1, r_t_2)
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)
        return len(labels)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_reward(self):
        ensemble_losses = [[] for _ in range(self.de)]
        ensemble_acc = np.array([0 for _ in range(self.de)])

        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = [np.random.permutation(max_len) for _ in range(self.de)]

        num_epochs = int(np.ceil(max_len / self.train_batch_size))
        total = 0

        for epoch in range(num_epochs):
            self.opt.zero_grad()
            loss = 0.0

            last_index = min((epoch + 1) * self.train_batch_size, max_len)

            for member in range(self.de):
                idxs = total_batch_index[member][epoch * self.train_batch_size:last_index]
                sa_t_1 = self.buffer_seg1[idxs]    # (B, T, D)
                sa_t_2 = self.buffer_seg2[idxs]
                labels = self.buffer_label[idxs]
                labels = torch.from_numpy(labels.flatten()).long().to(device)

                if member == 0:
                    total += labels.size(0)

                self.ensemble[member].train()
                r_hat1 = self.r_hat_member(sa_t_1, member=member).sum(axis=1)  # (B, 1)
                r_hat2 = self.r_hat_member(sa_t_2, member=member).sum(axis=1)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)                   # (B, 2)

                # ignore -1 (equal/skipped) labels
                valid = labels != -1
                if valid.sum() == 0:
                    continue
                curr_loss = self.CEloss(r_hat[valid], labels[valid])
                loss += curr_loss
                ensemble_losses[member].append(curr_loss.item())

                _, predicted = torch.max(r_hat[valid].data, 1)
                ensemble_acc[member] += (predicted == labels[valid]).sum().item()

            loss.backward()
            self.opt.step()

        ensemble_acc = ensemble_acc / max(total, 1)
        return ensemble_acc

    def train_soft_reward(self):
        ensemble_losses = [[] for _ in range(self.de)]
        ensemble_acc = np.array([0 for _ in range(self.de)])

        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = [np.random.permutation(max_len) for _ in range(self.de)]

        num_epochs = int(np.ceil(max_len / self.train_batch_size))
        total = 0

        for epoch in range(num_epochs):
            self.opt.zero_grad()
            loss = 0.0

            last_index = min((epoch + 1) * self.train_batch_size, max_len)

            for member in range(self.de):
                idxs = total_batch_index[member][epoch * self.train_batch_size:last_index]
                sa_t_1 = self.buffer_seg1[idxs]
                sa_t_2 = self.buffer_seg2[idxs]
                labels = self.buffer_label[idxs]
                labels = torch.from_numpy(labels.flatten()).long().to(device)

                if member == 0:
                    total += labels.size(0)

                self.ensemble[member].train()
                r_hat1 = self.r_hat_member(sa_t_1, member=member).sum(axis=1)
                r_hat2 = self.r_hat_member(sa_t_2, member=member).sum(axis=1)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)

                uniform_index = labels == -1
                labels[uniform_index] = 0
                target_onehot = torch.zeros_like(r_hat).scatter(
                    1, labels.unsqueeze(1), self.label_target)
                target_onehot += self.label_margin
                if sum(uniform_index) > 0:
                    target_onehot[uniform_index] = 0.5
                curr_loss = self.softXEnt_loss(r_hat, target_onehot)
                loss += curr_loss
                ensemble_losses[member].append(curr_loss.item())

                _, predicted = torch.max(r_hat.data, 1)
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct

            loss.backward()
            self.opt.step()

        ensemble_acc = ensemble_acc / max(total, 1)
        return ensemble_acc

    def get_train_acc(self):
        ensemble_acc = np.array([0 for _ in range(self.de)])
        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = np.random.permutation(max_len)
        batch_size = 256
        num_epochs = int(np.ceil(max_len / batch_size))
        total = 0

        for epoch in range(num_epochs):
            last_index = min((epoch + 1) * batch_size, max_len)
            sa_t_1 = self.buffer_seg1[epoch * batch_size:last_index]
            sa_t_2 = self.buffer_seg2[epoch * batch_size:last_index]
            labels = self.buffer_label[epoch * batch_size:last_index]
            labels = torch.from_numpy(labels.flatten()).long().to(device)
            total += labels.size(0)

            for member in range(self.de):
                with torch.no_grad():
                    self.ensemble[member].eval()
                    r_hat1 = self.r_hat_member(sa_t_1, member=member).sum(axis=1)
                    r_hat2 = self.r_hat_member(sa_t_2, member=member).sum(axis=1)
                    r_hat = torch.cat([r_hat1, r_hat2], axis=-1)
                    self.ensemble[member].train()
                _, predicted = torch.max(r_hat.data, 1)
                ensemble_acc[member] += (predicted == labels).sum().item()

        ensemble_acc = ensemble_acc / total
        return np.mean(ensemble_acc)

    # ------------------------------------------------------------------
    # Logging (stubs — transformer has no MLP-specific dormant metrics)
    # ------------------------------------------------------------------

    def pre_relabel_logging(self, step):
        pass

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save(self, model_dir, step):
        for member in range(self.de):
            torch.save(
                self.ensemble[member].state_dict(),
                '%s/reward_model_pt_%s_%s.pt' % (model_dir, step, member),
            )

    def load(self, model_dir, step):
        for member in range(self.de):
            self.ensemble[member].load_state_dict(
                torch.load('%s/reward_model_pt_%s_%s.pt' % (model_dir, step, member))
            )
