"""TandemReader — reads a baseline HDF5 log to drive tandem training conditions.

Public API used by train_PEBBLE.py:

  get_transition(step)          → (obs, action, env_reward, next_obs, done, done_no_max)
  get_query_event(event_idx)    → dict with pairs / labels / pool IDs / scores
  load_baseline_rm(event_idx, reward_model)   → load RM weights into reward_model in-place
  reconstruct_episode_pool(pool_episode_ids)  → (inputs, targets) lists matching self.inputs format
  num_query_events              → int
  close()
"""

from __future__ import annotations

import io

import h5py
import numpy as np
import torch

_DEVICE = "cuda"


class TandemReader:
    def __init__(self, log_path: str):
        self.f = h5py.File(log_path, "r", locking=False)
        self.num_steps = int(self.f["replay_buffer"].attrs["length"])
        self.num_query_events = int(self.f["query_events"].attrs["count"])

        # Build episode index once: episode_id → (start_step, length)
        ep = self.f["episodes"]
        ep_ids  = ep["episode_ids"][:]
        starts  = ep["start_steps"][:]
        lengths = ep["lengths"][:]
        self.episode_meta: dict[int, tuple[int, int]] = {
            int(eid): (int(s), int(l))
            for eid, s, l in zip(ep_ids, starts, lengths)
        }

    # ────────────────────────────────────────────────────────────────────────
    # Replay buffer access
    # ────────────────────────────────────────────────────────────────────────

    def get_transition(self, step: int):
        """Return one (obs, action, env_reward, next_obs, done, done_no_max) tuple."""
        rb = self.f["replay_buffer"]
        return (
            rb["obs"][step].copy(),
            rb["actions"][step].copy(),
            float(rb["env_rewards"][step]),
            rb["next_obs"][step].copy(),
            float(rb["dones"][step]),
            float(rb["dones_no_max"][step]),
        )

    # ────────────────────────────────────────────────────────────────────────
    # Query event access
    # ────────────────────────────────────────────────────────────────────────

    def get_query_event(self, event_idx: int) -> dict:
        """Return all logged data for query event `event_idx`."""
        grp = self.f["query_events"][f"event_{event_idx}"]
        return {
            "env_step":         int(grp.attrs["env_step"]),
            "pool_episode_ids": grp["pool_episode_ids"][:],
            "sa_t_1":           grp["sa_t_1"][:],
            "sa_t_2":           grp["sa_t_2"][:],
            "r_t_1":            grp["r_t_1"][:],
            "r_t_2":            grp["r_t_2"][:],
            "labels":           grp["labels"][:],
            "disagree_scores":  grp["disagree_scores"][:],
        }

    def load_baseline_rm(self, event_idx: int, reward_model):
        """Load the baseline RM weights recorded *before* query event `event_idx` into
        `reward_model` in-place.  Used in condition iv-a to score tandem candidate pairs."""
        grp = self.f["query_events"][f"event_{event_idx}"]["rm_before_train"]
        for member in range(reward_model.de):
            raw = grp[f"member_{member}"][:].tobytes()
            try:
                state_dict = torch.load(io.BytesIO(raw),
                                        map_location=_DEVICE, weights_only=True)
            except TypeError:  # PyTorch < 1.13 doesn't have weights_only
                state_dict = torch.load(io.BytesIO(raw), map_location=_DEVICE)
            reward_model.ensemble[member].load_state_dict(state_dict)

    # ────────────────────────────────────────────────────────────────────────
    # Episode pool reconstruction (condition iv-b)
    # ────────────────────────────────────────────────────────────────────────

    def reconstruct_episode_pool(self, pool_episode_ids) -> tuple[list, list]:
        """Reconstruct the baseline RM's self.inputs / self.targets lists from pointers.

        Each entry in the returned lists is a (T_i, ds+da) / (T_i, 1) numpy array,
        exactly matching the format that RewardModel.get_queries() expects.
        """
        inputs:  list[np.ndarray] = []
        targets: list[np.ndarray] = []
        rb = self.f["replay_buffer"]

        for ep_id in pool_episode_ids:
            ep_id = int(ep_id)
            if ep_id not in self.episode_meta:
                continue
            start, length = self.episode_meta[ep_id]
            obs     = rb["obs"][start:start + length]
            actions = rb["actions"][start:start + length]
            env_rew = rb["env_rewards"][start:start + length]
            sa = np.concatenate([obs, actions], axis=-1).astype(np.float32)
            r  = env_rew.reshape(-1, 1).astype(np.float32)
            inputs.append(sa)
            targets.append(r)

        return inputs, targets

    # ────────────────────────────────────────────────────────────────────────
    # Schedule
    # ────────────────────────────────────────────────────────────────────────

    def get_schedule_events(self):
        """Return (steps, fracs, mb_sizes) arrays for all change_batch calls."""
        sched = self.f["schedule"]
        return (sched["steps"][:], sched["fracs"][:], sched["mb_sizes"][:])

    # ────────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ────────────────────────────────────────────────────────────────────────

    def close(self):
        self.f.close()
