"""TandemLogger — records everything needed to replay a PEBBLE baseline deterministically.

Call convention (all calls come from train_PEBBLE.py's Workspace):

  Per env step:
    log_transition(step, obs, action, env_reward, next_obs, done, done_no_max)

  When an episode ends (done==True triggers the *next* reset):
    log_episode_end(episode_id, start_step, length)

  At the start of every learn_reward() call, before any querying:
    log_query_event_start(rm_update_idx, env_step, episode_ids_in_pool, reward_model)
      → saves pool episode IDs + RM weights that will be used to score candidates (iv-a)

  After querying and labelling, still inside learn_reward():
    log_query_event_pairs(rm_update_idx, sa_t_1, sa_t_2, r_t_1, r_t_2, labels, disagree_scores)

  At every change_batch() call:
    log_schedule(env_step, frac, mb_size)

  At end of run:
    close()
"""

import io
import os

import h5py
import numpy as np
import torch


class TandemLogger:
    FLUSH_INTERVAL = 5_000   # transitions buffered before writing to disk

    def __init__(self, log_path: str, obs_dim: int, act_dim: int, max_steps: int):
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        self.f = h5py.File(log_path, "w", libver="latest", locking=False)
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # ── Replay buffer (pre-allocated, row-indexed by env step) ──────────
        rb = self.f.create_group("replay_buffer")
        cz = min(max_steps, 10_000)
        rb.create_dataset("obs",          shape=(max_steps, obs_dim), dtype="f4", chunks=(cz, obs_dim))
        rb.create_dataset("actions",      shape=(max_steps, act_dim), dtype="f4", chunks=(cz, act_dim))
        rb.create_dataset("env_rewards",  shape=(max_steps,),         dtype="f4", chunks=(cz,))
        rb.create_dataset("next_obs",     shape=(max_steps, obs_dim), dtype="f4", chunks=(cz, obs_dim))
        rb.create_dataset("dones",        shape=(max_steps,),         dtype="f4", chunks=(cz,))
        rb.create_dataset("dones_no_max", shape=(max_steps,),         dtype="f4", chunks=(cz,))
        rb.attrs["length"] = 0

        # In-memory write buffers (flushed every FLUSH_INTERVAL steps)
        self._buf_obs          = np.empty((self.FLUSH_INTERVAL, obs_dim), dtype=np.float32)
        self._buf_act          = np.empty((self.FLUSH_INTERVAL, act_dim), dtype=np.float32)
        self._buf_rew          = np.empty(self.FLUSH_INTERVAL, dtype=np.float32)
        self._buf_nobs         = np.empty((self.FLUSH_INTERVAL, obs_dim), dtype=np.float32)
        self._buf_dones        = np.empty(self.FLUSH_INTERVAL, dtype=np.float32)
        self._buf_dones_no_max = np.empty(self.FLUSH_INTERVAL, dtype=np.float32)
        self._buf_idx   = 0   # write head inside buffer
        self._rb_offset = 0   # how many rows already flushed

        # ── Episode metadata (resizable) ────────────────────────────────────
        ep = self.f.create_group("episodes")
        ep.create_dataset("episode_ids", shape=(0,), maxshape=(None,), dtype="i4", chunks=(1000,))
        ep.create_dataset("start_steps", shape=(0,), maxshape=(None,), dtype="i4", chunks=(1000,))
        ep.create_dataset("lengths",     shape=(0,), maxshape=(None,), dtype="i4", chunks=(1000,))

        # ── Query events ────────────────────────────────────────────────────
        self.f.create_group("query_events")
        self.f["query_events"].attrs["count"] = 0

        # ── Schedule events (change_batch calls) ────────────────────────────
        sched = self.f.create_group("schedule")
        sched.create_dataset("steps",    shape=(0,), maxshape=(None,), dtype="i4", chunks=(100,))
        sched.create_dataset("fracs",    shape=(0,), maxshape=(None,), dtype="f4", chunks=(100,))
        sched.create_dataset("mb_sizes", shape=(0,), maxshape=(None,), dtype="i4", chunks=(100,))

    # ────────────────────────────────────────────────────────────────────────
    # Transition logging (buffered)
    # ────────────────────────────────────────────────────────────────────────

    def log_transition(self, step: int, obs, action, env_reward: float,
                       next_obs, done: float, done_no_max: float):
        i = self._buf_idx
        self._buf_obs[i]          = obs
        self._buf_act[i]          = action
        self._buf_rew[i]          = env_reward
        self._buf_nobs[i]         = next_obs
        self._buf_dones[i]        = done
        self._buf_dones_no_max[i] = done_no_max
        self._buf_idx += 1
        if self._buf_idx == self.FLUSH_INTERVAL:
            self._flush_replay_buffer()

    def _flush_replay_buffer(self):
        n = self._buf_idx
        if n == 0:
            return
        s = self._rb_offset
        rb = self.f["replay_buffer"]
        rb["obs"][s:s+n]          = self._buf_obs[:n]
        rb["actions"][s:s+n]      = self._buf_act[:n]
        rb["env_rewards"][s:s+n]  = self._buf_rew[:n]
        rb["next_obs"][s:s+n]     = self._buf_nobs[:n]
        rb["dones"][s:s+n]        = self._buf_dones[:n]
        rb["dones_no_max"][s:s+n] = self._buf_dones_no_max[:n]
        self._rb_offset += n
        self._buf_idx = 0
        rb.attrs["length"] = self._rb_offset

    # ────────────────────────────────────────────────────────────────────────
    # Episode metadata
    # ────────────────────────────────────────────────────────────────────────

    def log_episode_end(self, episode_id: int, start_step: int, length: int):
        ep = self.f["episodes"]
        for ds_name, val in (("episode_ids", episode_id),
                              ("start_steps", start_step),
                              ("lengths",     length)):
            ds = ep[ds_name]
            n  = ds.shape[0]
            ds.resize((n + 1,))
            ds[n] = val

    # ────────────────────────────────────────────────────────────────────────
    # Query event logging
    # ────────────────────────────────────────────────────────────────────────

    def log_query_event_start(self, rm_update_idx: int, env_step: int,
                               episode_ids_in_pool, reward_model):
        """Called before querying.  Saves pool membership + RM weights (scoring RM for iv-a)."""
        grp = self.f["query_events"].require_group(f"event_{rm_update_idx}")
        grp.attrs["env_step"] = env_step

        pool_ids = np.array(list(episode_ids_in_pool), dtype=np.int32)
        if "pool_episode_ids" not in grp:
            grp.create_dataset("pool_episode_ids", data=pool_ids)

        # RM weights — stored as raw bytes so we don't need to manage .pt files
        rm_grp = grp.require_group("rm_before_train")
        for member in range(reward_model.de):
            buf = io.BytesIO()
            torch.save(reward_model.ensemble[member].state_dict(), buf)
            data = np.frombuffer(buf.getvalue(), dtype=np.uint8)
            if f"member_{member}" not in rm_grp:
                rm_grp.create_dataset(f"member_{member}", data=data)

    def log_query_event_pairs(self, rm_update_idx: int,
                               sa_t_1, sa_t_2, r_t_1, r_t_2, labels, disagree_scores):
        """Called after labelling, before RM training."""
        grp = self.f["query_events"][f"event_{rm_update_idx}"]
        for name, arr in (("sa_t_1",          sa_t_1),
                           ("sa_t_2",          sa_t_2),
                           ("r_t_1",           r_t_1),
                           ("r_t_2",           r_t_2),
                           ("labels",          labels),
                           ("disagree_scores", disagree_scores)):
            if name not in grp:
                grp.create_dataset(name, data=np.asarray(arr, dtype=np.float32))
        self.f["query_events"].attrs["count"] = rm_update_idx + 1

    # ────────────────────────────────────────────────────────────────────────
    # Schedule events
    # ────────────────────────────────────────────────────────────────────────

    def log_schedule(self, env_step: int, frac: float, mb_size: int):
        sched = self.f["schedule"]
        for ds_name, val in (("steps", env_step), ("fracs", frac), ("mb_sizes", mb_size)):
            ds = sched[ds_name]
            n  = ds.shape[0]
            ds.resize((n + 1,))
            ds[n] = val

    # ────────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ────────────────────────────────────────────────────────────────────────

    def flush(self):
        self._flush_replay_buffer()
        self.f.flush()

    def close(self):
        self._flush_replay_buffer()
        self.f.close()
