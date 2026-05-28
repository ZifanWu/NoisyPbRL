from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch

import utils


def _sanitize_env_name(env_name: str) -> str:
    return env_name.replace("/", "_").replace(":", "_").replace("-", "_")


def _make_env_from_cfg(cfg):
    if "metaworld" in cfg.env:
        return utils.make_metaworld_env(cfg)
    return utils.make_env(cfg)


def build_or_load_reference_dataset(
    cfg,
    env_name: str,
    n_ref: int = 10_000,
    out_dir: str = "reference_dataset",
    seed: Optional[int] = None,
) -> np.ndarray:
    """Create/load frozen reference (obs, action) tuples for mean_ref gauge."""
    os.makedirs(out_dir, exist_ok=True)
    safe_env = _sanitize_env_name(env_name)
    out_path = os.path.join(out_dir, f"{safe_env}.pt")
    if seed is None:
        seed = int(getattr(cfg, "reference_seed", 0))

    if os.path.exists(out_path):
        payload = torch.load(out_path, map_location="cpu")
        sa = np.asarray(payload["sa"], dtype=np.float32)
        stored_seed = payload.get("seed")
        stored_n_ref = payload.get("n_ref", len(sa))
        if stored_seed == int(seed) and int(stored_n_ref) == int(n_ref) and len(sa) == int(n_ref):
            return sa
        print(
            f"Rebuilding reference dataset {out_path}: "
            f"stored seed/n_ref/len=({stored_seed}, {stored_n_ref}, {len(sa)}) "
            f"requested=({int(seed)}, {int(n_ref)}, {int(n_ref)})"
        )
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = _make_env_from_cfg(cfg)
    # Seed the action-space sampler explicitly so random rollouts are
    # reproducible even when cfg.seed differs from the seed argument.
    try:
        env.action_space.seed(seed)
    except Exception:
        pass
    obs = env.reset()
    obs_list = []
    action_list = []
    for _ in range(int(n_ref)):
        action = env.action_space.sample()
        obs_list.append(np.asarray(obs, dtype=np.float32))
        action_list.append(np.asarray(action, dtype=np.float32))
        step_out = env.step(action)
        if len(step_out) == 5:
            next_obs, _, terminated, truncated, _ = step_out
            done = bool(terminated or truncated)
        else:
            next_obs, _, done, _ = step_out  # old 4-tuple Gym API (dmc2gym)
            done = bool(done)
        obs = env.reset() if done else next_obs

    obs_arr = np.asarray(obs_list, dtype=np.float32)
    action_arr = np.asarray(action_list, dtype=np.float32)
    sa = np.concatenate([obs_arr, action_arr], axis=-1).astype(np.float32)
    torch.save(
        {
            "env_name": env_name,
            "seed": int(seed),
            "n_ref": int(n_ref),
            "obs": obs_arr,
            "action": action_arr,
            "sa": sa,
        },
        out_path,
    )
    return sa
