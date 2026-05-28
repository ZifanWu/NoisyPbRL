import os
import time
import uuid
from types import SimpleNamespace
from typing import Dict

import numpy as np
import torch

import utils


def env_file_name(env_name: str) -> str:
    return env_name.replace("/", "_").replace(":", "_")


def reference_path(root: str, env_name: str) -> str:
    return os.path.join(root, "{}.pt".format(env_file_name(env_name)))


def _reset_env(env):
    out = env.reset()
    if isinstance(out, tuple):
        return out[0]
    return out


def _step_env(env, action):
    out = env.step(action)
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        return obs, reward, bool(terminated or truncated), info
    obs, reward, done, info = out
    return obs, reward, bool(done), info


def make_env_for_reference(env_name: str, seed: int):
    cfg = SimpleNamespace(env=env_name, seed=seed)
    if "metaworld" in env_name:
        return utils.make_metaworld_env(cfg)
    return utils.make_env(cfg)


def generate_reference_dataset(env_name: str, n_ref: int, seed: int) -> Dict:
    env = make_env_for_reference(env_name, seed)
    obs = _reset_env(env)
    obses = []
    actions = []
    rng = np.random.RandomState(seed)

    while len(obses) < int(n_ref):
        if hasattr(env.action_space, "seed"):
            env.action_space.seed(int(rng.randint(0, 2 ** 31 - 1)))
        action = env.action_space.sample()
        obses.append(np.asarray(obs, dtype=np.float32))
        actions.append(np.asarray(action, dtype=np.float32))
        next_obs, _reward, done, _info = _step_env(env, action)
        obs = _reset_env(env) if done else next_obs

    if hasattr(env, "close"):
        env.close()

    return {
        "obs": np.asarray(obses, dtype=np.float32),
        "action": np.asarray(actions, dtype=np.float32),
        "env": env_name,
        "seed": int(seed),
        "n_ref": int(n_ref),
    }


def load_or_create_reference_dataset(
    env_name: str,
    root: str = "reference_dataset",
    n_ref: int = 10000,
    seed: int = 0,
    auto_create: bool = True,
) -> Dict:
    os.makedirs(root, exist_ok=True)
    path = reference_path(root, env_name)
    if os.path.exists(path):
        return _load_reference_dataset(path)
    if not auto_create:
        raise FileNotFoundError(
            "reference dataset missing at {}; rerun with auto_create=True".format(path)
        )
    data = generate_reference_dataset(env_name, n_ref=n_ref, seed=seed)
    tmp_path = "{}.tmp.{}.{}".format(path, os.getpid(), uuid.uuid4().hex)
    try:
        torch.save(data, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    return data


def _load_reference_dataset(path: str, attempts: int = 5, sleep_s: float = 1.0) -> Dict:
    last_error = None
    for _attempt in range(int(attempts)):
        try:
            return torch.load(path, map_location="cpu")
        except Exception as exc:
            last_error = exc
            time.sleep(float(sleep_s))
    raise last_error


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate a frozen random-policy reference dataset.")
    parser.add_argument("--env", required=True)
    parser.add_argument("--root", default="reference_dataset")
    parser.add_argument("--n-ref", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    path = reference_path(args.root, args.env)
    load_or_create_reference_dataset(
        args.env,
        root=args.root,
        n_ref=args.n_ref,
        seed=args.seed,
        auto_create=True,
    )
    print(path)


if __name__ == "__main__":
    main()
