#!/usr/bin/env python3
import argparse
import os
import shlex
import subprocess
import sys


DEFAULT_SLURM_LOG_DIR = "/uufs/chpc.utah.edu/common/home/dbrown-group1/zifan/logs"
ENVS = ("walker_walk", "metaworld_drawer-open-v2", "metaworld_hammer-v2")
ALPHA_MODES = ("auto", "low", "high")
SEEDS = (12345, 23451, 34512, 45123, 51234, 67890)
TANH_BASE_GAUGES = ("none", "no_tanh")
PG_GAUGES = ("none", "mean_buf", "mean_ref", "no_tanh", "no_tanh_mean_buf", "no_tanh_mean_ref")


def agent_override(env_name: str) -> str:
    return "agent=sac_metaworld" if "metaworld" in env_name else "agent=sac"


def build_command(env_name, alpha_mode, seed, method, active_gauge, python_bin, extra_overrides=None):
    cmd = [
        python_bin,
        "-m",
        "pebble_gauge.train_gauge",
        agent_override(env_name),
        "env={}".format(env_name),
        "seed={}".format(seed),
        "gauge.method={}".format(method),
        "gauge.alpha_mode={}".format(alpha_mode),
        "gauge.active={}".format(active_gauge),
    ]
    if extra_overrides:
        cmd.extend(extra_overrides)
    return cmd


def iter_jobs(phase, python_bin, extra_overrides=None):
    if phase in ("rrm", "all"):
        for env_name in ENVS:
            for alpha_mode in ALPHA_MODES:
                for seed in SEEDS:
                    for active in TANH_BASE_GAUGES:
                        yield build_command(env_name, alpha_mode, seed, "rrm", active, python_bin, extra_overrides)

    if phase in ("pg", "all"):
        for env_name in ENVS:
            for alpha_mode in ALPHA_MODES:
                for seed in SEEDS:
                    for active in PG_GAUGES:
                        yield build_command(env_name, alpha_mode, seed, "perfg", active, python_bin, extra_overrides)


def slurm_script(command, job_name, log_dir, repo_dir, gpu="a100"):
    quoted = " ".join(shlex.quote(x) for x in command)
    quoted_repo_dir = shlex.quote(repo_dir)
    return """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={log_dir}/{job_name}.out
#SBATCH --error={log_dir}/{job_name}.err
#SBATCH --gres=gpu:1
#SBATCH --partition={gpu}

set -euo pipefail
cd {repo_dir}
export MUJOCO_GL="${{MUJOCO_GL:-egl}}"
export LD_LIBRARY_PATH="${{LD_LIBRARY_PATH:-}}:/home/zifan/.mujoco/mujoco210/bin:/usr/lib/nvidia"
{quoted}
""".format(job_name=job_name, log_dir=log_dir, gpu=gpu, repo_dir=quoted_repo_dir, quoted=quoted)


def main():
    parser = argparse.ArgumentParser(description="Generate or launch gauge experiment sweeps.")
    parser.add_argument("--phase", choices=["rrm", "pg", "all"], default="all")
    parser.add_argument("--mode", choices=["dry-run", "local", "slurm"], default="dry-run")
    parser.add_argument("--limit", type=int, default=0, help="maximum jobs to emit/run; 0 means all")
    parser.add_argument("--log-dir", default=DEFAULT_SLURM_LOG_DIR)
    parser.add_argument("--slurm-partition", default="a100")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--repo-dir", default=os.getcwd(), help="repo path used by generated Slurm scripts")
    parser.add_argument("overrides", nargs="*", help="extra Hydra overrides appended to every job")
    args = parser.parse_args()

    jobs = list(iter_jobs(args.phase, python_bin=args.python, extra_overrides=args.overrides))
    if args.limit:
        jobs = jobs[: args.limit]

    if args.mode == "dry-run":
        for job in jobs:
            print(" ".join(shlex.quote(x) for x in job))
        print("# jobs: {}".format(len(jobs)))
        return

    if args.mode == "local":
        for job in jobs:
            subprocess.check_call(job)
        return

    os.makedirs(args.log_dir, exist_ok=True)
    script_dir = os.path.join("slurm_scripts", "gauge")
    os.makedirs(script_dir, exist_ok=True)
    for idx, job in enumerate(jobs):
        fields = {}
        for part in job:
            if "=" in part:
                key, value = part.split("=", 1)
                fields[key] = value
        job_name = "gauge_{:04d}_{}_{}_{}_{}".format(
            idx,
            fields.get("gauge.method", "m"),
            fields.get("env", "env").replace("metaworld_", "mw_").replace("-", "_"),
            fields.get("gauge.active", "g"),
            fields.get("seed", "seed"),
        )
        path = os.path.join(script_dir, "{}.sh".format(job_name))
        with open(path, "w") as f:
            f.write(slurm_script(job, job_name, args.log_dir, repo_dir=args.repo_dir, gpu=args.slurm_partition))
        subprocess.check_call(["sbatch", path])


if __name__ == "__main__":
    main()
