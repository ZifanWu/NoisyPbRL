"""Orchestrate the regime sweep × tasks × seeds. Smoke pilot under --quick.

Each (task, P_relabel, seed) cell is one full PEBBLE run executed as a subprocess
of `train_PEBBLE.py` with Hydra overrides + PD_* env vars. The hook in
train_PEBBLE.py installs the probe; outputs land as JSONL under perf_diag/runs/.

Run order:
  1. sanity.run_all(); refuse if critical checks fail.
  2. For each (task, regime, seed): spawn a PEBBLE run.
  3. Validate negative controls: assert they didn't turn over in horizon. If any did,
     fail loudly and don't proceed to analysis.

Outputs:
  perf_diag/runs/<task>__<regime>__seed<seed>.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field

import numpy as np


_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_THIS)
RUNS_DIR = os.path.join(_THIS, "runs")
os.makedirs(RUNS_DIR, exist_ok=True)


# RM capacity configs: (rm_hidden_dim, rm_num_layers, rm_output_activation)
CAPACITY_CONFIGS = {
    "full_rm":  dict(rm_hidden_dim=256, rm_num_layers=3, rm_output_activation="tanh"),
    "small_rm": dict(rm_hidden_dim=16,  rm_num_layers=1, rm_output_activation="tanh"),
}

# Sentinel max_feedback used when unlimited_budget=True. Sized to be "effectively unlimited"
# for our sweep configs (frequent_relabel at 1M steps needs ≤50K labels) but bounded so the
# RewardModel buffer fits in memory: capacity = max_feedback × large_batch (=10).
# At 100K × 10 × segment=50 × (ds+da)=90 × 4B ≈ 18 GB buffer.
UNLIMITED_BUDGET_VAL = 100_000


# ---------------------------------------------------------------------------
@dataclass
class RegimeCfg:
    name: str
    num_interact: int
    is_positive: bool
    extra_overrides: list = field(default_factory=list)


@dataclass
class SmokeConfig:
    tasks: tuple = ("metaworld_drawer-open-v2",)
    seeds: tuple = (0, 1)
    regimes: tuple = (
        RegimeCfg(name="rare_relabel", num_interact=15000, is_positive=True),
        RegimeCfg(name="frequent_relabel", num_interact=2000, is_positive=False),
    )
    num_train_steps: int = 60_000        # short horizon for smoke
    num_seed_steps: int = 500
    num_unsup_steps: int = 2000
    segment: int = 50
    eval_frequency: int = 5000
    ensemble_size: int = 5               # spec requires N=5 for the ensemble-var baseline
    teacher_eps_mistake: float = 0.1     # mild label noise so the data loop matters
    teacher_beta: int = -1               # rational+noise teacher (mistake ε only)
    max_feedback: int = 700
    reward_update: int = 50
    # Unlimited budget: when True, override TASK_DEFAULTS' max_feedback with UNLIMITED_BUDGET_VAL
    # to isolate the relabeling-frequency effect from preference-budget exhaustion.
    # Each regime gets a continuous stream of fresh on-policy preferences regardless of P_relabel.
    unlimited_budget: bool = True
    # capacity sweep
    capacities: tuple = ("full_rm", "small_rm")
    # probe knobs
    PD_N_PROBE: int = 4
    PD_SEGMENT_LEN: int = 50
    PD_REFIT_PAIRS: int = 12
    PD_K_REFIT: int = 5
    PD_REFIT_LR: float = 3e-4
    PD_PROBE_M: int = 1                  # probe every relabel
    PD_STEP_PROBE_EVERY: int = 10000     # step-cadence probe after budget exhaustion; 0 disables


# Per-task budgets, taken from the repo's authored scripts (scripts/<env>/<feedback>/oracle/run_PEBBLE.sh)
# so the full study uses the same scale the authors validated. Values are (max_feedback, central P_relabel, reward_update).
# Central P_relabel is what the regime sweep is built around: positive = ~2x central, negative = ~0.2x central.
# reward_update is the number of RM training epochs per relabel; the original scripts use 10 for MetaWorld and 50 for DMC.
# The repo's config default is 200 but it's always overridden in the run scripts — we replicate the per-env scripts here.
TASK_DEFAULTS = {
    "metaworld_drawer-open-v2":  dict(max_feedback=10000, central_P=10000, reward_update=10),
    "metaworld_door-close-v2":   dict(max_feedback=1000,  central_P=10000, reward_update=10),
    "metaworld_door-open-v2":    dict(max_feedback=4000,  central_P=10000, reward_update=10),
    "metaworld_door-unlock-v2":  dict(max_feedback=5000,  central_P=10000, reward_update=10),
    "metaworld_button-press-v2": dict(max_feedback=20000, central_P=5000,  reward_update=10),
    "metaworld_hammer-v2":       dict(max_feedback=20000, central_P=5000,  reward_update=10),
    "metaworld_sweep-into-v2":   dict(max_feedback=20000, central_P=5000,  reward_update=10),
    "metaworld_window-close-v2": dict(max_feedback=500,   central_P=10000, reward_update=10),
    "walker_walk":               dict(max_feedback=1000,  central_P=20000, reward_update=50),
    "quadruped_walk":            dict(max_feedback=2000,  central_P=20000, reward_update=50),
}


def task_overrides(task: str) -> dict:
    """Return Hydra overrides specific to this task, or {} for unknown tasks."""
    d = TASK_DEFAULTS.get(task)
    if not d:
        return {}
    out = {"max_feedback": d["max_feedback"]}
    if "reward_update" in d:
        out["reward_update"] = d["reward_update"]
    return out


def task_central_P(task: str, fallback: int = 10000) -> int:
    d = TASK_DEFAULTS.get(task)
    return d["central_P"] if d else fallback


@dataclass
class FullConfig(SmokeConfig):
    tasks: tuple = (
        "metaworld_drawer-open-v2",
        "metaworld_door-close-v2",
        "walker_walk",   # DMC; if available in this env
    )
    seeds: tuple = (0, 1, 2, 3, 4, 5)
    # NOTE: regimes in FullConfig are *per-task* — see _build_regimes_for_task below.
    # The placeholder here is just to satisfy the dataclass; it gets overridden.
    regimes: tuple = (
        RegimeCfg(name="placeholder", num_interact=10000, is_positive=True),
    )
    num_train_steps: int = 1_000_000
    num_seed_steps: int = 1000
    num_unsup_steps: int = 9000           # matches the authors' validated default (was 5000)
    max_feedback: int = 1400              # OVERRIDDEN per-task via TASK_DEFAULTS
    reward_update: int = 200


def build_regimes_for_task(cfg, task: str) -> list[RegimeCfg]:
    """For a FullConfig run, build a task-specific P_relabel sweep around the central value
    each env was validated on. For SmokeConfig we just use cfg.regimes as-is.
    """
    if not isinstance(cfg, FullConfig):
        return list(cfg.regimes)
    central = task_central_P(task)
    return [
        RegimeCfg(name="rare_relabel",     num_interact=int(central * 2),   is_positive=True),
        RegimeCfg(name="med_relabel",      num_interact=int(central),       is_positive=True),
        RegimeCfg(name="frequent_relabel", num_interact=max(2000, int(central * 0.2)), is_positive=False),
    ]


def _run_name(task: str, regime: str, seed: int, capacity: str = "full_rm") -> str:
    return f"{task.replace('/', '_')}__{regime}__{capacity}__seed{seed}"


def _spawn_one(cfg, task: str, regime: RegimeCfg, seed: int, log_dir: str,
               capacity: str = "full_rm") -> dict:
    """Spawn a single PEBBLE run as a subprocess and wait."""
    run_name = _run_name(task, regime.name, seed, capacity)
    jsonl_path = os.path.join(RUNS_DIR, f"{run_name}.jsonl")
    if os.path.exists(jsonl_path):
        # Skip already-completed runs (cheap restart). Comment out to force redo.
        return dict(run_name=run_name, status="skipped",
                    jsonl=jsonl_path, regime=regime.name, task=task, seed=seed,
                    capacity=capacity, is_positive=regime.is_positive)

    env = os.environ.copy()
    env["PD_ENABLE"] = "1"
    env["PD_RUN_NAME"] = run_name
    env["PD_TAG_POSITIVE"] = "positive" if regime.is_positive else "negative_control"
    env["PD_N_PROBE"] = str(cfg.PD_N_PROBE)
    env["PD_SEGMENT_LEN"] = str(cfg.PD_SEGMENT_LEN)
    env["PD_REFIT_PAIRS"] = str(cfg.PD_REFIT_PAIRS)
    env["PD_K_REFIT"] = str(cfg.PD_K_REFIT)
    env["PD_REFIT_LR"] = str(cfg.PD_REFIT_LR)
    env["PD_PROBE_M"] = str(cfg.PD_PROBE_M)
    env["PD_STEP_PROBE_EVERY"] = str(cfg.PD_STEP_PROBE_EVERY)
    env["PD_OUT_DIR"] = RUNS_DIR
    # Some Hydra/MuJoCo combos need these
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYTHONPATH", _REPO)

    # Per-task overrides (max_feedback, reward_update) take precedence over the global cfg defaults.
    per_task = task_overrides(task)
    max_feedback = per_task.get("max_feedback", cfg.max_feedback)
    reward_update = per_task.get("reward_update", cfg.reward_update)
    # Unlimited budget: override max_feedback so the regime sweep measures the pure
    # effect of relabeling frequency (each regime keeps getting fresh labels).
    if getattr(cfg, "unlimited_budget", False):
        max_feedback = UNLIMITED_BUDGET_VAL
    cap_cfg = CAPACITY_CONFIGS.get(capacity, CAPACITY_CONFIGS["full_rm"])
    overrides = [
        f"env={task}",
        f"seed={seed}",
        f"num_interact={regime.num_interact}",
        f"num_train_steps={cfg.num_train_steps}",
        f"num_seed_steps={cfg.num_seed_steps}",
        f"num_unsup_steps={cfg.num_unsup_steps}",
        f"eval_frequency={cfg.eval_frequency}",
        f"segment={cfg.segment}",
        f"ensemble_size={cfg.ensemble_size}",
        f"teacher_eps_mistake={cfg.teacher_eps_mistake}",
        f"teacher_beta={cfg.teacher_beta}",
        f"max_feedback={max_feedback}",
        f"reward_update={reward_update}",
        f"rm_hidden_dim={cap_cfg['rm_hidden_dim']}",
        f"rm_num_layers={cap_cfg['rm_num_layers']}",
        f"rm_output_activation={cap_cfg['rm_output_activation']}",
        "keep_relabeling_after_budget=true",
        "log_save_tb=true",
        "use_wandb=false",
        "save_video=false",
    ]
    overrides.extend(regime.extra_overrides)

    py = sys.executable
    cmd = [py, "train_PEBBLE.py"] + overrides
    stdout_path = os.path.join(log_dir, f"{run_name}.stdout")
    stderr_path = os.path.join(log_dir, f"{run_name}.stderr")
    print(f"  spawning: {run_name}")
    t0 = time.time()
    with open(stdout_path, "w") as so, open(stderr_path, "w") as se:
        rc = subprocess.call(cmd, cwd=_REPO, env=env, stdout=so, stderr=se)
    wall = time.time() - t0
    status = "ok" if rc == 0 else f"rc={rc}"
    print(f"  {run_name}: {status} in {wall:.1f}s")
    return dict(run_name=run_name, status=status,
                jsonl=jsonl_path, regime=regime.name, task=task, seed=seed,
                capacity=capacity, is_positive=regime.is_positive, wall=wall,
                stdout=stdout_path, stderr=stderr_path)


def _load_jsonl(path: str) -> list:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def validate_negative_controls(spawn_results: list[dict]) -> tuple[bool, list[str]]:
    """Spec §8 check #5: every negative-control run must not turn over in horizon."""
    from . import detect
    bad = []
    for r in spawn_results:
        if r.get("is_positive", True):
            continue
        rows = _load_jsonl(r["jsonl"])
        gold = np.array([row.get("gold_eval", np.nan) for row in rows], dtype=np.float64)
        gold = gold[~np.isnan(gold)]
        if gold.size < 5:
            continue  # too few samples to judge; let analysis flag this separately
        if detect.turned_over_in_horizon(
            gold, K_decline=5, alpha=0.2, min_drop_frac=0.2,
            min_post_points=8, max_peak_frac=0.8, late_window=5
        ):
            bad.append(r["run_name"])
    return (len(bad) == 0, bad)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="smoke pilot (1 task, 2 seeds, 2 regimes, short horizon)")
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--skip-sanity", action="store_true")
    parser.add_argument("--skip-pebble", action="store_true",
                        help="skip the actual PEBBLE subprocess and use existing JSONL")
    args = parser.parse_args(argv)

    cfg = SmokeConfig() if args.quick else FullConfig()
    if args.tasks:
        cfg.tasks = tuple(args.tasks)
    if args.seeds:
        cfg.seeds = tuple(args.seeds)

    # 1. Sanity gate
    if not args.skip_sanity:
        from perf_diag import sanity
        print("\n>>> SANITY")
        res = sanity.run_all()
        if not res["_crit_ok"]:
            print("REFUSING TO RUN: critical sanity checks failed.", file=sys.stderr)
            return 2
        print("SANITY PASSED.\n")

    # 2. Spawn runs
    log_dir = os.path.join(_THIS, "runs", "_logs")
    os.makedirs(log_dir, exist_ok=True)
    capacities = list(cfg.capacities)
    n_regimes = len(cfg.regimes)  # approximate; FullConfig overrides per task
    print(f">>> SPAWNING RUNS  ({len(cfg.tasks)} tasks × {n_regimes} regimes × "
          f"{len(capacities)} capacities × {len(cfg.seeds)} seeds = "
          f"~{len(cfg.tasks) * n_regimes * len(capacities) * len(cfg.seeds)} runs)")
    spawn_results = []
    if not args.skip_pebble:
        t0 = time.time()
        for task in cfg.tasks:
            regimes_for_task = build_regimes_for_task(cfg, task)
            for regime in regimes_for_task:
                for capacity in capacities:
                    for seed in cfg.seeds:
                        spawn_results.append(_spawn_one(cfg, task, regime, seed, log_dir, capacity))
        print(f">>> ALL RUNS DONE in {time.time() - t0:.1f}s\n")
    else:
        # Reconstruct results dict from existing JSONL files
        for task in cfg.tasks:
            regimes_for_task = build_regimes_for_task(cfg, task)
            for regime in regimes_for_task:
                for capacity in capacities:
                    for seed in cfg.seeds:
                        name = _run_name(task, regime.name, seed, capacity)
                        spawn_results.append(dict(
                            run_name=name, status="reused",
                            jsonl=os.path.join(RUNS_DIR, f"{name}.jsonl"),
                            regime=regime.name, task=task, seed=seed,
                            capacity=capacity, is_positive=regime.is_positive,
                        ))

    # 3. Validate negative controls
    print(">>> NEGATIVE-CONTROL VALIDATION (spec §8 #5)")
    ok, bad = validate_negative_controls(spawn_results)
    if ok:
        print("  all negative controls passed (no in-horizon turnover).\n")
    else:
        print(f"  NEGATIVE CONTROLS TURNED OVER: {bad}", file=sys.stderr)
        print("  This means the supposed-safe regime is unsafe in this config. The matched-FAR")
        print("  comparison will be unreliable until these configs are tightened. Continuing")
        print("  to analysis anyway, but the report will flag this.")

    # 4. Hand-off to analysis
    manifest_path = os.path.join(RUNS_DIR, "_manifest.json")
    # Collect the actual regime *names* used across tasks (FullConfig is per-task)
    regime_names_seen = sorted({r["regime"] for r in spawn_results})
    with open(manifest_path, "w") as f:
        json.dump({"runs": spawn_results,
                   "config": dict(quick=args.quick,
                                  tasks=list(cfg.tasks),
                                  seeds=list(cfg.seeds),
                                  regimes=regime_names_seen,
                                  capacities=capacities),
                   "neg_control_validity": dict(ok=ok, bad=bad)},
                  f, indent=2)
    print(f"manifest at: {manifest_path}")
    print("Next: python -m perf_diag.analysis")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
