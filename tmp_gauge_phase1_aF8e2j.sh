#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=20g
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --job-name=gauge_phase1
#SBATCH --time=24:00:00
##SBATCH --qos=dbrown-gpu-grn
#SBATCH --partition=dbrown-gpu-np
#SBATCH --account=dbrown-gpu-np
##SBATCH --exclude=notch372,notch369,notch475,notch371
#SBATCH --output=/uufs/chpc.utah.edu/common/home/dbrown-group1/zifan/logs/gauge_pbrl/gauge_phase1_%A_%a.out

set -euo pipefail

module load cuda/12.4.0
cd "/uufs/chpc.utah.edu/common/home/u1520755/NoisyPbRL"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

mapfile -t CELL_INFO < <("/uufs/chpc.utah.edu/common/home/dbrown-group1/zifan/miniconda3/envs/bpref/bin/python" - "phase1" "${SLURM_ARRAY_TASK_ID}" "/uufs/chpc.utah.edu/common/home/u1520755/NoisyPbRL" "/uufs/chpc.utah.edu/common/home/dbrown-group1/zifan/NoisyPbRL/results" "/uufs/chpc.utah.edu/common/home/dbrown-group1/zifan/NoisyPbRL/reference_dataset" "false" <<'PY'
import shlex
import sys
from pathlib import Path

import yaml

python = "/uufs/chpc.utah.edu/common/home/dbrown-group1/zifan/miniconda3/envs/bpref/bin/python"
phase = sys.argv[1]
task_id = int(sys.argv[2])
script_dir = Path(sys.argv[3])
results_dir = Path(sys.argv[4])
reference_dir = Path(sys.argv[5])
use_wandb = sys.argv[6]

config_dir = script_dir / "configs" / "gauge_experiment"
base = yaml.safe_load(open(config_dir / "base.yaml", "r", encoding="utf-8"))
phase_cfg = yaml.safe_load(open(config_dir / f"{phase}.yaml", "r", encoding="utf-8"))

cells = []
for env_key in phase_cfg["envs"]:
    env_meta = base["envs"][env_key]
    for alpha_mode in phase_cfg["alpha_modes"]:
        alpha_meta = base["alpha_modes"][alpha_mode]
        for seed in phase_cfg["seeds"]:
            for method in phase_cfg["methods"]:
                method_meta = base["methods"][method]
                for gauge in method_meta["gauges"]:
                    cells.append({
                        "env_key": env_key,
                        "train_env": env_meta["train_env"],
                        "method": method,
                        "gauge": gauge,
                        "alpha_mode": alpha_mode,
                        "seed": int(seed),
                        "horizon": int(env_meta["horizon"]),
                        "use_perf_correction": bool(method_meta["use_perf_correction"]),
                        "learnable_temperature": bool(alpha_meta["learnable_temperature"]),
                        "init_temperature": float(alpha_meta["init_temperature"]),
                    })

cell = cells[task_id]
run_name = (
    f"{cell['env_key']}__{cell['method']}__{cell['gauge']}"
    f"__{cell['alpha_mode']}__seed{cell['seed']}"
)
run_dir = results_dir / "runs" / run_name

overrides = []
for k, v in base.get("default_overrides", {}).items():
    if k == "use_wandb":
        v = use_wandb
    overrides.append(f"{k}={v}")

overrides.extend([
    f"env={cell['train_env']}",
    f"seed={cell['seed']}",
    f"method={cell['method']}",
    f"gauge={cell['gauge']}",
    f"alpha_mode={cell['alpha_mode']}",
    f"use_perf_correction={'true' if cell['use_perf_correction'] else 'false'}",
    f"h_horizon={cell['horizon']}",
    f"agent.params.learnable_temperature={'true' if cell['learnable_temperature'] else 'false'}",
    f"agent.params.init_temperature={cell['init_temperature']}",
    "gpu=0",
    f"exp_dir={results_dir / 'runs'}",
    f"reference_dataset_dir={reference_dir}",
    f"hydra.run.dir={run_dir}",
])

cmd = [python, "train_PEBBLE.py", *overrides]
print(run_name)
print(run_dir)
print(" ".join(shlex.quote(str(x)) for x in cmd))
PY
)

RUN_NAME="${CELL_INFO[0]}"
RUN_DIR="${CELL_INFO[1]}"
CMD="${CELL_INFO[2]}"

echo "=== Gauge PbRL task ==="
echo "  phase    : phase1"
echo "  array id : ${SLURM_ARRAY_TASK_ID}"
echo "  run      : ${RUN_NAME}"
echo "  run dir  : ${RUN_DIR}"
echo "  command  : ${CMD}"
echo ""

if [ "true" = "true" ] && [ -f "${RUN_DIR}/train.csv" ]; then
    LAST_STEP="$("/uufs/chpc.utah.edu/common/home/dbrown-group1/zifan/miniconda3/envs/bpref/bin/python" - "${RUN_DIR}/train.csv" <<'PY'
import csv
import sys
last = -1
try:
    with open(sys.argv[1], newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("step"):
                last = int(float(row["step"]))
except Exception:
    last = -1
print(last)
PY
)"
    if [ "$LAST_STEP" -ge "990000" ]; then
        echo "[SKIP] ${RUN_NAME} appears complete: last step ${LAST_STEP}"
        exit 0
    fi
fi

eval "$CMD"
