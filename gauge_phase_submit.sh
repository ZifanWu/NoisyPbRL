#!/bin/bash
# ==============================================================
# Gauge PbRL SLURM array submission
#
# One array task = one complete run from configs/gauge_experiment/{phase}.yaml.
# By default there is no Slurm array throttle; the scheduler can run as
# many tasks as partition/account/GPU availability allows. Set MAX_CONCURRENT
# to a positive integer to add an explicit array throttle.
#
# Usage:
#   bash gauge_phase1_submit.sh
#   bash gauge_phase2_submit.sh
#   bash gauge_phase3_submit.sh
#
# Useful overrides:
#   MAX_CONCURRENT=8 bash gauge_phase1_submit.sh
#   MAX_CONCURRENT=all bash gauge_phase1_submit.sh
#   DRY_RUN=true bash gauge_phase1_submit.sh
#   SKIP_DONE=false bash gauge_phase1_submit.sh
#   TIME_LIMIT=36:00:00 bash gauge_phase3_submit.sh
#   PARTITION=dbrown-gpu-grn bash gauge_phase1_submit.sh
#
# PARTITION selects one of three cluster/account modes:
#   dbrown-gpu-grn -> qos=dbrown-gpu-grn, partition=dbrown-gpu-grn, account=dbrown
#   dbrown-gpu-np  -> no qos, partition=dbrown-gpu-np, account=dbrown-gpu-np
#   soc-gpu-np     -> no qos, partition=soc-gpu-np, account=soc-gpu-np, exclude nodes
# ==============================================================

set -euo pipefail

PHASE="${1:-${PHASE:-phase1}}"
MAX_CONCURRENT="${MAX_CONCURRENT:-all}"
TIME_LIMIT="${TIME_LIMIT:-12:00:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
PARTITION="${PARTITION:-dbrown-gpu-np}"
case "$PARTITION" in
    dbrown-gpu-grn)
        SLURM_QOS_LINE="#SBATCH --qos=dbrown-gpu-grn"
        SLURM_PARTITION="dbrown-gpu-grn"
        SLURM_ACCOUNT="dbrown"
        SLURM_EXCLUDE_LINE="##SBATCH --exclude=${EXCLUDE_NODES:-notch372,notch369,notch475,notch371}"
        ;;
    dbrown-gpu-np)
        SLURM_QOS_LINE="##SBATCH --qos=dbrown-gpu-grn"
        SLURM_PARTITION="dbrown-gpu-np"
        SLURM_ACCOUNT="dbrown-gpu-np"
        SLURM_EXCLUDE_LINE="##SBATCH --exclude=${EXCLUDE_NODES:-notch372,notch369,notch475,notch371}"
        ;;
    soc-gpu-np)
        SLURM_QOS_LINE="##SBATCH --qos=dbrown-gpu-grn"
        SLURM_PARTITION="soc-gpu-np"
        SLURM_ACCOUNT="soc-gpu-np"
        SLURM_EXCLUDE_LINE="#SBATCH --exclude=${EXCLUDE_NODES:-notch372,notch369,notch475,notch371}"
        ;;
    *)
        echo "ERROR: PARTITION must be one of: dbrown-gpu-grn | dbrown-gpu-np | soc-gpu-np"
        exit 1
        ;;
esac
USE_WANDB="${USE_WANDB:-true}"
SKIP_DONE="${SKIP_DONE:-true}"
DONE_STEP="${DONE_STEP:-990000}"
DRY_RUN="${DRY_RUN:-false}"

# Prefer the CHPC paths used by gauge_ambiguity_existence.sh, but fall back to
# the current checkout so this script can be dry-run on 10.18.135.54.
DEFAULT_CHPC_SCRIPT_DIR="/uufs/chpc.utah.edu/common/home/u1520755/NoisyPbRL"
DEFAULT_CHPC_PYTHON="/uufs/chpc.utah.edu/common/home/dbrown-group1/zifan/miniconda3/envs/bpref/bin/python"
DEFAULT_CHPC_RESULTS_DIR="/uufs/chpc.utah.edu/common/home/dbrown-group1/zifan/NoisyPbRL/results"
DEFAULT_CHPC_REFERENCE_DIR="/uufs/chpc.utah.edu/common/home/dbrown-group1/zifan/NoisyPbRL/reference_dataset"
DEFAULT_CHPC_LOG_DIR="/uufs/chpc.utah.edu/common/home/dbrown-group1/zifan/logs/gauge_pbrl"

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -d "$DEFAULT_CHPC_SCRIPT_DIR" ]; then
    SCRIPT_DIR="${SCRIPT_DIR:-$DEFAULT_CHPC_SCRIPT_DIR}"
else
    SCRIPT_DIR="${SCRIPT_DIR:-$THIS_DIR}"
fi

if [ -x "$DEFAULT_CHPC_PYTHON" ]; then
    PYTHON="${PYTHON:-$DEFAULT_CHPC_PYTHON}"
else
    PYTHON="${PYTHON:-$HOME/miniconda3/envs/bpref/bin/python}"
fi

if [ -d "$(dirname "$DEFAULT_CHPC_RESULTS_DIR")" ]; then
    RESULTS_DIR="${RESULTS_DIR:-$DEFAULT_CHPC_RESULTS_DIR}"
    REFERENCE_DIR="${REFERENCE_DIR:-$DEFAULT_CHPC_REFERENCE_DIR}"
    LOG_DIR="${LOG_DIR:-$DEFAULT_CHPC_LOG_DIR}"
else
    RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/results}"
    REFERENCE_DIR="${REFERENCE_DIR:-$SCRIPT_DIR/reference_dataset}"
    LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/logs/gauge_pbrl}"
fi

mkdir -p "$RESULTS_DIR/manifests" "$REFERENCE_DIR" "$LOG_DIR"

case "$PHASE" in
    phase1|phase2|phase3) ;;
    *) echo "ERROR: PHASE must be phase1, phase2, or phase3; got '$PHASE'"; exit 1 ;;
esac

echo "=== Gauge PbRL SLURM submission ==="
echo "  phase          : $PHASE"
echo "  max concurrent : $MAX_CONCURRENT"
echo "  time limit     : $TIME_LIMIT"
echo "  partition mode : $PARTITION"
echo "  slurm partition: $SLURM_PARTITION"
echo "  slurm account  : $SLURM_ACCOUNT"
echo "  slurm qos line : $SLURM_QOS_LINE"
echo "  slurm exclude  : $SLURM_EXCLUDE_LINE"
echo "  script dir     : $SCRIPT_DIR"
echo "  python         : $PYTHON"
echo "  results dir    : $RESULTS_DIR"
echo "  reference dir  : $REFERENCE_DIR"
echo "  log dir        : $LOG_DIR"
echo "  use_wandb      : $USE_WANDB"
echo "  skip done      : $SKIP_DONE (DONE_STEP=$DONE_STEP)"
echo ""

N_CELLS="$("$PYTHON" - "$PHASE" "$SCRIPT_DIR" "$RESULTS_DIR" <<'PY'
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

import yaml

phase = sys.argv[1]
script_dir = Path(sys.argv[2])
results_dir = Path(sys.argv[3])
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
                    run_name = f"{env_key}__{method}__{gauge}__{alpha_mode}__seed{int(seed)}"
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
                        "run_name": run_name,
                    })

manifest_json = results_dir / "manifests" / f"manifest_{phase}.json"
manifest_csv = results_dir / "manifests" / f"manifest_{phase}.csv"
manifest_json.parent.mkdir(parents=True, exist_ok=True)
json.dump(cells, open(manifest_json, "w", encoding="utf-8"), indent=2)
with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(cells[0].keys()) if cells else [])
    if cells:
        writer.writeheader()
        writer.writerows(cells)

print(len(cells))
PY
)"

if [ "$N_CELLS" -le 0 ]; then
    echo "ERROR: no cells generated for $PHASE"
    exit 1
fi

ARRAY_MAX=$((N_CELLS - 1))
case "$MAX_CONCURRENT" in
    all|none|unlimited)
        ARRAY_SPEC="0-${ARRAY_MAX}"
        ;;
    ''|0)
        ARRAY_SPEC="0-${ARRAY_MAX}"
        ;;
    *[!0-9]*)
        echo "ERROR: MAX_CONCURRENT must be a positive integer or one of: all | none | unlimited"
        exit 1
        ;;
    *)
        ARRAY_SPEC="0-${ARRAY_MAX}%${MAX_CONCURRENT}"
        ;;
esac
JOB_NAME="gauge_${PHASE}"
TMP_SCRIPT="$(mktemp "${SCRIPT_DIR}/tmp_${JOB_NAME}_XXXXXX.sh")"

cat > "$TMP_SCRIPT" <<'EOT'
#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=20g
#SBATCH --cpus-per-task=__CPUS_PER_TASK__
#SBATCH --ntasks=1
#SBATCH --job-name=__JOB_NAME__
#SBATCH --time=__TIME_LIMIT__
__SLURM_QOS_LINE__
#SBATCH --partition=__SLURM_PARTITION__
#SBATCH --account=__SLURM_ACCOUNT__
__SLURM_EXCLUDE_LINE__
#SBATCH --output=__LOG_DIR__/__JOB_NAME___%A_%a.out

set -euo pipefail

module load cuda/12.4.0
cd "__SCRIPT_DIR__"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-__CPUS_PER_TASK__}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

mapfile -t CELL_INFO < <("__PYTHON__" - "__PHASE__" "${SLURM_ARRAY_TASK_ID}" "__SCRIPT_DIR__" "__RESULTS_DIR__" "__REFERENCE_DIR__" "__USE_WANDB__" <<'PY'
import shlex
import sys
from pathlib import Path

import yaml

python = "__PYTHON__"
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
echo "  phase    : __PHASE__"
echo "  array id : ${SLURM_ARRAY_TASK_ID}"
echo "  run      : ${RUN_NAME}"
echo "  run dir  : ${RUN_DIR}"
echo "  command  : ${CMD}"
echo ""

if [ "__SKIP_DONE__" = "true" ] && [ -f "${RUN_DIR}/train.csv" ]; then
    LAST_STEP="$("__PYTHON__" - "${RUN_DIR}/train.csv" <<'PY'
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
    if [ "$LAST_STEP" -ge "__DONE_STEP__" ]; then
        echo "[SKIP] ${RUN_NAME} appears complete: last step ${LAST_STEP}"
        exit 0
    fi
fi

eval "$CMD"
EOT

python_replace() {
    "$PYTHON" - "$TMP_SCRIPT" "$1" "$2" <<'PY'
from pathlib import Path
import sys
path = Path(sys.argv[1])
old = sys.argv[2]
new = sys.argv[3]
text = path.read_text()
path.write_text(text.replace(old, new))
PY
}

python_replace "__CPUS_PER_TASK__" "$CPUS_PER_TASK"
python_replace "__JOB_NAME__" "$JOB_NAME"
python_replace "__TIME_LIMIT__" "$TIME_LIMIT"
python_replace "__SLURM_QOS_LINE__" "$SLURM_QOS_LINE"
python_replace "__SLURM_PARTITION__" "$SLURM_PARTITION"
python_replace "__SLURM_ACCOUNT__" "$SLURM_ACCOUNT"
python_replace "__SLURM_EXCLUDE_LINE__" "$SLURM_EXCLUDE_LINE"
python_replace "__LOG_DIR__" "$LOG_DIR"
python_replace "__SCRIPT_DIR__" "$SCRIPT_DIR"
python_replace "__PYTHON__" "$PYTHON"
python_replace "__PHASE__" "$PHASE"
python_replace "__RESULTS_DIR__" "$RESULTS_DIR"
python_replace "__REFERENCE_DIR__" "$REFERENCE_DIR"
python_replace "__USE_WANDB__" "$USE_WANDB"
python_replace "__SKIP_DONE__" "$SKIP_DONE"
python_replace "__DONE_STEP__" "$DONE_STEP"

chmod +x "$TMP_SCRIPT"

echo "Generated manifest:"
echo "  ${RESULTS_DIR}/manifests/manifest_${PHASE}.json"
echo "Submitting array:"
echo "  sbatch --array=${ARRAY_SPEC} ${TMP_SCRIPT}"
echo ""

if [ "$DRY_RUN" = "true" ]; then
    echo "[DRY_RUN] Not submitting. Temporary Slurm script kept at:"
    echo "  $TMP_SCRIPT"
else
    sbatch --array="${ARRAY_SPEC}" "$TMP_SCRIPT"
    echo "Submitted ${N_CELLS} tasks for ${PHASE}; array spec = ${ARRAY_SPEC}."
    echo "Temporary Slurm script: $TMP_SCRIPT"
fi
