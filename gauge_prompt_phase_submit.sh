#!/bin/bash
# ==============================================================
# Prompt gauge validation SLURM array submission
#
# Matrix source: configs/gauge_experiment/matrix.yaml
# Runtime defaults: config/train_gauge.yaml
#
# Phase slicing:
#   phase1: first 2 envs, alpha=auto, first 3 seeds
#   phase2: all envs, all alpha modes, first 3 seeds
#   phase3: all envs, all alpha modes, all seeds
#
# Methods/gauges:
#   rrm   -> rrm_active_gauges from matrix.yaml
#   perfg -> pg_active_gauges from matrix.yaml, including no_tanh_*
#
# Usage:
#   bash gauge_prompt_phase1_submit.sh
#   bash gauge_prompt_phase2_submit.sh
#   bash gauge_prompt_phase3_submit.sh
#
# Useful overrides:
#   DRY_RUN=true bash gauge_prompt_phase1_submit.sh
#   MAX_CONCURRENT=8 bash gauge_prompt_phase1_submit.sh
#   TIME_LIMIT=36:00:00 bash gauge_prompt_phase3_submit.sh
#   VALIDATE=false bash gauge_prompt_phase1_submit.sh
#   EXTRA_OVERRIDES='num_train_steps=200000 max_feedback=400' bash gauge_prompt_phase1_submit.sh
# ==============================================================

set -euo pipefail

PHASE="${1:-${PHASE:-phase1}}"
MAX_CONCURRENT="${MAX_CONCURRENT:-all}"
TIME_LIMIT="${TIME_LIMIT:-12:00:00}"
ANALYSIS_TIME_LIMIT="${ANALYSIS_TIME_LIMIT:-01:00:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
ANALYSIS_CPUS_PER_TASK="${ANALYSIS_CPUS_PER_TASK:-2}"
PARTITION="${PARTITION:-dbrown-gpu-np}"
USE_WANDB="${USE_WANDB:-true}"
SKIP_DONE="${SKIP_DONE:-true}"
DONE_STEP="${DONE_STEP:-990000}"
STEP_WINDOW="${STEP_WINDOW:-200000}"
DRY_RUN="${DRY_RUN:-false}"
VALIDATE="${VALIDATE:-true}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"

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

DEFAULT_CHPC_SCRIPT_DIR="/uufs/chpc.utah.edu/common/home/u1520755/NoisyPbRL"
DEFAULT_CHPC_PYTHON="/uufs/chpc.utah.edu/common/home/dbrown-group1/zifan/miniconda3/envs/bpref/bin/python"
DEFAULT_CHPC_RESULTS_DIR="/uufs/chpc.utah.edu/common/home/dbrown-group1/zifan/NoisyPbRL/results/gauge_prompt"
DEFAULT_CHPC_REFERENCE_DIR="/uufs/chpc.utah.edu/common/home/dbrown-group1/zifan/NoisyPbRL/reference_dataset"
DEFAULT_CHPC_LOG_DIR="/uufs/chpc.utah.edu/common/home/dbrown-group1/zifan/logs/gauge_prompt"

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
    RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/results/gauge_prompt}"
    REFERENCE_DIR="${REFERENCE_DIR:-$SCRIPT_DIR/reference_dataset}"
    LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/logs/gauge_prompt}"
fi

case "$PHASE" in
    phase1|phase2|phase3) ;;
    *) echo "ERROR: PHASE must be phase1, phase2, or phase3; got '$PHASE'"; exit 1 ;;
esac

mkdir -p "$RESULTS_DIR/manifests" "$RESULTS_DIR/analysis" "$REFERENCE_DIR" "$LOG_DIR"

MANIFEST_JSON="$RESULTS_DIR/manifests/prompt_${PHASE}.json"
MANIFEST_CSV="$RESULTS_DIR/manifests/prompt_${PHASE}.csv"

N_CELLS="$("$PYTHON" - "$PHASE" "$SCRIPT_DIR" "$RESULTS_DIR" "$REFERENCE_DIR" "$USE_WANDB" "$EXTRA_OVERRIDES" "$MANIFEST_JSON" "$MANIFEST_CSV" <<'PY'
import csv
import json
import shlex
import sys
from pathlib import Path

import yaml

phase = sys.argv[1]
script_dir = Path(sys.argv[2])
results_dir = Path(sys.argv[3])
reference_dir = Path(sys.argv[4])
use_wandb = sys.argv[5]
extra_overrides = shlex.split(sys.argv[6]) if sys.argv[6] else []
manifest_json = Path(sys.argv[7])
manifest_csv = Path(sys.argv[8])

matrix_path = script_dir / "configs" / "gauge_experiment" / "matrix.yaml"
matrix = yaml.safe_load(open(matrix_path, "r", encoding="utf-8"))

envs = list(matrix["envs"])
alpha_modes = list(matrix["alpha_modes"])
seeds = [int(s) for s in matrix["seeds"]]

if phase == "phase1":
    phase_envs = envs[:2]
    phase_alpha_modes = ["auto"]
    phase_seeds = seeds[:3]
elif phase == "phase2":
    phase_envs = envs
    phase_alpha_modes = alpha_modes
    phase_seeds = seeds[:3]
elif phase == "phase3":
    phase_envs = envs
    phase_alpha_modes = alpha_modes
    phase_seeds = seeds
else:
    raise ValueError(phase)

methods = [
    ("rrm", list(matrix.get("rrm_active_gauges", ["none", "no_tanh"]))),
    ("perfg", list(matrix.get("pg_active_gauges", []))),
]

def agent_override(env_name):
    return "sac_metaworld" if "metaworld" in env_name else "sac"

cells = []
for env_name in phase_envs:
    for alpha_mode in phase_alpha_modes:
        for seed in phase_seeds:
            for method, gauges in methods:
                for gauge in gauges:
                    run_name = f"{env_name}__{method}__{gauge}__{alpha_mode}__seed{seed}"
                    run_dir = results_dir / "runs" / run_name
                    overrides = [
                        f"agent={agent_override(env_name)}",
                        f"env={env_name}",
                        f"seed={seed}",
                        f"gauge.method={method}",
                        f"gauge.active={gauge}",
                        f"gauge.alpha_mode={alpha_mode}",
                        f"gauge.reference_root={reference_dir}",
                        f"use_wandb={use_wandb}",
                        "gpu=0",
                        f"hydra.run.dir={run_dir}",
                    ]
                    overrides.extend(extra_overrides)
                    cells.append({
                        "phase": phase,
                        "env": env_name,
                        "method": method,
                        "gauge": gauge,
                        "alpha_mode": alpha_mode,
                        "seed": seed,
                        "run_name": run_name,
                        "run_dir": str(run_dir),
                        "command": " ".join(shlex.quote(x) for x in (["__PYTHON_BIN__", "-m", "pebble_gauge.train_gauge"] + overrides)),
                    })

manifest_json.parent.mkdir(parents=True, exist_ok=True)
json.dump(cells, open(manifest_json, "w", encoding="utf-8"), indent=2)
with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
    fieldnames = list(cells[0].keys()) if cells else ["phase", "env", "method", "gauge", "alpha_mode", "seed", "run_name", "run_dir", "command"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
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
    all|none|unlimited|''|0)
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

JOB_NAME="gauge_prompt_${PHASE}"
TMP_SCRIPT="$(mktemp "${SCRIPT_DIR}/tmp_${JOB_NAME}_XXXXXX.sh")"
ANALYSIS_JOB_NAME="gauge_prompt_${PHASE}_validate"
TMP_ANALYSIS_SCRIPT="$(mktemp "${SCRIPT_DIR}/tmp_${ANALYSIS_JOB_NAME}_XXXXXX.sh")"

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

mapfile -t CELL_INFO < <("__PYTHON__" - "__MANIFEST_JSON__" "${SLURM_ARRAY_TASK_ID}" <<'PY'
import json
import sys
manifest = json.load(open(sys.argv[1], "r", encoding="utf-8"))
cell = manifest[int(sys.argv[2])]
cmd = cell["command"].replace("__PYTHON_BIN__", "__PYTHON__")
print(cell["run_name"])
print(cell["run_dir"])
print(cmd)
PY
)

RUN_NAME="${CELL_INFO[0]}"
RUN_DIR="${CELL_INFO[1]}"
CMD="${CELL_INFO[2]}"

echo "=== Prompt gauge task ==="
echo "  phase    : __PHASE__"
echo "  array id : ${SLURM_ARRAY_TASK_ID}"
echo "  run      : ${RUN_NAME}"
echo "  run dir  : ${RUN_DIR}"
echo "  command  : ${CMD}"
echo ""

if [ "__SKIP_DONE__" = "true" ]; then
    LAST_STEP="$("__PYTHON__" - "${RUN_DIR}" <<'PY'
import csv
import os
import sys
run_dir = sys.argv[1]
last = -1
for name in ("eval.csv", "train.csv"):
    path = os.path.join(run_dir, name)
    if not os.path.exists(path):
        continue
    try:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("step"):
                    last = max(last, int(float(row["step"])))
    except Exception:
        pass
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

cat > "$TMP_ANALYSIS_SCRIPT" <<'EOT'
#!/bin/bash
#SBATCH --mem=8g
#SBATCH --cpus-per-task=__ANALYSIS_CPUS_PER_TASK__
#SBATCH --ntasks=1
#SBATCH --job-name=__ANALYSIS_JOB_NAME__
#SBATCH --time=__ANALYSIS_TIME_LIMIT__
__SLURM_QOS_LINE__
#SBATCH --partition=__SLURM_PARTITION__
#SBATCH --account=__SLURM_ACCOUNT__
__SLURM_EXCLUDE_LINE__
#SBATCH --output=__LOG_DIR__/__ANALYSIS_JOB_NAME___%j.out

set -euo pipefail
cd "__SCRIPT_DIR__"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-__ANALYSIS_CPUS_PER_TASK__}"

"__PYTHON__" -m pebble_gauge.validate_prompt_phase \
    --phase "__PHASE__" \
    --manifest "__MANIFEST_JSON__" \
    --root "__RESULTS_DIR__/runs" \
    --out "__RESULTS_DIR__/analysis/__PHASE__" \
    --done-step "__DONE_STEP__" \
    --step-window "__STEP_WINDOW__"
EOT

python_replace() {
    "$PYTHON" - "$1" "$2" "$3" <<'PY'
from pathlib import Path
import sys
path = Path(sys.argv[1])
old = sys.argv[2]
new = sys.argv[3]
path.write_text(path.read_text().replace(old, new))
PY
}

for script in "$TMP_SCRIPT" "$TMP_ANALYSIS_SCRIPT"; do
    python_replace "$script" "__CPUS_PER_TASK__" "$CPUS_PER_TASK"
    python_replace "$script" "__ANALYSIS_CPUS_PER_TASK__" "$ANALYSIS_CPUS_PER_TASK"
    python_replace "$script" "__JOB_NAME__" "$JOB_NAME"
    python_replace "$script" "__ANALYSIS_JOB_NAME__" "$ANALYSIS_JOB_NAME"
    python_replace "$script" "__TIME_LIMIT__" "$TIME_LIMIT"
    python_replace "$script" "__ANALYSIS_TIME_LIMIT__" "$ANALYSIS_TIME_LIMIT"
    python_replace "$script" "__SLURM_QOS_LINE__" "$SLURM_QOS_LINE"
    python_replace "$script" "__SLURM_PARTITION__" "$SLURM_PARTITION"
    python_replace "$script" "__SLURM_ACCOUNT__" "$SLURM_ACCOUNT"
    python_replace "$script" "__SLURM_EXCLUDE_LINE__" "$SLURM_EXCLUDE_LINE"
    python_replace "$script" "__LOG_DIR__" "$LOG_DIR"
    python_replace "$script" "__SCRIPT_DIR__" "$SCRIPT_DIR"
    python_replace "$script" "__PYTHON__" "$PYTHON"
    python_replace "$script" "__PHASE__" "$PHASE"
    python_replace "$script" "__RESULTS_DIR__" "$RESULTS_DIR"
    python_replace "$script" "__MANIFEST_JSON__" "$MANIFEST_JSON"
    python_replace "$script" "__SKIP_DONE__" "$SKIP_DONE"
    python_replace "$script" "__DONE_STEP__" "$DONE_STEP"
    python_replace "$script" "__STEP_WINDOW__" "$STEP_WINDOW"
    chmod +x "$script"
done

echo "=== Prompt gauge SLURM submission ==="
echo "  phase          : $PHASE"
echo "  cells          : $N_CELLS"
echo "  array spec     : $ARRAY_SPEC"
echo "  partition mode : $PARTITION"
echo "  script dir     : $SCRIPT_DIR"
echo "  python         : $PYTHON"
echo "  results dir    : $RESULTS_DIR"
echo "  reference dir  : $REFERENCE_DIR"
echo "  log dir        : $LOG_DIR"
echo "  manifest       : $MANIFEST_JSON"
echo "  validation     : $VALIDATE"
echo "  extra overrides: ${EXTRA_OVERRIDES:-<none>}"
echo ""

if [ "$DRY_RUN" = "true" ]; then
    echo "[DRY_RUN] Not submitting. Temporary scripts kept at:"
    echo "  $TMP_SCRIPT"
    echo "  $TMP_ANALYSIS_SCRIPT"
else
    ARRAY_JOB_ID="$(sbatch --parsable --array="${ARRAY_SPEC}" "$TMP_SCRIPT")"
    echo "Submitted training array: $ARRAY_JOB_ID"
    if [ "$VALIDATE" = "true" ]; then
        ANALYSIS_JOB_ID="$(sbatch --parsable --dependency=afterany:${ARRAY_JOB_ID} "$TMP_ANALYSIS_SCRIPT")"
        echo "Submitted validation job: $ANALYSIS_JOB_ID (afterany:$ARRAY_JOB_ID)"
        echo "Report will be written to: ${RESULTS_DIR}/analysis/${PHASE}/phase_report.md"
    else
        echo "Validation submission disabled. Run manually:"
        echo "  $PYTHON -m pebble_gauge.validate_prompt_phase --phase $PHASE --manifest $MANIFEST_JSON --root $RESULTS_DIR/runs --out $RESULTS_DIR/analysis/$PHASE --done-step $DONE_STEP --step-window $STEP_WINDOW"
    fi
fi
