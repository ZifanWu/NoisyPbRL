#!/bin/bash
# ==============================================================
# Axis 2 Tier B SLURM submission: README steps 2-4
#
# Step 2: full Axis 2 training matrix on Metaworld door-open
#   - base conditions: control, shift, epi, mis, seeds 1..6
#   - mis RM-capacity sweep: rm_hidden_dim in {8,16,32,64}, seeds 1..3
#     h=16 seed 1..3 duplicate the base mis runs and are skipped
#   - confirmatory mis runs: linear-head and mistake-teacher, seed 1
#
# Step 3: axis2_tier_b.analysis after training array
# Step 4: axis2_tier_b.payoff after training array, plus aggregate plot
#
# Usage:
#   bash axis2_steps2_4_submit.sh
#
# Useful overrides:
#   DRY_RUN=true bash axis2_steps2_4_submit.sh
#   MAX_CONCURRENT=8 bash axis2_steps2_4_submit.sh
#   TIME_LIMIT=36:00:00 bash axis2_steps2_4_submit.sh
#   AXIS2_ENV=metaworld_button-press-v2 bash axis2_steps2_4_submit.sh
#   RUN_ANALYSIS=false bash axis2_steps2_4_submit.sh
#   RUN_PAYOFF=false bash axis2_steps2_4_submit.sh
#   PAYOFF_SCOPE=base bash axis2_steps2_4_submit.sh
#   PAYOFF_SCOPE=all PAYOFF_MAX_CONCURRENT=4 bash axis2_steps2_4_submit.sh
#   EXTRA_OVERRIDES='num_train_steps=200000 monitor_frequency=20000' bash axis2_steps2_4_submit.sh
#
# Payoff scopes:
#   readme : only README Step 4 example, shift/seed1 checkpoint (default)
#   base   : base control/shift/epi/mis checkpoints for seeds 1..6
#   all    : every generated training checkpoint
#   none   : skip payoff submission
# ==============================================================

set -euo pipefail

MAX_CONCURRENT="${MAX_CONCURRENT:-all}"
PAYOFF_MAX_CONCURRENT="${PAYOFF_MAX_CONCURRENT:-1}"
TIME_LIMIT="${TIME_LIMIT:-36:00:00}"
ANALYSIS_TIME_LIMIT="${ANALYSIS_TIME_LIMIT:-01:00:00}"
PAYOFF_TIME_LIMIT="${PAYOFF_TIME_LIMIT:-06:00:00}"
PAYOFF_AGG_TIME_LIMIT="${PAYOFF_AGG_TIME_LIMIT:-00:30:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
ANALYSIS_CPUS_PER_TASK="${ANALYSIS_CPUS_PER_TASK:-2}"
PAYOFF_CPUS_PER_TASK="${PAYOFF_CPUS_PER_TASK:-4}"
PARTITION="${PARTITION:-dbrown-gpu-np}"
USE_WANDB="${USE_WANDB:-true}"
SKIP_DONE="${SKIP_DONE:-true}"
SKIP_PAYOFF_DONE="${SKIP_PAYOFF_DONE:-true}"
DONE_STEP="${DONE_STEP:-990000}"
CKPT_STEP="${CKPT_STEP:-500000}"
DRY_RUN="${DRY_RUN:-false}"
RUN_TRAIN="${RUN_TRAIN:-true}"
RUN_ANALYSIS="${RUN_ANALYSIS:-true}"
RUN_PAYOFF="${RUN_PAYOFF:-true}"
PAYOFF_SCOPE="${PAYOFF_SCOPE:-readme}"
DEPENDENCY_TYPE="${DEPENDENCY_TYPE:-afterany}"
AXIS2_ENV="${AXIS2_ENV:-metaworld_door-open-v2}"
SEEDS="${SEEDS:-1 2 3 4 5 6}"
MIS_SWEEP_SEEDS="${MIS_SWEEP_SEEDS:-1 2 3}"
MIS_WIDTHS="${MIS_WIDTHS:-8 16 32 64}"
CONFIRM_SEED="${CONFIRM_SEED:-1}"
INCLUDE_CONFIRMATORY="${INCLUDE_CONFIRMATORY:-true}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"
ANALYSIS_N_BOOT="${ANALYSIS_N_BOOT:-1000}"
PAYOFF_K_STEPS="${PAYOFF_K_STEPS:-20000}"
PAYOFF_REMEDY_LABELS="${PAYOFF_REMEDY_LABELS:-1000 5000}"

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

case "$PAYOFF_SCOPE" in
    readme|base|all|none) ;;
    *) echo "ERROR: PAYOFF_SCOPE must be one of: readme | base | all | none"; exit 1 ;;
esac

DEFAULT_CHPC_SCRIPT_DIR="/uufs/chpc.utah.edu/common/home/u1520755/NoisyPbRL"
DEFAULT_CHPC_PYTHON="/uufs/chpc.utah.edu/common/home/dbrown-group1/zifan/miniconda3/envs/bpref/bin/python"
DEFAULT_CHPC_RESULTS_DIR="/uufs/chpc.utah.edu/common/home/dbrown-group1/zifan/NoisyPbRL/exp/axis2_tier_b"
DEFAULT_CHPC_LOG_DIR="/uufs/chpc.utah.edu/common/home/dbrown-group1/zifan/logs/axis2_tier_b"

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
    LOG_DIR="${LOG_DIR:-$DEFAULT_CHPC_LOG_DIR}"
else
    RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/exp/axis2_tier_b}"
    LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/logs/axis2_tier_b}"
fi

MANIFEST_DIR="$RESULTS_DIR/manifests"
ANALYSIS_OUT_DIR="${ANALYSIS_OUT_DIR:-$RESULTS_DIR/results}"
PAYOFF_OUT_DIR="${PAYOFF_OUT_DIR:-$RESULTS_DIR/payoff}"
mkdir -p "$MANIFEST_DIR" "$ANALYSIS_OUT_DIR" "$PAYOFF_OUT_DIR" "$LOG_DIR"

TRAIN_MANIFEST_JSON="$MANIFEST_DIR/axis2_step2_train.json"
TRAIN_MANIFEST_CSV="$MANIFEST_DIR/axis2_step2_train.csv"
PAYOFF_MANIFEST_JSON="$MANIFEST_DIR/axis2_step4_payoff_${PAYOFF_SCOPE}.json"
PAYOFF_MANIFEST_CSV="$MANIFEST_DIR/axis2_step4_payoff_${PAYOFF_SCOPE}.csv"

read -r N_TRAIN N_PAYOFF < <("$PYTHON" - \
    "$AXIS2_ENV" "$SEEDS" "$MIS_SWEEP_SEEDS" "$MIS_WIDTHS" "$CONFIRM_SEED" \
    "$INCLUDE_CONFIRMATORY" "$RESULTS_DIR" "$PAYOFF_OUT_DIR" "$USE_WANDB" \
    "$EXTRA_OVERRIDES" "$CKPT_STEP" "$PAYOFF_SCOPE" \
    "$TRAIN_MANIFEST_JSON" "$TRAIN_MANIFEST_CSV" "$PAYOFF_MANIFEST_JSON" "$PAYOFF_MANIFEST_CSV" <<'PY'
import csv
import json
import shlex
import sys
from pathlib import Path

axis2_env = sys.argv[1]
seeds = [int(x) for x in shlex.split(sys.argv[2])]
mis_sweep_seeds = [int(x) for x in shlex.split(sys.argv[3])]
mis_widths = [int(x) for x in shlex.split(sys.argv[4])]
confirm_seed = int(sys.argv[5])
include_confirmatory = sys.argv[6].lower() == "true"
results_dir = Path(sys.argv[7])
payoff_out_dir = Path(sys.argv[8])
use_wandb = sys.argv[9]
extra_overrides = shlex.split(sys.argv[10]) if sys.argv[10] else []
ckpt_step = int(sys.argv[11])
payoff_scope = sys.argv[12]
train_manifest_json = Path(sys.argv[13])
train_manifest_csv = Path(sys.argv[14])
payoff_manifest_json = Path(sys.argv[15])
payoff_manifest_csv = Path(sys.argv[16])

def qjoin(parts):
    return " ".join(shlex.quote(str(x)) for x in parts)

cells = []
seen_run_dirs = set()

def add_cell(kind, condition, seed, config_name, run_dir, overrides, base=False):
    run_dir = Path(run_dir)
    key = str(run_dir)
    if key in seen_run_dirs:
        return
    seen_run_dirs.add(key)
    run_name = f"{axis2_env}__{kind}__seed{seed}"
    cli = [
        "__PYTHON_BIN__", "train_PEBBLE_axis2.py",
        f"env={axis2_env}",
        f"seed={seed}",
        f"use_wandb={use_wandb}",
        "gpu=0",
        f"hydra.run.dir={run_dir}",
    ]
    cli.extend(overrides)
    cli.extend(extra_overrides)
    cli.extend(["--config-name", config_name])
    cells.append({
        "kind": kind,
        "condition": condition,
        "env": axis2_env,
        "seed": seed,
        "config_name": config_name,
        "run_name": run_name,
        "run_dir": str(run_dir),
        "ckpt_dir": str(run_dir / f"ckpt_{ckpt_step}"),
        "command": qjoin(cli),
        "base": base,
    })

for seed in seeds:
    add_cell("control", "control", seed, "train_PEBBLE_axis2",
             results_dir / axis2_env / "control" / f"seed{seed}", [], base=True)
    add_cell("shift", "shift", seed, "train_PEBBLE_axis2_shift",
             results_dir / axis2_env / "shift" / f"seed{seed}", [], base=True)
    add_cell("epi", "epi", seed, "train_PEBBLE_axis2_epi",
             results_dir / axis2_env / "epi" / f"seed{seed}", [], base=True)
    add_cell("mis_h16_l1", "mis", seed, "train_PEBBLE_axis2_mis",
             results_dir / axis2_env / "mis_h16_l1" / f"seed{seed}", [], base=True)

for width in mis_widths:
    for seed in mis_sweep_seeds:
        add_cell(f"mis_h{width}_l1", "mis", seed, "train_PEBBLE_axis2_mis",
                 results_dir / axis2_env / f"mis_h{width}_l1" / f"seed{seed}",
                 [f"rm_hidden_dim={width}"], base=False)

if include_confirmatory:
    add_cell("mis_linear", "mis", confirm_seed, "train_PEBBLE_axis2_mis",
             results_dir / axis2_env / "mis_linear" / f"seed{confirm_seed}",
             ["rm_num_layers=0", "rm_output_activation=none"], base=False)
    add_cell("mis_mistake", "mis", confirm_seed, "train_PEBBLE_axis2_mis",
             results_dir / axis2_env / "mis_mistake" / f"seed{confirm_seed}",
             ["teacher_beta=-1", "teacher_eps_mistake=0.1", "bt_normalize_by_gap_std=false"],
             base=False)

train_manifest_json.parent.mkdir(parents=True, exist_ok=True)
json.dump(cells, open(train_manifest_json, "w", encoding="utf-8"), indent=2)
with open(train_manifest_csv, "w", newline="", encoding="utf-8") as f:
    fieldnames = list(cells[0].keys()) if cells else ["kind", "condition", "env", "seed", "config_name", "run_name", "run_dir", "ckpt_dir", "command", "base"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(cells)

if payoff_scope == "none":
    payoff_cells = []
elif payoff_scope == "readme":
    payoff_cells = [c for c in cells if c["kind"] == "shift" and c["seed"] == 1]
elif payoff_scope == "base":
    payoff_cells = [c for c in cells if c["base"]]
elif payoff_scope == "all":
    payoff_cells = list(cells)
else:
    raise ValueError(payoff_scope)

payoff_rows = []
for c in payoff_cells:
    out_dir = payoff_out_dir / "runs" / c["run_name"]
    payoff_rows.append({
        "run_name": c["run_name"],
        "kind": c["kind"],
        "condition": c["condition"],
        "env": c["env"],
        "seed": c["seed"],
        "ckpt_dir": c["ckpt_dir"],
        "out_dir": str(out_dir),
    })

json.dump(payoff_rows, open(payoff_manifest_json, "w", encoding="utf-8"), indent=2)
with open(payoff_manifest_csv, "w", newline="", encoding="utf-8") as f:
    fieldnames = list(payoff_rows[0].keys()) if payoff_rows else ["run_name", "kind", "condition", "env", "seed", "ckpt_dir", "out_dir"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(payoff_rows)

print(len(cells), len(payoff_rows))
PY
)

if [ "$N_TRAIN" -le 0 ]; then
    echo "ERROR: no Axis 2 training cells generated"
    exit 1
fi

array_spec() {
    local n="$1"
    local limit="$2"
    local max=$((n - 1))
    case "$limit" in
        all|none|unlimited|''|0)
            echo "0-${max}"
            ;;
        *[!0-9]*)
            echo "ERROR: concurrency must be a positive integer or one of: all | none | unlimited" >&2
            return 1
            ;;
        *)
            echo "0-${max}%${limit}"
            ;;
    esac
}

TRAIN_ARRAY_SPEC="$(array_spec "$N_TRAIN" "$MAX_CONCURRENT")"
if [ "$N_PAYOFF" -gt 0 ]; then
    PAYOFF_ARRAY_SPEC="$(array_spec "$N_PAYOFF" "$PAYOFF_MAX_CONCURRENT")"
else
    PAYOFF_ARRAY_SPEC=""
fi

TRAIN_JOB_NAME="axis2_step2"
ANALYSIS_JOB_NAME="axis2_step3_analysis"
PAYOFF_JOB_NAME="axis2_step4_payoff"
PAYOFF_AGG_JOB_NAME="axis2_step4_payoff_agg"
TMP_TRAIN_SCRIPT="$(mktemp "${SCRIPT_DIR}/tmp_${TRAIN_JOB_NAME}_XXXXXX.sh")"
TMP_ANALYSIS_SCRIPT="$(mktemp "${SCRIPT_DIR}/tmp_${ANALYSIS_JOB_NAME}_XXXXXX.sh")"
TMP_PAYOFF_SCRIPT="$(mktemp "${SCRIPT_DIR}/tmp_${PAYOFF_JOB_NAME}_XXXXXX.sh")"
TMP_PAYOFF_AGG_SCRIPT="$(mktemp "${SCRIPT_DIR}/tmp_${PAYOFF_AGG_JOB_NAME}_XXXXXX.sh")"

cat > "$TMP_TRAIN_SCRIPT" <<'EOT'
#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=20g
#SBATCH --cpus-per-task=__CPUS_PER_TASK__
#SBATCH --ntasks=1
#SBATCH --job-name=__TRAIN_JOB_NAME__
#SBATCH --time=__TIME_LIMIT__
__SLURM_QOS_LINE__
#SBATCH --partition=__SLURM_PARTITION__
#SBATCH --account=__SLURM_ACCOUNT__
__SLURM_EXCLUDE_LINE__
#SBATCH --output=__LOG_DIR__/__TRAIN_JOB_NAME___%A_%a.out

set -euo pipefail

module load cuda/12.4.0
cd "__SCRIPT_DIR__"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-__CPUS_PER_TASK__}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export LD_LIBRARY_PATH="${HOME}/.mujoco/mujoco210/bin:/usr/lib/nvidia:${LD_LIBRARY_PATH:-}"

mapfile -t CELL_INFO < <("__PYTHON__" - "__TRAIN_MANIFEST_JSON__" "${SLURM_ARRAY_TASK_ID}" <<'PY'
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

echo "=== Axis 2 Step 2 task ==="
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
path = os.path.join(sys.argv[1], "axis2_metrics.csv")
last = -1
if os.path.exists(path):
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
        echo "[SKIP] ${RUN_NAME} appears complete: last monitor step ${LAST_STEP}"
        exit 0
    fi
fi

eval "$CMD"
EOT

cat > "$TMP_ANALYSIS_SCRIPT" <<'EOT'
#!/bin/bash
#SBATCH --mem=12g
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

"__PYTHON__" -m axis2_tier_b.analysis \
    --results_dir "__RESULTS_DIR__" \
    --out_dir "__ANALYSIS_OUT_DIR__" \
    --n_boot "__ANALYSIS_N_BOOT__"
EOT

cat > "$TMP_PAYOFF_SCRIPT" <<'EOT'
#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=20g
#SBATCH --cpus-per-task=__PAYOFF_CPUS_PER_TASK__
#SBATCH --ntasks=1
#SBATCH --job-name=__PAYOFF_JOB_NAME__
#SBATCH --time=__PAYOFF_TIME_LIMIT__
__SLURM_QOS_LINE__
#SBATCH --partition=__SLURM_PARTITION__
#SBATCH --account=__SLURM_ACCOUNT__
__SLURM_EXCLUDE_LINE__
#SBATCH --output=__LOG_DIR__/__PAYOFF_JOB_NAME___%A_%a.out

set -euo pipefail

module load cuda/12.4.0
cd "__SCRIPT_DIR__"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-__PAYOFF_CPUS_PER_TASK__}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export LD_LIBRARY_PATH="${HOME}/.mujoco/mujoco210/bin:/usr/lib/nvidia:${LD_LIBRARY_PATH:-}"

mapfile -t CELL_INFO < <("__PYTHON__" - "__PAYOFF_MANIFEST_JSON__" "${SLURM_ARRAY_TASK_ID}" <<'PY'
import json
import sys
manifest = json.load(open(sys.argv[1], "r", encoding="utf-8"))
cell = manifest[int(sys.argv[2])]
print(cell["run_name"])
print(cell["ckpt_dir"])
print(cell["out_dir"])
PY
)

RUN_NAME="${CELL_INFO[0]}"
CKPT_DIR="${CELL_INFO[1]}"
OUT_DIR="${CELL_INFO[2]}"

echo "=== Axis 2 Step 4 payoff task ==="
echo "  array id : ${SLURM_ARRAY_TASK_ID}"
echo "  run      : ${RUN_NAME}"
echo "  ckpt dir : ${CKPT_DIR}"
echo "  out dir  : ${OUT_DIR}"
echo ""

if [ ! -d "$CKPT_DIR" ]; then
    echo "ERROR: checkpoint directory not found: ${CKPT_DIR}"
    exit 1
fi

if [ "__SKIP_PAYOFF_DONE__" = "true" ] && [ -f "${OUT_DIR}/payoff_results.json" ]; then
    echo "[SKIP] payoff already exists: ${OUT_DIR}/payoff_results.json"
    exit 0
fi

"__PYTHON__" -m axis2_tier_b.payoff \
    --ckpt_dir "$CKPT_DIR" \
    --k_steps "__PAYOFF_K_STEPS__" \
    --n_remedy_labels __PAYOFF_REMEDY_LABELS__ \
    --out_dir "$OUT_DIR"
EOT

cat > "$TMP_PAYOFF_AGG_SCRIPT" <<'EOT'
#!/bin/bash
#SBATCH --mem=8g
#SBATCH --cpus-per-task=__ANALYSIS_CPUS_PER_TASK__
#SBATCH --ntasks=1
#SBATCH --job-name=__PAYOFF_AGG_JOB_NAME__
#SBATCH --time=__PAYOFF_AGG_TIME_LIMIT__
__SLURM_QOS_LINE__
#SBATCH --partition=__SLURM_PARTITION__
#SBATCH --account=__SLURM_ACCOUNT__
__SLURM_EXCLUDE_LINE__
#SBATCH --output=__LOG_DIR__/__PAYOFF_AGG_JOB_NAME___%j.out

set -euo pipefail
cd "__SCRIPT_DIR__"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-__ANALYSIS_CPUS_PER_TASK__}"

"__PYTHON__" -m axis2_tier_b.payoff \
    --aggregate \
    --ckpt_dir "__PAYOFF_OUT_DIR__/runs" \
    --out_dir "__PAYOFF_OUT_DIR__/aggregate"
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

for script in "$TMP_TRAIN_SCRIPT" "$TMP_ANALYSIS_SCRIPT" "$TMP_PAYOFF_SCRIPT" "$TMP_PAYOFF_AGG_SCRIPT"; do
    python_replace "$script" "__CPUS_PER_TASK__" "$CPUS_PER_TASK"
    python_replace "$script" "__ANALYSIS_CPUS_PER_TASK__" "$ANALYSIS_CPUS_PER_TASK"
    python_replace "$script" "__PAYOFF_CPUS_PER_TASK__" "$PAYOFF_CPUS_PER_TASK"
    python_replace "$script" "__TRAIN_JOB_NAME__" "$TRAIN_JOB_NAME"
    python_replace "$script" "__ANALYSIS_JOB_NAME__" "$ANALYSIS_JOB_NAME"
    python_replace "$script" "__PAYOFF_JOB_NAME__" "$PAYOFF_JOB_NAME"
    python_replace "$script" "__PAYOFF_AGG_JOB_NAME__" "$PAYOFF_AGG_JOB_NAME"
    python_replace "$script" "__TIME_LIMIT__" "$TIME_LIMIT"
    python_replace "$script" "__ANALYSIS_TIME_LIMIT__" "$ANALYSIS_TIME_LIMIT"
    python_replace "$script" "__PAYOFF_TIME_LIMIT__" "$PAYOFF_TIME_LIMIT"
    python_replace "$script" "__PAYOFF_AGG_TIME_LIMIT__" "$PAYOFF_AGG_TIME_LIMIT"
    python_replace "$script" "__SLURM_QOS_LINE__" "$SLURM_QOS_LINE"
    python_replace "$script" "__SLURM_PARTITION__" "$SLURM_PARTITION"
    python_replace "$script" "__SLURM_ACCOUNT__" "$SLURM_ACCOUNT"
    python_replace "$script" "__SLURM_EXCLUDE_LINE__" "$SLURM_EXCLUDE_LINE"
    python_replace "$script" "__LOG_DIR__" "$LOG_DIR"
    python_replace "$script" "__SCRIPT_DIR__" "$SCRIPT_DIR"
    python_replace "$script" "__PYTHON__" "$PYTHON"
    python_replace "$script" "__RESULTS_DIR__" "$RESULTS_DIR"
    python_replace "$script" "__ANALYSIS_OUT_DIR__" "$ANALYSIS_OUT_DIR"
    python_replace "$script" "__PAYOFF_OUT_DIR__" "$PAYOFF_OUT_DIR"
    python_replace "$script" "__TRAIN_MANIFEST_JSON__" "$TRAIN_MANIFEST_JSON"
    python_replace "$script" "__PAYOFF_MANIFEST_JSON__" "$PAYOFF_MANIFEST_JSON"
    python_replace "$script" "__SKIP_DONE__" "$SKIP_DONE"
    python_replace "$script" "__SKIP_PAYOFF_DONE__" "$SKIP_PAYOFF_DONE"
    python_replace "$script" "__DONE_STEP__" "$DONE_STEP"
    python_replace "$script" "__ANALYSIS_N_BOOT__" "$ANALYSIS_N_BOOT"
    python_replace "$script" "__PAYOFF_K_STEPS__" "$PAYOFF_K_STEPS"
    python_replace "$script" "__PAYOFF_REMEDY_LABELS__" "$PAYOFF_REMEDY_LABELS"
    chmod +x "$script"
done

echo "=== Axis 2 steps 2-4 SLURM submission ==="
echo "  env              : $AXIS2_ENV"
echo "  train cells      : $N_TRAIN"
echo "  train array spec : $TRAIN_ARRAY_SPEC"
echo "  payoff scope     : $PAYOFF_SCOPE"
echo "  payoff cells     : $N_PAYOFF"
echo "  payoff array spec: ${PAYOFF_ARRAY_SPEC:-<none>}"
echo "  partition mode   : $PARTITION"
echo "  script dir       : $SCRIPT_DIR"
echo "  python           : $PYTHON"
echo "  results dir      : $RESULTS_DIR"
echo "  analysis out     : $ANALYSIS_OUT_DIR"
echo "  payoff out       : $PAYOFF_OUT_DIR"
echo "  log dir          : $LOG_DIR"
echo "  train manifest   : $TRAIN_MANIFEST_JSON"
echo "  payoff manifest  : $PAYOFF_MANIFEST_JSON"
echo "  run train        : $RUN_TRAIN"
echo "  run analysis     : $RUN_ANALYSIS"
echo "  run payoff       : $RUN_PAYOFF"
echo "  extra overrides  : ${EXTRA_OVERRIDES:-<none>}"
echo ""

if [ "$DRY_RUN" = "true" ]; then
    echo "[DRY_RUN] Not submitting. Temporary scripts kept at:"
    echo "  $TMP_TRAIN_SCRIPT"
    echo "  $TMP_ANALYSIS_SCRIPT"
    echo "  $TMP_PAYOFF_SCRIPT"
    echo "  $TMP_PAYOFF_AGG_SCRIPT"
    exit 0
fi

TRAIN_JOB_ID=""
if [ "$RUN_TRAIN" = "true" ]; then
    TRAIN_JOB_ID="$(sbatch --parsable --array="${TRAIN_ARRAY_SPEC}" "$TMP_TRAIN_SCRIPT")"
    echo "Submitted Step 2 training array: $TRAIN_JOB_ID"
else
    echo "Training submission disabled. Temporary script: $TMP_TRAIN_SCRIPT"
fi

analysis_dependency_args=()
payoff_dependency_args=()
if [ -n "$TRAIN_JOB_ID" ]; then
    analysis_dependency_args=(--dependency=${DEPENDENCY_TYPE}:${TRAIN_JOB_ID})
    payoff_dependency_args=(--dependency=${DEPENDENCY_TYPE}:${TRAIN_JOB_ID})
fi

if [ "$RUN_ANALYSIS" = "true" ]; then
    ANALYSIS_JOB_ID="$(sbatch --parsable "${analysis_dependency_args[@]}" "$TMP_ANALYSIS_SCRIPT")"
    echo "Submitted Step 3 analysis job: $ANALYSIS_JOB_ID"
    echo "Analysis report will be written to: ${ANALYSIS_OUT_DIR}/results.md"
else
    echo "Analysis submission disabled. Run manually:"
    echo "  $PYTHON -m axis2_tier_b.analysis --results_dir $RESULTS_DIR --out_dir $ANALYSIS_OUT_DIR --n_boot $ANALYSIS_N_BOOT"
fi

if [ "$RUN_PAYOFF" = "true" ] && [ "$PAYOFF_SCOPE" != "none" ] && [ "$N_PAYOFF" -gt 0 ]; then
    PAYOFF_JOB_ID="$(sbatch --parsable "${payoff_dependency_args[@]}" --array="${PAYOFF_ARRAY_SPEC}" "$TMP_PAYOFF_SCRIPT")"
    echo "Submitted Step 4 payoff array: $PAYOFF_JOB_ID"
    PAYOFF_AGG_JOB_ID="$(sbatch --parsable --dependency=${DEPENDENCY_TYPE}:${PAYOFF_JOB_ID} "$TMP_PAYOFF_AGG_SCRIPT")"
    echo "Submitted Step 4 payoff aggregate job: $PAYOFF_AGG_JOB_ID"
    echo "Payoff aggregate plots will be written to: ${PAYOFF_OUT_DIR}/aggregate"
else
    echo "Payoff submission disabled. Run a single checkpoint manually, e.g.:"
    echo "  $PYTHON -m axis2_tier_b.payoff --ckpt_dir $RESULTS_DIR/$AXIS2_ENV/shift/seed1/ckpt_$CKPT_STEP --out_dir $PAYOFF_OUT_DIR/runs/${AXIS2_ENV}__shift__seed1"
fi
