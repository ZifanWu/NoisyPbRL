#!/bin/bash
# ==============================================================
# perf_diag SLURM submission: train array + analysis
#
# Cells: (task × RM capacity × P_relabel regime × seed), with per-task
# max_feedback from perf_diag.run.TASK_DEFAULTS (matches the authors' scripts/).
#
# Step 1: training array (one cell = one full PEBBLE run with PD_ENABLE=1).
# Step 2: perf_diag.analysis after the training array completes.
#
# Usage:
#   bash perf_diag/submit.sh
#
# Useful overrides / filters:
#   DRY_RUN=true bash perf_diag/submit.sh
#   MAX_CONCURRENT=8 bash perf_diag/submit.sh
#   TIME_LIMIT=36:00:00 bash perf_diag/submit.sh
#   ENVS='metaworld_drawer-open-v2 metaworld_door-close-v2' bash perf_diag/submit.sh
#   SEEDS='0 1 2 3 4 5' bash perf_diag/submit.sh
#   CAPACITIES='full_rm small_rm' REGIMES='rare_relabel frequent_relabel' bash perf_diag/submit.sh
#   NUM_TRAIN_STEPS=200000 bash perf_diag/submit.sh
#   PD_N_PROBE=4 PD_K_REFIT=3 bash perf_diag/submit.sh
#   PD_STEP_PROBE_EVERY=10000 bash perf_diag/submit.sh
#   RUN_ANALYSIS=false bash perf_diag/submit.sh
#   PARTITION=soc-gpu-np bash perf_diag/submit.sh
#   EXTRA_OVERRIDES='reward_update=200 segment=50' bash perf_diag/submit.sh
# ==============================================================

set -euo pipefail

MAX_CONCURRENT="${MAX_CONCURRENT:-all}"
TIME_LIMIT="${TIME_LIMIT:-12:00:00}"
ANALYSIS_TIME_LIMIT="${ANALYSIS_TIME_LIMIT:-01:00:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
ANALYSIS_CPUS_PER_TASK="${ANALYSIS_CPUS_PER_TASK:-2}"
# Train-cell memory. With UNLIMITED_BUDGET_VAL=100K + large_batch=10 the RewardModel
# buffer alone is ~18 GB, so we default to 32 GB. Override for smaller envs if needed.
TRAIN_MEM="${TRAIN_MEM:-20g}"
PARTITION="${PARTITION:-dbrown-gpu-np}"
USE_WANDB="${USE_WANDB:-true}"
SKIP_DONE="${SKIP_DONE:-true}"
# Unlimited budget: when true (default), override per-task max_feedback with a huge
# sentinel so the regime sweep measures relabeling frequency in isolation from
# budget exhaustion. Set to false to use original B-Pref budgets from TASK_DEFAULTS.
UNLIMITED_BUDGET="${UNLIMITED_BUDGET:-true}"
# Sentinel max_feedback used when UNLIMITED_BUDGET=true. Sized to be "effectively unlimited"
# for our sweep configs (frequent_relabel at 1M steps needs ≤50K labels) but bounded so the
# RewardModel buffer fits in memory: buffer ≈ max_feedback × large_batch × segment × (ds+da) × 4 bytes.
# At max_feedback=100K, large_batch=10, segment=50, ds+da≈90: ~18 GB. Pair with --mem=32g.
UNLIMITED_BUDGET_VAL="${UNLIMITED_BUDGET_VAL:-100000}"
DRY_RUN="${DRY_RUN:-false}"
RUN_TRAIN="${RUN_TRAIN:-true}"
RUN_ANALYSIS="${RUN_ANALYSIS:-true}"
DEPENDENCY_TYPE="${DEPENDENCY_TYPE:-afterany}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_${RANDOM}}"
# Default false: clean reports should only aggregate this submission. Set
# MERGE_PREVIOUS=true only when intentionally combining old JSONLs.
MERGE_PREVIOUS="${MERGE_PREVIOUS:-false}"

# Per-cell training budget (matches FullConfig in perf_diag/run.py)
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-1000000}"
NUM_SEED_STEPS="${NUM_SEED_STEPS:-1000}"
NUM_UNSUP_STEPS="${NUM_UNSUP_STEPS:-9000}"
EVAL_FREQUENCY="${EVAL_FREQUENCY:-10000}"
SEGMENT="${SEGMENT:-50}"
ENSEMBLE_SIZE="${ENSEMBLE_SIZE:-5}"
TEACHER_EPS_MISTAKE="${TEACHER_EPS_MISTAKE:-0.1}"
TEACHER_BETA="${TEACHER_BETA:--1}"
REWARD_UPDATE="${REWARD_UPDATE:-200}"

# Probe knobs (env vars consumed inside train_PEBBLE.py via perf_diag.hooks)
PD_N_PROBE="${PD_N_PROBE:-8}"
PD_SEGMENT_LEN="${PD_SEGMENT_LEN:-50}"
PD_REFIT_PAIRS="${PD_REFIT_PAIRS:-16}"
PD_K_REFIT="${PD_K_REFIT:-5}"
PD_REFIT_LR="${PD_REFIT_LR:-0.0003}"
PD_PROBE_M="${PD_PROBE_M:-1}"
PD_STEP_PROBE_EVERY="${PD_STEP_PROBE_EVERY:-10000}"

# Filters (space-separated; empty = use defaults)
ENVS="${ENVS:-walker_walk}"
SEEDS="${SEEDS:-0 1 2 3 4 5}"
CAPACITIES="${CAPACITIES:-full_rm small_rm}"
REGIMES="${REGIMES:-rare_relabel frequent_relabel}"
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

# Path resolution: prefer CHPC layout if it exists, else local.
DEFAULT_CHPC_SCRIPT_DIR="/uufs/chpc.utah.edu/common/home/u1520755/NoisyPbRL"
DEFAULT_CHPC_PYTHON="/uufs/chpc.utah.edu/common/home/dbrown-group1/zifan/miniconda3/envs/bpref/bin/python"
DEFAULT_CHPC_RESULTS_DIR="/uufs/chpc.utah.edu/common/home/dbrown-group1/zifan/NoisyPbRL/exp/perf_diag"
DEFAULT_CHPC_LOG_DIR="/uufs/chpc.utah.edu/common/home/dbrown-group1/zifan/logs/perf_diag"

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${THIS_DIR}/.." && pwd)"
if [ -d "$DEFAULT_CHPC_SCRIPT_DIR" ]; then
    SCRIPT_DIR="${SCRIPT_DIR:-$DEFAULT_CHPC_SCRIPT_DIR}"
else
    SCRIPT_DIR="${SCRIPT_DIR:-$REPO_DIR}"
fi

if [ -x "$DEFAULT_CHPC_PYTHON" ]; then
    PYTHON="${PYTHON:-$DEFAULT_CHPC_PYTHON}"
else
    PYTHON="${PYTHON:-$HOME/miniconda3/envs/bpref/bin/python}"
fi

if [ -d "$DEFAULT_CHPC_SCRIPT_DIR" ]; then
    RESULTS_DIR="${RESULTS_DIR:-$DEFAULT_CHPC_RESULTS_DIR}"
    LOG_DIR="${LOG_DIR:-$DEFAULT_CHPC_LOG_DIR}"
else
    RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/exp/perf_diag}"
    LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/logs/perf_diag}"
fi

# Output of the probe JSONL traces (and the final figs/tables/results.md).
PD_OUT_DIR="${PD_OUT_DIR:-$SCRIPT_DIR/perf_diag/runs}"

MANIFEST_DIR="$RESULTS_DIR/manifests"
mkdir -p "$MANIFEST_DIR" "$LOG_DIR" "$PD_OUT_DIR"

TRAIN_MANIFEST_JSON="$MANIFEST_DIR/perf_diag_train_${RUN_ID}.json"
TRAIN_MANIFEST_CSV="$MANIFEST_DIR/perf_diag_train_${RUN_ID}.csv"
REPORT_DIR="${REPORT_DIR:-$SCRIPT_DIR/perf_diag/reports/$RUN_ID}"

# ==============================================================
# Build the per-cell training manifest.
# Cells come from (ENVS × CAPACITIES × REGIMES × SEEDS); per-task
# max_feedback + central P_relabel are looked up from perf_diag.run.TASK_DEFAULTS.
# Each cell's command is a full hydra train_PEBBLE.py invocation.
# ==============================================================
N_TRAIN="$("$PYTHON" - \
    "$REPO_DIR" "$RESULTS_DIR" "$PD_OUT_DIR" "$USE_WANDB" "$EXTRA_OVERRIDES" \
    "$ENVS" "$SEEDS" "$CAPACITIES" "$REGIMES" \
    "$NUM_TRAIN_STEPS" "$NUM_SEED_STEPS" "$NUM_UNSUP_STEPS" "$EVAL_FREQUENCY" \
    "$SEGMENT" "$ENSEMBLE_SIZE" "$TEACHER_EPS_MISTAKE" "$TEACHER_BETA" "$REWARD_UPDATE" \
    "$PD_N_PROBE" "$PD_SEGMENT_LEN" "$PD_REFIT_PAIRS" "$PD_K_REFIT" "$PD_REFIT_LR" "$PD_PROBE_M" "$PD_STEP_PROBE_EVERY" \
    "$TRAIN_MANIFEST_JSON" "$TRAIN_MANIFEST_CSV" "$UNLIMITED_BUDGET" "$UNLIMITED_BUDGET_VAL" <<'PY'
import csv
import json
import shlex
import sys
from pathlib import Path

repo_dir = Path(sys.argv[1])
results_dir = Path(sys.argv[2])
pd_out_dir = Path(sys.argv[3])
use_wandb = sys.argv[4]
extra_overrides = shlex.split(sys.argv[5]) if sys.argv[5] else []
envs = shlex.split(sys.argv[6]) if sys.argv[6] else []
seeds = [int(x) for x in shlex.split(sys.argv[7])] if sys.argv[7] else []
capacity_names = shlex.split(sys.argv[8]) if sys.argv[8] else []
regime_names = shlex.split(sys.argv[9]) if sys.argv[9] else []

num_train_steps = int(sys.argv[10])
num_seed_steps = int(sys.argv[11])
num_unsup_steps = int(sys.argv[12])
eval_frequency = int(sys.argv[13])
segment = int(sys.argv[14])
ensemble_size = int(sys.argv[15])
teacher_eps_mistake = float(sys.argv[16])
teacher_beta = int(sys.argv[17])
reward_update = int(sys.argv[18])

pd_n_probe = int(sys.argv[19])
pd_segment_len = int(sys.argv[20])
pd_refit_pairs = int(sys.argv[21])
pd_k_refit = int(sys.argv[22])
pd_refit_lr = float(sys.argv[23])
pd_probe_m = int(sys.argv[24])
pd_step_probe_every = int(sys.argv[25])

train_manifest_json = Path(sys.argv[26])
train_manifest_csv = Path(sys.argv[27])
unlimited_budget = sys.argv[28].lower() in {"1", "true", "yes", "y", "on"}
unlimited_budget_val = int(sys.argv[29])

sys.path.insert(0, str(repo_dir))
from perf_diag.run import TASK_DEFAULTS, task_central_P, task_overrides, RegimeCfg

# Per-task regime construction (same logic as perf_diag.run.build_regimes_for_task
# but driven by the user-selected regime NAMES so the sweep stays consistent).
def regimes_for_task(task: str):
    central = task_central_P(task, fallback=10000)
    mapping = {
        "rare_relabel":     RegimeCfg("rare_relabel",     num_interact=int(central * 2),   is_positive=True),
        "med_relabel":      RegimeCfg("med_relabel",      num_interact=int(central),       is_positive=True),
        "frequent_relabel": RegimeCfg("frequent_relabel", num_interact=max(2000, int(central * 0.2)), is_positive=False),
        "ultra_frequent_relabel": RegimeCfg("ultra_frequent_relabel", num_interact=max(500, int(central * 0.05)), is_positive=False),
    }
    return [mapping[name] for name in regime_names if name in mapping]


def capacity_cfgs(names):
    """Named RM-capacity sweep. Custom tokens may be name:hidden_dim:num_layers:activation."""
    presets = {
        "full_rm": dict(capacity="full_rm", rm_hidden_dim=256, rm_num_layers=3,
                        rm_output_activation="tanh", is_capacity_limited=False),
        "small_rm": dict(capacity="small_rm", rm_hidden_dim=16, rm_num_layers=1,
                         rm_output_activation="tanh", is_capacity_limited=True),
        "linear_rm": dict(capacity="linear_rm", rm_hidden_dim=0, rm_num_layers=0,
                          rm_output_activation="none", is_capacity_limited=True),
    }
    out = []
    for token in names:
        if token in presets:
            out.append(presets[token])
            continue
        parts = token.split(":")
        if len(parts) != 4:
            raise ValueError(
                f"unknown capacity {token!r}; use one of {sorted(presets)} "
                "or name:hidden_dim:num_layers:activation"
            )
        name, hidden_dim, num_layers, activation = parts
        out.append(dict(capacity=name, rm_hidden_dim=int(hidden_dim),
                        rm_num_layers=int(num_layers),
                        rm_output_activation=activation,
                        is_capacity_limited=(name != "full_rm")))
    return out


def qjoin(parts):
    return " ".join(shlex.quote(str(x)) for x in parts)


def dedupe_cli_overrides(parts):
    """Last-write-wins for `key=value` overrides — lets EXTRA_OVERRIDES override defaults."""
    result = []
    key_to_idx = {}
    for item in parts:
        if isinstance(item, str) and "=" in item and not item.startswith("--"):
            key = item.split("=", 1)[0]
            old_idx = key_to_idx.get(key)
            if old_idx is not None:
                result[old_idx] = None
            key_to_idx[key] = len(result)
        result.append(item)
    return [item for item in result if item is not None]


cells = []
capacities = capacity_cfgs(capacity_names)
for task in envs:
    overrides_per_task = task_overrides(task)
    max_feedback = overrides_per_task.get("max_feedback", 1400)
    if unlimited_budget:
        # Decouple regime sweep from B-Pref budget exhaustion: each regime keeps
        # collecting fresh on-policy preferences regardless of P_relabel.
        max_feedback = unlimited_budget_val
    # Per-task reward_update takes precedence over the global REWARD_UPDATE env override.
    # Original MetaWorld scripts use 10, DMC scripts use 50; the YAML default of 200 is
    # never used by the authored scripts.
    task_reward_update = overrides_per_task.get("reward_update", reward_update)
    for cap in capacities:
        for regime in regimes_for_task(task):
            for seed in seeds:
                run_name = f"{task}__{cap['capacity']}__{regime.name}__seed{seed}"
                run_dir = results_dir / task / cap["capacity"] / regime.name / f"seed{seed}"
                jsonl_path = pd_out_dir / f"{run_name}.jsonl"
                cli = [
                    "__PYTHON_BIN__", "train_PEBBLE.py",
                    f"env={task}",
                    f"seed={seed}",
                    f"num_interact={regime.num_interact}",
                    f"num_train_steps={num_train_steps}",
                    f"num_seed_steps={num_seed_steps}",
                    f"num_unsup_steps={num_unsup_steps}",
                    f"eval_frequency={eval_frequency}",
                    f"segment={segment}",
                    f"ensemble_size={ensemble_size}",
                    f"teacher_eps_mistake={teacher_eps_mistake}",
                    f"teacher_beta={teacher_beta}",
                    f"max_feedback={max_feedback}",
                    f"reward_update={task_reward_update}",
                    f"rm_hidden_dim={cap['rm_hidden_dim']}",
                    f"rm_num_layers={cap['rm_num_layers']}",
                    f"rm_output_activation={cap['rm_output_activation']}",
                    "keep_relabeling_after_budget=true",
                    f"use_wandb={use_wandb}",
                    "log_save_tb=false",
                    "save_video=false",
                    "gpu=0",
                    f"hydra.run.dir={run_dir}",
                ]
                cli.extend(extra_overrides)
                cli = dedupe_cli_overrides(cli)
                cells.append({
                    "task": task,
                    "capacity": cap["capacity"],
                    "is_capacity_limited": bool(cap["is_capacity_limited"]),
                    "rm_hidden_dim": int(cap["rm_hidden_dim"]),
                    "rm_num_layers": int(cap["rm_num_layers"]),
                    "rm_output_activation": cap["rm_output_activation"],
                    "regime": regime.name,
                    "is_positive": bool(regime.is_positive),
                    "seed": int(seed),
                    "max_feedback": int(max_feedback),
                    "num_interact": int(regime.num_interact),
                    "run_name": run_name,
                    "run_dir": str(run_dir),
                    "jsonl_path": str(jsonl_path),
                    "command": qjoin(cli),
                    # PD_* env vars per-cell so each run gets the right run name/tag
                    "env_pd": {
                        "PD_ENABLE": "1",
                        "PD_RUN_NAME": run_name,
                        "PD_TAG_POSITIVE": "positive" if regime.is_positive else "negative_control",
                        "PD_N_PROBE": str(pd_n_probe),
                        "PD_SEGMENT_LEN": str(pd_segment_len),
                        "PD_REFIT_PAIRS": str(pd_refit_pairs),
                        "PD_K_REFIT": str(pd_k_refit),
                        "PD_REFIT_LR": str(pd_refit_lr),
                        "PD_PROBE_M": str(pd_probe_m),
                        "PD_STEP_PROBE_EVERY": str(pd_step_probe_every),
                        "PD_OUT_DIR": str(pd_out_dir),
                    },
                })

train_manifest_json.parent.mkdir(parents=True, exist_ok=True)
json.dump(cells, open(train_manifest_json, "w", encoding="utf-8"), indent=2)
if cells:
    fieldnames = ["task", "capacity", "is_capacity_limited", "rm_hidden_dim",
                  "rm_num_layers", "rm_output_activation", "regime", "is_positive", "seed", "max_feedback",
                  "num_interact", "run_name", "run_dir", "jsonl_path", "command"]
    with open(train_manifest_csv, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        for c in cells:
            wr.writerow({k: c[k] for k in fieldnames})

print(len(cells))
PY
)"

if [ "$N_TRAIN" -le 0 ]; then
    echo "ERROR: no perf_diag training cells generated"
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

TRAIN_JOB_NAME="perf_diag_train"
ANALYSIS_JOB_NAME="perf_diag_analysis"
TMP_TRAIN_SCRIPT="$(mktemp "${SCRIPT_DIR}/tmp_${TRAIN_JOB_NAME}_XXXXXX")"
TMP_ANALYSIS_SCRIPT="$(mktemp "${SCRIPT_DIR}/tmp_${ANALYSIS_JOB_NAME}_XXXXXX")"

# ==============================================================
# Per-cell training array script.
# Reads its own cell from the manifest using SLURM_ARRAY_TASK_ID.
# Exports the PD_* env vars then evals the hydra command.
# Skip-done: skip if the cell's JSONL already has at least __MIN_DONE_ROWS__ rows.
# ==============================================================
cat > "$TMP_TRAIN_SCRIPT" <<'EOT'
#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=__TRAIN_MEM__
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
export PYTHONPATH="__SCRIPT_DIR__:${PYTHONPATH:-}"

mapfile -t CELL_INFO < <("__PYTHON__" - "__TRAIN_MANIFEST_JSON__" "${SLURM_ARRAY_TASK_ID}" <<'PY'
import json, sys, shlex
manifest = json.load(open(sys.argv[1], "r", encoding="utf-8"))
cell = manifest[int(sys.argv[2])]
cmd = cell["command"].replace("__PYTHON_BIN__", "__PYTHON__")
print(cell["run_name"])
print(cell["jsonl_path"])
print(cmd)
# Emit PD_* env vars in KEY=VALUE form for the bash side to consume
for k, v in cell["env_pd"].items():
    print(f"__PD_KV__:{k}={v}")
PY
)

RUN_NAME="${CELL_INFO[0]}"
JSONL_PATH="${CELL_INFO[1]}"
CMD="${CELL_INFO[2]}"
for line in "${CELL_INFO[@]:3}"; do
    case "$line" in
        __PD_KV__:*) export "${line#__PD_KV__:}";;
    esac
done

echo "=== perf_diag training task ==="
echo "  array id   : ${SLURM_ARRAY_TASK_ID}"
echo "  run        : ${RUN_NAME}"
echo "  JSONL path : ${JSONL_PATH}"
echo "  PD_RUN_NAME: ${PD_RUN_NAME:-<unset>}"
echo "  PD_PROBE_M : ${PD_PROBE_M:-<unset>}"
echo "  command    : ${CMD}"
echo ""

if [ "__SKIP_DONE__" = "true" ] && [ -f "${JSONL_PATH}" ]; then
    N_ROWS="$(wc -l < "${JSONL_PATH}" || echo 0)"
    if [ "${N_ROWS}" -ge "__MIN_DONE_ROWS__" ]; then
        echo "[SKIP] ${RUN_NAME} appears complete: ${N_ROWS} probe rows in ${JSONL_PATH}"
        exit 0
    fi
fi

eval "$CMD"
EOT

# ==============================================================
# Combined analysis job (runs after the training array).
# ==============================================================
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
export PYTHONPATH="__SCRIPT_DIR__:${PYTHONPATH:-}"
export PD_OUT_DIR="__PD_OUT_DIR__"

# Step 0: rebuild a manifest of completed runs from the JSONL traces in PD_OUT_DIR.
# This is the file that perf_diag.analysis reads.
# When MERGE_PREVIOUS=true, also scan any JSONL in PD_OUT_DIR that wasn't part of
# this submission's train_manifest (i.e., runs left over from previous submissions).
# Filename convention:
#   new: <task>__<capacity>__<regime>__seed<N>.jsonl
#   old: <task>__<regime>__seed<N>.jsonl
"__PYTHON__" - "__TRAIN_MANIFEST_JSON__" "__PD_OUT_DIR__" "__MERGE_PREVIOUS__" <<'PY'
import json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from perf_diag import detect

train_manifest = json.load(open(sys.argv[1], "r", encoding="utf-8"))
out_dir = Path(sys.argv[2])
merge_previous = sys.argv[3].lower() in {"1", "true", "yes", "y", "on"}

POSITIVE_REGIMES = {"rare_relabel", "med_relabel"}
NEGATIVE_REGIMES = {"frequent_relabel", "ultra_frequent_relabel"}
KNOWN_REGIMES = POSITIVE_REGIMES | NEGATIVE_REGIMES
FILENAME_RE = re.compile(r"^(?P<prefix>.+)__seed(?P<seed>\d+)\.jsonl$")


def parse_jsonl_name(path: Path):
    m = FILENAME_RE.match(path.name)
    if not m:
        return None
    parts = m.group("prefix").split("__")
    seed = int(m.group("seed"))
    if len(parts) >= 3 and parts[-1] in KNOWN_REGIMES:
        task = "__".join(parts[:-2])
        capacity = parts[-2]
        regime = parts[-1]
    elif len(parts) >= 2 and parts[-1] in KNOWN_REGIMES:
        task = "__".join(parts[:-1])
        capacity = "default_rm"
        regime = parts[-1]
    else:
        return None
    return task, capacity, regime, seed


def load_gold(path: str):
    vals = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except Exception:
                continue
            val = row.get("gold_eval")
            if isinstance(val, (int, float)):
                vals.append(float(val))
    return vals

runs = []
seen_paths = set()

# (a) Cells from this submission's train manifest first.
for cell in train_manifest:
    jsonl = Path(cell["jsonl_path"])
    if not jsonl.exists() or jsonl.stat().st_size == 0:
        continue
    seen_paths.add(jsonl.resolve())
    runs.append({
        "run_name": cell["run_name"],
        "status": "completed",
        "jsonl": str(jsonl),
        "regime": cell["regime"],
        "task": cell["task"],
        "capacity": cell.get("capacity", "default_rm"),
        "is_capacity_limited": bool(cell.get("is_capacity_limited", False)),
        "rm_hidden_dim": cell.get("rm_hidden_dim"),
        "rm_num_layers": cell.get("rm_num_layers"),
        "rm_output_activation": cell.get("rm_output_activation"),
        "seed": cell["seed"],
        "is_positive": cell["is_positive"],
        "source": "this_submission",
    })

# (b) When merging, scan PD_OUT_DIR for previously-completed JSONLs.
n_merged_in = 0
n_skipped_unknown = 0
if merge_previous and out_dir.exists():
    for jsonl in sorted(out_dir.glob("*.jsonl")):
        if jsonl.resolve() in seen_paths:
            continue
        if jsonl.stat().st_size == 0:
            continue
        parsed = parse_jsonl_name(jsonl)
        if not parsed:
            # Skip files like smoke_test.jsonl that don't match the convention.
            n_skipped_unknown += 1
            continue
        task, capacity, regime, seed = parsed
        is_positive = regime in POSITIVE_REGIMES
        runs.append({
            "run_name": jsonl.stem,
            "status": "completed",
            "jsonl": str(jsonl),
            "regime": regime,
            "task": task,
            "capacity": capacity,
            "is_capacity_limited": capacity not in {"full_rm", "default_rm"},
            "rm_hidden_dim": None,
            "rm_num_layers": None,
            "rm_output_activation": None,
            "seed": seed,
            "is_positive": is_positive,
            "source": "previous_submission",
        })
        n_merged_in += 1

bad_neg = []
for r in runs:
    if r.get("is_positive", True):
        continue
    gold = load_gold(r["jsonl"])
    if len(gold) < 5:
        continue
    if detect.turned_over_in_horizon(
        gold, K_decline=5, alpha=0.2, min_drop_frac=0.2,
        min_post_points=8, max_peak_frac=0.8, late_window=5
    ):
        bad_neg.append(r["run_name"])

manifest = {
    "runs": runs,
    "config": {
        "quick": False,
        "tasks": sorted({r["task"] for r in runs}),
        "capacities": sorted({r.get("capacity", "default_rm") for r in runs}),
        "seeds": sorted({r["seed"] for r in runs}),
        "regimes": sorted({r["regime"] for r in runs}),
        "merge_previous": bool(merge_previous),
    },
    "neg_control_validity": {"ok": len(bad_neg) == 0, "bad": bad_neg},
}
manifest_path = out_dir / "_manifest.json"
manifest_path.parent.mkdir(parents=True, exist_ok=True)
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)
n_this = len(runs) - n_merged_in
print(f"wrote {manifest_path}: {len(runs)} runs total "
      f"({n_this} from this submission, {n_merged_in} merged from previous; "
      f"{n_skipped_unknown} unrelated JSONLs skipped)")
PY

echo "=== perf_diag analysis: combined report ==="
"__PYTHON__" -m perf_diag.analysis --out_dir "__REPORT_DIR__"

# Per-cell reports: one results.md / Figs / Tables per task × capacity.
mapfile -t TASK_CAPS < <("__PYTHON__" - "__PD_OUT_DIR__/_manifest.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1], "r", encoding="utf-8"))
for task, capacity in sorted({(r["task"], r.get("capacity", "default_rm")) for r in m.get("runs", [])}):
    print(f"{task}\t{capacity}")
PY
)
for TASK_CAP in "${TASK_CAPS[@]}"; do
    [ -z "$TASK_CAP" ] && continue
    IFS=$'\t' read -r TASK CAPACITY <<< "$TASK_CAP"
    echo ""
    echo "=== perf_diag analysis: per-cell report for ${TASK} / ${CAPACITY} ==="
    SAFE_TASK="${TASK//\//_}"
    "__PYTHON__" -m perf_diag.analysis \
        --task "${TASK}" \
        --capacity "${CAPACITY}" \
        --out_dir "__REPORT_DIR__/per_cell/${SAFE_TASK}/${CAPACITY}"
done
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

# Skip-done threshold: a run is "complete" if it has at least this many probe rows.
# Raised from 3 to 10: crashed runs often produce 0-2 rows before failure, so 3 was
# too close to the crash floor — rerun would silently skip those cells.
MIN_DONE_ROWS="${MIN_DONE_ROWS:-10}"

for script in "$TMP_TRAIN_SCRIPT" "$TMP_ANALYSIS_SCRIPT"; do
    python_replace "$script" "__CPUS_PER_TASK__" "$CPUS_PER_TASK"
    python_replace "$script" "__ANALYSIS_CPUS_PER_TASK__" "$ANALYSIS_CPUS_PER_TASK"
    python_replace "$script" "__TRAIN_JOB_NAME__" "$TRAIN_JOB_NAME"
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
    python_replace "$script" "__PD_OUT_DIR__" "$PD_OUT_DIR"
    python_replace "$script" "__TRAIN_MANIFEST_JSON__" "$TRAIN_MANIFEST_JSON"
    python_replace "$script" "__REPORT_DIR__" "$REPORT_DIR"
    python_replace "$script" "__SKIP_DONE__" "$SKIP_DONE"
    python_replace "$script" "__MIN_DONE_ROWS__" "$MIN_DONE_ROWS"
    python_replace "$script" "__MERGE_PREVIOUS__" "$MERGE_PREVIOUS"
    python_replace "$script" "__TRAIN_MEM__" "$TRAIN_MEM"
    chmod +x "$script"
done

echo "=== perf_diag SLURM submission ==="
echo "  envs             : $ENVS"
echo "  seeds            : $SEEDS"
echo "  capacities       : $CAPACITIES"
echo "  regimes          : $REGIMES"
echo "  num_train_steps  : $NUM_TRAIN_STEPS"
echo "  unlimited_budget : $UNLIMITED_BUDGET  (max_feedback override = $UNLIMITED_BUDGET_VAL when true)"
echo "  train cells      : $N_TRAIN"
echo "  train array spec : $TRAIN_ARRAY_SPEC"
echo "  time limit       : $TIME_LIMIT  (per cell)"
echo "  partition        : $PARTITION"
echo "  script dir       : $SCRIPT_DIR"
echo "  python           : $PYTHON"
echo "  results dir      : $RESULTS_DIR"
echo "  PD out dir       : $PD_OUT_DIR"
echo "  log dir          : $LOG_DIR"
echo "  run id           : $RUN_ID"
echo "  train manifest   : $TRAIN_MANIFEST_JSON"
echo "  report dir       : $REPORT_DIR"
echo "  run train        : $RUN_TRAIN"
echo "  run analysis     : $RUN_ANALYSIS"
echo "  merge previous   : $MERGE_PREVIOUS  (combined report scans all $PD_OUT_DIR/*.jsonl)"
echo "  extra overrides  : ${EXTRA_OVERRIDES:-<none>}"
echo "  probe knobs      : N_PROBE=$PD_N_PROBE  SEGMENT_LEN=$PD_SEGMENT_LEN  REFIT_PAIRS=$PD_REFIT_PAIRS  K_REFIT=$PD_K_REFIT  PROBE_M=$PD_PROBE_M  STEP_PROBE_EVERY=$PD_STEP_PROBE_EVERY"
echo ""

if [ "$DRY_RUN" = "true" ]; then
    echo "[DRY_RUN] Not submitting. Temporary scripts kept at:"
    echo "  $TMP_TRAIN_SCRIPT"
    echo "  $TMP_ANALYSIS_SCRIPT"
    exit 0
fi

TRAIN_JOB_ID=""
if [ "$RUN_TRAIN" = "true" ]; then
    TRAIN_JOB_ID="$(sbatch --parsable --array="${TRAIN_ARRAY_SPEC}" "$TMP_TRAIN_SCRIPT")"
    echo "Submitted perf_diag training array: $TRAIN_JOB_ID"
else
    echo "Training submission disabled. Temporary script: $TMP_TRAIN_SCRIPT"
fi

if [ "$RUN_ANALYSIS" = "true" ]; then
    analysis_dependency_args=()
    if [ -n "$TRAIN_JOB_ID" ]; then
        analysis_dependency_args=(--dependency=${DEPENDENCY_TYPE}:${TRAIN_JOB_ID})
    fi
    ANALYSIS_JOB_ID="$(sbatch --parsable "${analysis_dependency_args[@]}" "$TMP_ANALYSIS_SCRIPT")"
    echo "Submitted perf_diag analysis job: $ANALYSIS_JOB_ID"
    echo "Report will be written to: ${SCRIPT_DIR}/perf_diag/results.md"
else
    echo "Analysis submission disabled. Run manually:"
    echo "  PYTHONPATH=$SCRIPT_DIR $PYTHON -m perf_diag.analysis"
fi
