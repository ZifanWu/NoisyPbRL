#!/bin/bash
# ==============================================================
# Tandem Experiment SLURM Submission
#
# For each (env × teacher × seed):
#   1. Submit tandem_mode=baseline  (writes HDF5 log)
#   2. Submit all 5 tandem conditions with --dependency=afterok on baseline
#
# Skip logic:
#   baseline  — skip if HDF5 already exists in tandem_logs/
#   condition — skip if output directory already exists
#
# Usage:
#   Edit the CONFIGURE block, then run:
#     bash tandem_slurm_submit.sh
#   Dry-run (print commands, don't submit):
#     DRY_RUN=1 bash tandem_slurm_submit.sh
# ==============================================================

# ===================== CONFIGURE BELOW ========================

envs=(walker_walk)
# Full env names passed to env= argument (see resolve_env_name for mapping)

teachers=(oracle stochastic mistake)
# oracle | stochastic | mistake  (see README for hyperparams)

seeds=(12345 23451 34512 45123 51234)

# ── Training hyperparams (identical across ALL 6 conditions) ──
num_train_steps=500000
num_seed_steps=1000
num_unsup_steps=5000
num_interact=5000
max_feedback=1400
reward_batch=128
reward_update=200
feed_type=1
segment=50
ensemble_size=3
large_batch=10
rm_reset=false
activation=tanh
gradient_update=1

sanity_mode=false       # enable only for debugging short runs
use_wandb=true

# ── SLURM resource settings ───────────────────────────────────
TIME_LIMIT="12:00:00"
PARTITION="soc-gpu-np"
ACCOUNT="soc-gpu-np"
EXCLUDE_NODES="notch372,notch369,notch475,notch371"
CPUS=4

# ── Paths (edit for each machine) ────────────────────────────
PYTHON="/uufs/chpc.utah.edu/common/home/dbrown-group1/zifan/miniconda3/envs/bpref/bin/python"
SCRIPT_DIR="/uufs/chpc.utah.edu/common/home/u1520755/NoisyPbRL"
EXP_DIR="/uufs/chpc.utah.edu/common/home/dbrown-group1/zifan/NoisyPbRL/exp"
LOG_DIR="/uufs/chpc.utah.edu/common/home/dbrown-group1/zifan/logs/tandem"

# ==============================================================

DRY_RUN="${DRY_RUN:-0}"

TANDEM_CONDITIONS=(
    all_passive
    passive_pol_active_rm
    active_pol_passive_rm
    active_pol_passive_query
    active_pol_passive_dist
)

mkdir -p "$LOG_DIR"

# ── Helper: short env key → full env name (matches cfg.env) ──
resolve_env_name() {
    case "$1" in
        walker_walk)    echo "walker_walk" ;;
        quadruped_walk) echo "quadruped_walk" ;;
        button_press)   echo "metaworld_button-press-v2" ;;
        sweep_into)     echo "metaworld_sweep-into-v2" ;;
        hammer)         echo "metaworld_hammer-v2" ;;
        door_close)     echo "metaworld_door-close-v2" ;;
        door_open)      echo "metaworld_door-open-v2" ;;
        door_unlock)    echo "metaworld_door-unlock-v2" ;;
        drawer_open)    echo "metaworld_drawer-open-v2" ;;
        window_close)   echo "metaworld_window-close-v2" ;;
        *) echo "$1" ;;  # pass through unknown names unchanged
    esac
}

# ── Helper: teacher name → (beta gamma eps_mistake eps_skip eps_equal) ──
resolve_teacher() {
    case "$1" in
        oracle)     echo "-1 1 0 0 0" ;;
        stochastic) echo "1 1 0 0 0" ;;
        mistake)    echo "-1 1 0.1 0 0" ;;
        *) echo "ERROR: unknown teacher '$1'" >&2; exit 1 ;;
    esac
}

# ── Helper: build the python argument string for one run ──────
build_args() {
    local env_name=$1 seed=$2 mode=$3
    local beta gamma eps_m eps_s eps_e
    read -r beta gamma eps_m eps_s eps_e <<< "$teacher_params"

    printf '%s' \
        "env=${env_name} seed=${seed} tandem_mode=${mode}" \
        " num_train_steps=${num_train_steps}" \
        " num_seed_steps=${num_seed_steps}" \
        " num_unsup_steps=${num_unsup_steps}" \
        " num_interact=${num_interact}" \
        " max_feedback=${max_feedback}" \
        " reward_batch=${reward_batch}" \
        " reward_update=${reward_update}" \
        " feed_type=${feed_type}" \
        " segment=${segment}" \
        " ensemble_size=${ensemble_size}" \
        " large_batch=${large_batch}" \
        " rm_reset=${rm_reset}" \
        " activation=${activation}" \
        " gradient_update=${gradient_update}" \
        " teacher_beta=${beta}" \
        " teacher_gamma=${gamma}" \
        " teacher_eps_mistake=${eps_m}" \
        " teacher_eps_skip=${eps_s}" \
        " teacher_eps_equal=${eps_e}" \
        " sanity_mode=${sanity_mode}" \
        " use_wandb=${use_wandb}" \
        " exp_dir=${EXP_DIR}" \
        " gpu=0"
}

# ── Helper: write and submit one SLURM job ────────────────────
# Returns the SLURM job ID via stdout.
# dep: job ID to wait on (pass "" for no dependency).
submit_slurm() {
    local job_name=$1 dep=$2 cmd=$3
    local dep_directive=""
    [ -n "$dep" ] && dep_directive="#SBATCH --dependency=afterok:${dep}"

    local tmp
    tmp=$(mktemp "${SCRIPT_DIR}/tmp_slurm_XXXXXX.sh")
    cat > "$tmp" << EOT
#!/bin/bash
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --ntasks=1
#SBATCH --job-name=${job_name}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --partition=${PARTITION}
#SBATCH --account=${ACCOUNT}
#SBATCH --exclude=${EXCLUDE_NODES}
#SBATCH --output=${LOG_DIR}/${job_name}_%j.out
${dep_directive}

module load cuda/12.4.0
cd ${SCRIPT_DIR}
${cmd}
EOT

    if [ "$DRY_RUN" = "1" ]; then
        echo "DRY_RUN"
        cat "$tmp"
        rm "$tmp"
        return
    fi

    local jid
    jid=$(sbatch "$tmp" | awk '{print $NF}')
    rm "$tmp"
    echo "$jid"
}

# ── Summary counters ─────────────────────────────────────────
total_submitted=0
total_skipped=0

echo "=== Tandem Experiment SLURM Submission ==="
echo "  envs     : ${envs[*]}"
echo "  teachers : ${teachers[*]}"
echo "  seeds    : ${seeds[*]}"
echo "  conditions: baseline + ${TANDEM_CONDITIONS[*]}"
echo "  dry_run  : ${DRY_RUN}"
echo ""

# ── Main loop ─────────────────────────────────────────────────
for env in "${envs[@]}"; do
    env_name=$(resolve_env_name "$env")

    for teacher in "${teachers[@]}"; do
        teacher_params=$(resolve_teacher "$teacher")
        read -r t_beta t_gamma t_mistake t_skip t_equal <<< "$teacher_params"
        teacher_dir="teacher_b${t_beta}_g${t_gamma}_m${t_mistake}_s${t_skip}_e${t_equal}"

        for seed in "${seeds[@]}"; do
            echo "--- env=${env}  teacher=${teacher}  seed=${seed} ---"

            # ── Step 1: Baseline ─────────────────────────────────────────
            baseline_h5="${EXP_DIR}/tandem_logs/tandem_baseline_${env_name}_seed${seed}.h5"
            baseline_jid=""

            if [ -f "$baseline_h5" ]; then
                echo "  [SKIP]   baseline  →  HDF5 exists"
                total_skipped=$((total_skipped + 1))
            else
                job_name="${env:0:4}_${teacher:0:3}_base_s${seed}"
                args=$(build_args "$env_name" "$seed" "baseline")
                cmd="${PYTHON} train_PEBBLE.py ${args}"
                baseline_jid=$(submit_slurm "$job_name" "" "$cmd")
                echo "  [SUBMIT] baseline  →  job ${baseline_jid:-DRY}"
                total_submitted=$((total_submitted + 1))
            fi

            # ── Step 2: Tandem conditions ────────────────────────────────
            for mode in "${TANDEM_CONDITIONS[@]}"; do

                # Skip if output directory already exists for this condition
                existing=$(find "${EXP_DIR}/${env_name}" -maxdepth 8 -type d \
                    -path "*/${teacher_dir}/*/tandem_${mode}/*seed${seed}" \
                    2>/dev/null | head -n1)

                if [ -n "$existing" ]; then
                    echo "  [SKIP]   ${mode}"
                    total_skipped=$((total_skipped + 1))
                    continue
                fi

                # Use a compact job name (SLURM limits are system-dependent)
                # Format: {env4}_{t2}_{mode_code}_{seed5}
                case "$mode" in
                    all_passive)              mode_code="ap"   ;;
                    passive_pol_active_rm)    mode_code="ppar" ;;
                    active_pol_passive_rm)    mode_code="appr" ;;
                    active_pol_passive_query) mode_code="appq" ;;
                    active_pol_passive_dist)  mode_code="appd" ;;
                esac
                job_name="${env:0:4}_${teacher:0:2}_${mode_code}_${seed}"

                args=$(build_args "$env_name" "$seed" "$mode")
                cmd="${PYTHON} train_PEBBLE.py ${args}"

                dep_info=""
                if [ -n "$baseline_jid" ]; then
                    dep_info=" (after job ${baseline_jid})"
                fi

                cond_jid=$(submit_slurm "$job_name" "$baseline_jid" "$cmd")
                echo "  [SUBMIT] ${mode}  →  job ${cond_jid:-DRY}${dep_info}"
                total_submitted=$((total_submitted + 1))
            done

            echo ""
        done
    done
done

echo "=== Done. Submitted: ${total_submitted} | Skipped: ${total_skipped} ==="
