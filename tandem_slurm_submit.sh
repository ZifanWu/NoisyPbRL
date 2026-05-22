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
#   Edit the CONFIGURE block, then:
#     bash tandem_slurm_submit.sh          # submit for real
#     DRY_RUN=1 bash tandem_slurm_submit.sh  # print without submitting
# ==============================================================

# ===================== CONFIGURE BELOW ========================

envs=(walker_walk)
# Supported: walker_walk quadruped_walk
#            door_open door_close door_unlock
#            drawer_open button_press sweep_into hammer window_close

teachers=(oracle stochastic mistake)
# oracle | stochastic | mistake

seeds=(12345 23451 34512 45123 51234 67890 78906 89067 90678 6789)

# Shared across all conditions for a given run (these are fixed by design)
feed_type=1
rm_reset=false
sanity_mode=false
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

# ── Per-env hyperparameters (sourced from scripts/ reference runs) ──────────
#
# Sets these variables in caller scope:
#   ENV_NAME          full env name passed to env=
#   ACTOR_LR CRITIC_LR
#   NUM_TRAIN_STEPS NUM_SEED_STEPS NUM_UNSUP_STEPS
#   NUM_INTERACT MAX_FEEDBACK REWARD_BATCH REWARD_UPDATE
#   HIDDEN_DIM HIDDEN_DEPTH BATCH_SIZE   (agent architecture)
#   LARGE_BATCH SEGMENT ENSEMBLE_SIZE ACTIVATION GRADIENT_UPDATE
#
resolve_env_hyperparams() {
    local env=$1

    # Defaults shared by all envs
    NUM_SEED_STEPS=1000
    LARGE_BATCH=10
    SEGMENT=50
    ENSEMBLE_SIZE=3
    ACTIVATION=tanh
    GRADIENT_UPDATE=1
    # Architecture defaults (dm_control)
    HIDDEN_DIM=1024
    HIDDEN_DEPTH=2
    BATCH_SIZE=""        # empty = use agent config default (256)

    case "$env" in
        walker_walk)
            ENV_NAME="walker_walk"
            ACTOR_LR=0.0005;  CRITIC_LR=0.0005
            NUM_TRAIN_STEPS=500000;  NUM_UNSUP_STEPS=9000
            NUM_INTERACT=20000;  MAX_FEEDBACK=1000
            REWARD_BATCH=100;    REWARD_UPDATE=50
            ;;
        quadruped_walk)
            ENV_NAME="quadruped_walk"
            ACTOR_LR=0.0001;  CRITIC_LR=0.0001
            NUM_TRAIN_STEPS=1000000; NUM_UNSUP_STEPS=9000
            NUM_INTERACT=30000;  MAX_FEEDBACK=1000
            REWARD_BATCH=100;    REWARD_UPDATE=50
            ;;
        door_open)
            ENV_NAME="metaworld_door-open-v2"
            ACTOR_LR=0.0003;  CRITIC_LR=0.0003
            NUM_TRAIN_STEPS=1000000; NUM_UNSUP_STEPS=9000
            NUM_INTERACT=10000;  MAX_FEEDBACK=2000
            REWARD_BATCH=50;     REWARD_UPDATE=10
            HIDDEN_DIM=256; HIDDEN_DEPTH=3; BATCH_SIZE=512
            ;;
        door_close)
            ENV_NAME="metaworld_door-close-v2"
            ACTOR_LR=0.0003;  CRITIC_LR=0.0003
            NUM_TRAIN_STEPS=500000;  NUM_UNSUP_STEPS=9000
            NUM_INTERACT=10000;  MAX_FEEDBACK=1000
            REWARD_BATCH=50;     REWARD_UPDATE=10
            HIDDEN_DIM=256; HIDDEN_DEPTH=3; BATCH_SIZE=512
            ;;
        door_unlock)
            ENV_NAME="metaworld_door-unlock-v2"
            ACTOR_LR=0.0003;  CRITIC_LR=0.0003
            NUM_TRAIN_STEPS=1000000; NUM_UNSUP_STEPS=9000
            NUM_INTERACT=10000;  MAX_FEEDBACK=2500
            REWARD_BATCH=25;     REWARD_UPDATE=10
            HIDDEN_DIM=256; HIDDEN_DEPTH=3; BATCH_SIZE=512
            ;;
        drawer_open)
            ENV_NAME="metaworld_drawer-open-v2"
            ACTOR_LR=0.0003;  CRITIC_LR=0.0003
            NUM_TRAIN_STEPS=1000000; NUM_UNSUP_STEPS=9000
            NUM_INTERACT=10000;  MAX_FEEDBACK=10000
            REWARD_BATCH=100;    REWARD_UPDATE=10
            HIDDEN_DIM=256; HIDDEN_DEPTH=3; BATCH_SIZE=512
            ;;
        button_press)
            ENV_NAME="metaworld_button-press-v2"
            ACTOR_LR=0.0003;  CRITIC_LR=0.0003
            NUM_TRAIN_STEPS=1000000; NUM_UNSUP_STEPS=9000
            NUM_INTERACT=5000;   MAX_FEEDBACK=10000
            REWARD_BATCH=50;     REWARD_UPDATE=10
            HIDDEN_DIM=256; HIDDEN_DEPTH=3; BATCH_SIZE=512
            ;;
        sweep_into)
            ENV_NAME="metaworld_sweep-into-v2"
            ACTOR_LR=0.0003;  CRITIC_LR=0.0003
            NUM_TRAIN_STEPS=1000000; NUM_UNSUP_STEPS=9000
            NUM_INTERACT=5000;   MAX_FEEDBACK=10000
            REWARD_BATCH=50;     REWARD_UPDATE=10
            HIDDEN_DIM=256; HIDDEN_DEPTH=3; BATCH_SIZE=512
            ;;
        hammer)
            ENV_NAME="metaworld_hammer-v2"
            ACTOR_LR=0.0003;  CRITIC_LR=0.0003
            NUM_TRAIN_STEPS=2000000; NUM_UNSUP_STEPS=9000
            NUM_INTERACT=5000;   MAX_FEEDBACK=10000
            REWARD_BATCH=50;     REWARD_UPDATE=10
            HIDDEN_DIM=256; HIDDEN_DEPTH=3; BATCH_SIZE=512
            ;;
        window_close)
            ENV_NAME="metaworld_window-close-v2"
            ACTOR_LR=0.0003;  CRITIC_LR=0.0003
            NUM_TRAIN_STEPS=500000;  NUM_UNSUP_STEPS=9000
            NUM_INTERACT=10000;  MAX_FEEDBACK=1000
            REWARD_BATCH=10;     REWARD_UPDATE=10
            HIDDEN_DIM=256; HIDDEN_DEPTH=3; BATCH_SIZE=512
            ;;
        *)
            echo "ERROR: unknown env '$env'" >&2
            echo "Supported: walker_walk quadruped_walk door_open door_close" >&2
            echo "           door_unlock drawer_open button_press sweep_into" >&2
            echo "           hammer window_close" >&2
            exit 1
            ;;
    esac
}

# ── Helper: teacher name → (beta gamma eps_mistake eps_skip eps_equal) ──────
resolve_teacher() {
    case "$1" in
        oracle)     echo "-1 1 0 0 0" ;;
        stochastic) echo "1 1 0 0 0" ;;
        mistake)    echo "-1 1 0.1 0 0" ;;
        *) echo "ERROR: unknown teacher '$1' (oracle|stochastic|mistake)" >&2; exit 1 ;;
    esac
}

# ── Helper: build the full python argument string ────────────────────────────
build_args() {
    local seed=$1 mode=$2
    local beta gamma eps_m eps_s eps_e
    read -r beta gamma eps_m eps_s eps_e <<< "$teacher_params"

    # Architecture args: only pass hidden_dim/hidden_depth/batch_size when
    # they differ from the agent config defaults (i.e. for metaworld envs).
    local arch_args=""
    arch_args="${arch_args} diag_gaussian_actor.params.hidden_dim=${HIDDEN_DIM}"
    arch_args="${arch_args} diag_gaussian_actor.params.hidden_depth=${HIDDEN_DEPTH}"
    arch_args="${arch_args} double_q_critic.params.hidden_dim=${HIDDEN_DIM}"
    arch_args="${arch_args} double_q_critic.params.hidden_depth=${HIDDEN_DEPTH}"
    if [ -n "$BATCH_SIZE" ]; then
        arch_args="${arch_args} agent.params.batch_size=${BATCH_SIZE}"
    fi

    printf '%s' \
        "env=${ENV_NAME} seed=${seed} tandem_mode=${mode}" \
        " agent.params.actor_lr=${ACTOR_LR}" \
        " agent.params.critic_lr=${CRITIC_LR}" \
        " ${arch_args}" \
        " gradient_update=${GRADIENT_UPDATE}" \
        " activation=${ACTIVATION}" \
        " num_train_steps=${NUM_TRAIN_STEPS}" \
        " num_seed_steps=${NUM_SEED_STEPS}" \
        " num_unsup_steps=${NUM_UNSUP_STEPS}" \
        " num_interact=${NUM_INTERACT}" \
        " max_feedback=${MAX_FEEDBACK}" \
        " reward_batch=${REWARD_BATCH}" \
        " reward_update=${REWARD_UPDATE}" \
        " feed_type=${feed_type}" \
        " segment=${SEGMENT}" \
        " ensemble_size=${ENSEMBLE_SIZE}" \
        " large_batch=${LARGE_BATCH}" \
        " rm_reset=${rm_reset}" \
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

# ── Helper: write and submit one SLURM job ───────────────────────────────────
# dep: job ID to wait on ("" = no dependency).
# Returns the SLURM job ID via stdout.
submit_slurm() {
    local job_name=$1 dep=$2 cmd=$3
    local dep_directive=""
    [ -n "$dep" ] && dep_directive="#SBATCH --dependency=afterok:${dep}"

    local tmp
    tmp=$(mktemp "${SCRIPT_DIR}/tmp_slurm_XXXXXX.sh")
    cat > "$tmp" << EOT
#!/bin/bash
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --job-name=${job_name}
#SBATCH --time=12:00:00
##SBATCH --qos=dbrown-gpu-grn
#SBATCH --partition=soc-gpu-np
#SBATCH --account=soc-gpu-np
#SBATCH --exclude=notch372,notch369,notch475,notch371
#SBATCH --output=${LOG_DIR}/${job_name}_%j.out
${dep_directive}

module load cuda/12.4.0
cd ${SCRIPT_DIR}
${cmd}
EOT

    if [ "$DRY_RUN" = "1" ]; then
        echo "--- DRY_RUN: $job_name ---"
        cat "$tmp"
        rm "$tmp"
        echo "DRY_RUN"   # stand-in for job ID
        return
    fi

    local jid
    jid=$(sbatch "$tmp" | awk '{print $NF}')
    rm "$tmp"
    echo "$jid"
}

# ── Summary counters ─────────────────────────────────────────────────────────
total_submitted=0
total_skipped=0

echo "=== Tandem Experiment SLURM Submission ==="
echo "  envs      : ${envs[*]}"
echo "  teachers  : ${teachers[*]}"
echo "  seeds     : ${seeds[*]}"
echo "  dry_run   : ${DRY_RUN}"
echo ""

# ── Main loop ─────────────────────────────────────────────────────────────────
for env in "${envs[@]}"; do

    resolve_env_hyperparams "$env"  # populates ENV_NAME, ACTOR_LR, etc.

    for teacher in "${teachers[@]}"; do
        teacher_params=$(resolve_teacher "$teacher")
        read -r t_beta t_gamma t_mistake t_skip t_equal <<< "$teacher_params"
        teacher_dir="teacher_b${t_beta}_g${t_gamma}_m${t_mistake}_s${t_skip}_e${t_equal}"

        for seed in "${seeds[@]}"; do
            echo "--- env=${env}  teacher=${teacher}  seed=${seed} ---"

            # ── Step 1: Baseline ─────────────────────────────────────────────
            baseline_h5="${EXP_DIR}/tandem_logs/tandem_baseline_${ENV_NAME}_seed${seed}.h5"
            baseline_jid=""

            if [ -f "$baseline_h5" ]; then
                echo "  [SKIP]   baseline  →  HDF5 already exists"
                total_skipped=$((total_skipped + 1))
            else
                job_name="${env:0:4}_${teacher:0:3}_base_${seed}"
                args=$(build_args "$seed" "baseline")
                cmd="${PYTHON} train_PEBBLE.py ${args}"
                baseline_jid=$(submit_slurm "$job_name" "" "$cmd")
                echo "  [SUBMIT] baseline  →  job ${baseline_jid}"
                total_submitted=$((total_submitted + 1))
            fi

            # ── Step 2: Tandem conditions ────────────────────────────────────
            for mode in "${TANDEM_CONDITIONS[@]}"; do

                # Skip if output directory already exists for this (condition, seed)
                existing=$(find "${EXP_DIR}/${ENV_NAME}" -maxdepth 8 -type d \
                    -path "*/${teacher_dir}/*/tandem_${mode}/*seed${seed}" \
                    2>/dev/null | head -n1)

                if [ -n "$existing" ]; then
                    echo "  [SKIP]   ${mode}"
                    total_skipped=$((total_skipped + 1))
                    continue
                fi

                case "$mode" in
                    all_passive)              mode_code="ap"   ;;
                    passive_pol_active_rm)    mode_code="ppar" ;;
                    active_pol_passive_rm)    mode_code="appr" ;;
                    active_pol_passive_query) mode_code="appq" ;;
                    active_pol_passive_dist)  mode_code="appd" ;;
                esac
                job_name="${env:0:4}_${teacher:0:2}_${mode_code}_${seed}"

                args=$(build_args "$seed" "$mode")
                cmd="${PYTHON} train_PEBBLE.py ${args}"

                dep_info=""
                [ -n "$baseline_jid" ] && dep_info=" (after job ${baseline_jid})"

                cond_jid=$(submit_slurm "$job_name" "$baseline_jid" "$cmd")
                echo "  [SUBMIT] ${mode}  →  job ${cond_jid}${dep_info}"
                total_submitted=$((total_submitted + 1))
            done

            echo ""
        done
    done
done

echo "=== Done. Submitted: ${total_submitted} | Skipped: ${total_skipped} ==="
