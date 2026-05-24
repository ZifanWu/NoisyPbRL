#!/bin/bash
# ==============================================================
# Tandem Baseline SLURM Submission
#
# Submits tandem_mode=baseline jobs for each (env × teacher × seed).
# Skips if the HDF5 log already exists and is complete (rb_length >= num_train_steps).
# Deletes and resubmits if the HDF5 exists but is incomplete.
#
# Usage:
#   bash tandem_baseline_submit.sh          # submit for real
#   DRY_RUN=1 bash tandem_baseline_submit.sh  # print without submitting
#
# After all baselines finish, run tandem_conditions_submit.sh.
# ==============================================================

# ===================== CONFIGURE BELOW ========================

envs=(door_open door_close door_unlock drawer_open button_press sweep_into hammer window_close walker_walk quadruped_walk)
# Supported: walker_walk quadruped_walk
#            door_open door_close door_unlock
#            drawer_open button_press sweep_into hammer window_close

teachers=(oracle stochastic)
# oracle | stochastic | mistake

seeds=(78906 89067)

# Shared across all conditions for a given run (these are fixed by design)
feed_type=1
rm_reset=false
sanity_mode=false
use_wandb=true

# ── Paths (edit for each machine) ────────────────────────────
PYTHON="/uufs/chpc.utah.edu/common/home/dbrown-group1/zifan/miniconda3/envs/bpref/bin/python"
SCRIPT_DIR="/uufs/chpc.utah.edu/common/home/u1520755/NoisyPbRL"
EXP_DIR="/uufs/chpc.utah.edu/common/home/dbrown-group1/zifan/NoisyPbRL/exp"
LOG_DIR="/uufs/chpc.utah.edu/common/home/dbrown-group1/zifan/logs/tandem"

# ==============================================================

DRY_RUN="${DRY_RUN:-0}"
mkdir -p "$LOG_DIR"

resolve_env_hyperparams() {
    local env=$1
    NUM_SEED_STEPS=1000
    LARGE_BATCH=10
    SEGMENT=50
    ENSEMBLE_SIZE=3
    ACTIVATION=tanh
    GRADIENT_UPDATE=1
    HIDDEN_DIM=1024
    HIDDEN_DEPTH=2
    BATCH_SIZE=""

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
            echo "ERROR: unknown env '$env'" >&2; exit 1 ;;
    esac
}

resolve_teacher() {
    case "$1" in
        oracle)     echo "-1 1 0 0 0" ;;
        stochastic) echo "1 1 0 0 0" ;;
        mistake)    echo "-1 1 0.1 0 0" ;;
        *) echo "ERROR: unknown teacher '$1'" >&2; exit 1 ;;
    esac
}

build_args() {
    local seed=$1 mode=$2
    local beta gamma eps_m eps_s eps_e
    read -r beta gamma eps_m eps_s eps_e <<< "$teacher_params"

    local arch_args=""
    arch_args="${arch_args} diag_gaussian_actor.params.hidden_dim=${HIDDEN_DIM}"
    arch_args="${arch_args} diag_gaussian_actor.params.hidden_depth=${HIDDEN_DEPTH}"
    arch_args="${arch_args} double_q_critic.params.hidden_dim=${HIDDEN_DIM}"
    arch_args="${arch_args} double_q_critic.params.hidden_depth=${HIDDEN_DEPTH}"
    [ -n "$BATCH_SIZE" ] && arch_args="${arch_args} agent.params.batch_size=${BATCH_SIZE}"

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

submit_slurm() {
    local job_name=$1 cmd=$2
    local tmp
    tmp=$(mktemp "${LOG_DIR}/tmp_slurm_XXXXXX.sh")
    cat > "$tmp" << EOT
#!/bin/bash
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --job-name=${job_name}
#SBATCH --time=12:00:00
#SBATCH --qos=dbrown-gpu-grn
#SBATCH --partition=dbrown-gpu-grn
#SBATCH --account=dbrown
##SBATCH --exclude=notch372,notch369,notch475,notch371
#SBATCH --output=${LOG_DIR}/${job_name}_%j.out

module load cuda/12.4.0
cd ${SCRIPT_DIR}
${cmd}
EOT

    if [ "$DRY_RUN" = "1" ]; then
        echo "--- DRY_RUN: $job_name ---"
        cat "$tmp"
        rm "$tmp"
        echo "DRY_RUN"
        return
    fi

    local jid
    jid=$(sbatch "$tmp" | awk '{print $NF}')
    rm "$tmp"
    echo "$jid"
}

# ── Completeness check ───────────────────────────────────────────────────────
h5_is_complete() {
    local h5_path=$1 min_steps=$2
    local rb_len
    rb_len=$("$PYTHON" -c "
import h5py
try:
    f = h5py.File('$h5_path', 'r', locking=False)
    print(int(f['replay_buffer'].attrs['length']))
    f.close()
except:
    print(0)
" 2>/dev/null)
    [ "${rb_len:-0}" -ge "$min_steps" ]
}

# ── Summary counters ─────────────────────────────────────────────────────────
total_submitted=0
total_skipped=0

echo "=== Tandem Baseline SLURM Submission ==="
echo "  envs     : ${envs[*]}"
echo "  teachers : ${teachers[*]}"
echo "  seeds    : ${seeds[*]}"
echo "  dry_run  : ${DRY_RUN}"
echo ""

# ── Main loop ────────────────────────────────────────────────────────────────
for env in "${envs[@]}"; do
    resolve_env_hyperparams "$env"

    for teacher in "${teachers[@]}"; do
        teacher_params=$(resolve_teacher "$teacher")

        for seed in "${seeds[@]}"; do
            baseline_h5="${EXP_DIR}/tandem_logs/tandem_baseline_${ENV_NAME}_seed${seed}.h5"

            if [ -f "$baseline_h5" ]; then
                if h5_is_complete "$baseline_h5" "$NUM_TRAIN_STEPS"; then
                    echo "[SKIP]   ${env}  ${teacher}  seed=${seed}  →  baseline complete"
                    total_skipped=$((total_skipped + 1))
                    continue
                else
                    echo "[RERUN]  ${env}  ${teacher}  seed=${seed}  →  incomplete HDF5, deleting"
                    rm -f "$baseline_h5"
                fi
            fi

            job_name="${env:0:4}_${teacher:0:3}_base_${seed}"
            args=$(build_args "$seed" "baseline")
            cmd="${PYTHON} train_PEBBLE.py ${args}"
            jid=$(submit_slurm "$job_name" "$cmd")
            echo "[SUBMIT] ${env}  ${teacher}  seed=${seed}  →  job ${jid}"
            total_submitted=$((total_submitted + 1))
        done
    done
done

echo ""
echo "=== Done. Submitted: ${total_submitted} | Skipped: ${total_skipped} ==="
