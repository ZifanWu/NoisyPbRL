#!/bin/bash
# ==============================================================
# Track 1 SLURM Submission — MetaWorld standard regime
#
# Sweeps: envs × teacher_types × feed_types × 6 seeds
# Packs 2 runs per GPU job (both seeds run concurrently on 1 GPU)
#
# Usage: bash slurm_submit_track1.sh
# ==============================================================

# ===================== CONFIGURE BELOW ========================
envs=(drawer_open door_open door_close window_close)
teacher_types=(oracle noisy mistake myopic skip)
feed_types=(0 1)
gauge_modes=(l2 none zero_mean_ref) # l2 none zero_mean_ref
conditions=(oneshot)  # which RM conditions to run: (iterative oneshot)
seeds=(34512)   # 6 seeds
#  12345 23451 34512 45123 51234 67890 78906 89067 90678 6789

# Default teacher-specific params (extend to arrays to sweep multiple values)
teacher_gammas=(0.9)        # myopic:  teacher_gamma values
teacher_eps_mistakes=(0.1)  # mistake: teacher_eps_mistake values

rm_reset=false
use_wandb=true
# ==============================================================

PYTHON="/uufs/chpc.utah.edu/common/home/dbrown-group1/zifan/miniconda3/envs/bpref/bin/python"
SCRIPT_DIR="/uufs/chpc.utah.edu/common/home/u1520755/NoisyPbRL"
EXP_DIR="/uufs/chpc.utah.edu/common/home/dbrown-group1/zifan/NoisyPbRL/exp"
LOG_DIR="/uufs/chpc.utah.edu/common/home/dbrown-group1/zifan/logs"
mkdir -p "$LOG_DIR"

echo "=== Track 1 SLURM Submission (MetaWorld, 2 runs/GPU) ==="
echo "  envs         : ${envs[*]}"
echo "  teachers     : ${teacher_types[*]}"
echo "  feed_types   : ${feed_types[*]}"
echo "  gauge_modes  : ${gauge_modes[*]}"
echo "  conditions   : ${conditions[*]}"
echo "  seeds        : ${seeds[*]}"
echo "  rm_reset     : $rm_reset"
echo "  use_wandb    : $use_wandb"
echo ""

total_jobs=0
total_submitted=0
total_skipped=0

# Helper: build a concrete command for one seed from a base template command.
# Mirrors the substitution logic in slurm_submit.sh.
build_cmd() {
    local raw="$1"
    local seed="$2"
    local feed_type="$3"
    local teacher="$4"
    local teacher_gamma="$5"
    local teacher_eps_mistake="$6"
    local gauge_mode="$7"

    local cmd
    cmd=$(echo "$raw" | sed "s|^[[:space:]]*python |${PYTHON} |")
    cmd=$(echo "$cmd" | sed 's/\$seed/'"${seed}"'/g')
    cmd=$(echo "$cmd" | sed 's/gpu=\$2/gpu=0/g; s/gpu=\$1/gpu=0/g')
    cmd=$(echo "$cmd" | sed 's/gpu=\${[12]:-0}/gpu=0/g')
    cmd=$(echo "$cmd" | sed 's/feed_type=\$1/feed_type='"${feed_type}"'/g')
    cmd=$(echo "$cmd" | sed 's/use_wandb=[^ ]*/use_wandb='"${use_wandb}"'/g')
    if ! echo "$cmd" | grep -q "use_wandb="; then
        cmd="${cmd} use_wandb=${use_wandb}"
    fi
    if ! echo "$cmd" | grep -q "gpu="; then
        cmd="${cmd} gpu=0"
    fi
    case "$teacher" in
        myopic)  cmd=$(echo "$cmd" | sed 's/teacher_gamma=\$3/teacher_gamma='"${teacher_gamma}"'/g') ;;
        mistake) cmd=$(echo "$cmd" | sed 's/teacher_eps_mistake=\$3/teacher_eps_mistake='"${teacher_eps_mistake}"'/g') ;;
    esac
    cmd="${cmd} rm_reset=${rm_reset} gauge_mode=${gauge_mode} exp_dir=${EXP_DIR}"
    echo "$cmd"
}

for env in "${envs[@]}"; do

    # Map short name → full env name used in exp/ directories
    case "$env" in
        button_press) env_name="metaworld_button-press-v2" ;;
        sweep_into)   env_name="metaworld_sweep-into-v2"   ;;
        hammer)       env_name="metaworld_hammer-v2"        ;;
        drawer_open)  env_name="metaworld_drawer-open-v2"   ;;
        door_unlock)  env_name="metaworld_door-unlock-v2"   ;;
        door_open)    env_name="metaworld_door-open-v2"     ;;
        door_close)   env_name="metaworld_door-close-v2"    ;;
        window_close) env_name="metaworld_window-close-v2"  ;;
        *) echo "ERROR: Unknown env '$env'"; exit 1 ;;
    esac

    # Resolve max_feedback for this env
    case "$env" in
        button_press|sweep_into|hammer) resolved_feedback=20000 ;;
        drawer_open)                    resolved_feedback=10000 ;;
        door_unlock)                    resolved_feedback=5000  ;;
        door_open)                      resolved_feedback=4000  ;;
        door_close|window_close)        resolved_feedback=1000  ;;
    esac

    for teacher in "${teacher_types[@]}"; do

        # Resolve per-teacher parameter sweep
        case "$teacher" in
            myopic)                  params=("${teacher_gammas[@]}") ;;
            mistake)                 params=("${teacher_eps_mistakes[@]}") ;;
            oracle|noisy|skip|equal) params=("_") ;;
        esac

        for param in "${params[@]}"; do

            # Resolve teacher command-line values and exp-dir label
            teacher_gamma=1.0
            teacher_eps_mistake=0.0
            case "$teacher" in
                myopic)  teacher_gamma="$param" ;;
                mistake) teacher_eps_mistake="$param" ;;
            esac

            case "$teacher" in
                myopic)  t_beta=-1; t_gamma="$teacher_gamma";       t_mistake=0;                     t_skip=0;   t_equal=0   ;;
                mistake) t_beta=-1; t_gamma=1;                      t_mistake="$teacher_eps_mistake"; t_skip=0;   t_equal=0   ;;
                oracle)  t_beta=-1; t_gamma=1;                      t_mistake=0;                     t_skip=0;   t_equal=0   ;;
                noisy)   t_beta=1;  t_gamma=1;                      t_mistake=0;                     t_skip=0;   t_equal=0   ;;
                skip)    t_beta=-1; t_gamma=1;                      t_mistake=0;                     t_skip=0.1; t_equal=0   ;;
                equal)   t_beta=-1; t_gamma=1;                      t_mistake=0;                     t_skip=0;   t_equal=0.1 ;;
            esac
            teacher_dir="teacher_b${t_beta}_g${t_gamma}_m${t_mistake}_s${t_skip}_e${t_equal}"

            # Locate reference script
            script_path="${SCRIPT_DIR}/scripts/${env}/${resolved_feedback}/${teacher}/run_PEBBLE.sh"
            if [ ! -f "$script_path" ]; then
                echo "  [MISSING] $script_path — skipping"
                continue
            fi
            raw_cmd=$(grep -m1 "python train" "$script_path")
            if [ -z "$raw_cmd" ]; then
                echo "  [NO CMD] $script_path — skipping"
                continue
            fi

            for feed_type in "${feed_types[@]}"; do
            for gauge_mode in "${gauge_modes[@]}"; do
            for condition in "${conditions[@]}"; do

                # Abbreviated labels for job names and log output
                case "$gauge_mode" in
                    l2)            g_label="l2" ;;
                    none)          g_label="no" ;;
                    zero_mean_ref) g_label="zm" ;;
                    *)             g_label="${gauge_mode:0:2}" ;;
                esac
                case "$condition" in
                    iterative) c_label="it"; dir_prefix="PEBBLE_init" ;;
                    oneshot)   c_label="os"; dir_prefix="PEBBLE_oneshot_init" ;;
                esac

                echo "--- env=${env}  teacher=${teacher}  feed_type=${feed_type}  gauge=${gauge_mode}  cond=${condition} ---"

                # Full path to the gauge-specific experiment directory (same for both conditions)
                gauge_base="${EXP_DIR}/${env_name}/H256_L3_lr0.0003/${teacher_dir}/label_smooth_0.0/schedule_0/tandem_baseline/gauge_${gauge_mode}"

                # Collect seeds that still need to run (condition-specific dir prefix)
                pending=()
                for seed in "${seeds[@]}"; do
                    found=$(find "$gauge_base" -maxdepth 1 -type d \
                        -name "${dir_prefix}*maxfeed${resolved_feedback}*sample${feed_type}*_rm${rm_reset}_seed${seed}" \
                        2>/dev/null | head -n1)
                    if [ -z "$found" ] && [ "$rm_reset" = "false" ]; then
                        found=$(find "$gauge_base" -maxdepth 1 -type d \
                            -name "${dir_prefix}*maxfeed${resolved_feedback}*sample${feed_type}*seed${seed}" \
                            2>/dev/null | grep -v "_rm" | head -n1)
                    fi
                    if [ -n "$found" ]; then
                        echo "  [SKIP] seed=${seed} → $(basename "$found")"
                        total_skipped=$((total_skipped + 1))
                    else
                        pending+=("$seed")
                    fi
                done

                [ ${#pending[@]} -eq 0 ] && { echo "  [ALL DONE]"; echo ""; continue; }

                # Submit one job per pending seed
                for seed in "${pending[@]}"; do
                    cmd=$(build_cmd "$raw_cmd" "$seed" "$feed_type" "$teacher" \
                          "$teacher_gamma" "$teacher_eps_mistake" "$gauge_mode")
                    # For one-shot: swap in the oneshot script (config picked up from its yaml)
                    if [ "$condition" = "oneshot" ]; then
                        cmd=$(echo "$cmd" | sed 's|train_PEBBLE\.py|train_PEBBLE_oneshot.py|')
                    fi
                    job_name="${env:0:4}_${teacher:0:3}_f${feed_type}_${g_label}_${c_label}_s${seed}"
                    tmp_script=$(mktemp "${SCRIPT_DIR}/tmp_slurm_XXXXXX.sh")
                    cat > "$tmp_script" << EOT
#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20g
#SBATCH --ntasks=1
#SBATCH --job-name=${job_name}
#SBATCH --time=12:00:00
##SBATCH --qos=dbrown-gpu-grn
#SBATCH --partition=dbrown-gpu-np
#SBATCH --account=dbrown-gpu-np
#SBATCH --exclude=notch372,notch369,notch475,notch371
#SBATCH --output=${LOG_DIR}/${job_name}_%j.out

module load cuda/12.4.0
cd ${SCRIPT_DIR}

${cmd}
EOT
                    echo "  [SUBMIT] gauge=${gauge_mode}  seed=${seed}"
                    sbatch "$tmp_script"
                    rm "$tmp_script"
                    total_jobs=$((total_jobs + 1))
                    total_submitted=$((total_submitted + 1))
                done

                echo ""
            done  # condition
            done  # gauge_mode
            done  # feed_type
        done  # param
    done  # teacher
done  # env

echo "=== Done.  Jobs: ${total_jobs}  |  Runs submitted: ${total_submitted}  |  Skipped: ${total_skipped} ==="
