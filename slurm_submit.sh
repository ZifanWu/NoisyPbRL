#!/bin/bash
# ==============================================================
# NoisyPbRL SLURM Submission Script
# Submits one job per (env × teacher_param × seed); skips already-done runs.
# Usage: edit the CONFIGURE block below, then run: bash slurm_submit.sh
# ==============================================================

# ===================== CONFIGURE BELOW ========================
envs=(door_close door_open door_unlock drawer_open hammer window_close button_press sweep_into)  # any subset of: button_press sweep_into quadruped_walk walker_walk hammer door_close door_open door_unlock drawer_open window_close

max_feedback="auto"       # feedback budget: auto (max per env) | 500 | 1000 | 2000 | 4000 | 5000 | 10000 | 20000
teacher="oracle"          # mistake | myopic | noisy | oracle | skip | equal
algorithm="PEBBLE"        # PEBBLE | RUNE | SURF | static_SAC
feed_type=1               # query selection type (PEBBLE/RUNE/SURF): 0=uniform, 1=entropy, ...

# Sweep arrays — only the one matching the teacher type is used:
teacher_gammas=(0.9 0.95 0.99)   # myopic: values to sweep over teacher_gamma
teacher_eps_mistakes=(0.1 0.2 0.3)   # mistake: values to sweep over teacher_eps_mistake

rm_reset=false            # reset reward model periodically (PEBBLE only): true | false
use_wandb=true            # log to Weights & Biases: true | false
seeds=(12345 23451 34512 45123 51234 67890 78906 89067 90678 6789)
# ==============================================================

PYTHON="/uufs/chpc.utah.edu/common/home/dbrown-group1/zifan/miniconda3/envs/bpref/bin/python"
SCRIPT_DIR="/uufs/chpc.utah.edu/common/home/u1520755/NoisyPbRL"
EXP_DIR="/uufs/chpc.utah.edu/common/home/dbrown-group1/zifan/NoisyPbRL/exp"
LOG_DIR="/uufs/chpc.utah.edu/common/home/dbrown-group1/zifan/logs"
mkdir -p "$LOG_DIR"

# Validate algorithm
case "$algorithm" in
    PEBBLE|RUNE|SURF|static_SAC) ;;
    *) echo "ERROR: Unknown algorithm '$algorithm'. Choose: PEBBLE | RUNE | SURF | static_SAC"; exit 1 ;;
esac

# Resolve which teacher_param array to sweep, and a display label for it.
# RUNE and SURF hardcode teacher param values in their scripts — no sweep.
case "$teacher" in
    myopic)  param_label="teacher_gamma";       params=("${teacher_gammas[@]}") ;;
    mistake) param_label="teacher_eps_mistake"; params=("${teacher_eps_mistakes[@]}") ;;
    oracle|noisy|skip|equal) param_label=""; params=("_") ;;  # no sweep; single pass
    *) echo "ERROR: Unknown teacher '$teacher'. Choose: mistake | myopic | noisy | oracle | skip | equal"; exit 1 ;;
esac
if [[ "$algorithm" == "RUNE" || "$algorithm" == "SURF" ]]; then
    param_label=""
    params=("_")
fi

echo "=== NoisyPbRL SLURM Submission ==="
echo "  envs         : ${envs[*]}"
echo "  max_feedback : $max_feedback$([ "$max_feedback" = "auto" ] && echo " (resolved per env)")"
echo "  teacher      : $teacher"
echo "  algorithm    : $algorithm"
if [ "$algorithm" != "static_SAC" ]; then
    echo "  feed_type    : $feed_type"
fi
if [ "$algorithm" = "PEBBLE" ]; then
    echo "  rm_reset     : $rm_reset"
fi
[ -n "$param_label" ] && echo "  ${param_label}s : ${params[*]}"
echo "  seeds        : ${seeds[*]}"
echo ""

total_submitted=0
total_skipped=0

for env in "${envs[@]}"; do

    # Map short env name → full environment name used in exp/ directories
    case "$env" in
        button_press)   env_name="metaworld_button-press-v2"  ;;
        sweep_into)     env_name="metaworld_sweep-into-v2"    ;;
        quadruped_walk) env_name="quadruped_walk"              ;;
        walker_walk)    env_name="walker_walk"                 ;;
        hammer)         env_name="metaworld_hammer-v2"         ;;
        door_close)     env_name="metaworld_door-close-v2"     ;;
        door_open)      env_name="metaworld_door-open-v2"      ;;
        door_unlock)    env_name="metaworld_door-unlock-v2"    ;;
        drawer_open)    env_name="metaworld_drawer-open-v2"    ;;
        window_close)   env_name="metaworld_window-close-v2"   ;;
        *) echo "ERROR: Unknown env '$env'. Choose: button_press | sweep_into | quadruped_walk | walker_walk | hammer | door_close | door_open | door_unlock | drawer_open | window_close"; exit 1 ;;
    esac

    # Resolve max_feedback: "auto" picks the largest available budget for this env
    if [ "$max_feedback" = "auto" ]; then
        case "$env" in
            button_press|sweep_into|hammer)  resolved_feedback=20000 ;;
            drawer_open)                     resolved_feedback=10000 ;;
            door_unlock)                     resolved_feedback=5000  ;;
            door_open)                       resolved_feedback=4000  ;;
            quadruped_walk)                  resolved_feedback=2000  ;;
            walker_walk|door_close|window_close) resolved_feedback=1000 ;;
        esac
    else
        resolved_feedback="$max_feedback"
    fi

    [ "$max_feedback" = "auto" ] && echo "  [auto] ${env}: max_feedback=${resolved_feedback}"

    for param in "${params[@]}"; do

        # Resolve teacher_gamma / teacher_eps_mistake for this iteration
        teacher_gamma=1.0
        teacher_eps_mistake=0.0
        case "$teacher" in
            myopic)  teacher_gamma="$param" ;;
            mistake) teacher_eps_mistake="$param" ;;
        esac

        # Build teacher directory name (matches hydra output path)
        case "$teacher" in
            myopic)  t_beta=-1; t_gamma="$teacher_gamma"; t_mistake=0;   t_skip=0;   t_equal=0   ;;
            mistake) t_beta=-1; t_gamma=1;                t_mistake="$teacher_eps_mistake"; t_skip=0; t_equal=0 ;;
            oracle)  t_beta=-1; t_gamma=1;                t_mistake=0;   t_skip=0;   t_equal=0   ;;
            noisy)   t_beta=1;  t_gamma=1;                t_mistake=0;   t_skip=0;   t_equal=0   ;;
            skip)    t_beta=-1; t_gamma=1;                t_mistake=0;   t_skip=0.1; t_equal=0   ;;
            equal)   t_beta=-1; t_gamma=1;                t_mistake=0;   t_skip=0;   t_equal=0.1 ;;
        esac
        teacher_dir="teacher_b${t_beta}_g${t_gamma}_m${t_mistake}_s${t_skip}_e${t_equal}"

        # Locate the reference script
        script_path="${SCRIPT_DIR}/scripts/${env}/${resolved_feedback}/${teacher}/run_${algorithm}.sh"
        if [ ! -f "$script_path" ]; then
            echo "WARNING: Script not found, skipping: $script_path"
            continue
        fi

        # Extract the single python training command
        raw_cmd=$(grep -m1 "python train" "$script_path")
        if [ -z "$raw_cmd" ]; then
            echo "WARNING: No 'python train' line found in $script_path, skipping"
            continue
        fi

        echo "--- env=${env}  teacher=${teacher_dir} ---"

        for seed in "${seeds[@]}"; do

            # --------------------------------------------------------------
            # Check if this seed was already completed by looking for its exp dir.
            #
            # PEBBLE (new naming, includes rm_reset):
            #   exp/{env}/H*/teacher_dir/label_smooth*/schedule_*/*_rm{rm_reset}_seed{seed}   (maxdepth 3)
            # PEBBLE (old naming, no rm_reset in dir — created before this change):
            #   Fallback when rm_reset=false to stay backward-compatible.
            # RUNE:
            #   exp/{env}/H*/teacher_dir/label_smooth*/schedule_*/PEBBLE-RUNE_*/*maxfeed*_seed{seed}  (maxdepth 4)
            # SURF:
            #   exp/{env}/H*/teacher_dir/ratio_*/schedule_*/data_aug_*/*maxfeed*_seed{seed}  (maxdepth 4)
            # static_SAC:
            #   exp/{env}/static_sac/teacher_dir/maxfeed*seed{seed}
            # --------------------------------------------------------------
            found=""
            if [ "$algorithm" = "PEBBLE" ]; then
                base="${EXP_DIR}/${env_name}/H256_L3_lr0.0003/${teacher_dir}"
                found=$(find "$base" -maxdepth 3 -type d \
                    -name "*maxfeed${resolved_feedback}*sample${feed_type}*_rm${rm_reset}_seed${seed}" 2>/dev/null | head -n1)
                if [ -z "$found" ] && [ "$rm_reset" = "false" ]; then
                    found=$(find "$base" -maxdepth 3 -type d \
                        -name "*maxfeed${resolved_feedback}*sample${feed_type}*seed${seed}" 2>/dev/null \
                        | grep -v "_rm" | head -n1)
                fi
            elif [[ "$algorithm" == "RUNE" || "$algorithm" == "SURF" ]]; then
                base="${EXP_DIR}/${env_name}/H256_L3_lr0.0003/${teacher_dir}"
                found=$(find "$base" -maxdepth 4 -type d \
                    -name "*maxfeed${resolved_feedback}*sample${feed_type}*seed${seed}" 2>/dev/null | head -n1)
            else
                base="${EXP_DIR}/${env_name}/static_sac/${teacher_dir}"
                found=$(find "$base" -maxdepth 1 -type d \
                    -name "maxfeed${resolved_feedback}*seed${seed}" 2>/dev/null | head -n1)
            fi

            if [ -n "$found" ]; then
                echo "  [SKIP]   seed=${seed}  →  $(basename "$found")"
                total_skipped=$((total_skipped + 1))
                continue
            fi

            # --------------------------------------------------------------
            # Build the concrete training command for this seed.
            # --------------------------------------------------------------
            cmd=$(echo "$raw_cmd" | sed "s|^[[:space:]]*python |${PYTHON} |")
            cmd=$(echo "$cmd" | sed 's/\$seed/'"${seed}"'/g')
            cmd=$(echo "$cmd" | sed 's/gpu=\$2/gpu=0/g')
            cmd=$(echo "$cmd" | sed 's/gpu=\$1/gpu=0/g')
            cmd=$(echo "$cmd" | sed 's/gpu=\${2:-0}/gpu=0/g')
            cmd=$(echo "$cmd" | sed 's/gpu=\${1:-0}/gpu=0/g')

            if [ "$algorithm" != "static_SAC" ]; then
                cmd=$(echo "$cmd" | sed 's/feed_type=\$1/feed_type='"${feed_type}"'/g')
            fi
            if [ "$algorithm" = "PEBBLE" ]; then
                case "$teacher" in
                    myopic)
                        cmd=$(echo "$cmd" | sed 's/teacher_gamma=\$3/teacher_gamma='"${teacher_gamma}"'/g')
                        ;;
                    mistake)
                        cmd=$(echo "$cmd" | sed 's/teacher_eps_mistake=\$3/teacher_eps_mistake='"${teacher_eps_mistake}"'/g')
                        ;;
                esac
                cmd="${cmd} rm_reset=${rm_reset}"
            fi

            if ! echo "$cmd" | grep -q "gpu="; then
                cmd="${cmd} gpu=0"
            fi
            cmd=$(echo "$cmd" | sed 's/use_wandb=[^ ]*/use_wandb='"${use_wandb}"'/g')
            if ! echo "$cmd" | grep -q "use_wandb="; then
                cmd="${cmd} use_wandb=${use_wandb}"
            fi

            # --------------------------------------------------------------
            # Write and submit the SLURM job script for this seed.
            # --------------------------------------------------------------
            job_name="${env:0:4}_${teacher:0:3}_${algorithm:0:4}_s${seed}"
            tmp_script=$(mktemp "${SCRIPT_DIR}/tmp_slurm_XXXXXX.sh")

            cat > "$tmp_script" << EOT
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

module load cuda/12.4.0

cd ${SCRIPT_DIR}
${cmd}
EOT

            echo "  [SUBMIT] seed=${seed}  →  $cmd"
            sbatch "$tmp_script"
            rm "$tmp_script"
            total_submitted=$((total_submitted + 1))
        done

        echo ""
    done
done

echo "=== Done. Submitted: ${total_submitted} | Skipped (already done): ${total_skipped} ==="
