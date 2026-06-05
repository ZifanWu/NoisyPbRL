#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=20g
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --job-name=axis2_step4_payoff
#SBATCH --time=06:00:00
##SBATCH --qos=dbrown-gpu-grn
#SBATCH --partition=dbrown-gpu-np
#SBATCH --account=dbrown-gpu-np
##SBATCH --exclude=notch372,notch369,notch475,notch371
#SBATCH --output=/home/zifan/NoisyPbRL/logs/axis2_tier_b/axis2_step4_payoff_%A_%a.out

set -euo pipefail

module load cuda/12.4.0
cd "/home/zifan/NoisyPbRL"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export LD_LIBRARY_PATH="${HOME}/.mujoco/mujoco210/bin:/usr/lib/nvidia:${LD_LIBRARY_PATH:-}"

mapfile -t CELL_INFO < <("/home/zifan/miniconda3/envs/bpref/bin/python" - "/home/zifan/NoisyPbRL/exp/axis2_tier_b/manifests/axis2_step4_payoff_readme.json" "${SLURM_ARRAY_TASK_ID}" <<'PY'
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

if [ "true" = "true" ] && [ -f "${OUT_DIR}/payoff_results.json" ]; then
    echo "[SKIP] payoff already exists: ${OUT_DIR}/payoff_results.json"
    exit 0
fi

"/home/zifan/miniconda3/envs/bpref/bin/python" -m axis2_tier_b.payoff \
    --ckpt_dir "$CKPT_DIR" \
    --k_steps "20000" \
    --n_remedy_labels 1000 5000 \
    --out_dir "$OUT_DIR"
