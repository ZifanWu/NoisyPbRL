#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=20g
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --job-name=axis2_step2
#SBATCH --time=12:00:00
##SBATCH --qos=dbrown-gpu-grn
#SBATCH --partition=dbrown-gpu-np
#SBATCH --account=dbrown-gpu-np
##SBATCH --exclude=notch372,notch369,notch475,notch371
#SBATCH --output=/home/zifan/NoisyPbRL/logs/axis2_tier_b/axis2_step2_%A_%a.out

set -euo pipefail

module load cuda/12.4.0
cd "/home/zifan/NoisyPbRL"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export LD_LIBRARY_PATH="${HOME}/.mujoco/mujoco210/bin:/usr/lib/nvidia:${LD_LIBRARY_PATH:-}"

mapfile -t CELL_INFO < <("/home/zifan/miniconda3/envs/bpref/bin/python" - "/home/zifan/NoisyPbRL/exp/axis2_tier_b/manifests/axis2_step2_train.json" "${SLURM_ARRAY_TASK_ID}" <<'PY'
import json
import sys
manifest = json.load(open(sys.argv[1], "r", encoding="utf-8"))
cell = manifest[int(sys.argv[2])]
cmd = cell["command"].replace("__PYTHON_BIN__", "/home/zifan/miniconda3/envs/bpref/bin/python")
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

if [ "true" = "true" ]; then
    LAST_STEP="$("/home/zifan/miniconda3/envs/bpref/bin/python" - "${RUN_DIR}" <<'PY'
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
    if [ "$LAST_STEP" -ge "990000" ]; then
        echo "[SKIP] ${RUN_NAME} appears complete: last monitor step ${LAST_STEP}"
        exit 0
    fi
fi

eval "$CMD"
