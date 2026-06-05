#!/bin/bash
#SBATCH --mem=20g
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=1
#SBATCH --job-name=axis2_step3_analysis
#SBATCH --time=01:00:00
##SBATCH --qos=dbrown-gpu-grn
#SBATCH --partition=dbrown-gpu-np
#SBATCH --account=dbrown-gpu-np
##SBATCH --exclude=notch372,notch369,notch475,notch371
#SBATCH --output=/home/zifan/NoisyPbRL/logs/axis2_tier_b/axis2_step3_analysis_%j.out

set -euo pipefail
cd "/home/zifan/NoisyPbRL"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"

echo "=== Axis 2 combined analysis ==="
"/home/zifan/miniconda3/envs/bpref/bin/python" -m axis2_tier_b.analysis \
    --results_dir "/home/zifan/NoisyPbRL/exp/axis2_tier_b" \
    --out_dir "/home/zifan/NoisyPbRL/exp/axis2_tier_b/results" \
    --n_boot "1000"

mapfile -t ENV_RESULT_DIRS < <("/home/zifan/miniconda3/envs/bpref/bin/python" - "/home/zifan/NoisyPbRL/exp/axis2_tier_b" <<'PY'
import sys
from pathlib import Path
root = Path(sys.argv[1])
for path in sorted(root.iterdir() if root.exists() else []):
    if path.is_dir() and any(path.rglob('axis2_metrics.csv')):
        print(path)
PY
)

for ENV_RESULTS_DIR in "${ENV_RESULT_DIRS[@]}"; do
    ENV_NAME="$(basename "${ENV_RESULTS_DIR}")"
    echo "=== Axis 2 per-env analysis: ${ENV_NAME} ==="
    "/home/zifan/miniconda3/envs/bpref/bin/python" -m axis2_tier_b.analysis \
        --results_dir "${ENV_RESULTS_DIR}" \
        --out_dir "${ENV_RESULTS_DIR}/results" \
        --n_boot "1000"
done
