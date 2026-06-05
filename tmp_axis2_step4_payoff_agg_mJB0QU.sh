#!/bin/bash
#SBATCH --mem=8g
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=1
#SBATCH --job-name=axis2_step4_payoff_agg
#SBATCH --time=00:30:00
##SBATCH --qos=dbrown-gpu-grn
#SBATCH --partition=dbrown-gpu-np
#SBATCH --account=dbrown-gpu-np
##SBATCH --exclude=notch372,notch369,notch475,notch371
#SBATCH --output=/home/zifan/NoisyPbRL/logs/axis2_tier_b/axis2_step4_payoff_agg_%j.out

set -euo pipefail
cd "/home/zifan/NoisyPbRL"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"

"/home/zifan/miniconda3/envs/bpref/bin/python" -m axis2_tier_b.payoff \
    --aggregate \
    --ckpt_dir "/home/zifan/NoisyPbRL/exp/axis2_tier_b/payoff/runs" \
    --out_dir "/home/zifan/NoisyPbRL/exp/axis2_tier_b/payoff/aggregate"
