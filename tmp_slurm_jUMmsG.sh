#!/bin/bash
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --job-name=butt_myo_PEBB_s90678
#SBATCH --time=12:00:00
##SBATCH --qos=dbrown-gpu-grn
#SBATCH --partition=soc-gpu-np
#SBATCH --account=soc-gpu-np
##SBATCH --exclude=notch372,notch369
#SBATCH --output=/uufs/chpc.utah.edu/common/home/dbrown-group1/zifan/logs/butt_myo_PEBB_s90678_%j.out

cd /uufs/chpc.utah.edu/common/home/u1520755/NoisyPbRL
/uufs/chpc.utah.edu/common/home/dbrown-group1/zifan/miniconda3/envs/bpref/bin/python train_PEBBLE.py gpu=0 use_wandb=true env=metaworld_button-press-v2 seed=90678 agent.params.actor_lr=0.0003 agent.params.critic_lr=0.0003 gradient_update=1 activation=tanh num_unsup_steps=9000 num_train_steps=1000000 agent.params.batch_size=512 double_q_critic.params.hidden_dim=256 double_q_critic.params.hidden_depth=3 diag_gaussian_actor.params.hidden_dim=256 diag_gaussian_actor.params.hidden_depth=3 reward_update=10  num_interact=5000 max_feedback=20000 reward_batch=100 reward_update=10 feed_type=0 teacher_beta=-1 teacher_gamma=0.9 teacher_eps_mistake=0 teacher_eps_skip=0 teacher_eps_equal=0 rm_reset=true
