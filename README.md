## Requirements

- NVIDIA GPU with driver supporting CUDA 12.x
- Conda (Miniconda or Anaconda)

## Install

```bash
conda env create -f conda_env.yml
conda activate bpref

# Install stable-baselines3 (used by PPO scripts)
pip install -e .[docs,tests,extra]

# Install the local dm_control and gym wrappers
cd custom_dmc2gym && pip install -e . && cd ..
cd Metaworld-2.0.0 && pip install -e . && cd ..
```

> **Note on MuJoCo:** This repo uses the open-source `mujoco` pip package (3.x) and
> `dm-control` 1.x, replacing the proprietary MuJoCo 2.0 binaries required by the
> original B-Pref codebase. Physics simulation differs from MuJoCo 2.0, so absolute
> reward numbers will not match the original paper — but algorithm comparisons within
> this setup are internally consistent.


## Logging

Results are logged to [Weights & Biases](https://wandb.ai) by default.
Log in once before running experiments:

```bash
wandb login
```

To disable wandb and fall back to CSV-only logging:

```bash
python train_SAC.py ... use_wandb=false
```

To use TensorBoard instead of (or alongside) wandb:

```bash
python train_SAC.py ... log_save_tb=true
```

## Run experiments using GT rewards

### SAC & SAC + unsupervised pre-training

```bash
./scripts/[env_name]/run_sac.sh
./scripts/[env_name]/run_sac_unsuper.sh
```

### PPO & PPO + unsupervised pre-training

```bash
./scripts/[env_name]/run_ppo.sh
./scripts/[env_name]/run_ppo_unsuper.sh
```

## GPU selection

Use the `gpu` flag to select which GPU to run on (default: `0`):

```bash
python train_SAC.py env=quadruped_walk gpu=1 ...
```

This sets the training device to `cuda:1`. The same flag works for all train scripts.

## Run experiments on irrational teacher

To design more realistic models of human teachers, we consider a common stochastic
model and systematically manipulate its terms and operators:

```
teacher_beta: rationality constant of stochastic preference model (default: -1 for perfectly rational model)
teacher_gamma: discount factor to model myopic behavior (default: 1)
teacher_eps_mistake: probability of making a mistake (default: 0)
teacher_eps_skip: hyperparameters to control skip threshold (\in [0,1])
teacher_eps_equal: hyperparameters to control equal threshold (\in [0,1])
```

In B-Pref, we tried the following teachers:

`Oracle teacher`: (teacher_beta=-1, teacher_gamma=1, teacher_eps_mistake=0, teacher_eps_skip=0, teacher_eps_equal=0)

`Mistake teacher`: (teacher_beta=-1, teacher_gamma=1, teacher_eps_mistake=0.1, teacher_eps_skip=0, teacher_eps_equal=0)

`Noisy teacher`: (teacher_beta=1, teacher_gamma=1, teacher_eps_mistake=0, teacher_eps_skip=0, teacher_eps_equal=0)

`Skip teacher`: (teacher_beta=-1, teacher_gamma=1, teacher_eps_mistake=0, teacher_eps_skip=0.1, teacher_eps_equal=0)

`Myopic teacher`: (teacher_beta=-1, teacher_gamma=0.9, teacher_eps_mistake=0, teacher_eps_skip=0, teacher_eps_equal=0)

`Equal teacher`: (teacher_beta=-1, teacher_gamma=1, teacher_eps_mistake=0, teacher_eps_skip=0, teacher_eps_equal=0.1)

### PEBBLE

```bash
./scripts/[env_name]/[teacher_type]/[max_budget]/run_PEBBLE.sh [sampling_scheme: 0=uniform, 1=disagreement, 2=entropy]
```

### PrefPPO

```bash
./scripts/[env_name]/[teacher_type]/[max_budget]/run_PrefPPO.sh [sampling_scheme: 0=uniform, 1=disagreement, 2=entropy]
```

### Static SAC

```bash
./scripts/[env_name]/[max_budget]/[teacher_type]/run_static_SAC.sh [gpu_id]
```

`run_static_SAC.sh` does not take a `sampling_scheme` argument because the reward
model is always trained offline with uniform sampling.

## Static SAC — three-phase offline-then-online pipeline

`train_static_sac.py` implements a decoupled baseline for preference-based RL:

1. **Phase 1 — Data collection (SAC with ground-truth rewards)**  
   Runs a standard SAC agent with true environment rewards for `num_train_steps`
   steps and records every completed episode as a raw trajectory (observation–action
   concatenations and true rewards, one array per episode).  
   The trajectory pool is cached at:

   ```text
   trajectory_cache/{env}/seed{seed}/trajectories.pkl
   ```

   The cache key is `(env, seed)` only — teacher parameters do not affect data
   collection, so re-running with a different teacher reuses the same trajectories
   automatically without re-running Phase 1.

2. **Phase 2 — Offline reward-model training**  
   Loads the cached trajectory pool into the reward model, generates `max_feedback`
   labelled preference pairs in one shot using uniform sampling, then trains the
   ensemble reward model.  
   Stopping criterion (same as each PEBBLE round): up to `reward_update` epochs,
   early-stop when accuracy exceeds 0.97.  
   The reward model is **fixed** after this phase — no further updates.  
   Teacher behavior (rational / noisy / myopic / skip / mistake / equal) is
   determined by the same `teacher_*` parameters as PEBBLE.

3. **Phase 3 — Online policy training with fixed reward model**  
   Initialises a fresh SAC agent and an empty replay buffer, then interacts with
   the environment using the fixed reward model's predicted rewards in place of
   true rewards.  Logs both `train/episode_reward` (RM-predicted) and
   `train/true_episode_reward` (ground-truth) for comparison.

### Configuration

The config file is `config/train_static_sac.yaml`.  Key parameters beyond the
shared SAC/teacher ones:

| Parameter | Default | Description |
|---|---|---|
| `rm_log_interval` | `5000` | Log RM metrics (dormant rate, feature rank, BT weights) every this many gradient steps during Phase 2 |
| `traj_cache_dir` | `trajectory_cache` | Root directory for trajectory caches, relative to the project root |
| `max_feedback` | `1400` | Number of preference pairs generated for offline RM training |

### Running a single experiment

```bash
# quadruped_walk, noisy teacher, 2000 feedback pairs, GPU 1
seed=12345 python train_static_sac.py \
    use_wandb=true gpu=1 \
    env=quadruped_walk seed=$seed \
    agent.params.actor_lr=0.0001 agent.params.critic_lr=0.0001 \
    gradient_update=1 activation=tanh \
    num_unsup_steps=9000 num_train_steps=1000000 \
    max_feedback=2000 reward_batch=200 reward_update=50 \
    teacher_beta=1 teacher_gamma=1 \
    teacher_eps_mistake=0 teacher_eps_skip=0 teacher_eps_equal=0
```

### Script directory layout

```text
scripts/
  {quadruped_walk, button_press, sweep_into}/
    {1000|2000, 10000|20000}/          ← max_feedback budget
      {oracle, noisy, myopic, skip, mistake, equal}/
        run_static_SAC.sh
```

Feedback budgets and per-environment hyper-parameters match those of the
corresponding PEBBLE scripts.

## PEBBLE + RM Reset — online PbRL with per-round reward model re-initialisation

`train_PEBBLE.py` with `rm_reset=true` adds a single behavioural change to the
standard PEBBLE loop: the reward model's weights and optimizer are re-initialised
from scratch at the start of every reward learning round **after the first**.

### RM Reset — configuration

Set `rm_reset=true` on the command line or in `config/train_PEBBLE.yaml`:

| Parameter | Default | Description |
|---|---|---|
| `rm_reset` | `false` | Re-initialise reward model weights and optimizer before each reward learning round (excluding the first) |

### RM Reset — running

```bash
./scripts/[env_name]/[teacher_type]/[max_budget]/run_PEBBLE.sh [sampling_scheme] [gpu_id] rm_reset=true
```

Or inline:

```bash
# button_press, noisy teacher, 20 000 feedback pairs, GPU 0, disagreement sampling
seed=12345 python train_PEBBLE.py \
    use_wandb=true gpu=0 \
    env=metaworld_button-press-v2 seed=$seed \
    agent.params.actor_lr=0.0003 agent.params.critic_lr=0.0003 \
    gradient_update=1 activation=tanh \
    num_unsup_steps=9000 num_train_steps=1000000 \
    agent.params.batch_size=512 \
    double_q_critic.params.hidden_dim=256 double_q_critic.params.hidden_depth=3 \
    diag_gaussian_actor.params.hidden_dim=256 diag_gaussian_actor.params.hidden_depth=3 \
    reward_update=10 num_interact=5000 max_feedback=20000 reward_batch=100 \
    feed_type=1 \
    teacher_beta=1 teacher_gamma=1 \
    teacher_eps_mistake=0 teacher_eps_skip=0 teacher_eps_equal=0 \
    rm_reset=true
```

### Implementation

The reset is a two-liner in `RewardModel` (`reward_model.py`):

```python
def reset_ensemble(self):
    self.ensemble = []
    self.paramlst = []
    self.construct_ensemble()
```

Called in `learn_reward()` (`train_PEBBLE.py`) after feedback collection and
before the training loop, conditioned on `cfg.rm_reset and first_flag != 1`.

## Dependency versions (tested)

| Package | Version |
|---|---|
| Python | 3.8 |
| PyTorch | 2.4.1 |
| CUDA toolkit | 12.4 |
| intel-openmp / MKL | 2023.1.0 |
| mujoco | 3.2.3 |
| dm-control | 1.0.23 |
| gym | 0.26.2 |
| wandb | 0.24.2 |
