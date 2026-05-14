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

> **Note on MetaWorld:** `metaworld` is not installed by default. It is only needed
> for MetaWorld environments (`metaworld_*`). Install it separately if required.

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
