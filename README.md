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

### PEBBLE + RUNE (reward uncertainty exploration bonus)

`train_PEBBLE_explore.py` adds an intrinsic exploration bonus equal to the **std of the reward model ensemble** scaled by a decaying β coefficient. The exploration buffer tracks extrinsic and intrinsic rewards separately.

```bash
# walker_walk, oracle teacher, 100 feedback pairs, disagreement sampling
CUDA_VISIBLE_DEVICES=0 python train_PEBBLE_explore.py \
    env=walker_walk seed=12345 \
    agent.params.actor_lr=0.0005 agent.params.critic_lr=0.0005 \
    gradient_update=1 activation=tanh \
    num_unsup_steps=9000 num_train_steps=500000 \
    num_interact=20000 max_feedback=100 reward_batch=10 reward_update=50 \
    feed_type=1 \
    teacher_beta=-1 teacher_gamma=1 \
    teacher_eps_mistake=0 teacher_eps_skip=0 teacher_eps_equal=0 \
    agent.params.beta_schedule=linear_decay agent.params.beta_init=0.05 agent.params.beta_decay=0.00001

# metaworld_hammer-v2, oracle teacher, 10000 feedback pairs
CUDA_VISIBLE_DEVICES=0 python train_PEBBLE_explore.py \
    env=metaworld_hammer-v2 seed=12345 \
    agent.params.actor_lr=0.0003 agent.params.critic_lr=0.0003 \
    gradient_update=1 activation=tanh num_unsup_steps=9000 num_train_steps=2000000 \
    agent.params.batch_size=512 \
    double_q_critic.params.hidden_dim=256 double_q_critic.params.hidden_depth=3 \
    diag_gaussian_actor.params.hidden_dim=256 diag_gaussian_actor.params.hidden_depth=3 \
    reward_update=10 num_interact=5000 max_feedback=10000 reward_batch=50 \
    feed_type=1 \
    teacher_beta=-1 teacher_gamma=1 \
    teacher_eps_mistake=0 teacher_eps_skip=0 teacher_eps_equal=0
```

Key RUNE hyperparameters (set in config or command line):

| Parameter | Default | Description |
|---|---|---|
| `agent.params.beta_schedule` | `linear_decay` | Schedule for exploration bonus weight (`constant` or `linear_decay`) |
| `agent.params.beta_init` | `0.05` | Initial β for exploration bonus |
| `agent.params.beta_decay` | `0.00001` | Per-step multiplicative decay of β |

### SURF (semi-supervised reward learning with data augmentation)

`train_PEBBLE_semi_dataaug.py` improves reward model sample efficiency by:

1. collecting `inv_label_ratio × mb_size` **unlabeled** queries each round in addition to labeled ones
2. training with pseudo-labels above `threshold_u` confidence and temporal crop augmentation

```bash
# walker_walk, oracle teacher, 100 feedback pairs
CUDA_VISIBLE_DEVICES=0 python train_PEBBLE_semi_dataaug.py \
    env=walker_walk seed=12345 \
    agent.params.actor_lr=0.0005 agent.params.critic_lr=0.0005 \
    gradient_update=1 activation=tanh \
    num_unsup_steps=9000 num_train_steps=500000 \
    num_interact=20000 max_feedback=100 reward_batch=10 \
    inv_label_ratio=100 reward_update=1000 \
    feed_type=1 \
    teacher_beta=-1 teacher_gamma=1 \
    teacher_eps_mistake=0 teacher_eps_skip=0 teacher_eps_equal=0 \
    threshold_u=0.99 mu=4

# metaworld_hammer-v2, oracle teacher, 10000 feedback pairs
CUDA_VISIBLE_DEVICES=0 python train_PEBBLE_semi_dataaug.py \
    env=metaworld_hammer-v2 seed=12345 \
    agent.params.actor_lr=0.0003 agent.params.critic_lr=0.0003 \
    gradient_update=1 activation=tanh num_unsup_steps=9000 num_train_steps=2000000 \
    agent.params.batch_size=512 \
    double_q_critic.params.hidden_dim=256 double_q_critic.params.hidden_depth=3 \
    diag_gaussian_actor.params.hidden_dim=256 diag_gaussian_actor.params.hidden_depth=3 \
    reward_update=20 num_interact=5000 max_feedback=10000 reward_batch=50 \
    feed_type=1 \
    teacher_beta=-1 teacher_gamma=1 \
    teacher_eps_mistake=0 teacher_eps_skip=0 teacher_eps_equal=0 \
    threshold_u=0.99 mu=4 inv_label_ratio=10
```

Key SURF hyperparameters:

| Parameter | Default | Description |
|---|---|---|
| `inv_label_ratio` | `10` | Ratio of unlabeled to labeled queries collected each round |
| `threshold_u` | `0.95` | Confidence threshold for pseudo-label acceptance |
| `mu` | `1` | Unlabeled batch size multiplier relative to labeled batch |
| `lambda_u` | `1` | Weight of the pseudo-label loss term |
| `dataaug_window` | `5` | Half-window added to each side of segment for temporal cropping |
| `crop_range` | `5` | ±range for random crop length around the original segment size |

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

## Scripts overview

All run scripts live under `scripts/` and follow the layout:

```text
scripts/
  {env}/
    {max_feedback}/
      {teacher}/
        run_PEBBLE.sh        [feed_type] [gpu]
        run_RUNE.sh          [feed_type] [gpu]
        run_SURF.sh          [feed_type] [gpu]
        run_PrefPPO.sh       [feed_type] [gpu]
        run_static_SAC.sh    [gpu]
```

`$1` selects the query sampling scheme (0 = uniform, 1 = disagreement, 2 = entropy).
`$2` selects the GPU (default: 0).

### Environments

| Env directory | Gym / DMC id | Algorithms |
|---|---|---|
| `button_press` | `metaworld_button-press-v2` | PEBBLE, RUNE, SURF, PrefPPO, Static SAC |
| `sweep_into` | `metaworld_sweep-into-v2` | PEBBLE, RUNE, SURF, PrefPPO, Static SAC |
| `walker_walk` | `walker_walk` | PEBBLE, RUNE, SURF, PrefPPO |
| `quadruped_walk` | `quadruped_walk` | PEBBLE, RUNE, SURF, PrefPPO, Static SAC |
| `hammer` | `metaworld_hammer-v2` | PEBBLE, RUNE, SURF, PrefPPO |
| `door_close` | `metaworld_door-close-v2` | PEBBLE, RUNE, SURF, PrefPPO |
| `door_open` | `metaworld_door-open-v2` | PEBBLE, RUNE, SURF, PrefPPO |
| `door_unlock` | `metaworld_door-unlock-v2` | PEBBLE, RUNE, SURF, PrefPPO |
| `drawer_open` | `metaworld_drawer-open-v2` | PEBBLE, RUNE, SURF, PrefPPO |
| `window_close` | `metaworld_window-close-v2` | PEBBLE, RUNE, SURF, PrefPPO |

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

---

## Axis 2 Tier B — Reward Model Error Dissociation

This experiment validates whether three gold-free instruments (κ̂, ensemble spread, GoF residual)
can each uniquely detect a distinct reward model error component (e_shift, e_epi, e_mis) in a
realistic PEBBLE/Metaworld setup.  The headline claim: misspecification is in the *blind spot* of both
κ̂ and ensemble spread — only GoF catches it.

### Design choices

#### Query sampling — `feed_type=0` (uniform)

All four conditions use uniform sampling deliberately.  Disagreement or entropy sampling actively
reduce epistemic uncertainty by targeting high-variance queries; this would suppress e_epi in the
epi condition and interfere with the clean three-way dissociation.  Uniform sampling holds the
query distribution constant across conditions so that the only variables are the experimental
knobs (relabeling frequency, label budget, RM capacity).

#### On-policy window — `buffer_window_rounds`

Each condition is designed to elevate exactly one error component while keeping the other two at
control levels.  Using different buffer regimes across conditions would violate this: if the shift
condition windowed while control accumulated, the shift RM would see fewer total training pairs —
elevating ensemble spread for a reason unrelated to support staleness and contaminating the
off-diagonal entries of the 3×3 matrix.

All four conditions therefore use `buffer_window_rounds=3`: the buffer keeps only the last 3
relabeling rounds' labeled pairs (≈3 × 50 = 150 pairs), evicting the oldest round whenever a
new one is added.  The RM fitting regime is identical across conditions; the only things that vary
are the three experimental knobs.

Effect per condition:

- **Control / mis**: policy drift per round is small (`num_interact=10000`); the 3-round window
  is approximately on-policy.  The 16×1 mis RM saturates at ~150 pairs regardless — its bias
  floor is set by capacity, not label count.
- **Shift**: the window becomes stale as the policy drifts over 100 000 steps between rounds.
  ψ_cur's training distribution no longer covers the current policy's support, so κ̂ elevates
  cleanly without a confounding epistemic signal from mixed-support training.
- **Epi**: four 50-query rounds of data can exist (`max_feedback=200`), so k=3
  retains the latest 150 active pairs and keeps the finite-label regime information-starved.

k=3 balances isolation (small k → cleaner staleness signal) against RM fit quality (large k →
more pairs).  With 150 active pairs on a 30-dimensional obs+action space the 256×3 ensemble is
reasonably fitted; elevated spread in the control condition at any monitoring step would signal
underfitting and warrant increasing k.  Override on the command line with
`buffer_window_rounds=2` or `5` to sweep this trade-off.

#### Teacher — Bradley-Terry for all four conditions; β as the epi knob

All four conditions use the Bradley-Terry (BT) noisy teacher.  Control / shift / mis hold the
same *high* β (≈ clean labels) for comparability; epi *lowers* β as its only knob.

| Condition | teacher_beta | bt_normalize | max_feedback | Rationale |
|---|---|---|---|---|
| control | 5 | true | 2000 | high β → ~0.7% mistake rate at median \|ΔR\|; instruments should read near-zero |
| shift | 5 | true | 2000 | same teacher cleanliness as control; only `num_interact` differs |
| epi | **1** | true | 200 | **EPI KNOB**: P(mistake) at median ≈ 27%; well-specified BT noise → pure epistemic |
| mis | 5 | true | 2000 | same teacher cleanliness as control; only `rm_hidden_dim` differs |

**Why BT noise is *pure* epistemic, while eps_mistake is not.**  Under standard BT loss, BT-noisy
labels admit a *correctly specified* MLE: with `n → ∞`, the recovered reward function converges to
the true reward up to a global scale factor β.  All disagreement at finite n is purely
finite-sample variance — the textbook definition of epistemic uncertainty.  In contrast,
`teacher_eps_mistake=ε` (i.i.d. flip on oracle labels) is *mis-specified* under BT loss: the
asymptotic MLE saturates at `r̂₂ − r̂₁ → logit(1−ε)` for every preferred pair, losing all
fine-grained reward information.  This irreducible saturation contaminates e_epi with
misspecification artifacts — exactly what e_mis is supposed to own.  BT (β>0) avoids this.

**Why β is gap-std-normalized (`bt_normalize_by_gap_std: true`).**  The BT mistake rate depends
on the product `β · |ΔR|`.  A fixed numeric β implies a *different* effective cleanliness as the
policy improves and segment-return scales drift — control might end up noisier than mis simply
because the policy performs differently in each condition.  With normalization, the running EMA
of `std(|ΔR|)` (computed inside `get_label`, EMA decay 0.9) is divided into β.  The configured
`teacher_beta` becomes a *unitless* target: P(mistake) at a typical pair (|ΔR| ≈ std) is
`σ(−teacher_beta)`, condition-invariant and time-invariant.

The mistake teacher (`teacher_eps_mistake > 0`) is reserved for the **mis-condition confirmatory
run only** — to verify that the misspecification signal under capacity limits is robust to a
qualitatively different noise model.  Run it via CLI override:

```bash
python train_PEBBLE_axis2.py env=metaworld_door-open-v2 seed=1 \
    teacher_beta=-1 teacher_eps_mistake=0.1 bt_normalize_by_gap_std=false \
    --config-name train_PEBBLE_axis2_mis
```

Skip and Myopic teachers are not used because they introduce systematic distortions — temporal
discounting (myopic) and abstentions that depend on reward magnitude (skip) — both of which
induce a preference function different from the gold reward.  That is closer to misspecification
than to finite-sample variance.

### How to run

W&B is the primary live dashboard for Axis 2. Each run logs normal train/eval metrics plus
`axis2/*` monitor metrics (`axis2/kappa_hat`, `axis2/ensemble_spread`, `axis2/gof_deployed`,
`axis2/e_shift`, `axis2/e_epi`, `axis2/e_mis`, etc.). The local `axis2_metrics.csv` mirrors the
same monitor rows for offline analysis and reproducibility.

Log in once before launching runs:

```bash
wandb login
```

The commands below still use CLI overrides for configuration. `--config-name` is just a condition
selector shim for this repo's older config parser; valid names are `train_PEBBLE_axis2`,
`train_PEBBLE_axis2_shift`, `train_PEBBLE_axis2_epi`, and `train_PEBBLE_axis2_mis`. Set
`use_wandb=false` or `WANDB_MODE=offline` only when you intentionally want local-only logging.

#### Step 0 — pre-flight sanity checks (run once before any full experiment)

```bash
cd /home/zifan/NoisyPbRL
/home/zifan/miniconda3/envs/bpref/bin/python -m axis2_tier_b.sanity_checks
# Expected: 7/7 passed
```

#### Step 1 — pilot run (verify output paths and monitoring)

```bash
cd /home/zifan/NoisyPbRL
/home/zifan/miniconda3/envs/bpref/bin/python train_PEBBLE_axis2.py \
    env=metaworld_door-open-v2 seed=12345 \
    num_train_steps=200000 monitor_frequency=20000
# Output: /home/zifan/NoisyPbRL/exp/axis2_tier_b/metaworld_door-open-v2/control/seed12345/axis2_metrics.csv
```

For the same control run with the explicit config selector:

```bash
/home/zifan/miniconda3/envs/bpref/bin/python train_PEBBLE_axis2.py \
    env=metaworld_door-open-v2 seed=12345 \
    num_train_steps=200000 monitor_frequency=20000 \
    --config-name train_PEBBLE_axis2
```

In W&B, check the `axis2/*` panel first. The mirrored CSV should have monitoring rows with all 13 columns present and finite where expected. `floor_he` and `floor_ref` are logged as a GoF floor calibration check.

#### Step 2 — full runs (6 seeds × 4 conditions on Metaworld door-open)

```bash
cd /home/zifan/NoisyPbRL
ENV=metaworld_door-open-v2

for seed in 1 2 3 4 5 6; do

  # control
  /home/zifan/miniconda3/envs/bpref/bin/python train_PEBBLE_axis2.py \
      env=$ENV seed=$seed --config-name train_PEBBLE_axis2 &

  # distribution shift: 10x rarer relabeling than control
  /home/zifan/miniconda3/envs/bpref/bin/python train_PEBBLE_axis2.py \
      env=$ENV seed=$seed --config-name train_PEBBLE_axis2_shift &

  # epistemic: low-beta BT teacher + 10x fewer labels
  /home/zifan/miniconda3/envs/bpref/bin/python train_PEBBLE_axis2.py \
      env=$ENV seed=$seed --config-name train_PEBBLE_axis2_epi &

  # misspecification: default 16x1 RM
  /home/zifan/miniconda3/envs/bpref/bin/python train_PEBBLE_axis2.py \
      env=$ENV seed=$seed --config-name train_PEBBLE_axis2_mis &

done
wait
```

To sweep RM capacity in the mis condition:

```bash
cd /home/zifan/NoisyPbRL
ENV=metaworld_door-open-v2

for h in 8 16 32 64; do
  for seed in 1 2 3; do
    /home/zifan/miniconda3/envs/bpref/bin/python train_PEBBLE_axis2.py \
        env=$ENV seed=$seed rm_hidden_dim=$h \
        --config-name train_PEBBLE_axis2_mis &
  done
done
wait
```

Confirmatory linear-head run (pure affine RM, convex BT loss):

```bash
cd /home/zifan/NoisyPbRL
/home/zifan/miniconda3/envs/bpref/bin/python train_PEBBLE_axis2.py \
    env=metaworld_door-open-v2 seed=1 \
    rm_num_layers=0 rm_output_activation=none \
    --config-name train_PEBBLE_axis2_mis
```

Confirmatory mistake-teacher run in the mis condition:

```bash
cd /home/zifan/NoisyPbRL
/home/zifan/miniconda3/envs/bpref/bin/python train_PEBBLE_axis2.py \
    env=metaworld_door-open-v2 seed=1 \
    teacher_beta=-1 teacher_eps_mistake=0.1 bt_normalize_by_gap_std=false \
    --config-name train_PEBBLE_axis2_mis
```

#### Step 3 — analysis

```bash
cd /home/zifan/NoisyPbRL
/home/zifan/miniconda3/envs/bpref/bin/python -m axis2_tier_b.analysis \
    --results_dir /home/zifan/NoisyPbRL/exp/axis2_tier_b
```

Output: `exp/axis2_tier_b/results/results.md` with the 3×3 Spearman matrix, partial correlations,
9 scatter plots, and H2.1–H2.5 verdicts.

#### Step 4 — payoff test (H2.5, requires step 2 checkpoints)

```bash
cd /home/zifan/NoisyPbRL
/home/zifan/miniconda3/envs/bpref/bin/python -m axis2_tier_b.payoff \
    --ckpt_dir /home/zifan/NoisyPbRL/exp/axis2_tier_b/metaworld_door-open-v2/shift/seed1/ckpt_500000
```

### Outputs

Primary live tracking is in W&B under the `NoisyPbRL` project. Local artifacts are written as:

```text
exp/axis2_tier_b/
  metaworld_door-open-v2/
    control/seed{1..6}/
      axis2_metrics.csv          <- 13-column monitoring log (one row per monitor_frequency steps)
      ckpt_500000/               <- agent + RM + replay buffer checkpoint (H2.5 payoff test)
    shift/seed{1..6}/
    epi/seed{1..6}/
    mis_h{8,16,32,64}_l{1}/seed{1..6}/
  results/
    results.md                   <- H2.1-H2.5 verdicts + 3x3 matrix
    spearman_matrix.png
    partial_corr_matrix.png
    scatter_*.png                <- 9 instrument x component scatter plots
```

### Metrics CSV columns

| Column | Description |
|---|---|
| `step` | Environment step at which measurement was taken |
| `kappa_hat` | κ̂ — REINFORCE-style policy gradient ratio; detects distribution shift |
| `sigma_hat` | Cosine alignment of g_delta and g0 |
| `ensemble_spread` | Mean per-step reward variance across ensemble members; detects epistemic uncertainty |
| `gof_deployed` | GoF residual of deployed RM against teacher-labeled held-out pairs (epistemic + misspec) |
| `gof_inf` | GoF residual of ψ_inf against clean held-out pairs (pure misspecification floor) |
| `gof_epi` | `gof_deployed − gof_inf` (epistemic contribution to GoF) |
| `floor_used` | CE floor used for `gof_deployed` (max of H(ε) and ψ_inf held-out loss) |
| `e_shift` | Distribution shift component — L2 distance between ψ_cur and ψ_fin on current support |
| `e_epi` | Epistemic component — L2 distance between ψ_fin and ψ_inf |
| `e_mis` | Misspecification component — L2 distance between ψ_inf and R_star |
| `floor_he` | Analytic CE floor from `teacher_eps_mistake` when applicable |
| `floor_ref` | Empirical clean-pair CE floor from ψ_inf |

### What conclusions to reach

The experiment is designed to produce a 3×3 instrument × component Spearman matrix that is
**diagonally dominant**:

| | e_shift | e_epi | e_mis |
|---|---|---|---|
| **κ̂** | **high** | low | ~0 |
| **ensemble spread** | low | **high** | ~0 |
| **gof_deployed** | low | low | **high** |

Each hypothesis below has a pass/fail verdict in `results/results.md`:

**H2.1 (shift → κ̂):** In the shift condition, κ̂ should be significantly elevated vs control
and should track e_shift across monitoring steps.  Ensemble spread and `gof_deployed` should stay near
control levels.  This shows κ̂ is the unique leading indicator of support drift.

**H2.2 (epi → ensemble spread):** In the epi condition (low-beta BT teacher, 10× fewer labels), ensemble
spread should be elevated and should correlate with e_epi.  κ̂ may be mildly elevated (policy is
uncertain about the RM signal) but should not dominate.  `gof_epi = gof_deployed - gof_inf` may
show the finite-label contribution, while `gof_inf` should stay near the control floor.

**H2.3 (mis → GoF, κ̂/spread blind spot):** In the mis condition (16×1 RM, clean labels),
`gof_deployed` should be significantly elevated while κ̂ and ensemble spread remain near control
levels.  This is the headline result: a capacity-limited RM has high irreducible GoF but looks fine
to the other two instruments.  The RM capacity sweep (h in {8,16,32,64}) should show GoF decreasing
as capacity increases, while κ̂ and ensemble spread stay flat.

**H2.4 (partial correlations):** After controlling for the other two error components, the partial
correlation of each instrument with its matched component should be significantly positive, while
partial correlations with unmatched components should not be significantly different from zero.
This rules out the alternative explanation that one instrument detects everything.

**H2.5 (payoff — instrument predicts recoverability):** At the step-500k checkpoint, apply three
remedies and measure Δgold (change in true episode return over 20k further steps):

- Remedy 1 (relabel at current θ): recovers gold in shift condition; κ̂ predicts Δgold(1).
- Remedy 2 (more clean labels): recovers gold in epi condition; ensemble spread predicts Δgold(2).
- Remedy 3 (bigger RM): recovers gold in mis condition; gof_deployed predicts *un-recoverability*
  (high gof → near-zero Δgold from remedies 1 and 2; only remedy 3 helps).

A null result on any off-diagonal entry (e.g., κ̂ also correlates with e_mis) would suggest the
instruments are not as cleanly dissociated as hypothesized and warrants checking whether the
experimental knobs sufficiently isolate the corresponding error component.

### Implementation notes

**How is κ̂ computed?**

κ̂ answers: "how much would the actor's gradient direction change if ψ_cur were replaced by a
fresh refit on current data?"  It is computed in five steps at every monitoring event, with the
actor and RM frozen in eval mode:

1. **Sample probe state-actions** — draw `probe_size=1024` (s, a) pairs from the most recent
   on-policy episodes.  The probe is refreshed every call so it stays on the current support.

2. **Compute g0** — the REINFORCE policy gradient under the current RM:

   ```text
   g0 = ∇_θ [ mean_i( r_ψ(sᵢ, aᵢ) · log π_θ(aᵢ|sᵢ) ) ]
   ```

   Actions `aᵢ ~ π_θ(·|sᵢ)` are sampled once and held fixed for both gradient passes.

3. **Approximate ψ'** — deep-copy ψ_cur, then run `kappa_warmup_steps=10` Adam steps on
   `kappa_fresh_prefs=64` preference pairs labeled from current on-policy data using the
   deployed teacher settings (BT beta/normalization plus any configured mistake flips).  This warm-started clone is not a full refit
   but moves directionally toward what a fresh refit at the current support would predict.

4. **Compute g_delta** — the REINFORCE gradient of the reward *shift* (ψ' minus ψ_cur), using
   the same fixed probe actions:

   ```text
   g_delta = ∇_θ [ mean_i( (r_ψ'(sᵢ, aᵢ) − r_ψ(sᵢ, aᵢ)) · log π_θ(aᵢ|sᵢ) ) ]
   ```

5. **Return the ratio and alignment:**

   ```text
   κ̂ = ‖g_delta‖ / ‖g0‖
   σ̂ = cos(g0, g_delta)
   ```

κ̂ large means the reward shift between ψ_cur and ψ' would substantially redirect the actor's
gradient — the RM is stale enough to matter for policy learning.  σ̂ < 0 means the shift opposes
the current update direction (actively misleading rather than merely noisy).

**Approximation caveat:** step 3 uses 10 warm-start steps rather than a full convergent refit.
When distribution shift is severe, the clone has not moved far enough from ψ_cur, so `r_ψ' − r_ψ`
underestimates the true reward shift and κ̂ is a **conservative lower bound** on the actual
gradient sensitivity.  In practice this means κ̂ rises clearly in the shift condition but may not
saturate even when shift is large — interpret the absolute value with caution, use relative
comparisons across conditions and monitoring steps.

**Why REINFORCE-style gradient for κ̂, not SAC's reparameterization trick?**

The κ̂ probe computes `g0 = ∇_θ[r_ψ · log π_θ(a|s)]` — the REINFORCE policy gradient under the
current RM. SAC's own update uses reparameterization (`∇_θ[r_ψ(s, a(ε,θ))]` with a
noise-reparameterized action), which embeds the RM signal *inside* the action computation. If
κ̂ used reparameterization, the gradient ratio would conflate the RM's signal with the policy's
own structural gradient (the Jacobian of the action w.r.t. θ). REINFORCE keeps those separate:
`log π` is the only place θ appears, so `||g_delta|| / ||g0||` cleanly measures how much a RM
shift would change the actor's ascent direction, independent of the action sampling mechanism.
The probe does not need to match SAC's update rule; it needs to answer "how much does the RM
move the actor?" — which REINFORCE answers directly.

**Why chained gauge fix for error decomposition, not gold-anchoring all three RMs?**

The three error components are measured against an ordered chain of reference points:

```text
e_shift = ||gauge_fix(ψ_cur, ψ_fin) − ψ_fin|| / ||R*||   (ψ_cur vs fresh refit at current support)
e_epi   = ||gauge_fix(ψ_fin, ψ_inf) − ψ_inf|| / ||R*||   (finite-label vs infinite-label refit)
e_mis   = ||gauge_fix(ψ_inf, R*)    − R*||    / ||R*||   (best-fit RM vs true reward)
```

If instead all three were anchored directly to R*, they would not be orthogonal. In the shift
condition, ψ_cur is well-fitted to its stale support, so `||ψ_cur − R*||` is dominated by the
support gap *plus* the epistemic gap *plus* the misspecification floor — e_shift would absorb
all of them. In the epi condition, ψ_fin would look identical to ψ_cur under gold-anchoring
because both fit the same support; e_epi would collapse to near-zero. Chaining ensures each
component measures *only* its own gap: shift = staleness, epi = finite-sample variance,
misspecification = irreducible capacity floor.

**ψ_fin / ψ_inf are refits of the *deployed* RM class (not a full-capacity yardstick).**

By spec, the nested refits inside `compute_error_components` use the same architecture as the
deployed RM (`rm_hidden_dim`, `rm_num_layers`, `rm_output_activation`).  In the mis condition that
means ψ_fin and ψ_inf are *also* 16×1 — ψ_inf trained on abundant clean labels is the
"best achievable function in the 16×1 class," and `e_mis = ||gauge_fix(ψ_inf, R*) − R*|| / ||R*||`
measures the irreducible **capacity floor of the diagnosed class**.

An earlier version of this design forced ψ_fin/ψ_inf to a full-capacity (256×3) yardstick out of
concern that "e_mis collapses to zero by construction."  That concern was wrong: `gauge_fix` only
removes the per-segment additive constant and the global scale; it cannot absorb a *functional-form*
mismatch.  A 16×1 MLP cannot represent the nonlinear shape of R\*, and that shape error survives
gauge fixing — exactly what we want `e_mis` to measure.  Forcing the yardstick to full capacity
instead would make:

- `e_mis` measure the 256×3 yardstick's residual against R\* (near zero — uninformative).
- `e_shift` mix in a capacity *upgrade* (small deployed vs full-capacity refit) on top of support drift.
- `e_epi` measure the 256×3 yardstick's finite-sample noise, not the deployed class's.

The yardstick override (`yardstick_hidden_dim`, `yardstick_num_layers`, `yardstick_output_activation`)
is now `null` in all main configs and is **only** used by sanity-check smoke tests to keep runtime
manageable.  Production runs always refit in the deployed class.

**GoF instrument is `gof_deployed`, not `gof_inf`.**  `axis2_tier_b/analysis.py` uses
`gof_deployed` — the deployed RM's CE residual on held-out pairs (minus the empirical floor) — as
the GoF instrument in the 3×3 matrix.  `gof_inf` is logged for the chained decomposition
(`gof_epi = gof_deployed − gof_inf`) but is the floor *of the deployed class*; using it as the
instrument would diagnose the floor, not the deployed RM's distance from it.

**What does `gauge_fix(ψ, R_ref)` do?**

BT-trained reward models are only identified up to two ambiguities: (1) a per-segment additive
constant (BT labels depend only on the *difference* in cumulative return, so shifting each
segment's rewards by an arbitrary constant leaves all preferences unchanged), and (2) a global
global scale (BT identifies reward scale only relative to the teacher noise/temperature, and
independently trained RMs can drift to different numeric scales). Without removing these, `||ψ_cur − ψ_fin||` would be dominated by whichever ambiguity happens
to be largest — making e_shift, e_epi, and e_mis incomparable across seeds or conditions and
sensitive to initialization rather than genuine reward disagreement.

`gauge_fix(R_ψ, R_ref)` resolves both:

1. **Per-segment centering** — subtract each segment's own temporal mean from both arrays,
   removing the additive-constant degree of freedom.
2. **Global scale alignment** — find the scalar `s* = ⟨R_ψ_c, R_ref_c⟩ / ||R_ψ_c||²` (OLS)
   and return `s* · R_ψ_c`, aligning ψ's scale to the reference before measuring distance.

After gauge-fixing, L2 distance measures only the part of the disagreement that actually changes
preferences — the part the teacher can, in principle, distinguish.


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
