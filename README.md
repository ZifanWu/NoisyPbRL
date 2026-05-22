## Requirements

- NVIDIA GPU with driver supporting CUDA 12.x
- Conda (Miniconda or Anaconda)

## Install

```bash
conda env create -f conda_env.yml
conda activate bpref

cd custom_dmc2gym && pip install -e . && cd ..
cd Metaworld-2.0.0 && pip install -e . && cd ..
```

> **Note on MuJoCo:** This repo uses the open-source `mujoco` pip package (3.x) and
> `dm-control` 1.x. Physics simulation differs from MuJoCo 2.0, so absolute reward
> numbers will not match prior papers — but algorithm comparisons within this setup
> are internally consistent.

## Dependency versions (tested)

| Package | Version |
|---|---|
| Python | 3.8 |
| PyTorch | 2.4.1 |
| CUDA toolkit | 12.4 |
| mujoco | 3.2.3 |
| dm-control | 1.0.23 |
| gym | 0.26.2 |
| wandb | 0.24.2 |

## Logging

```bash
wandb login          # log in once; set use_wandb=false to fall back to CSV
```

---

## Tandem experiments

The tandem experiment disentangles the contribution of policy activeness, query strategy
activeness, and reward-model data-distribution activeness in iterative PbRL.
All conditions run through `train_PEBBLE.py` with a single `tandem_mode` flag.

### Conditions

| `tandem_mode` | Policy data | RM preference data | Query scorer |
|---|---|---|---|
| `baseline` | own env | own segments (disagreement) | self RM |
| `all_passive` | baseline replay | baseline pairs | — |
| `passive_pol_active_rm` | baseline replay | own segments | self RM |
| `active_pol_passive_rm` | own env | baseline pairs | — |
| `active_pol_passive_query` | own env | own segments | baseline RM |
| `active_pol_passive_dist` | own env | baseline segment pool | self RM |

### Step 1 — Run the baseline

The baseline run must come first; it logs the full replay buffer, preference pairs, RM
checkpoints, and episode metadata to a single HDF5 file that all tandem conditions read.

```bash
python train_PEBBLE.py \
    env=walker_walk seed=1 \
    tandem_mode=baseline \
    agent.params.actor_lr=0.0005 agent.params.critic_lr=0.0005 \
    gradient_update=1 activation=tanh \
    num_seed_steps=1000 num_unsup_steps=5000 num_train_steps=500000 \
    num_interact=5000 max_feedback=500 reward_batch=50 reward_update=200 \
    feed_type=1 \
    teacher_beta=-1 teacher_gamma=1 \
    teacher_eps_mistake=0 teacher_eps_skip=0 teacher_eps_equal=0
```

The HDF5 log is written to the Hydra output directory under
`tandem_baseline_{env}_seed{seed}.h5`. Pass its full path to all tandem conditions
via `tandem_log_path=`.

### Step 2 — Run each tandem condition

All conditions use **identical hyperparameters** to the baseline. The only differences
are `tandem_mode` and `tandem_log_path`.

```bash
LOG=/path/to/tandem_baseline_walker_walk_seed1.h5

# (i) All passive
python train_PEBBLE.py env=walker_walk seed=1 tandem_mode=all_passive \
    tandem_log_path=$LOG <...same hyperparams as baseline...>

# (ii) Passive policy, active RM
python train_PEBBLE.py env=walker_walk seed=1 tandem_mode=passive_pol_active_rm \
    tandem_log_path=$LOG <...same hyperparams...>

# (iii) Active policy, passive RM
python train_PEBBLE.py env=walker_walk seed=1 tandem_mode=active_pol_passive_rm \
    tandem_log_path=$LOG <...same hyperparams...>

# (iv-a) Active policy, passive query (tandem pool scored by baseline RM)
python train_PEBBLE.py env=walker_walk seed=1 tandem_mode=active_pol_passive_query \
    tandem_log_path=$LOG <...same hyperparams...>

# (iv-b) Active policy, passive data distribution (baseline pool scored by tandem RM)
python train_PEBBLE.py env=walker_walk seed=1 tandem_mode=active_pol_passive_dist \
    tandem_log_path=$LOG <...same hyperparams...>
```

### Teacher types

Three synthetic teachers from B-Pref are supported. Fix `seed` and teacher
hyperparameters identically across all conditions for a fair comparison.

| Teacher | `teacher_beta` | `teacher_gamma` | `teacher_eps_mistake` | `teacher_eps_skip` | `teacher_eps_equal` |
|---|---|---|---|---|---|
| Oracle | -1 | 1 | 0 | 0 | 0 |
| Stochastic | 1 | 1 | 0 | 0 | 0 |
| Mistake | -1 | 1 | 0.1 | 0 | 0 |

### GPU selection

```bash
python train_PEBBLE.py ... gpu=1    # uses cuda:1
```

### Key hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `num_seed_steps` | 1000 | Random-action warm-up steps |
| `num_unsup_steps` | 5000 | State-entropy unsupervised exploration steps |
| `num_interact` | 5000 | Env steps between RM update rounds |
| `max_feedback` | 1400 | Total preference-label budget |
| `reward_batch` | 128 | Labels queried per RM update round |
| `reward_update` | 200 | Max RM gradient steps per round (early-stop at acc > 0.97) |
| `feed_type` | 1 | Query strategy (1 = disagreement; all tandem experiments use 1) |
| `large_batch` | 10 | Candidate pool multiplier for disagreement sampling |
| `segment` | 50 | Trajectory segment length (timesteps) |
| `ensemble_size` | 3 | Number of RM ensemble members |
| `rm_reset` | false | Re-initialise RM weights before each update round (except the first) |

### What the baseline log contains

The HDF5 file written by `tandem_mode=baseline` stores everything needed to replay
the run deterministically in any tandem condition:

- **`replay_buffer/`** — full (obs, action, env_reward, next_obs, done) stream indexed by env step
- **`episodes/`** — (episode_id, start_step, length) pointers for pool reconstruction
- **`query_events/event_k/`** — per RM update round:
  - `pool_episode_ids` — which episodes were in the segment pool when querying
  - `rm_before_train/member_{0,1,2}` — RM weights before training (used as scorer in condition iv-a)
  - `sa_t_1`, `sa_t_2`, `r_t_1`, `r_t_2`, `labels`, `disagree_scores` — selected pairs and labels
- **`schedule/`** — `change_batch()` calls (env_step, frac, mb_size)

### Implementation notes

**Data routing.** Three module-level sets in `train_PEBBLE.py` govern routing without
nested if-else trees:

```python
_PASSIVE_POLICY   = {"all_passive", "passive_pol_active_rm"}
_PASSIVE_RM_PAIRS = {"all_passive", "active_pol_passive_rm"}
_ACTIVE_RM_DATA   = {"baseline", "passive_pol_active_rm", "active_pol_passive_query"}
```

**Condition ii (passive_pol_active_rm).** A second environment handle (`_tandem_env`)
is created for RM data collection. The policy's SAC gradient updates use baseline
transitions from the log; the policy's *behavioral* outputs (for the RM segment pool)
come from `_tandem_env`. During the unsupervised phase both data streams mirror
condition (i) — the RM part is moot until RM training begins.

**Condition iv-a (active_pol_passive_query).** A scratch `RewardModel` instance
(`_baseline_rm_helper`) is populated from the logged RM checkpoint before each query
event and used only to score candidate pairs drawn from the tandem's own segment pool.

**Condition iv-b (active_pol_passive_dist).** `TandemReader.reconstruct_episode_pool`
rebuilds the baseline's rolling episode window from (start_step, length) pointers into
the logged replay buffer. These episodes are passed to
`RewardModel.disagreement_sampling_external_pool`, which temporarily swaps `self.inputs`
for querying and restores it immediately after.

**Relabeling invariant.** In every condition, `replay_buffer.relabel_with_predictor`
always uses the *tandem* RM — never the baseline RM. This is enforced architecturally:
the baseline RM is only ever loaded into `_baseline_rm_helper` and never replaces
`self.reward_model`.
