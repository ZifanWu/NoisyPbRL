# Gauge-Ambiguity Experiment — Reproduction Guide

## Overview

This experiment tests whether the **gauge choice** in iterative PbRL
(how the reward model's additive constant is fixed) causes systematically
different policy learning trajectories.

Three gauge modes are compared:

| Mode | Optimizer | Post-step projection | Description |
|------|-----------|----------------------|-------------|
| `l2` | Adam, `weight_decay=1e-4` | None | Standard RLHF ℓ₂ regularisation |
| `none` | Adam, `weight_decay=0` | None | No regularisation (original BPref) |
| `zero_mean_ref` | Adam, `weight_decay=0` | Subtract mean on frozen D_ref | Explicit zero-mean constraint |

Experiment matrix: 2 envs × 3 gauge modes × 5 seeds = **30 runs** (Track 1 standard).

---

## Prerequisites

```bash
# Existing BPref dependencies assumed.  Additional requirements:
pip install wandb matplotlib pandas scipy tabulate
wandb login          # authenticate once
```

Set the W&B project:
```bash
export WANDB_PROJECT=pbrl_gauge_ambiguity
```

---

## RM update count by environment

The number of RM retraining events is `ceil(max_feedback / reward_batch)`.  This
varies significantly across environment families:

| Environment family | `max_feedback` | `reward_batch` | `reward_update` | RM updates |
|-------------------|---------------|----------------|-----------------|------------|
| DMControl (`walker_walk`, `cheetah_run`) | 1 400 | 128 | 200 | **~11** |
| MetaWorld (`button_press`, `window_close`, …) | 20 000 | 100 | 10 | **~200** |

For the gauge hypothesis (divergence accumulates over iterations) the MetaWorld
setting already provides the dense iterative regime.  **MetaWorld environments are
the better primary choice** for this experiment; the DMControl stress test
(`num_interact=1000`) is only needed if MetaWorld is unavailable or results there
are inconclusive.

---

## Track 1 — MetaWorld, standard regime (~200 RM updates)

Recommended primary track.  Uses the existing `button_press` script parameters.

### Single run (sanity check)
```bash
python train_PEBBLE.py \
    env=metaworld_button-press-v2 \
    seed=1 \
    gauge_mode=l2 \
    tandem_mode=baseline \
    use_wandb=true \
    num_unsup_steps=9000 \
    num_interact=5000 \
    max_feedback=20000 \
    reward_batch=100 \
    reward_update=10 \
    num_train_steps=1000000 \
    exp_dir=/path/to/exp
```

### Full 30-run matrix (use SLURM or a parallel launcher)

```bash
for ENV in metaworld_button-press-v2 metaworld_window-close-v2; do
for GAUGE in l2 none zero_mean_ref; do
for SEED in 1 2 3 4 5; do
  sbatch tandem_slurm_submit.sh \
    env=$ENV \
    gauge_mode=$GAUGE \
    seed=$SEED \
    num_unsup_steps=9000 \
    num_interact=5000 \
    max_feedback=20000 \
    reward_batch=100 \
    reward_update=10 \
    tandem_mode=baseline \
    use_wandb=true
done; done; done
```

Estimated compute: each run ≈ **2–3 GPU-hours** on a single A100/V100.
Total: 30 × 2.5 h ≈ **75 GPU-hours**.

---

## Track 2 — DMControl, stress test (~50 RM updates, optional)

Only needed if MetaWorld is unavailable or as a cross-domain replication.
Set `num_interact=1000` to increase RM retraining frequency from ~11 to ~50
with the DMControl default `max_feedback=1400`.

```bash
for ENV in walker_walk cheetah_run; do
for GAUGE in l2 none zero_mean_ref; do
for SEED in 1 2 3; do
  sbatch tandem_slurm_submit.sh \
    env=$ENV \
    gauge_mode=$GAUGE \
    seed=$SEED \
    num_interact=1000 \
    tandem_mode=baseline \
    use_wandb=true
done; done; done
```

Estimated compute: 18 runs × 2.5 h ≈ **45 GPU-hours**.

---

## Logged metrics

Every RM training event (`~learn_reward` call, ≈ every `num_interact` steps):

| Metric | Description |
|--------|-------------|
| `rm/mean_on_ref` | Mean RM-predicted return on frozen D_ref (256 segments) |
| `rm/std_on_ref` | Std of RM-predicted return on D_ref |
| `rm/mean_on_policy` | Mean RM-predicted return on 64 fresh on-policy segments |
| `rm/std_on_policy` | Std of on-policy return |
| `rm/gauge_gap` | `mean_on_policy − mean_on_ref` (key diagnostic) |
| `rm/param_norm` | ℓ₂ norm of all RM parameters (all ensemble members) |

Every `eval_frequency=10 000` env steps:

| Metric | Description |
|--------|-------------|
| `eval/true_episode_reward` | True environment return (primary outcome) |
| `eval/proxy_return` | RM-predicted episode return (different scale from true) |
| `eval/return_gap` | `proxy_return − true_return` (Goodhart indicator) |
| `train/reward_model_acc` | BT classification accuracy (sanity check C3) |

---

## Sanity checks before trusting results

Run these before producing final plots:

| Check | What to verify | Failure action |
|-------|---------------|----------------|
| C1 | Mode `l2`: `rm/param_norm` stays bounded over training | Weight decay not applied — check optimizer config |
| C2 | Mode `zero_mean_ref`: `rm/gauge_gap < 0.1 × ‖r‖_typical` | Projection broken — check `_apply_zero_mean_projection` |
| C3 | All modes: `train/reward_model_acc` converges to similar values | One mode is worse at fitting BT — investigate LR/WD interaction |
| C4 | All modes: same number of RM training events | Feedback budget or `num_interact` mismatch |
| C5 | Same env steps, same seeds per mode | Verify Hydra output dirs are separate (gauge_mode in path) |
| C6 | Mode `l2` reproduces published PEBBLE performance (within seed noise) | Baseline setup is wrong — compare against paper numbers |

---

## Analysis

After all runs finish:

```bash
python analyze_gauge.py \
    --project pbrl_gauge_ambiguity \
    --envs walker_walk cheetah_run \
    --seeds 1 2 3 4 5 \
    --output results/ \
    --track both
```

Outputs:

```
results/
  track1/
    plot1_true_return.png     # Primary figure: true return per env per gauge mode
    plot2_gauge_gap.png       # rm/gauge_gap over time
    plot3_param_norm.png      # rm/param_norm over time
    plot4_return_gap.png      # Goodhart indicator
    results.md                # Summary + verdict + stats table
  track2/                     # Same structure (if Track 2 runs exist)
    ...
```

---

## Implementation notes

### Where seeding happens

`utils.set_seed_everywhere(cfg.seed)` is called in `Workspace.__init__` before
any environment or model construction — this covers PyTorch, NumPy, and the
environment RNG.  The gauge eval env (`_gauge_eval_env`) is created in the same
`__init__`, so it shares the same seeding context.

### Reference set timing

`D_ref` (256 segments) is collected via `_collect_on_policy_segments(256)` at
the **first** `learn_reward` call (`first_flag=1`), which fires at
`num_seed_steps + num_unsup_steps = 6 000` env steps.  At this point the policy
has completed unsupervised exploration but has not yet seen any preference labels.
This is the intended π_ref (unsup-pretrained initial policy).

### RM update count

The RM is retrained as long as `total_feedback < max_feedback`, once per
`num_interact` env steps.  The number of updates is `ceil(max_feedback /
reward_batch)` — this depends on environment configuration, not just
`num_interact`:

- **MetaWorld** (`max_feedback=20000`, `reward_batch=100`): **~200 updates** —
  the full iterative regime. Recommended primary track.
- **DMControl** (`max_feedback=1400`, `reward_batch=128`): **~11 updates** —
  too sparse for the gauge effect to accumulate meaningfully without adjusting
  `num_interact`. Use `num_interact=1000` (stress test) to bring this to ~50.

### Dependencies on existing code

No other training scripts (`train_PT.py`, `train_static_sac.py`, etc.) were
modified.  The gauge experiment is isolated to:

- `reward_model.py` — `gauge_mode` param, `set_ref_segments`, `_apply_zero_mean_projection`, `get_gauge_diagnostics`
- `train_PEBBLE.py` — gauge env, `_collect_on_policy_segments`, `evaluate` proxy logging, `learn_reward` diagnostic dispatch
- `config/train_PEBBLE.yaml` — `gauge_mode` field, Hydra run dir updated
