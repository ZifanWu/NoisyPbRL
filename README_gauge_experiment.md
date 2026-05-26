# Gauge-Ambiguity Experiment — Reproduction Guide

## Overview

This experiment tests whether the **gauge choice** in iterative PbRL
(how the reward model's additive constant is fixed) causes systematically
different policy learning trajectories, and whether that difference is
*amplified by iterative retraining* or is already present in a static,
one-shot RM.

Two scripts are compared:

| Script | RM training | Role |
|--------|------------|------|
| `train_PEBBLE.py` | Periodic retraining as policy evolves (**iterative**) | Signal 1 — standard BPref |
| `train_PEBBLE_oneshot.py` | Single event from pi_ref data, then **frozen** (**one-shot**) | Signal 2 — control condition |

Three gauge modes are compared within each script:

| Mode | Optimizer | Post-step projection | Description |
|------|-----------|----------------------|-------------|
| `none` | Adam, `weight_decay=0` | None | **Baseline** — original BPref, no gauge fixing |
| `l2` | Adam, `weight_decay=1e-4` | None | Standard RLHF ℓ₂ regularisation |
| `zero_mean_ref` | Adam, `weight_decay=0` | Subtract mean on frozen D_ref | Explicit zero-mean constraint |

Default config: `gauge_mode=none` (backward-compatible with BPref's original behavior).

### Key hypothesis

Compute `sigma_iter` = std of per-gauge mean final J(π) across gauge modes (iterative),
and `sigma_oneshot` = same for the one-shot control.

- **Ratio > 1.5** → iterative retraining *amplifies* gauge sensitivity (**PERFORMATIVE_AMPLIFICATION**)
- **Ratio < 1.2** → gauge effect is static; iteration does not amplify it (**NO_PERFORMATIVE_AMPLIFICATION**)
- Otherwise → **INCONCLUSIVE**

Full matrix: 2 envs × 3 gauge modes × 2 scripts × 5 seeds = **60 runs** total.

---

## Prerequisites

```bash
pip install wandb matplotlib pandas scipy tabulate
wandb login          # authenticate once (only if use_wandb=true)
```

Set the W&B project (optional):
```bash
export WANDB_PROJECT=pbrl_gauge_ambiguity
```

---

## RM update count by environment

The number of RM retraining events is `ceil(max_feedback / reward_batch)`.

| Environment family | `max_feedback` | `reward_batch` | `reward_update` | RM updates (iterative) |
|-------------------|---------------|----------------|-----------------|------------------------|
| DMControl (`walker_walk`, `cheetah_run`) | 1 400 | 128 | 200 | **~11** |
| MetaWorld (`button_press`, `window_close`, …) | 20 000 | 100 | 10 | **~200** |

One-shot trains the RM exactly **once** regardless of environment.
MetaWorld provides the denser iterative regime and is the recommended primary track.

---

## Track 1 — MetaWorld, standard regime (~200 RM updates)

### Signal 1 — Iterative (train_PEBBLE.py)

```bash
# Single sanity-check run
python train_PEBBLE.py \
    env=metaworld_button-press-v2 seed=1 gauge_mode=l2 \
    tandem_mode=baseline use_wandb=true \
    num_unsup_steps=9000 num_interact=5000 \
    max_feedback=20000 reward_batch=100 reward_update=10 \
    num_train_steps=1000000 exp_dir=/path/to/exp

# Full 3 gauge × 5 seeds × 2 envs matrix
for ENV in metaworld_button-press-v2 metaworld_window-close-v2; do
for GAUGE in l2 none zero_mean_ref; do
for SEED in 1 2 3 4 5; do
  sbatch tandem_slurm_submit.sh \
    env=$ENV gauge_mode=$GAUGE seed=$SEED \
    num_unsup_steps=9000 num_interact=5000 \
    max_feedback=20000 reward_batch=100 reward_update=10 \
    tandem_mode=baseline use_wandb=true
done; done; done
```

### Signal 2 — One-shot control (train_PEBBLE_oneshot.py)

```bash
# Single sanity-check run
python train_PEBBLE_oneshot.py \
    env=metaworld_button-press-v2 seed=1 gauge_mode=l2 \
    tandem_mode=baseline use_wandb=true \
    num_unsup_steps=9000 \
    max_feedback=20000 reward_batch=100 reward_update=10 \
    num_train_steps=1000000 exp_dir=/path/to/exp

# Full matrix
for ENV in metaworld_button-press-v2 metaworld_window-close-v2; do
for GAUGE in l2 none zero_mean_ref; do
for SEED in 1 2 3 4 5; do
  sbatch tandem_slurm_submit.sh \
    SCRIPT=train_PEBBLE_oneshot.py \
    env=$ENV gauge_mode=$GAUGE seed=$SEED \
    num_unsup_steps=9000 \
    max_feedback=20000 reward_batch=100 reward_update=10 \
    tandem_mode=baseline use_wandb=true
done; done; done
```

`num_interact` is irrelevant for the one-shot script (the RM is never retrained),
but matches the config default so you can pass it without error.

Estimated compute: 60 runs × 2.5 h ≈ **150 GPU-hours** (A100/V100).

---

## Track 2 — DMControl, stress test (optional)

```bash
for SCRIPT in train_PEBBLE.py train_PEBBLE_oneshot.py; do
for ENV in walker_walk cheetah_run; do
for GAUGE in l2 none zero_mean_ref; do
for SEED in 1 2 3; do
  python $SCRIPT \
    env=$ENV gauge_mode=$GAUGE seed=$SEED \
    num_interact=1000 tandem_mode=baseline use_wandb=true \
    exp_dir=/path/to/exp
done; done; done; done
```

18 iterative + 18 one-shot = 36 runs × 2.5 h ≈ **90 GPU-hours**.

---

## Logged metrics

### Every RM training event (iterative: ~every `num_interact` steps; one-shot: once at warmup)

| Metric | Description |
|--------|-------------|
| `train/rm_mean_on_ref` | Mean RM-predicted return on frozen D_ref (256 segments from pi_ref) |
| `train/rm_std_on_ref` | Std of RM-predicted return on D_ref |
| `train/rm_mean_on_policy` | Mean RM-predicted return on 64 fresh on-policy segments |
| `train/rm_std_on_policy` | Std of on-policy return |
| `train/rm_gauge_gap` | `mean_on_policy − mean_on_ref` — **key gauge diagnostic** |
| `train/rm_param_norm` | ℓ₂ norm of all RM parameters (C1 check for weight-decay effect) |
| `train/rm_bt_loss_final` | Mean BT loss on the final training epoch (C3 comparability check) |
| `train/rm_update_count` | Cumulative number of `learn_reward` calls (C7 check) |

### Every `eval_frequency` env steps (both scripts)

| Metric | Description |
|--------|-------------|
| `eval/true_episode_reward` | True environment return — **primary outcome** |
| `eval/proxy_return` | RM-predicted episode return (different scale, track trend only) |
| `eval/return_gap` | `proxy_return − true_return` (Goodhart indicator) |
| `train/reward_model_acc` | BT classification accuracy (C3 sanity check) |

### One-shot script additional metrics (every `gauge_diag_frequency` steps during policy phase)

| Metric | Description |
|--------|-------------|
| `train/oneshot_n_preferences` | Total preference pairs collected in the single RM event |
| `train/oneshot_policy_drift` | `rm_mean_on_policy − rm_mean_on_ref` — how far policy moved from pi_ref in RM-return units |
| `train/oneshot_policy_drift_normalized` | `policy_drift / rm_std_on_ref` — drift in units of the reference distribution's std |

`gauge_diag_frequency` defaults to `${eval_frequency}` (10 000 steps).
It is ignored by the iterative script (diagnostics there fire on each RM training event).

---

## What to look at

### Primary: sigma_iter vs sigma_oneshot (performativity test)

Run `analyze_performativity.py` after all runs finish:

```bash
python analyze_performativity.py \
    --exp_dir /path/to/exp \
    --envs metaworld_button-press-v2 metaworld_window-close-v2 \
    --seeds 1 2 3 4 5 \
    --output results/
```

Outputs:

```
results/
  plot_A_bar.png                  Per-gauge mean final J(π) — iterative vs one-shot side by side
  plot_B_bootstrap_{env}.png      Bootstrap distribution of sigma_iter/sigma_oneshot
  plot_C_trajectories_{env}.png   True-return learning curves per gauge mode per condition
  plot_D_gauge_gap_{env}.png      rm/gauge_gap over training per gauge mode per condition
  performativity_summary.md       Verdict + statistics table
```

Look for:
- **Plot A**: Do the three gauge-mode bars differ in height? Do they differ *more* for iterative than one-shot?
- **Plot B**: Is the ratio CI entirely above 1.5 → PERFORMATIVE_AMPLIFICATION?  Entirely below 1.2 → NO_PERFORMATIVE_AMPLIFICATION?
- **Plot C**: Do learning curves separate by gauge mode more under iterative than one-shot?
- **Plot D**: Does `rm/gauge_gap` drift systematically across RM updates (iterative) vs stay flat (one-shot)?

### Secondary: iterative-only gauge analysis

```bash
python analyze_gauge.py \
    --project pbrl_gauge_ambiguity \
    --envs button_press sweep_into \
    --seeds 1 2 3 4 5 \
    --output results/iterative_only/
```

---

## Sanity checks before trusting results

| Check | What to verify | Pass criterion | Failure action |
|-------|---------------|----------------|----------------|
| C1 | `rm/param_norm` constrained by weight decay | End of training: `param_norm(l2) < 0.7 × param_norm(none)`, both finite | Weight decay too weak — bump to `5e-4` and re-run |
| C2 | `zero_mean_ref`: zero-mean projection works | After each `learn_reward`: `|rm/mean_on_ref| < 0.01 × rm/std_on_ref` | Projection broken — check `_apply_zero_mean_projection` |
| C3 | BT loss comparability across modes | `median(rm/bt_loss_final[zero_mean_ref]) / median(rm/bt_loss_final[none]) < 1.5` | Constraint interferes with BT optimization |
| C4 | All modes: same number of RM training events | `train/rm_update_count` identical across all (mode, env, seed) for matched seeds | Feedback scheduling bug |
| C5 | Same env steps, same seeds per mode | Identical `num_train_steps`, `num_seed_steps`, RNG seeds | Configuration confound |
| C6 | Mode `none` reproduces published PEBBLE baseline | Final `eval/true_episode_reward` within ±15% of paper's PEBBLE numbers | New modes broke the pipeline |
| C7 | One-shot: `rm_update_count == 1` throughout training | `train/rm_update_count` stays at 1 after the warmup event | Second RM event fired unexpectedly |
| C8 | One-shot: `policy_drift_normalized` grows over time | `train/oneshot_policy_drift_normalized` should increase as policy improves | If flat, RM may not be aligned with task reward |

---

## Implementation notes

### Where seeding happens

`utils.set_seed_everywhere(cfg.seed)` is called in `Workspace.__init__` before
any environment or model construction.  The gauge eval env (`_gauge_eval_env`) is
created in the same `__init__`, sharing the same seeding context.

### Reference set timing

`D_ref` (256 segments) is collected from `_gauge_eval_env` at the **first**
`learn_reward` call:
- **Iterative**: fires at `num_seed_steps + num_unsup_steps` steps.
- **One-shot**: fires immediately before the single RM training event (same timing).

At this point the policy has completed unsupervised exploration (pi_ref) but has
not seen any preference labels — this is the intended reference policy.

### One-shot RM freeze

`set_frozen(True)` is called on the RM after the single training event.  This is
belt-and-suspenders: `total_feedback >= max_feedback` already prevents any further
`learn_reward` calls in `_policy_training_loop`.

### RM update count by condition

- **Iterative**: `ceil(max_feedback / reward_batch)` events — ~11 (DMControl) or ~200 (MetaWorld).
- **One-shot**: always **1** event.  `train/rm_update_count` should stay at 1 for the entire run.

### Dependencies

The gauge/performativity experiment touches only:

- `reward_model.py` — `gauge_mode`, `set_ref_segments`, `_apply_zero_mean_projection`, `get_gauge_diagnostics`, `get_rm_parameters`, `set_frozen`
- `train_PEBBLE.py` — refactored into `_unsup_pretrain_loop` + `_policy_training_loop` + `run`
- `train_PEBBLE_oneshot.py` — `OneShotWorkspace(Workspace)` subclass
- `config/train_PEBBLE.yaml` — `gauge_mode`, `gauge_diag_frequency` fields
- `config/train_PEBBLE_oneshot.yaml` — standalone config, `experiment: PEBBLE_oneshot`
- `analyze_performativity.py` — sigma_iter/sigma_oneshot ratio analysis (local CSV)
- `analyze_gauge.py` — iterative-only gauge analysis (W&B)

No other training scripts (`train_PT.py`, `train_static_sac.py`, etc.) were modified.
