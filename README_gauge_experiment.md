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

## Track 1 — Standard regime (~11 RM updates across 1 M steps)

Default config: `max_feedback=1400`, `num_interact=5000`, `reward_update=200`.

### Single run (sanity check)
```bash
python train_PEBBLE.py \
    env=walker_walk \
    seed=1 \
    gauge_mode=l2 \
    tandem_mode=baseline \
    use_wandb=true \
    num_train_steps=1000000 \
    exp_dir=/path/to/exp
```

### Full 30-run matrix (use SLURM or a parallel launcher)

```bash
for ENV in walker_walk cheetah_run; do
for GAUGE in l2 none zero_mean_ref; do
for SEED in 1 2 3 4 5; do
  sbatch tandem_slurm_submit.sh \
    env=$ENV \
    gauge_mode=$GAUGE \
    seed=$SEED \
    tandem_mode=baseline \
    use_wandb=true
done; done; done
```

Estimated compute: each run ≈ **2–3 GPU-hours** on a single A100/V100 for 1 M steps
with `walker_walk`/`cheetah_run`.  Total: 30 × 2.5 h ≈ **75 GPU-hours**.

---

## Track 2 — Stress test (~50 RM updates, optional)

Increase RM retraining frequency by setting `num_interact=1000`.  This keeps the
feedback budget the same (`max_feedback=1400`) but the RM is retrained ≈ 50 times,
giving the gauge effect more iterations to accumulate.

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

Estimated compute: same per-run cost, 18 runs × 2.5 h ≈ **45 GPU-hours**.

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

In the standard regime, the RM is retrained as long as
`total_feedback < max_feedback`.  With `max_feedback=1400`, `mb_size=128` (after
schedule), `large_batch=10`, and `num_interact=5000`, the RM is updated
approximately **11 times** across 1 M steps.  This is fewer than the 100–200
quoted in the original PEBBLE paper because that paper uses a larger feedback
budget.  The stress test (`num_interact=1000`) brings this to approximately **50**
updates.

### Dependencies on existing code

No other training scripts (`train_PT.py`, `train_static_sac.py`, etc.) were
modified.  The gauge experiment is isolated to:

- `reward_model.py` — `gauge_mode` param, `set_ref_segments`, `_apply_zero_mean_projection`, `get_gauge_diagnostics`
- `train_PEBBLE.py` — gauge env, `_collect_on_policy_segments`, `evaluate` proxy logging, `learn_reward` diagnostic dispatch
- `config/train_PEBBLE.yaml` — `gauge_mode` field, Hydra run dir updated
