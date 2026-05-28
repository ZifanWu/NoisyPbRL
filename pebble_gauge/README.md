# NoisyPbRL Gauge Experiments

> **⚠️ BEFORE LAUNCHING THE FULL SLURM SWEEP — pre-generate reference datasets**
>
> `train_gauge.py` creates missing reference datasets on first use, but concurrent
> Slurm jobs for the same environment will race to generate the same file.  The
> write itself is atomic (`os.replace`), so no corruption occurs, but every losing
> job wastes GPU-minutes rolling out a random policy it will discard.  Run the
> three commands below **once** before submitting any jobs:
>
> ```bash
> PY=/home/zifan/miniconda3/envs/bpref/bin/python
> $PY -m pebble_gauge.reference_dataset --env walker_walk              --seed 0
> $PY -m pebble_gauge.reference_dataset --env metaworld_drawer-open-v2 --seed 0
> $PY -m pebble_gauge.reference_dataset --env metaworld_hammer-v2      --seed 0
> ```

This package adds gauge experiments on top of the existing PEBBLE/SAC code.

## Gauges

The runner supports six active gauges:

- `none`: default reward model with final `tanh`, no shift.
- `mean_buf`: final `tanh`, subtract replay-buffer mean reward.
- `mean_ref`: final `tanh`, subtract frozen reference-dataset mean reward.
- `no_tanh`: train the reward model without the final `tanh`, no shift.
- `no_tanh_mean_buf`: no final `tanh`, subtract replay-buffer mean reward.
- `no_tanh_mean_ref`: no final `tanh`, subtract reference-dataset mean reward.

RRM runs one job per activation family: `none` logs `none/mean_buf/mean_ref`,
and `no_tanh` logs `no_tanh/no_tanh_mean_buf/no_tanh_mean_ref`. PerfG runs
one job per active gauge.

## Environment

Use the `bpref` Python environment:

```bash
cd /home/zifan/NoisyPbRL
PY=/home/zifan/miniconda3/envs/bpref/bin/python
```

For Metaworld jobs, export MuJoCo paths before running locally:

```bash
export MUJOCO_GL=egl
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/home/zifan/.mujoco/mujoco210/bin:/usr/lib/nvidia"
```

`run_sweep.py --mode slurm` adds these exports automatically to generated Slurm scripts.

## Single Runs

RRM example:

```bash
$PY -m pebble_gauge.train_gauge \
  agent=sac \
  env=walker_walk \
  seed=12345 \
  gauge.method=rrm \
  gauge.alpha_mode=auto \
  gauge.active=none
```

PerfG example:

```bash
$PY -m pebble_gauge.train_gauge \
  agent=sac \
  env=walker_walk \
  seed=12345 \
  gauge.method=perfg \
  gauge.alpha_mode=auto \
  gauge.active=no_tanh_mean_ref
```

PG correction can be decomposed from config:

```bash
gauge.pg.perf_grad=true     # standard Section 4.7 performative-gradient term
gauge.pg.gauge_corr=true    # additive gauge-shift term, active only with perf_grad
gauge.pg.solver=cg          # cg for full IFT solve; ridge for first-order approximation
gauge.pg.merge_cg=true      # one CG solve for base+shift; false restores two-solve path
gauge.pg.fd_weights=true    # approximate per-example RM gradients by directional finite differences
gauge.pg.fd_eps=0.001
gauge.pg.u_source=recent      # recent online cache; set rollout for exact extra-env rollouts
gauge.pg.rm_param_scope=all   # all, final_layer, or final_bias
gauge.pg.apply_during_reset=false
gauge.pg.max_actor_grad_norm=1.0
```

This skips correction during the first post-RM `reset_update` burst and clips
the extra actor-gradient step. The normal SAC actor updates after training
resumes receive correction every `gauge.pg.k_perf` actor updates. Set either
`perf_grad=false` disables all performative correction; `gauge_corr=false` ablates only the gauge-shift component; `solver=ridge` replaces the IFT solve with the first-order approximation `w = u / ridge`; `fd_weights=false` restores exact per-sample RM backward for the preference weights; `u_source=rollout` restores the old extra-environment rollout estimator for `u`; `rm_param_scope=final_layer` or `final_bias` restricts the implicit-gradient solve to the reward model's final parameters.

Use `agent=sac_metaworld` for Metaworld environments:

```bash
$PY -m pebble_gauge.train_gauge \
  agent=sac_metaworld \
  env=metaworld_drawer-open-v2 \
  seed=12345 \
  gauge.method=rrm \
  gauge.alpha_mode=auto \
  gauge.active=no_tanh
```

Alpha modes:

- `gauge.alpha_mode=auto`: learnable SAC temperature.
- `gauge.alpha_mode=low`: fixed alpha `0.05`.
- `gauge.alpha_mode=high`: fixed alpha `0.5`.

Reward-model diagnostic logging is off by default for these experiments:

```bash
gauge.log_reward_diagnostics=false
```

This suppresses unrelated RM diagnostics such as dormant-neuron rates, BT weight
histograms/saturation metrics, RM grad norms, and weight-update-ratio logs. Main
train/eval logs, gauge shifts, proxy returns, and performative-correction logs
are still emitted. Set `gauge.log_reward_diagnostics=true` only when debugging
the reward model itself.

Gauge-specific scalar diagnostics are also written as a long CSV table in each
Hydra run directory:

```bash
gauge_metrics.csv
```

This file contains `step,split,key,value` rows for gauge shifts, eval proxy
returns, and performative-correction diagnostics. It avoids the base logger's
fixed CSV-header behavior for metrics that appear after the first train dump.

## Full Sweep

The default matrix is:

- Envs: `walker_walk`, `metaworld_drawer-open-v2`, `metaworld_hammer-v2`
- Alpha modes: `auto`, `low`, `high`
- Seeds: `12345`, `23451`, `34512`, `45123`, `51234`, `67890`
- RRM activation-family jobs: `none`, `no_tanh`
- PG active gauges: all six gauges

Expected total:

- RRM: `3 * 3 * 6 * 2 = 108`
- PerfG: `3 * 3 * 6 * 6 = 324`
- Total: `432`

Check commands without launching:

```bash
$PY -m pebble_gauge.run_sweep --phase all --mode dry-run | tail -1
```

Launch locally:

```bash
$PY -m pebble_gauge.run_sweep --phase rrm --mode local
$PY -m pebble_gauge.run_sweep --phase pg --mode local
```

Submit Slurm jobs:

```bash
$PY -m pebble_gauge.run_sweep --phase all --mode slurm
```

Generated Slurm scripts `cd` into the directory where `run_sweep.py` is invoked.
Override it when submitting from a different location:

```bash
$PY -m pebble_gauge.run_sweep --phase all --mode slurm \
  --repo-dir /path/to/NoisyPbRL
```

By default, Hydra writes gauge runs under:

```bash
/uufs/chpc.utah.edu/common/home/dbrown-group1/zifan/pg_gauge
```

By default, Slurm stdout/stderr logs from `run_sweep.py --mode slurm` go under:

```bash
/uufs/chpc.utah.edu/common/home/dbrown-group1/zifan/logs
```

Override `hydra.run.dir=...` explicitly for scratch or pilot runs.

Append Hydra overrides to every job by placing them after the options:

```bash
$PY -m pebble_gauge.run_sweep --phase pg --mode dry-run \
  num_train_steps=200000 gpu=0
```

## Reference Datasets

Reference datasets are saved under `reference_dataset/<env_name>.pt` by default
and are auto-created by `train_gauge.py` when missing. Writes are atomic, so
multiple Slurm jobs can safely race to create the same file; the recommended
workflow is still to pre-generate each environment once before launching the
full matrix.

To create one explicitly:

```bash
$PY -m pebble_gauge.reference_dataset \
  --env walker_walk \
  --root reference_dataset \
  --n-ref 10000 \
  --seed 0
```

## Analysis

Analyze Hydra run outputs:

```bash
$PY -m pebble_gauge.analyze \
  --root /home/zifan/NoisyPbRL/exp \
  --out results/gauge_analysis \
  --step-window 200000
```

Outputs include:

- `eval_merged.csv`
- `final_window_means.csv`
- `welch_pairwise.csv`
- `rrm_proxy_summary.csv`
- `true_return_curves.png`
- `final_true_return_box.png`

`rrm_proxy_summary.csv` checks the additive offset identity within each activation
family: `base_proxy_return - proxy_return ~= gauge_shift * episode_length`. The
analysis script uses seaborn if installed and falls back to matplotlib otherwise.

## Sanity Checks

Compile and synthetic checks. RRM runs also execute the final H1 gauge-offset
identity check at the end of training:

```bash
$PY -m py_compile pebble_gauge/*.py
$PY -m pebble_gauge.sanity_checks
```

Tiny runner smoke check:

```bash
$PY -m pebble_gauge.train_gauge \
  agent=sac \
  env=walker_walk \
  seed=101 \
  device=cpu \
  num_seed_steps=2 \
  num_unsup_steps=0 \
  num_train_steps=3 \
  replay_buffer_capacity=10 \
  segment=1 \
  reward_batch=2 \
  reward_update=1 \
  max_feedback=2 \
  large_batch=1 \
  ensemble_size=1 \
  reset_update=1 \
  agent.params.batch_size=4 \
  diag_gaussian_actor.params.hidden_dim=16 \
  diag_gaussian_actor.params.hidden_depth=1 \
  double_q_critic.params.hidden_dim=16 \
  double_q_critic.params.hidden_depth=1 \
  log_save_tb=false \
  use_wandb=false \
  eval_frequency=100000 \
  num_eval_episodes=1 \
  gauge.n_ref=4 \
  gauge.reference_root=/tmp/noisypbrl_ref_runner \
  gauge.method=perfg \
  gauge.active=no_tanh_mean_ref \
  gauge.pg.k_perf=1 \
  gauge.pg.n_traj=1 \
  gauge.pg.rollout_horizon=2 \
  gauge.pg.pref_batch_size=2 \
  gauge.pg.cg_iters=2 \
  hydra.run.dir=/tmp/noisypbrl_gauge_smoke
```

## Notes

- `no_tanh*` changes the reward model architecture during BT training; it is not
  an inference-only bypass.
- Shift values are recomputed after reward-model updates and held fixed between updates.
- PG correction uses the approved practical approximations documented in
  `modifications.md`.
