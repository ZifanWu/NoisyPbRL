# Axis 2 Tier B matrix

`axis2_steps2_4_submit.sh` reads `configs/axis2_tier_b/matrix.yaml` to generate the Step 2 training manifest and Step 4 payoff manifest.

The YAML controls:

- `envs`: environments to run.
- `env_profiles`: per-environment overrides inferred from `scripts/`, including environment-specific `max_feedback`, `reward_batch`, and feedback-budget sweep anchors.
- `seeds`: base condition seeds.
- `conditions`: base Step-2 conditions and their Hydra config names.
- `hyperparameter_sets`: optional global override sets. These cross every cell, so keep them small.
- `mis_capacity_sweep`: primary RM-width sweep for misspecification.
- `confirmatory`: linear-head and mistake-teacher confirmatory runs.
- `sensitivity_sweeps`: targeted hyperparameter probes selected with `SWEEP_GROUPS`.
- `payoff`: default payoff scope and payoff hyperparameters.

Current targeted sensitivity groups:

- `shift_num_interact`: graded distribution-shift relabeling interval.
- `buffer_window_rounds`: checks whether shift is sensitive to stale-pair mixture fitting.
- `epi_feedback_budget`: label-budget sweep for epistemic uncertainty.
- `epi_teacher_beta`: BT preference-informativeness sweep for epistemic uncertainty.

Common filters can be supplied at submit time without editing YAML:

```bash
DRY_RUN=true ENVS=metaworld_door-open-v2 SEEDS=12345 CONDITIONS=control \
  INCLUDE_MIS_SWEEP=false INCLUDE_CONFIRMATORY=false \
  RUN_ANALYSIS=false RUN_PAYOFF=false bash axis2_steps2_4_submit.sh
```

Run selected sensitivity groups:

```bash
DRY_RUN=true SWEEP_GROUPS=shift_num_interact buffer_window_rounds bash axis2_steps2_4_submit.sh
DRY_RUN=true SWEEP_GROUPS=all bash axis2_steps2_4_submit.sh
```

One Slurm array task is one `train_PEBBLE_axis2.py` run. The task exits when that run exits or when `SKIP_DONE=true` detects that the run already reached `DONE_STEP` in `axis2_metrics.csv`.


## Per-env profiles

Do not create one matrix per environment. Put shared experiment structure in this file and add environment-specific budgets under `env_profiles`.

For example, `metaworld_drawer-open-v2` uses high-feedback base runs:

```yaml
control/mis/shift: max_feedback=10000
epi:               max_feedback=1000
epi sweep anchor:  max_feedback=10000
```

The submitter applies profile overrides before sweep-specific overrides, so a sweep value such as `max_feedback=5000` cleanly replaces the profile's epi default.
