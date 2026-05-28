# Gauge implementation notes

This directory adds gauge experiments without changing the upstream PEBBLE reward
training loop in place.

## Gauges

- `none`: upstream reward model with final `tanh`, no additive shift.
- `mean_buf`: upstream final `tanh`, subtract the mean reward over a fresh replay-buffer sample.
- `mean_ref`: upstream final `tanh`, subtract the mean reward over the frozen reference dataset.
- `no_tanh`: same reward model hidden layers, but the final `tanh` is replaced by identity during BT training and policy reward.
- `no_tanh_mean_buf`: identity final layer plus replay-buffer mean subtraction.
- `no_tanh_mean_ref`: identity final layer plus reference-dataset mean subtraction.

Shifts are recomputed immediately after each reward-model update and held fixed
between updates. RRM runs are grouped by final-activation family, so additive
gauge invariance is only checked within `tanh` or within `identity`.

## Performative correction

The PerfG correction follows the prompt formulas:

- `G_perf_base = -(1/n) sum_i <H^-1 u, grad_phi ell_i> grad_theta log pi(pair_i)`
- `G_perf_shift = +(H_horizon/n) sum_i <H^-1 u_shift, grad_phi ell_i> grad_theta log pi(pair_i)`

The implemented total correction is applied as an extra actor ascent step every
`K_perf=5` SAC actor updates after the post-RM reset-update burst. The standard
performative term and the gauge-shift term are independently controlled by
`gauge.pg.perf_grad` and `gauge.pg.gauge_corr`; the gauge-shift term is active
only when `perf_grad=true`. When both base and shift terms are active,
`gauge.pg.solver=cg` uses the full IFT HVP/CG solve. `gauge.pg.solver=ridge`
is a first-order approximation that replaces `H^{-1}u` by `u / ridge` and skips
HVP/CG entirely. `gauge.pg.merge_cg=true` solves the combined right-hand side
with one CG solve instead of two separate solves when `solver=cg`.
`gauge.pg.fd_weights=true` approximates
`<w, grad_phi ell_i>` by central directional finite differences, replacing one
RM backward per preference pair with two RM forward passes per correction
vector. `gauge.pg.u_source=recent` estimates `u` from a cache of recent online
`(obs, action)` samples instead of launching extra environment rollouts; set it
to `rollout` to restore the previous estimator. `gauge.pg.rm_param_scope` can
restrict the implicit-gradient solve to `final_layer` or `final_bias` parameters
for NTK/last-layer-style approximations. The reset burst is skipped by default
because it applies many actor updates immediately after critic reset and
can destabilize Metaworld actors before the online SAC loop resumes. The extra
actor-gradient is finite-checked and globally clipped with
`gauge.pg.max_actor_grad_norm`.

Approved approximations:

- HVP is estimated on 256 preference pairs.
- Conjugate gradient uses 10 iterations and ridge regularization.
- `u` is estimated from 4 fresh current-policy rollouts in a separate env.
- The ensemble reward is the arithmetic mean of member outputs.
- Stored preference-buffer actions are scored under the current actor.
- Shift distribution samples are treated as stop-gradient samples. For
  `mean_buf`, the PG shift-correction sample is drawn fresh from the replay
  buffer at correction time; for `mean_ref`, it uses the frozen reference set.
- `solver=ridge` is a first-order approximation, not the full implicit-function
  PerfG correction; it is useful as a fast ablation and may require stronger
  clipping or a different ridge scale.
- Directional finite-difference preference weights are an approximation; set
  `gauge.pg.fd_weights=false` to restore exact per-sample autograd weights.
- `u_source=recent` treats recent online samples as approximately on-policy and
  may cross episode boundaries in the cached windows.
- `rm_param_scope=final_layer/final_bias` is a linearized approximation to the
  full RM implicit derivative. In particular, the pure final-bias component can
  cancel under mean-shift gauges when the shift horizon matches the reward
  trajectory horizon.
- PG correction is skipped during `update_after_reset` by default
  (`gauge.pg.apply_during_reset=false`) and clipped to the configured norm.

## Files

- `reward_model_gauge.py`: final-activation variants and additive gauges.
- `perf_correction.py`: HVP/CG and actor correction.
- `reference_dataset.py`: random-policy frozen reference data.
- `train_gauge.py`: Hydra runner that wraps the existing PEBBLE workspace.
- `run_sweep.py`: dry-run/local/slurm sweep command generator.
- `analyze.py`: Welch tests, Bonferroni correction, effect sizes, and plots.

## Logging

Gauge experiments default to `gauge.log_reward_diagnostics=false`. This disables
reward-model-only diagnostics unrelated to this experiment, including dormant
rates, BT weight metrics, RM gradient norms, and weight-update-ratio logs. Core
train/eval metrics, gauge shifts, proxy returns, and PG correction diagnostics
remain enabled.
