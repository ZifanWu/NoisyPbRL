# BPref Gauge Experiment Modifications

## What Was Added

### New package: `pebble_gauge/`

- `reward_model_gauge.py`
  - Adds `GaugedRewardModel`, a post-hoc wrapper around the base `RewardModel`.
  - Supports gauges: `none`, `mean_buf`, `mean_ref`.
  - Recomputes additive shifts after each RM update and applies them only at reward query time.
  - Exposes:
    - active-gauge `r_hat` / `r_hat_batch` (drop-in for replay relabel and online reward)
    - per-gauge proxy return computation
    - gauge consistency checks.

- `reference_dataset.py`
  - Builds or loads a frozen reference dataset at `reference_dataset/<env>.pt`.
  - Dataset is `(obs, action)` tuples sampled from a random policy.
  - Reused across all runs for the same environment.

- `perf_correction.py`
  - Adds PG-RLHF-style performative correction utilities:
    - base correction (`G_perf_base`) with BT Hessian-vector products and CG
    - shift correction (`G_perf_shift`) for non-`none` gauges
    - actor-gradient combination and norm diagnostics.

- `run_sweep.py`
  - Orchestrates phased matrix execution (`phase1`, `phase2`, `phase3`, `full`).
  - Implements RRM compute-sharing (single run per env/alpha/seed, no gauge replication).
  - Emits run manifests (`results/manifests/manifest_<phase>.json/.csv`).

- `analyze.py`
  - Loads runs from manifest and produces:
    - `headline_pg_rlhf_returns.png`
    - `rrm_baseline.png`
    - `final_pg_rlhf_bars.png`
    - `shift_evolution.png`
    - `alpha_sweep_summary.png`
    - `pg_vs_rrm.png`
    - `results/summary.txt`
  - Computes pairwise Welch t-tests for PG-RLHF gauge pairs (Bonferroni corrected), Cohen's d, and 95% CI for mean differences.

### New config bundle: `configs/gauge_experiment/`

- `base.yaml` (common defaults + env/method/alpha metadata)
- `phase1.yaml`, `phase2.yaml`, `phase3.yaml`, `full.yaml` (execution scopes)

## Upstream Files Modified

- `train_PEBBLE.py`
  - Added gauge-experiment flags and defaults integration:
    - `enable_gauge_experiment`, `method`, `gauge`, `alpha_mode`
    - performative correction controls (`use_perf_correction`, `k_perf`, CG/HVP settings)
  - Wraps base RM with `GaugedRewardModel` when enabled.
  - Builds/loads frozen reference dataset at startup.
  - Recomputes and logs shift values after RM updates:
    - `rm/shift_value_none`
    - `rm/shift_value_mean_buf`
    - `rm/shift_value_mean_ref`
  - Evaluation updates:
    - logs `eval/true_return`
    - RRM logs all three proxy returns (`eval/proxy_return_<gauge>`) in one run
    - PG-RLHF logs active-gauge proxy metrics.
  - Added PG-RLHF actor correction hook (`_on_actor_update`) and diagnostics:
    - `actor/perf_correction_base_norm`
    - `actor/perf_correction_shift_norm`
    - `actor/perf_correction_total_norm`
    - `rm/d_shift_norm`
  - Added end-of-run RRM identity sanity check across gauges.

- `agent/sac.py`
  - Added optional `actor_update_callback` to:
    - `update(...)`
    - `update_after_reset(...)`
  - Keeps existing behavior unchanged when callback is `None`.

- `config/train_PEBBLE.yaml`
  - Added gauge-experiment and performative-correction runtime fields.

## Why These Changes

1. **Exact post-hoc gauge test for RRM**
   - Train one base RM/policy trajectory and vary only additive shifts at query time.
   - This isolates the gauge operation to a pure constant shift and enables exact invariance checks.

2. **Gauge-dependent PG-RLHF path**
   - Introduces a dedicated shift-gradient correction branch so PG updates can depend on `c(theta)` through the reward model pathway.

3. **Efficient sweep execution**
   - Avoids redundant RRM-by-gauge training jobs.
   - Uses phase-based rollout to validate signal before full compute.

4. **Reproducible analysis**
   - Manifest-driven run indexing and deterministic output paths (`hydra.run.dir` override from sweep script).

## Sanity Checks Implemented

- Gauge shift logs each RM update (`rm/shift_value_*`).
- RRM post-training identity check for true returns under all three gauges.
- Per-correction norm logging to detect dead/unstable performative terms.

## Known Limitations And Caveats

1. **Hybrid SAC + correction is heuristic**
   - `G_perf_total` is applied as an extra actor step on top of SAC actor updates.
   - This is not equivalent to replacing SAC with an exact policy-gradient objective.

2. **Truncated RM optimization**
   - PEBBLE does not fully solve RM to argmin at each update, so IFT assumptions are approximate.

3. **Subsampled HVP/CG**
   - Preference and replay subsampling introduce variance into correction estimates.

4. **Stop-gradient simplification for replay/reference sampling**
   - Sampling distribution changes with policy, but correction treats sampled sets as fixed per call.

5. **Potential interaction with existing reward-model gauge mode**
   - Existing `gauge_mode` path in `RewardModel` remains active; experiment logic uses post-hoc wrapper shifts.
   - For cleanest interpretation, keep `gauge_mode=none` during this experiment.

6. **RRM identity tolerance**
   - Identity check uses near-machine precision (`np.isclose`), but tiny environment numerical differences can still appear depending on backend determinism.

## Failure Modes To Track In Writeup

- **Failure Mode A'**: gauge spread shrinks as alpha increases (`low -> auto -> high`).
- **Temperature compensation**: auto-alpha may partially absorb reward-scale effects.
- **Shift instability**: `mean_buf` shift may drift if replay distribution becomes narrow/nonstationary.
- **Weak shift regime**: if both shift magnitudes stay near zero, gauge effects are not practically testable.
