# Toy PerfAL Experiment

This folder contains a self-contained tabular toy experiment for PerfAL
(Performative Active Learning) in preference-based RL. It does not import or
modify the main B-Pref/PEBBLE training code.

## Quick Run

From the repository root:

```bash
~/miniconda3/envs/bpref/bin/python toy_perfal/run_experiment.py --preset quick
```

The quick preset runs a small preliminary sweep:

- seeds: `0,1,2`
- rounds: `10`
- initial labels: `10`
- labels per round: `10`
- teacher beta: `3`
- L2/Fisher damping: `1e-3`
- candidate query pairs per round: `2000`

Outputs are written to:

```text
toy_perfal/result_summary.md
toy_perfal/results/quick/config.json
toy_perfal/results/quick/metrics.csv
toy_perfal/results/quick/*.png
```

The implementation uses common random numbers for fair comparison:

- all methods with the same seed receive the same initial labeled dataset `D0`
- repeated/shared query pairs receive the same deterministic Bradley-Terry label
- self-pairs are excluded from initial data, on-policy random queries, and pools
- non-oracle PerfAL methods never fall back to the true gradient direction

## Methods

The script compares:

- `on_policy_random`: random preference pairs sampled from the current policy.
- `pool_uniform`: uniform random pairs from the candidate pool.
- `entropy`: highest Bradley-Terry label entropy under the current reward model.
- `disagreement`: highest Laplace predictive variance of pair preference logits.
- `d_optimal`: greedy log-det information gain.
- `perfal_topB`: top-B first-order PerfAL score `w_x (m^T delta_x)^2`.
- `perfal_greedy`: greedy Sherman-Morrison PerfAL marginal gain.
- `perfal_oracle`: PerfAL-greedy using the true policy-gradient direction.
- `perfal_random_v`: sanity baseline using a randomized reward-parameter direction.

## Larger Runs

The CLI exposes the main sweep controls:

```bash
~/miniconda3/envs/bpref/bin/python toy_perfal/run_experiment.py \
  --output-name beta3_b10_T30 \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --rounds 30 \
  --batch-size 10 \
  --candidate-pairs 10000 \
  --beta 3 \
  --l2 1e-3
```

The full prompt-style beta/damping sweep is available as:

```bash
~/miniconda3/envs/bpref/bin/python toy_perfal/run_experiment.py --preset full_sweep
```

This sweeps:

- beta: `1, 3, 10`
- L2 damping: `1e-4, 1e-3, 1e-2`
- seeds: `0..49`
- rounds: `30`
- candidate pairs per round: `10000`

The default summary is intentionally labeled preliminary when generated from
the quick preset.

## Fork-Decoy Direction-Limited Environment

`run_fork_decoy.py` adds a second toy environment intended to test the regime
where PerfAL should help. The policy starts near a goal/trap fork, while the
candidate pool contains many uncertain zero-reward decoy features. This asks
whether directional uncertainty reduction can avoid wasting labels on decoys.

Quick run:

```bash
~/miniconda3/envs/bpref/bin/python toy_perfal/run_fork_decoy.py --preset quick
```

Paper-style run:

```bash
~/miniconda3/envs/bpref/bin/python toy_perfal/run_fork_decoy.py --preset paper
```

The fork-decoy comparison includes `perfal_mixed`, a coverage-safeguarded rule
that greedily combines normalized PerfAL gain with normalized D-optimal gain:

```text
score = alpha * PerfAL_gain + (1 - alpha) * D_optimal_gain
```

Outputs are written to `toy_perfal/fork_decoy_summary.md` and
`toy_perfal/results/<output-name>/`.
