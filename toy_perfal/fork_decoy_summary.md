# Fork-Decoy PerfAL Result Summary

Generated: 2026-05-30T20:07:24

This direction-limited toy environment has a current policy near a goal/trap fork and a query pool containing many high-uncertainty zero-reward decoy features.

## Configuration

```json
{
  "preset": "paper",
  "output_name": "fork_decoy_paper",
  "seeds": [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29
  ],
  "rounds": 25,
  "init_labels": 16,
  "warmstart_critical": 6,
  "batch_size": 8,
  "candidate_pairs": 8000,
  "beta": 5.0,
  "l2": 0.001,
  "policy_lr": 0.5,
  "mixed_alpha": 0.7,
  "grid": [
    5,
    5
  ],
  "horizon": 7,
  "start": [
    0,
    2
  ],
  "fork": [
    2,
    2
  ],
  "goal": [
    4,
    3
  ],
  "trap": [
    4,
    1
  ],
  "safe": [
    4,
    2
  ],
  "decoys": [
    [
      0,
      4
    ],
    [
      1,
      4
    ],
    [
      2,
      4
    ],
    [
      3,
      4
    ],
    [
      0,
      0
    ],
    [
      1,
      0
    ],
    [
      2,
      0
    ],
    [
      3,
      0
    ]
  ],
  "feature_names": [
    "goal",
    "trap",
    "safe",
    "decoy_0_4_up",
    "decoy_0_4_down",
    "decoy_0_4_left",
    "decoy_0_4_right",
    "decoy_0_4_stay",
    "decoy_1_4_up",
    "decoy_1_4_down",
    "decoy_1_4_left",
    "decoy_1_4_right",
    "decoy_1_4_stay",
    "decoy_2_4_up",
    "decoy_2_4_down",
    "decoy_2_4_left",
    "decoy_2_4_right",
    "decoy_2_4_stay",
    "decoy_3_4_up",
    "decoy_3_4_down",
    "decoy_3_4_left",
    "decoy_3_4_right",
    "decoy_3_4_stay",
    "decoy_0_0_up",
    "decoy_0_0_down",
    "decoy_0_0_left",
    "decoy_0_0_right",
    "decoy_0_0_stay",
    "decoy_1_0_up",
    "decoy_1_0_down",
    "decoy_1_0_left",
    "decoy_1_0_right",
    "decoy_1_0_stay",
    "decoy_2_0_up",
    "decoy_2_0_down",
    "decoy_2_0_left",
    "decoy_2_0_right",
    "decoy_2_0_stay",
    "decoy_3_0_up",
    "decoy_3_0_down",
    "decoy_3_0_left",
    "decoy_3_0_right",
    "decoy_3_0_stay",
    "step"
  ],
  "true_reward": [
    1.0,
    -1.2,
    0.2,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    -0.01
  ],
  "methods": [
    "on_policy_random",
    "pool_uniform",
    "entropy",
    "disagreement",
    "d_optimal",
    "perfal_topB",
    "perfal_greedy",
    "perfal_mixed",
    "perfal_oracle",
    "perfal_random_v"
  ]
}
```

## Aggregate Results

| Method | Final return | Return AUC | Grad cosine | Dir. var | Critical RM err | Decoy |
|---|---:|---:|---:|---:|---:|---:|
| perfal_topB | 2.9186 +/- 0.0004 | 2.7006 +/- 0.0052 | 0.9989 | 0.001172 | 40.4638 | 3.8420 |
| perfal_random_v | 2.9173 +/- 0.0004 | 2.6680 +/- 0.0079 | 0.9988 | 0.001848 | 49.3399 | 5.9225 |
| perfal_mixed | 2.8885 +/- 0.0032 | 2.6563 +/- 0.0115 | 0.9989 | 0.002809 | 14.5686 | 1.2833 |
| d_optimal | 2.9194 +/- 0.0004 | 2.6446 +/- 0.0083 | 0.9987 | 0.002605 | 83.1710 | 10.4790 |
| perfal_oracle | 2.8813 +/- 0.0032 | 2.6435 +/- 0.0100 | 0.9981 | 0.001554 | 12.4154 | 1.6563 |
| perfal_greedy | 2.8816 +/- 0.0044 | 2.6335 +/- 0.0144 | 0.9990 | 0.002402 | 12.5721 | 1.3368 |
| on_policy_random | 2.8871 +/- 0.0024 | 2.6315 +/- 0.0083 | 0.9982 | 0.034358 | 11.2542 | 0.9028 |
| disagreement | 2.8959 +/- 0.0019 | 2.6237 +/- 0.0089 | 0.9942 | 0.047521 | 17.2303 | 3.5695 |
| pool_uniform | 2.8857 +/- 0.0016 | 2.6187 +/- 0.0113 | 0.9979 | 0.044951 | 11.1784 | 1.2065 |
| entropy | 2.8612 +/- 0.0047 | 2.5560 +/- 0.0166 | 0.9892 | 0.194093 | 8.0793 | 1.7311 |

## Readout

- Best return AUC: `perfal_topB`.
- Best final true return: `d_optimal`.
- Lowest final policy-direction posterior variance: `perfal_topB`.
- `perfal_greedy` beats `perfal_random_v` on AUC: `False`.
- `perfal_mixed` beats `perfal_random_v` on AUC: `False`.
- The intended positive signal is PerfAL or PerfAL+coverage improving return while avoiding decoy-focused uncertainty.

## Output Files

- `results/fork_decoy_paper/metrics.csv`
- `results/fork_decoy_paper/true_return.png`
- `results/fork_decoy_paper/posterior_directional_variance.png`
- `results/fork_decoy_paper/critical_rm_error.png`
