# PerfAL Toy Experiment Result Summary

Generated: 2026-05-30T19:17:18

This summary was generated from the configured CLI sweep.

## Configuration

```json
{
  "preset": "full_sweep",
  "output_name": "full_sweep_corrected",
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
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    49
  ],
  "rounds": 30,
  "init_labels": 10,
  "batch_size": 10,
  "candidate_pairs": 10000,
  "beta": 3.0,
  "betas": [
    1.0,
    3.0,
    10.0
  ],
  "l2": 0.001,
  "l2s": [
    0.0001,
    0.001,
    0.01
  ],
  "policy_lr": 0.8,
  "grid": [
    4,
    4
  ],
  "horizon": 6,
  "actions": [
    "up",
    "down",
    "left",
    "right",
    "stay"
  ],
  "true_reward": [
    1.0,
    -0.8,
    0.0,
    -0.02
  ],
  "goal": [
    3,
    3
  ],
  "trap": [
    2,
    3
  ],
  "side": [
    0,
    2
  ],
  "methods": [
    "on_policy_random",
    "pool_uniform",
    "entropy",
    "disagreement",
    "d_optimal",
    "perfal_topB",
    "perfal_greedy",
    "perfal_oracle",
    "perfal_random_v"
  ]
}
```

## Aggregate Results

| beta | l2 | Best AUC | Best final return | Lowest dir. variance | PerfAL-greedy AUC rank | PerfAL-greedy beats random-v AUC? |
|---:|---:|---|---|---|---:|---|
| 1 | 0.0001 | `d_optimal` | `d_optimal` | `perfal_topB` | 5 | no |
| 1 | 0.001 | `d_optimal` | `d_optimal` | `perfal_topB` | 8 | no |
| 1 | 0.01 | `d_optimal` | `on_policy_random` | `perfal_greedy` | 8 | no |
| 3 | 0.0001 | `entropy` | `entropy` | `perfal_topB` | 9 | no |
| 3 | 0.001 | `disagreement` | `pool_uniform` | `perfal_topB` | 8 | no |
| 3 | 0.01 | `disagreement` | `entropy` | `perfal_greedy` | 8 | no |
| 10 | 0.0001 | `perfal_oracle` | `perfal_oracle` | `entropy` | 5 | no |
| 10 | 0.001 | `disagreement` | `disagreement` | `d_optimal` | 4 | no |
| 10 | 0.01 | `perfal_random_v` | `perfal_random_v` | `perfal_greedy` | 5 | no |

## Per-Setting Results

### beta=1, l2=0.0001

| Method | Final true return | Final regret | Return AUC | Grad cosine | Dir. variance |
|---|---:|---:|---:|---:|---:|
| d_optimal | -0.1341 +/- 0.0009 | 1.0141 | -0.1421 +/- 0.0011 | 0.6036 | 0.000607 |
| perfal_random_v | -0.1357 +/- 0.0009 | 1.0157 | -0.1440 +/- 0.0010 | 0.6039 | 0.000975 |
| perfal_topB | -0.1365 +/- 0.0009 | 1.0165 | -0.1440 +/- 0.0012 | 0.8284 | 0.000199 |
| perfal_oracle | -0.1378 +/- 0.0006 | 1.0178 | -0.1463 +/- 0.0007 | 0.7291 | 0.000495 |
| perfal_greedy | -0.1381 +/- 0.0007 | 1.0181 | -0.1467 +/- 0.0011 | 0.8226 | 0.000210 |
| disagreement | -0.1378 +/- 0.0007 | 1.0178 | -0.1467 +/- 0.0007 | 0.6333 | 0.000960 |
| on_policy_random | -0.1382 +/- 0.0022 | 1.0182 | -0.1475 +/- 0.0028 | 0.6085 | 0.002880 |
| pool_uniform | -0.1415 +/- 0.0037 | 1.0215 | -0.1486 +/- 0.0037 | 0.6078 | 0.027722 |
| entropy | -0.2243 +/- 0.0576 | 1.1043 | -0.2125 +/- 0.0398 | 0.4720 | 0.246590 |

- Best return AUC: `d_optimal`.
- Best final true return: `d_optimal`.
- Lowest policy-direction posterior variance: `perfal_topB`.

### beta=1, l2=0.001

| Method | Final true return | Final regret | Return AUC | Grad cosine | Dir. variance |
|---|---:|---:|---:|---:|---:|
| d_optimal | -0.1345 +/- 0.0009 | 1.0145 | -0.1430 +/- 0.0011 | 0.6746 | 0.000485 |
| perfal_random_v | -0.1355 +/- 0.0008 | 1.0155 | -0.1439 +/- 0.0010 | 0.6660 | 0.000886 |
| disagreement | -0.1356 +/- 0.0008 | 1.0156 | -0.1452 +/- 0.0008 | 0.6576 | 0.000715 |
| perfal_topB | -0.1375 +/- 0.0009 | 1.0175 | -0.1452 +/- 0.0013 | 0.8296 | 0.000196 |
| pool_uniform | -0.1354 +/- 0.0022 | 1.0154 | -0.1454 +/- 0.0028 | 0.6515 | 0.001214 |
| perfal_oracle | -0.1385 +/- 0.0006 | 1.0185 | -0.1473 +/- 0.0007 | 0.7749 | 0.000548 |
| on_policy_random | -0.1394 +/- 0.0024 | 1.0194 | -0.1478 +/- 0.0023 | 0.5587 | 0.001378 |
| perfal_greedy | -0.1405 +/- 0.0012 | 1.0205 | -0.1491 +/- 0.0014 | 0.7471 | 0.000268 |
| entropy | -0.2088 +/- 0.0528 | 1.0888 | -0.1958 +/- 0.0325 | 0.4849 | 0.105655 |

- Best return AUC: `d_optimal`.
- Best final true return: `d_optimal`.
- Lowest policy-direction posterior variance: `perfal_topB`.

### beta=1, l2=0.01

| Method | Final true return | Final regret | Return AUC | Grad cosine | Dir. variance |
|---|---:|---:|---:|---:|---:|
| d_optimal | -0.1354 +/- 0.0008 | 1.0154 | -0.1439 +/- 0.0009 | 0.6081 | 0.000623 |
| perfal_random_v | -0.1364 +/- 0.0009 | 1.0164 | -0.1451 +/- 0.0011 | 0.6671 | 0.000847 |
| on_policy_random | -0.1354 +/- 0.0013 | 1.0154 | -0.1454 +/- 0.0015 | 0.6480 | 0.001065 |
| pool_uniform | -0.1358 +/- 0.0016 | 1.0158 | -0.1454 +/- 0.0018 | 0.6628 | 0.000978 |
| perfal_topB | -0.1379 +/- 0.0009 | 1.0179 | -0.1458 +/- 0.0011 | 0.7763 | 0.000227 |
| disagreement | -0.1366 +/- 0.0008 | 1.0166 | -0.1464 +/- 0.0007 | 0.6337 | 0.000830 |
| perfal_oracle | -0.1386 +/- 0.0006 | 1.0186 | -0.1474 +/- 0.0007 | 0.7505 | 0.000527 |
| perfal_greedy | -0.1396 +/- 0.0007 | 1.0196 | -0.1487 +/- 0.0008 | 0.8299 | 0.000212 |
| entropy | -0.1823 +/- 0.0411 | 1.0623 | -0.1754 +/- 0.0190 | 0.5283 | 0.037599 |

- Best return AUC: `d_optimal`.
- Best final true return: `on_policy_random`.
- Lowest policy-direction posterior variance: `perfal_greedy`.

### beta=3, l2=0.0001

| Method | Final true return | Final regret | Return AUC | Grad cosine | Dir. variance |
|---|---:|---:|---:|---:|---:|
| entropy | -0.1082 +/- 0.0064 | 0.9882 | -0.1277 +/- 0.0007 | 0.6319 | 0.083585 |
| disagreement | -0.1228 +/- 0.0005 | 1.0028 | -0.1308 +/- 0.0007 | 0.6741 | 0.007462 |
| pool_uniform | -0.1138 +/- 0.0092 | 0.9938 | -0.1313 +/- 0.0016 | 0.7015 | 0.029747 |
| d_optimal | -0.1252 +/- 0.0004 | 1.0052 | -0.1327 +/- 0.0007 | 0.7445 | 0.000596 |
| perfal_random_v | -0.1254 +/- 0.0004 | 1.0054 | -0.1331 +/- 0.0007 | 0.7425 | 0.001106 |
| on_policy_random | -0.1229 +/- 0.0012 | 1.0029 | -0.1332 +/- 0.0013 | 0.7677 | 0.014959 |
| perfal_oracle | -0.1259 +/- 0.0003 | 1.0059 | -0.1335 +/- 0.0006 | 0.8540 | 0.000363 |
| perfal_topB | -0.1268 +/- 0.0004 | 1.0068 | -0.1343 +/- 0.0008 | 0.8579 | 0.000183 |
| perfal_greedy | -0.1269 +/- 0.0004 | 1.0069 | -0.1355 +/- 0.0008 | 0.8731 | 0.000214 |

- Best return AUC: `entropy`.
- Best final true return: `entropy`.
- Lowest policy-direction posterior variance: `perfal_topB`.

### beta=3, l2=0.001

| Method | Final true return | Final regret | Return AUC | Grad cosine | Dir. variance |
|---|---:|---:|---:|---:|---:|
| disagreement | -0.1240 +/- 0.0004 | 1.0040 | -0.1327 +/- 0.0007 | 0.6672 | 0.005184 |
| perfal_random_v | -0.1256 +/- 0.0003 | 1.0056 | -0.1337 +/- 0.0006 | 0.7788 | 0.000618 |
| d_optimal | -0.1252 +/- 0.0004 | 1.0052 | -0.1337 +/- 0.0006 | 0.7454 | 0.000507 |
| pool_uniform | -0.1233 +/- 0.0017 | 1.0033 | -0.1338 +/- 0.0011 | 0.7221 | 0.008635 |
| on_policy_random | -0.1245 +/- 0.0007 | 1.0045 | -0.1339 +/- 0.0009 | 0.7135 | 0.002399 |
| perfal_oracle | -0.1265 +/- 0.0003 | 1.0065 | -0.1349 +/- 0.0005 | 0.8124 | 0.000611 |
| perfal_topB | -0.1272 +/- 0.0003 | 1.0072 | -0.1354 +/- 0.0007 | 0.9008 | 0.000149 |
| perfal_greedy | -0.1272 +/- 0.0003 | 1.0072 | -0.1364 +/- 0.0006 | 0.9120 | 0.000167 |
| entropy | -0.1445 +/- 0.0312 | 1.0245 | -0.1508 +/- 0.0181 | 0.6139 | 0.030919 |

- Best return AUC: `disagreement`.
- Best final true return: `pool_uniform`.
- Lowest policy-direction posterior variance: `perfal_topB`.

### beta=3, l2=0.01

| Method | Final true return | Final regret | Return AUC | Grad cosine | Dir. variance |
|---|---:|---:|---:|---:|---:|
| disagreement | -0.1248 +/- 0.0005 | 1.0048 | -0.1342 +/- 0.0007 | 0.7049 | 0.003154 |
| perfal_random_v | -0.1260 +/- 0.0003 | 1.0060 | -0.1346 +/- 0.0006 | 0.7675 | 0.000872 |
| d_optimal | -0.1259 +/- 0.0003 | 1.0059 | -0.1350 +/- 0.0006 | 0.7643 | 0.000519 |
| perfal_topB | -0.1271 +/- 0.0003 | 1.0071 | -0.1357 +/- 0.0006 | 0.8558 | 0.000223 |
| perfal_oracle | -0.1272 +/- 0.0002 | 1.0072 | -0.1363 +/- 0.0004 | 0.8292 | 0.000474 |
| pool_uniform | -0.1267 +/- 0.0005 | 1.0067 | -0.1363 +/- 0.0008 | 0.7480 | 0.001064 |
| on_policy_random | -0.1270 +/- 0.0007 | 1.0070 | -0.1370 +/- 0.0009 | 0.7954 | 0.000983 |
| perfal_greedy | -0.1279 +/- 0.0003 | 1.0079 | -0.1377 +/- 0.0005 | 0.9027 | 0.000177 |
| entropy | -0.1242 +/- 0.0028 | 1.0042 | -0.1394 +/- 0.0039 | 0.6247 | 0.030862 |

- Best return AUC: `disagreement`.
- Best final true return: `entropy`.
- Lowest policy-direction posterior variance: `perfal_greedy`.

### beta=10, l2=0.0001

| Method | Final true return | Final regret | Return AUC | Grad cosine | Dir. variance |
|---|---:|---:|---:|---:|---:|
| perfal_oracle | 0.0775 +/- 0.0482 | 0.8025 | -0.0944 +/- 0.0089 | 0.8816 | 0.611411 |
| perfal_random_v | 0.0033 +/- 0.0385 | 0.8767 | -0.1108 +/- 0.0053 | 0.8466 | 0.506006 |
| disagreement | -0.0364 +/- 0.0331 | 0.9164 | -0.1180 +/- 0.0040 | 0.7747 | 0.976049 |
| perfal_topB | -0.0529 +/- 0.0267 | 0.9329 | -0.1196 +/- 0.0026 | 0.9529 | 0.444562 |
| perfal_greedy | -0.0717 +/- 0.0138 | 0.9517 | -0.1213 +/- 0.0011 | 0.9511 | 0.637543 |
| pool_uniform | -0.0593 +/- 0.0226 | 0.9393 | -0.1227 +/- 0.0020 | 0.8667 | 1.480295 |
| on_policy_random | -0.0625 +/- 0.0226 | 0.9425 | -0.1235 +/- 0.0017 | 0.8750 | 1.535579 |
| d_optimal | -0.0967 +/- 0.0153 | 0.9767 | -0.1236 +/- 0.0016 | 0.9192 | 0.127517 |
| entropy | -0.1011 +/- 0.0074 | 0.9811 | -0.1275 +/- 0.0008 | 0.6609 | 0.105193 |

- Best return AUC: `perfal_oracle`.
- Best final true return: `perfal_oracle`.
- Lowest policy-direction posterior variance: `entropy`.

### beta=10, l2=0.001

| Method | Final true return | Final regret | Return AUC | Grad cosine | Dir. variance |
|---|---:|---:|---:|---:|---:|
| disagreement | -0.0841 +/- 0.0192 | 0.9641 | -0.1250 +/- 0.0027 | 0.7542 | 0.103763 |
| perfal_random_v | -0.1039 +/- 0.0093 | 0.9839 | -0.1257 +/- 0.0008 | 0.8224 | 0.026920 |
| perfal_oracle | -0.1127 +/- 0.0017 | 0.9927 | -0.1262 +/- 0.0003 | 0.9003 | 0.010046 |
| perfal_greedy | -0.1148 +/- 0.0012 | 0.9948 | -0.1270 +/- 0.0002 | 0.9450 | 0.006390 |
| perfal_topB | -0.1156 +/- 0.0013 | 0.9956 | -0.1276 +/- 0.0003 | 0.9393 | 0.006141 |
| d_optimal | -0.1180 +/- 0.0005 | 0.9980 | -0.1278 +/- 0.0003 | 0.8964 | 0.003243 |
| pool_uniform | -0.1140 +/- 0.0032 | 0.9940 | -0.1296 +/- 0.0003 | 0.8503 | 0.028905 |
| on_policy_random | -0.1177 +/- 0.0013 | 0.9977 | -0.1297 +/- 0.0002 | 0.8635 | 0.015321 |
| entropy | -0.1093 +/- 0.0055 | 0.9893 | -0.1357 +/- 0.0028 | 0.6673 | 0.050190 |

- Best return AUC: `disagreement`.
- Best final true return: `disagreement`.
- Lowest policy-direction posterior variance: `d_optimal`.

### beta=10, l2=0.01

| Method | Final true return | Final regret | Return AUC | Grad cosine | Dir. variance |
|---|---:|---:|---:|---:|---:|
| perfal_random_v | -0.1183 +/- 0.0007 | 0.9983 | -0.1295 +/- 0.0003 | 0.8320 | 0.001838 |
| perfal_oracle | -0.1192 +/- 0.0004 | 0.9992 | -0.1300 +/- 0.0001 | 0.8592 | 0.001683 |
| d_optimal | -0.1205 +/- 0.0003 | 1.0005 | -0.1306 +/- 0.0003 | 0.8863 | 0.000560 |
| perfal_topB | -0.1202 +/- 0.0004 | 1.0002 | -0.1307 +/- 0.0003 | 0.9397 | 0.000478 |
| perfal_greedy | -0.1208 +/- 0.0002 | 1.0008 | -0.1309 +/- 0.0001 | 0.9418 | 0.000468 |
| disagreement | -0.1194 +/- 0.0008 | 0.9994 | -0.1312 +/- 0.0005 | 0.7464 | 0.007798 |
| pool_uniform | -0.1233 +/- 0.0002 | 1.0033 | -0.1331 +/- 0.0004 | 0.8023 | 0.001182 |
| on_policy_random | -0.1230 +/- 0.0002 | 1.0030 | -0.1335 +/- 0.0003 | 0.8721 | 0.001046 |
| entropy | -0.1288 +/- 0.0050 | 1.0088 | -0.1423 +/- 0.0046 | 0.6205 | 0.041570 |

- Best return AUC: `perfal_random_v`.
- Best final true return: `perfal_random_v`.
- Lowest policy-direction posterior variance: `perfal_greedy`.


## Readout

- Non-oracle PerfAL directions use the proxy gradient only; if the proxy gradient is zero, they use a random unit policy direction, not the true gradient.
- All methods within a seed share the same initial dataset and deterministic per-query Bradley-Terry label noise.
- Self-pairs are excluded from initial data, on-policy random queries, and candidate pools.
- The `perfal_random_v` baseline is included as the randomized-direction sanity check; it should be interpreted as a control for whether the PerfAL direction is doing useful work.

## Output Files

- `results/full_sweep_corrected/metrics.csv`: per-seed, per-round metrics.
- Single-setting runs put plots directly in `results/full_sweep_corrected/`.
- Multi-setting sweeps put plots under `results/full_sweep_corrected/beta*_l2*/`.
