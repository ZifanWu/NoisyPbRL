#!/usr/bin/env python3
"""Direction-limited fork/decoy toy experiment for PerfAL.

This environment is designed to test the regime where PerfAL should help:
many uncertain decoy reward features are available to the query pool, but the
current policy update mostly depends on resolving a narrow goal-vs-trap fork.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize


GRID = 5
HORIZON = 7
ACTIONS = ("up", "down", "left", "right", "stay")
N_ACTIONS = len(ACTIONS)
N_STATES = GRID * GRID
START = (0, 2)
FORK = (2, 2)
GOAL = (4, 3)
TRAP = (4, 1)
SAFE = (4, 2)
DECOYS = ((0, 4), (1, 4), (2, 4), (3, 4), (0, 0), (1, 0), (2, 0), (3, 0))
DECOY_TRANSITIONS = tuple((cell, action) for cell in DECOYS for action in range(N_ACTIONS))
FEATURE_NAMES = (
    ("goal", "trap", "safe")
    + tuple(f"decoy_{cell[0]}_{cell[1]}_{ACTIONS[action]}" for cell, action in DECOY_TRANSITIONS)
    + ("step",)
)
N_FEATURES = len(FEATURE_NAMES)
STEP_IDX = N_FEATURES - 1
TRUE_W = np.zeros(N_FEATURES, dtype=np.float64)
TRUE_W[0] = 1.0
TRUE_W[1] = -1.2
TRUE_W[2] = 0.2
TRUE_W[STEP_IDX] = -0.01

METHODS = (
    "on_policy_random",
    "pool_uniform",
    "entropy",
    "disagreement",
    "d_optimal",
    "perfal_topB",
    "perfal_greedy",
    "perfal_mixed",
    "perfal_oracle",
    "perfal_random_v",
)


@dataclass
class PairData:
    i: np.ndarray
    j: np.ndarray
    y: np.ndarray


@dataclass
class PolicyEval:
    pi: np.ndarray
    probs: np.ndarray
    scores: np.ndarray
    G: np.ndarray
    true_return: float
    true_grad: np.ndarray


@dataclass
class FitResult:
    phi: np.ndarray
    precision: np.ndarray
    precision_inv: np.ndarray
    success: bool


def state_id(x: int, y: int) -> int:
    return y * GRID + x


def next_state(x: int, y: int, action: int) -> Tuple[int, int]:
    if ACTIONS[action] == "up":
        y += 1
    elif ACTIONS[action] == "down":
        y -= 1
    elif ACTIONS[action] == "left":
        x -= 1
    elif ACTIONS[action] == "right":
        x += 1
    return min(max(x, 0), GRID - 1), min(max(y, 0), GRID - 1)


def step_features(x: int, y: int, action: int) -> Tuple[Tuple[int, int], np.ndarray]:
    nx, ny = next_state(x, y, action)
    feats = np.zeros(N_FEATURES, dtype=np.float64)
    feats[0] = float((nx, ny) == GOAL)
    feats[1] = float((nx, ny) == TRAP)
    feats[2] = float((nx, ny) == SAFE)
    for decoy_idx, (cell, decoy_action) in enumerate(DECOY_TRANSITIONS):
        feats[3 + decoy_idx] = float((nx, ny) == cell and action == decoy_action)
    feats[STEP_IDX] = 1.0
    return (nx, ny), feats


def enumerate_trajectories() -> Dict[str, np.ndarray]:
    n = N_ACTIONS**HORIZON
    action_counts = np.zeros((n, N_STATES, N_ACTIONS), dtype=np.float64)
    features = np.zeros((n, N_FEATURES), dtype=np.float64)
    for traj_idx in range(n):
        code = traj_idx
        x, y = START
        for _ in range(HORIZON):
            action = code % N_ACTIONS
            code //= N_ACTIONS
            action_counts[traj_idx, state_id(x, y), action] += 1.0
            (x, y), feats = step_features(x, y, action)
            features[traj_idx] += feats
    true_rewards = features @ TRUE_W
    return {
        "action_counts": action_counts,
        "state_counts": action_counts.sum(axis=2),
        "features": features,
        "true_rewards": true_rewards,
        "optimal_return": np.array(float(true_rewards.max())),
    }


def softmax(theta: np.ndarray) -> np.ndarray:
    centered = theta - theta.max(axis=1, keepdims=True)
    exp = np.exp(centered)
    return exp / exp.sum(axis=1, keepdims=True)


def policy_template(kind: str) -> np.ndarray:
    theta = np.zeros((N_STATES, N_ACTIONS), dtype=np.float64)
    up = ACTIONS.index("up")
    down = ACTIONS.index("down")
    left = ACTIONS.index("left")
    right = ACTIONS.index("right")
    stay = ACTIONS.index("stay")
    for y in range(GRID):
        for x in range(GRID):
            s = state_id(x, y)
            if kind == "explore":
                continue
            if kind == "decoy":
                if y < GRID - 1:
                    theta[s, up] = 2.2
                elif x < GRID - 1:
                    theta[s, right] = 2.2
                else:
                    theta[s, stay] = 2.2
                continue
            if (x, y) == GOAL or (x, y) == TRAP or (x, y) == SAFE:
                theta[s, stay] = 2.2
            elif x < FORK[0] and y == START[1]:
                theta[s, right] = 2.7
            elif (x, y) == FORK:
                theta[s, up] = 1.2
                theta[s, down] = 1.2
                theta[s, right] = 0.2
            elif y == 3 and 2 <= x < 4:
                theta[s, right] = 2.4
            elif y == 1 and 2 <= x < 4:
                theta[s, right] = 2.4
            elif y == 2 and 2 <= x < 4:
                theta[s, right] = 1.2
                theta[s, up] = 0.7
                theta[s, down] = 0.7
            elif x > 0 and y not in (1, 2, 3):
                theta[s, left] = 0.6
            else:
                theta[s, right] = 0.5
    theta -= theta.mean(axis=1, keepdims=True)
    return theta


def initial_theta() -> np.ndarray:
    return policy_template("current")


def evaluate_policy(theta: np.ndarray, traj: Dict[str, np.ndarray]) -> PolicyEval:
    pi = softmax(theta)
    log_pi = np.log(np.maximum(pi, 1e-300))
    log_probs = np.tensordot(traj["action_counts"], log_pi, axes=([1, 2], [0, 1]))
    probs = np.exp(log_probs - log_probs.max())
    probs /= probs.sum()
    score_tensor = traj["action_counts"] - traj["state_counts"][:, :, None] * pi[None, :, :]
    scores = score_tensor.reshape(traj["action_counts"].shape[0], -1)
    G = (scores * probs[:, None]).T @ traj["features"]
    true_grad = G @ TRUE_W
    return PolicyEval(
        pi=pi,
        probs=probs,
        scores=scores,
        G=G,
        true_return=float(probs @ traj["true_rewards"]),
        true_grad=true_grad,
    )


def sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def splitmix64(x: np.ndarray) -> np.ndarray:
    z = x + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def deterministic_uniforms(label_seed: int, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    base = np.full_like(i, np.uint64(label_seed), dtype=np.uint64)
    mixed = base ^ (i.astype(np.uint64) * np.uint64(0xD1B54A32D192ED03))
    mixed ^= j.astype(np.uint64) * np.uint64(0xABC98388FB8FAC03)
    bits = splitmix64(mixed)
    return ((bits >> np.uint64(11)).astype(np.float64)) * (1.0 / float(1 << 53))


def deterministic_labels(
    label_seed: int,
    i: np.ndarray,
    j: np.ndarray,
    true_rewards: np.ndarray,
    beta: float,
) -> np.ndarray:
    p = sigmoid(beta * (true_rewards[j] - true_rewards[i]))
    lo = np.minimum(i, j)
    hi = np.maximum(i, j)
    u = deterministic_uniforms(label_seed, lo, hi)
    ordered_u = np.where(i <= j, u, 1.0 - u)
    return (ordered_u < p).astype(np.float64)


def sample_trajectory_indices(rng: np.random.Generator, probs: np.ndarray, n: int) -> np.ndarray:
    return rng.choice(len(probs), size=n, replace=True, p=probs)


def sample_nonself_pairs(
    rng: np.random.Generator,
    probs: np.ndarray,
    n: int,
) -> Tuple[np.ndarray, np.ndarray]:
    i = sample_trajectory_indices(rng, probs, n)
    j = sample_trajectory_indices(rng, probs, n)
    for _ in range(100):
        mask = i == j
        if not np.any(mask):
            return i, j
        j[mask] = sample_trajectory_indices(rng, probs, int(mask.sum()))
    mask = i == j
    if np.any(mask):
        j[mask] = (j[mask] + 1) % len(probs)
    return i, j


def append_pairs(data: PairData, i: np.ndarray, j: np.ndarray, y: np.ndarray) -> PairData:
    return PairData(
        i=np.concatenate([data.i, i.astype(np.int64)]),
        j=np.concatenate([data.j, j.astype(np.int64)]),
        y=np.concatenate([data.y, y.astype(np.float64)]),
    )


def initial_dataset(
    rng: np.random.Generator,
    label_seed: int,
    init_labels: int,
    warmstart_critical: int,
    current_probs: np.ndarray,
    explore_probs: np.ndarray,
    traj: Dict[str, np.ndarray],
    beta: float,
) -> PairData:
    mix = 0.7 * current_probs + 0.3 * explore_probs
    i, j = sample_nonself_pairs(rng, mix, init_labels)
    if warmstart_critical > 0:
        features = traj["features"]
        goal_idx = np.flatnonzero(features[:, 0] > 0)
        trap_idx = np.flatnonzero(features[:, 1] > 0)
        safe_idx = np.flatnonzero(features[:, 2] > 0)
        anchors_i: List[int] = []
        anchors_j: List[int] = []
        for _ in range(warmstart_critical):
            source = rng.choice(3)
            if source == 0 and len(goal_idx) and len(trap_idx):
                anchors_i.append(int(rng.choice(trap_idx)))
                anchors_j.append(int(rng.choice(goal_idx)))
            elif source == 1 and len(goal_idx) and len(safe_idx):
                anchors_i.append(int(rng.choice(safe_idx)))
                anchors_j.append(int(rng.choice(goal_idx)))
            elif len(trap_idx) and len(safe_idx):
                anchors_i.append(int(rng.choice(trap_idx)))
                anchors_j.append(int(rng.choice(safe_idx)))
        if anchors_i:
            i = np.concatenate([i, np.array(anchors_i, dtype=np.int64)])
            j = np.concatenate([j, np.array(anchors_j, dtype=np.int64)])
    y = deterministic_labels(label_seed, i, j, traj["true_rewards"], beta)
    return PairData(i=i, j=j, y=y)


def fit_reward_model(
    data: PairData,
    features: np.ndarray,
    l2: float,
    x0: np.ndarray | None = None,
) -> FitResult:
    if x0 is None:
        x0 = np.zeros(features.shape[1], dtype=np.float64)
    delta = features[data.j] - features[data.i]
    if len(data.y) == 0:
        precision = l2 * np.eye(features.shape[1])
        return FitResult(x0.copy(), precision, np.linalg.inv(precision), True)
    signs = 2.0 * data.y - 1.0

    def objective(phi: np.ndarray) -> Tuple[float, np.ndarray]:
        margins = signs * (delta @ phi)
        loss = float(np.logaddexp(0.0, -margins).sum() + 0.5 * l2 * (phi @ phi))
        coeff = -signs * sigmoid(-margins)
        grad = delta.T @ coeff + l2 * phi
        return loss, grad

    result = minimize(
        fun=lambda p: objective(p)[0],
        x0=x0,
        jac=lambda p: objective(p)[1],
        method="L-BFGS-B",
        options={"maxiter": 250, "ftol": 1e-10, "gtol": 1e-8},
    )
    phi = result.x.astype(np.float64)
    p = sigmoid(delta @ phi)
    w = p * (1.0 - p)
    precision = delta.T @ (delta * w[:, None]) + l2 * np.eye(features.shape[1])
    return FitResult(phi=phi, precision=precision, precision_inv=np.linalg.inv(precision), success=bool(result.success))


def pair_quantities(
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    features: np.ndarray,
    phi: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    delta = features[pair_j] - features[pair_i]
    p = sigmoid(delta @ phi)
    w = np.maximum(p * (1.0 - p), 1e-12)
    return delta, p, w


def build_candidate_pool(
    rng: np.random.Generator,
    current_probs: np.ndarray,
    decoy_probs: np.ndarray,
    explore_probs: np.ndarray,
    n_pairs: int,
) -> Tuple[np.ndarray, np.ndarray]:
    components = rng.choice(3, size=n_pairs, p=np.array([0.30, 0.55, 0.15]))
    i = np.empty(n_pairs, dtype=np.int64)
    j = np.empty(n_pairs, dtype=np.int64)
    for component, probs in enumerate((current_probs, decoy_probs, explore_probs)):
        mask = components == component
        if np.any(mask):
            pair_i, pair_j = sample_nonself_pairs(rng, probs, int(mask.sum()))
            i[mask] = pair_i
            j[mask] = pair_j
    return i, j


def normalized(x: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(x))
    if norm > 1e-12:
        return x / norm
    return np.zeros_like(x)


def proxy_direction(proxy_grad: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    norm = float(np.linalg.norm(proxy_grad))
    if norm > 1e-12:
        return proxy_grad / norm
    return normalized(rng.normal(size=proxy_grad.shape))


def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    k = min(k, len(scores))
    if k == len(scores):
        return np.argsort(-scores)
    idx = np.argpartition(-scores, kth=k - 1)[:k]
    return idx[np.argsort(-scores[idx])]


def greedy_rank_one(
    delta: np.ndarray,
    w: np.ndarray,
    A0: np.ndarray,
    batch_size: int,
    mode: str,
    v: np.ndarray | None = None,
    alpha: float = 0.7,
) -> np.ndarray:
    A = A0.copy()
    available = np.ones(len(delta), dtype=bool)
    selected: List[int] = []
    for _ in range(min(batch_size, len(delta))):
        q = np.einsum("ij,jk,ik->i", delta, A, delta)
        d_gain = np.log1p(w * q)
        if mode == "d_optimal":
            gains = d_gain
        else:
            assert v is not None
            m = A @ v
            projected = delta @ m
            p_gain = (w * projected * projected) / (1.0 + w * q)
            if mode == "perfal":
                gains = p_gain
            elif mode == "mixed":
                p_scale = max(float(np.max(p_gain[available])), 1e-12)
                d_scale = max(float(np.max(d_gain[available])), 1e-12)
                gains = alpha * (p_gain / p_scale) + (1.0 - alpha) * (d_gain / d_scale)
            else:
                raise ValueError(mode)
        gains = np.where(available, gains, -np.inf)
        chosen = int(np.argmax(gains))
        selected.append(chosen)
        available[chosen] = False
        d = delta[chosen]
        Ad = A @ d
        denom = 1.0 + w[chosen] * float(d @ Ad)
        A = A - (w[chosen] / denom) * np.outer(Ad, Ad)
    return np.array(selected, dtype=np.int64)


def choose_pairs(
    method: str,
    rng: np.random.Generator,
    eval_info: PolicyEval,
    decoy_probs: np.ndarray,
    explore_probs: np.ndarray,
    fit: FitResult,
    features: np.ndarray,
    batch_size: int,
    candidate_pairs: int,
    mixed_alpha: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if method == "on_policy_random":
        return sample_nonself_pairs(rng, eval_info.probs, batch_size)
    pool_i, pool_j = build_candidate_pool(rng, eval_info.probs, decoy_probs, explore_probs, candidate_pairs)
    delta, _, w = pair_quantities(pool_i, pool_j, features, fit.phi)
    A = fit.precision_inv
    proxy_grad = eval_info.G @ fit.phi
    v = eval_info.G.T @ proxy_direction(proxy_grad, rng)
    oracle_v = eval_info.G.T @ normalized(eval_info.true_grad)
    if method == "pool_uniform":
        selected = rng.choice(len(pool_i), size=min(batch_size, len(pool_i)), replace=False)
    elif method == "entropy":
        selected = topk_indices(w, batch_size)
    elif method == "disagreement":
        selected = topk_indices(np.einsum("ij,jk,ik->i", delta, A, delta), batch_size)
    elif method == "d_optimal":
        selected = greedy_rank_one(delta, w, A, batch_size, mode="d_optimal")
    elif method == "perfal_topB":
        score = w * np.square(delta @ (A @ v))
        selected = topk_indices(score, batch_size)
    elif method == "perfal_greedy":
        selected = greedy_rank_one(delta, w, A, batch_size, mode="perfal", v=v)
    elif method == "perfal_mixed":
        selected = greedy_rank_one(delta, w, A, batch_size, mode="mixed", v=v, alpha=mixed_alpha)
    elif method == "perfal_oracle":
        selected = greedy_rank_one(delta, w, A, batch_size, mode="perfal", v=oracle_v)
    elif method == "perfal_random_v":
        random_v = normalized(rng.normal(size=N_FEATURES))
        random_v *= max(float(np.linalg.norm(v)), 1e-12)
        selected = greedy_rank_one(delta, w, A, batch_size, mode="perfal", v=random_v)
    else:
        raise ValueError(method)
    return pool_i[selected], pool_j[selected]


def centered_rm_errors(
    phi: np.ndarray,
    features: np.ndarray,
    true_rewards: np.ndarray,
    probs: np.ndarray,
) -> Tuple[float, float, float, float]:
    pred = features @ phi
    full = float(np.sqrt(np.mean(np.square((pred - pred.mean()) - (true_rewards - true_rewards.mean())))))
    pred_policy = pred - float(probs @ pred)
    true_policy = true_rewards - float(probs @ true_rewards)
    policy = float(np.sqrt(probs @ np.square(pred_policy - true_policy)))
    critical = float(np.linalg.norm((phi[:3] - TRUE_W[:3]), ord=2))
    decoy = float(np.mean(np.abs(phi[3:STEP_IDX])))
    return full, policy, critical, decoy


def metric_row(
    method: str,
    seed: int,
    round_idx: int,
    labels: int,
    eval_info: PolicyEval,
    fit: FitResult,
    traj: Dict[str, np.ndarray],
) -> Dict[str, float | int | str]:
    proxy_grad = eval_info.G @ fit.phi
    proxy_hat = normalized(proxy_grad)
    v = eval_info.G.T @ proxy_hat
    proxy_norm = float(np.linalg.norm(proxy_grad))
    true_norm = float(np.linalg.norm(eval_info.true_grad))
    cosine = 0.0
    if proxy_norm > 1e-12 and true_norm > 1e-12:
        cosine = float((proxy_grad @ eval_info.true_grad) / (proxy_norm * true_norm))
    full_err, policy_err, critical_err, decoy_abs = centered_rm_errors(
        fit.phi, traj["features"], traj["true_rewards"], eval_info.probs
    )
    return {
        "method": method,
        "seed": seed,
        "round": round_idx,
        "labels": labels,
        "true_return": eval_info.true_return,
        "regret": float(traj["optimal_return"] - eval_info.true_return),
        "gradient_cosine": cosine,
        "directional_gradient_bias": float(abs(proxy_hat @ (proxy_grad - eval_info.true_grad))),
        "posterior_directional_variance": float(v @ fit.precision_inv @ v),
        "rm_error_full_centered": full_err,
        "rm_error_policy_centered": policy_err,
        "rm_error_critical": critical_err,
        "rm_decoy_abs_mean": decoy_abs,
        "proxy_grad_norm": proxy_norm,
        "true_grad_norm": true_norm,
    }


def update_policy(theta: np.ndarray, proxy_grad: np.ndarray, lr: float) -> np.ndarray:
    next_theta = theta + lr * proxy_grad.reshape(N_STATES, N_ACTIONS)
    next_theta -= next_theta.mean(axis=1, keepdims=True)
    return np.clip(next_theta, -8.0, 8.0)


def run_one_method(
    method: str,
    seed: int,
    args: argparse.Namespace,
    traj: Dict[str, np.ndarray],
    decoy_probs: np.ndarray,
    explore_probs: np.ndarray,
    initial_data: PairData,
) -> List[Dict[str, float | int | str]]:
    rng = np.random.default_rng(seed * 1009 + METHODS.index(method) * 9173 + 8753)
    theta = initial_theta()
    data = PairData(initial_data.i.copy(), initial_data.j.copy(), initial_data.y.copy())
    rows: List[Dict[str, float | int | str]] = []
    phi_start = np.zeros(N_FEATURES, dtype=np.float64)
    for round_idx in range(args.rounds + 1):
        eval_info = evaluate_policy(theta, traj)
        fit = fit_reward_model(data, traj["features"], args.l2, x0=phi_start)
        phi_start = fit.phi
        row = metric_row(method, seed, round_idx, len(data.y), eval_info, fit, traj)
        row["beta"] = args.beta
        row["l2"] = args.l2
        rows.append(row)
        if round_idx == args.rounds:
            break
        pair_i, pair_j = choose_pairs(
            method,
            rng,
            eval_info,
            decoy_probs,
            explore_probs,
            fit,
            traj["features"],
            args.batch_size,
            args.candidate_pairs,
            args.mixed_alpha,
        )
        labels = deterministic_labels(seed, pair_i, pair_j, traj["true_rewards"], args.beta)
        data = append_pairs(data, pair_i, pair_j, labels)
        fit_after = fit_reward_model(data, traj["features"], args.l2, x0=fit.phi)
        phi_start = fit_after.phi
        theta = update_policy(theta, eval_info.G @ fit_after.phi, args.policy_lr)
    return rows


def write_metrics_csv(path: Path, rows: Sequence[Dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def mean_stderr(values: Sequence[float]) -> Tuple[float, float]:
    arr = np.array(values, dtype=np.float64)
    if len(arr) <= 1:
        return float(arr.mean()), 0.0
    return float(arr.mean()), float(arr.std(ddof=1) / math.sqrt(len(arr)))


def auc_by_method_seed(rows: Sequence[Dict[str, float | int | str]]) -> Dict[Tuple[str, int], float]:
    grouped: Dict[Tuple[str, int], List[Dict[str, float | int | str]]] = {}
    for row in rows:
        grouped.setdefault((str(row["method"]), int(row["seed"])), []).append(row)
    aucs: Dict[Tuple[str, int], float] = {}
    for key, group in grouped.items():
        ordered = sorted(group, key=lambda r: int(r["labels"]))
        x = np.array([float(r["labels"]) for r in ordered])
        y = np.array([float(r["true_return"]) for r in ordered])
        aucs[key] = float(np.trapz(y, x) / max(float(x[-1] - x[0]), 1.0))
    return aucs


def aggregate_summary(rows: Sequence[Dict[str, float | int | str]], final_round: int) -> List[Dict[str, float | str]]:
    aucs = auc_by_method_seed(rows)
    summary = []
    for method in METHODS:
        final = [r for r in rows if r["method"] == method and int(r["round"]) == final_round]
        auc_values = [aucs[(method, int(r["seed"]))] for r in final]
        ret = mean_stderr([float(r["true_return"]) for r in final])
        regret = mean_stderr([float(r["regret"]) for r in final])
        cosine = mean_stderr([float(r["gradient_cosine"]) for r in final])
        var = mean_stderr([float(r["posterior_directional_variance"]) for r in final])
        critical = mean_stderr([float(r["rm_error_critical"]) for r in final])
        decoy = mean_stderr([float(r["rm_decoy_abs_mean"]) for r in final])
        auc = mean_stderr(auc_values)
        summary.append(
            {
                "method": method,
                "final_true_return_mean": ret[0],
                "final_true_return_se": ret[1],
                "final_regret_mean": regret[0],
                "auc_return_mean": auc[0],
                "auc_return_se": auc[1],
                "gradient_cosine_mean": cosine[0],
                "posterior_dir_var_mean": var[0],
                "critical_rm_error_mean": critical[0],
                "decoy_abs_mean": decoy[0],
            }
        )
    summary.sort(key=lambda r: float(r["auc_return_mean"]), reverse=True)
    return summary


def plot_metric(rows: Sequence[Dict[str, float | int | str]], metric: str, ylabel: str, path: Path) -> None:
    plt.figure(figsize=(9, 5.5))
    for method in METHODS:
        method_rows = [r for r in rows if r["method"] == method]
        by_round: Dict[int, List[Dict[str, float | int | str]]] = {}
        for row in method_rows:
            by_round.setdefault(int(row["round"]), []).append(row)
        xs, means, ses = [], [], []
        for round_idx in sorted(by_round):
            group = by_round[round_idx]
            xs.append(float(group[0]["labels"]))
            mean, se = mean_stderr([float(r[metric]) for r in group])
            means.append(mean)
            ses.append(se)
        x = np.array(xs)
        y = np.array(means)
        e = np.array(ses)
        plt.plot(x, y, label=method, linewidth=1.7)
        plt.fill_between(x, y - e, y + e, alpha=0.10)
    plt.xlabel("Preference labels")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def write_summary(path: Path, rows: Sequence[Dict[str, float | int | str]], args: argparse.Namespace, config: Dict[str, object]) -> None:
    aggregate = aggregate_summary(rows, args.rounds)
    best_auc = aggregate[0]
    best_return = max(aggregate, key=lambda r: float(r["final_true_return_mean"]))
    best_var = min(aggregate, key=lambda r: float(r["posterior_dir_var_mean"]))
    table = [
        "| Method | Final return | Return AUC | Grad cosine | Dir. var | Critical RM err | Decoy |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        table.append(
            "| {method} | {ret:.4f} +/- {ret_se:.4f} | {auc:.4f} +/- {auc_se:.4f} | {cos:.4f} | {var:.6f} | {crit:.4f} | {decoy:.4f} |".format(
                method=row["method"],
                ret=float(row["final_true_return_mean"]),
                ret_se=float(row["final_true_return_se"]),
                auc=float(row["auc_return_mean"]),
                auc_se=float(row["auc_return_se"]),
                cos=float(row["gradient_cosine_mean"]),
                var=float(row["posterior_dir_var_mean"]),
                crit=float(row["critical_rm_error_mean"]),
                decoy=float(row["decoy_abs_mean"]),
            )
        )
    greedy_auc = next(float(row["auc_return_mean"]) for row in aggregate if row["method"] == "perfal_greedy")
    topb_auc = next(float(row["auc_return_mean"]) for row in aggregate if row["method"] == "perfal_topB")
    mixed_auc = next(float(row["auc_return_mean"]) for row in aggregate if row["method"] == "perfal_mixed")
    random_auc = next(float(row["auc_return_mean"]) for row in aggregate if row["method"] == "perfal_random_v")
    text = f"""# Fork-Decoy PerfAL Result Summary

Generated: {datetime.now().isoformat(timespec="seconds")}

This direction-limited toy environment has a current policy near a goal/trap fork and a query pool containing many high-uncertainty zero-reward decoy features.

## Configuration

```json
{json.dumps(config, indent=2)}
```

## Aggregate Results

{chr(10).join(table)}

## Readout

- Best return AUC: `{best_auc['method']}`.
- Best final true return: `{best_return['method']}`.
- Lowest final policy-direction posterior variance: `{best_var['method']}`.
- `perfal_topB` beats `perfal_random_v` on AUC: `{topb_auc > random_auc}`.
- `perfal_greedy` beats `perfal_random_v` on AUC: `{greedy_auc > random_auc}`.
- `perfal_mixed` beats `perfal_random_v` on AUC: `{mixed_auc > random_auc}`.
- The intended positive signal is PerfAL or PerfAL+coverage improving return while avoiding decoy-focused uncertainty.

## Output Files

- `results/{args.output_name}/metrics.csv`
- `results/{args.output_name}/true_return.png`
- `results/{args.output_name}/posterior_directional_variance.png`
- `results/{args.output_name}/critical_rm_error.png`
"""
    path.write_text(text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("quick", "paper"), default="quick")
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--init-labels", type=int, default=16)
    parser.add_argument("--warmstart-critical", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--candidate-pairs", type=int, default=4000)
    parser.add_argument("--beta", type=float, default=5.0)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--policy-lr", type=float, default=0.5)
    parser.add_argument("--mixed-alpha", type=float, default=0.7)
    args = parser.parse_args()
    if args.preset == "paper":
        args.seeds = list(range(30))
        args.rounds = 25
        args.init_labels = 16
        args.warmstart_critical = 6
        args.batch_size = 8
        args.candidate_pairs = 8000
        args.beta = 5.0
        args.l2 = 1e-3
        args.policy_lr = 0.5
        args.mixed_alpha = 0.7
        if args.output_name is None:
            args.output_name = "fork_decoy_paper"
    if args.output_name is None:
        args.output_name = "fork_decoy_quick"
    return args


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    out_dir = root / "results" / args.output_name
    out_dir.mkdir(parents=True, exist_ok=True)

    traj = enumerate_trajectories()
    current_eval = evaluate_policy(initial_theta(), traj)
    decoy_eval = evaluate_policy(policy_template("decoy"), traj)
    explore_eval = evaluate_policy(policy_template("explore"), traj)
    rows: List[Dict[str, float | int | str]] = []

    for seed in args.seeds:
        init_rng = np.random.default_rng(seed * 1009 + 777)
        initial_data = initial_dataset(
            init_rng,
            label_seed=seed,
            init_labels=args.init_labels,
            warmstart_critical=args.warmstart_critical,
            current_probs=current_eval.probs,
            explore_probs=explore_eval.probs,
            traj=traj,
            beta=args.beta,
        )
        for method in METHODS:
            rows.extend(run_one_method(method, seed, args, traj, decoy_eval.probs, explore_eval.probs, initial_data))

    config = {
        "preset": args.preset,
        "output_name": args.output_name,
        "seeds": args.seeds,
        "rounds": args.rounds,
        "init_labels": args.init_labels,
        "warmstart_critical": args.warmstart_critical,
        "batch_size": args.batch_size,
        "candidate_pairs": args.candidate_pairs,
        "beta": args.beta,
        "l2": args.l2,
        "policy_lr": args.policy_lr,
        "mixed_alpha": args.mixed_alpha,
        "grid": [GRID, GRID],
        "horizon": HORIZON,
        "start": START,
        "fork": FORK,
        "goal": GOAL,
        "trap": TRAP,
        "safe": SAFE,
        "decoys": DECOYS,
        "feature_names": FEATURE_NAMES,
        "true_reward": TRUE_W.tolist(),
        "methods": METHODS,
    }
    write_metrics_csv(out_dir / "metrics.csv", rows)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    plot_metric(rows, "true_return", "Exact true return", out_dir / "true_return.png")
    plot_metric(rows, "posterior_directional_variance", "Posterior directional variance", out_dir / "posterior_directional_variance.png")
    plot_metric(rows, "rm_error_critical", "Critical reward-weight error", out_dir / "critical_rm_error.png")
    write_summary(root / "fork_decoy_summary.md", rows, args, config)
    print(f"Wrote {out_dir / 'metrics.csv'}")
    print(f"Wrote {root / 'fork_decoy_summary.md'}")


if __name__ == "__main__":
    main()
