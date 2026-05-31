#!/usr/bin/env python3
"""Exact tabular toy experiment for PerfAL.

The experiment is intentionally independent from the main B-Pref/PEBBLE stack.
It enumerates every length-6 trajectory in a 4x4 deterministic gridworld and
uses exact policy-gradient and reward-model information quantities.
"""

from __future__ import annotations

import argparse
import copy
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


GRID = 4
HORIZON = 6
ACTIONS = ("up", "down", "left", "right", "stay")
N_ACTIONS = len(ACTIONS)
N_STATES = GRID * GRID
N_FEATURES = 4
START = (0, 0)
GOAL = (3, 3)
TRAP = (2, 3)
SIDE = (0, 2)
TRUE_W = np.array([1.0, -0.8, 0.0, -0.02], dtype=np.float64)
METHODS = (
    "on_policy_random",
    "pool_uniform",
    "entropy",
    "disagreement",
    "d_optimal",
    "perfal_topB",
    "perfal_greedy",
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
    feats = np.array(
        [
            float((nx, ny) == GOAL),
            float((nx, ny) == TRAP),
            float((nx, ny) == SIDE),
            1.0,
        ],
        dtype=np.float64,
    )
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
            state = state_id(x, y)
            action_counts[traj_idx, state, action] += 1.0
            (x, y), feats = step_features(x, y, action)
            features[traj_idx] += feats

    state_counts = action_counts.sum(axis=2)
    true_rewards = features @ TRUE_W
    optimal_return = float(true_rewards.max())
    return {
        "action_counts": action_counts,
        "state_counts": state_counts,
        "features": features,
        "true_rewards": true_rewards,
        "optimal_return": np.array(optimal_return),
    }


def softmax(theta: np.ndarray) -> np.ndarray:
    centered = theta - theta.max(axis=1, keepdims=True)
    exp = np.exp(centered)
    return exp / exp.sum(axis=1, keepdims=True)


def initial_theta() -> np.ndarray:
    theta = np.zeros((N_STATES, N_ACTIONS), dtype=np.float64)
    for y in range(GRID):
        for x in range(GRID):
            s = state_id(x, y)
            if (x, y) == SIDE:
                theta[s, ACTIONS.index("stay")] = 1.9
                theta[s, ACTIONS.index("right")] = 0.5
                theta[s, ACTIONS.index("up")] = 0.3
            elif x == 0 and y < 2:
                theta[s, ACTIONS.index("up")] = 2.0
                theta[s, ACTIONS.index("stay")] = 0.2
            elif y == 3 and x < 3:
                theta[s, ACTIONS.index("right")] = 1.1
                theta[s, ACTIONS.index("stay")] = 0.2
            elif x < 3 and y < 3:
                theta[s, ACTIONS.index("up")] = 0.65
                theta[s, ACTIONS.index("right")] = 0.65
            elif (x, y) == GOAL:
                theta[s, ACTIONS.index("stay")] = 1.0
            else:
                theta[s, ACTIONS.index("up")] = 0.4
                theta[s, ACTIONS.index("right")] = 0.4
    theta -= theta.mean(axis=1, keepdims=True)
    return theta


def evaluate_policy(theta: np.ndarray, traj: Dict[str, np.ndarray]) -> PolicyEval:
    action_counts = traj["action_counts"]
    state_counts = traj["state_counts"]
    features = traj["features"]
    true_rewards = traj["true_rewards"]

    pi = softmax(theta)
    log_pi = np.log(np.maximum(pi, 1e-300))
    log_probs = np.tensordot(action_counts, log_pi, axes=([1, 2], [0, 1]))
    probs = np.exp(log_probs - log_probs.max())
    probs = probs / probs.sum()

    score_tensor = action_counts - state_counts[:, :, None] * pi[None, :, :]
    scores = score_tensor.reshape(action_counts.shape[0], -1)
    weighted_scores = scores * probs[:, None]
    G = weighted_scores.T @ features
    true_grad = G @ TRUE_W
    true_return = float(probs @ true_rewards)
    return PolicyEval(pi=pi, probs=probs, scores=scores, G=G, true_return=true_return, true_grad=true_grad)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def append_pairs(data: PairData, i: np.ndarray, j: np.ndarray, y: np.ndarray) -> PairData:
    return PairData(
        i=np.concatenate([data.i, i.astype(np.int64)]),
        j=np.concatenate([data.j, j.astype(np.int64)]),
        y=np.concatenate([data.y, y.astype(np.float64)]),
    )


def splitmix64(x: np.ndarray) -> np.ndarray:
    z = x + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def deterministic_uniforms(label_seed: int, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    base = np.full_like(i, np.uint64(label_seed), dtype=np.uint64)
    x = i.astype(np.uint64)
    y = j.astype(np.uint64)
    mixed = base ^ (x * np.uint64(0xD1B54A32D192ED03)) ^ (y * np.uint64(0xABC98388FB8FAC03))
    bits = splitmix64(mixed)
    return ((bits >> np.uint64(11)).astype(np.float64)) * (1.0 / float(1 << 53))


def deterministic_labels(
    label_seed: int,
    i: np.ndarray,
    j: np.ndarray,
    true_rewards: np.ndarray,
    beta: float,
) -> np.ndarray:
    """Common-random-number Bradley-Terry labels for fair method comparison."""
    logits = beta * (true_rewards[j] - true_rewards[i])
    p = sigmoid(logits)
    lo = np.minimum(i, j)
    hi = np.maximum(i, j)
    u = deterministic_uniforms(label_seed, lo, hi)
    ordered_u = np.where(i <= j, u, 1.0 - u)
    return (ordered_u < p).astype(np.float64)


def fit_reward_model(
    data: PairData,
    features: np.ndarray,
    l2: float,
    x0: np.ndarray | None = None,
) -> FitResult:
    if x0 is None:
        x0 = np.zeros(N_FEATURES, dtype=np.float64)
    if len(data.y) == 0:
        precision = l2 * np.eye(N_FEATURES)
        return FitResult(x0.copy(), precision, np.linalg.inv(precision), True)

    delta = features[data.j] - features[data.i]
    signs = 2.0 * data.y - 1.0

    def objective(phi: np.ndarray) -> Tuple[float, np.ndarray]:
        z = delta @ phi
        margins = signs * z
        loss_terms = np.logaddexp(0.0, -margins)
        loss = float(loss_terms.sum() + 0.5 * l2 * (phi @ phi))
        coeff = -signs * sigmoid(-margins)
        grad = delta.T @ coeff + l2 * phi
        return loss, grad

    result = minimize(
        fun=lambda p: objective(p)[0],
        x0=x0,
        jac=lambda p: objective(p)[1],
        method="L-BFGS-B",
        options={"maxiter": 200, "ftol": 1e-10, "gtol": 1e-8},
    )
    phi = result.x.astype(np.float64)
    p = sigmoid(delta @ phi)
    w = p * (1.0 - p)
    precision = delta.T @ (delta * w[:, None]) + l2 * np.eye(N_FEATURES)
    precision_inv = np.linalg.inv(precision)
    return FitResult(phi=phi, precision=precision, precision_inv=precision_inv, success=bool(result.success))


def sample_trajectory_indices(
    rng: np.random.Generator,
    probs: np.ndarray,
    n: int,
) -> np.ndarray:
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


def build_candidate_pool(
    rng: np.random.Generator,
    current_probs: np.ndarray,
    ref_probs: np.ndarray,
    explore_probs: np.ndarray,
    n_pairs: int,
) -> Tuple[np.ndarray, np.ndarray]:
    components = rng.choice(3, size=n_pairs, p=np.array([0.5, 0.25, 0.25]))
    i = np.empty(n_pairs, dtype=np.int64)
    j = np.empty(n_pairs, dtype=np.int64)
    for component, probs in enumerate((current_probs, ref_probs, explore_probs)):
        mask = components == component
        count = int(mask.sum())
        if count:
            pair_i, pair_j = sample_nonself_pairs(rng, probs, count)
            i[mask] = pair_i
            j[mask] = pair_j
    return i, j


def initial_dataset(
    rng: np.random.Generator,
    label_seed: int,
    init_labels: int,
    ref_probs: np.ndarray,
    explore_probs: np.ndarray,
    true_rewards: np.ndarray,
    beta: float,
) -> PairData:
    mix = 0.5 * ref_probs + 0.5 * explore_probs
    i, j = sample_nonself_pairs(rng, mix, init_labels)
    y = deterministic_labels(label_seed, i, j, true_rewards, beta)
    return PairData(i=i, j=j, y=y)


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
) -> np.ndarray:
    A = A0.copy()
    available = np.ones(len(delta), dtype=bool)
    selected: List[int] = []

    for _ in range(min(batch_size, len(delta))):
        q = np.einsum("ij,jk,ik->i", delta, A, delta)
        if mode == "d_optimal":
            gains = np.log1p(w * q)
        else:
            assert v is not None
            m = A @ v
            projected = delta @ m
            gains = (w * projected * projected) / (1.0 + w * q)
        gains = np.where(available, gains, -np.inf)
        chosen = int(np.argmax(gains))
        selected.append(chosen)
        available[chosen] = False

        d = delta[chosen]
        Ad = A @ d
        denom = 1.0 + w[chosen] * float(d @ Ad)
        A = A - (w[chosen] / denom) * np.outer(Ad, Ad)

    return np.array(selected, dtype=np.int64)


def normalized(x: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    norm = float(np.linalg.norm(x))
    if norm > 1e-12:
        return x / norm
    if fallback is not None:
        fallback_norm = float(np.linalg.norm(fallback))
        if fallback_norm > 1e-12:
            return fallback / fallback_norm
    return np.zeros_like(x)


def proxy_direction(
    proxy_grad: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    norm = float(np.linalg.norm(proxy_grad))
    if norm > 1e-12:
        return proxy_grad / norm
    return normalized(rng.normal(size=proxy_grad.shape))


def choose_pairs(
    method: str,
    rng: np.random.Generator,
    eval_info: PolicyEval,
    ref_probs: np.ndarray,
    explore_probs: np.ndarray,
    fit: FitResult,
    features: np.ndarray,
    batch_size: int,
    candidate_pairs: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if method == "on_policy_random":
        return sample_nonself_pairs(rng, eval_info.probs, batch_size)

    pool_i, pool_j = build_candidate_pool(
        rng,
        current_probs=eval_info.probs,
        ref_probs=ref_probs,
        explore_probs=explore_probs,
        n_pairs=candidate_pairs,
    )
    delta, _, w = pair_quantities(pool_i, pool_j, features, fit.phi)
    A = fit.precision_inv
    proxy_grad = eval_info.G @ fit.phi
    g_hat = proxy_direction(proxy_grad, rng)
    v = eval_info.G.T @ g_hat
    oracle_hat = normalized(eval_info.true_grad)
    oracle_v = eval_info.G.T @ oracle_hat

    if method == "pool_uniform":
        selected = rng.choice(len(pool_i), size=min(batch_size, len(pool_i)), replace=False)
    elif method == "entropy":
        selected = topk_indices(w, batch_size)
    elif method == "disagreement":
        variance = np.einsum("ij,jk,ik->i", delta, A, delta)
        selected = topk_indices(variance, batch_size)
    elif method == "d_optimal":
        selected = greedy_rank_one(delta, w, A, batch_size, mode="d_optimal")
    elif method == "perfal_topB":
        m = A @ v
        score = w * np.square(delta @ m)
        selected = topk_indices(score, batch_size)
    elif method == "perfal_greedy":
        selected = greedy_rank_one(delta, w, A, batch_size, mode="perfal", v=v)
    elif method == "perfal_oracle":
        selected = greedy_rank_one(delta, w, A, batch_size, mode="perfal", v=oracle_v)
    elif method == "perfal_random_v":
        random_v = rng.normal(size=N_FEATURES)
        random_v = normalized(random_v)
        random_v *= max(float(np.linalg.norm(v)), 1e-12)
        selected = greedy_rank_one(delta, w, A, batch_size, mode="perfal", v=random_v)
    else:
        raise ValueError(f"Unknown method {method}")

    return pool_i[selected], pool_j[selected]


def centered_rm_errors(
    phi: np.ndarray,
    features: np.ndarray,
    true_rewards: np.ndarray,
    probs: np.ndarray,
) -> Tuple[float, float]:
    pred = features @ phi
    pred_centered = pred - pred.mean()
    true_centered = true_rewards - true_rewards.mean()
    full = float(np.sqrt(np.mean(np.square(pred_centered - true_centered))))

    pred_policy = pred - float(probs @ pred)
    true_policy = true_rewards - float(probs @ true_rewards)
    policy = float(np.sqrt(probs @ np.square(pred_policy - true_policy)))
    return full, policy


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
    posterior_dir_var = float(v @ fit.precision_inv @ v)
    proxy_norm = float(np.linalg.norm(proxy_grad))
    true_norm = float(np.linalg.norm(eval_info.true_grad))
    if proxy_norm > 1e-12 and true_norm > 1e-12:
        cosine = float((proxy_grad @ eval_info.true_grad) / (proxy_norm * true_norm))
    else:
        cosine = 0.0
    directional_bias = float(abs(proxy_hat @ (proxy_grad - eval_info.true_grad)))
    rm_full, rm_policy = centered_rm_errors(
        fit.phi,
        traj["features"],
        traj["true_rewards"],
        eval_info.probs,
    )
    regret = float(traj["optimal_return"] - eval_info.true_return)
    return {
        "method": method,
        "seed": seed,
        "round": round_idx,
        "labels": labels,
        "true_return": eval_info.true_return,
        "regret": regret,
        "gradient_cosine": cosine,
        "directional_gradient_bias": directional_bias,
        "posterior_directional_variance": posterior_dir_var,
        "rm_error_full_centered": rm_full,
        "rm_error_policy_centered": rm_policy,
        "proxy_grad_norm": proxy_norm,
        "true_grad_norm": true_norm,
        "phi_goal": float(fit.phi[0]),
        "phi_trap": float(fit.phi[1]),
        "phi_side": float(fit.phi[2]),
        "phi_step": float(fit.phi[3]),
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
    ref_probs: np.ndarray,
    explore_probs: np.ndarray,
    initial_data: PairData,
) -> List[Dict[str, float | int | str]]:
    rng = np.random.default_rng(seed * 1009 + METHODS.index(method) * 9173 + 44497)
    theta = initial_theta()
    data = PairData(initial_data.i.copy(), initial_data.j.copy(), initial_data.y.copy())
    rows: List[Dict[str, float | int | str]] = []
    phi_start = np.zeros(N_FEATURES, dtype=np.float64)

    for round_idx in range(args.rounds + 1):
        eval_info = evaluate_policy(theta, traj)
        fit = fit_reward_model(data, traj["features"], args.l2, x0=phi_start)
        phi_start = fit.phi
        rows.append(
            metric_row(
                method=method,
                seed=seed,
                round_idx=round_idx,
                labels=len(data.y),
                eval_info=eval_info,
                fit=fit,
                traj=traj,
            )
        )
        rows[-1]["beta"] = args.beta
        rows[-1]["l2"] = args.l2
        if round_idx == args.rounds:
            break

        pair_i, pair_j = choose_pairs(
            method=method,
            rng=rng,
            eval_info=eval_info,
            ref_probs=ref_probs,
            explore_probs=explore_probs,
            fit=fit,
            features=traj["features"],
            batch_size=args.batch_size,
            candidate_pairs=args.candidate_pairs,
        )
        labels = deterministic_labels(seed, pair_i, pair_j, traj["true_rewards"], args.beta)
        data = append_pairs(data, pair_i, pair_j, labels)

        fit_after = fit_reward_model(data, traj["features"], args.l2, x0=fit.phi)
        phi_start = fit_after.phi
        proxy_grad = eval_info.G @ fit_after.phi
        theta = update_policy(theta, proxy_grad, args.policy_lr)

    return rows


def write_metrics_csv(path: Path, rows: Sequence[Dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def group_rows(rows: Sequence[Dict[str, float | int | str]], key: str) -> Dict[str, List[Dict[str, float | int | str]]]:
    groups: Dict[str, List[Dict[str, float | int | str]]] = {}
    for row in rows:
        groups.setdefault(str(row[key]), []).append(row)
    return groups


def mean_stderr(values: Sequence[float]) -> Tuple[float, float]:
    arr = np.array(values, dtype=np.float64)
    mean = float(arr.mean()) if len(arr) else float("nan")
    if len(arr) <= 1:
        return mean, 0.0
    return mean, float(arr.std(ddof=1) / math.sqrt(len(arr)))


def row_setting(row: Dict[str, float | int | str]) -> Tuple[float, float]:
    return float(row.get("beta", 3.0)), float(row.get("l2", 1e-3))


def setting_name(beta: float, l2: float) -> str:
    return f"beta{beta:g}_l2{l2:g}".replace("-", "m").replace(".", "p")


def unique_settings(rows: Sequence[Dict[str, float | int | str]]) -> List[Tuple[float, float]]:
    settings = sorted({row_setting(row) for row in rows})
    return settings


def auc_by_method_seed(rows: Sequence[Dict[str, float | int | str]]) -> Dict[Tuple[float, float, str, int], float]:
    grouped: Dict[Tuple[float, float, str, int], List[Dict[str, float | int | str]]] = {}
    for row in rows:
        beta, l2 = row_setting(row)
        grouped.setdefault((beta, l2, str(row["method"]), int(row["seed"])), []).append(row)
    aucs: Dict[Tuple[float, float, str, int], float] = {}
    for key, group in grouped.items():
        ordered = sorted(group, key=lambda r: int(r["labels"]))
        x = np.array([float(r["labels"]) for r in ordered], dtype=np.float64)
        y = np.array([float(r["true_return"]) for r in ordered], dtype=np.float64)
        span = max(float(x[-1] - x[0]), 1.0)
        aucs[key] = float(np.trapz(y, x) / span)
    return aucs


def aggregate_summary(
    rows: Sequence[Dict[str, float | int | str]],
    final_round: int,
    beta: float,
    l2: float,
) -> List[Dict[str, float | str]]:
    setting_rows = [row for row in rows if row_setting(row) == (beta, l2)]
    aucs = auc_by_method_seed(setting_rows)
    summary = []
    for method in METHODS:
        final = [r for r in setting_rows if r["method"] == method and int(r["round"]) == final_round]
        auc_values = [aucs[(beta, l2, method, int(r["seed"]))] for r in final]
        true_return = mean_stderr([float(r["true_return"]) for r in final])
        regret = mean_stderr([float(r["regret"]) for r in final])
        cosine = mean_stderr([float(r["gradient_cosine"]) for r in final])
        dir_var = mean_stderr([float(r["posterior_directional_variance"]) for r in final])
        auc = mean_stderr(auc_values)
        summary.append(
            {
                "method": method,
                "final_true_return_mean": true_return[0],
                "final_true_return_se": true_return[1],
                "final_regret_mean": regret[0],
                "gradient_cosine_mean": cosine[0],
                "posterior_dir_var_mean": dir_var[0],
                "auc_return_mean": auc[0],
                "auc_return_se": auc[1],
            }
        )
    summary.sort(key=lambda r: float(r["auc_return_mean"]), reverse=True)
    return summary


def plot_metric(
    rows: Sequence[Dict[str, float | int | str]],
    metric: str,
    ylabel: str,
    path: Path,
) -> None:
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
        xs_arr = np.array(xs)
        means_arr = np.array(means)
        ses_arr = np.array(ses)
        plt.plot(xs_arr, means_arr, label=method, linewidth=1.8)
        plt.fill_between(xs_arr, means_arr - ses_arr, means_arr + ses_arr, alpha=0.12)
    plt.xlabel("Preference labels")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def write_summary(
    path: Path,
    rows: Sequence[Dict[str, float | int | str]],
    args: argparse.Namespace,
    config: Dict[str, object],
) -> None:
    settings = unique_settings(rows)

    def method_table(aggregate: Sequence[Dict[str, float | str]]) -> str:
        table_lines = [
            "| Method | Final true return | Final regret | Return AUC | Grad cosine | Dir. variance |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for row in aggregate:
            table_lines.append(
                "| {method} | {ret:.4f} +/- {ret_se:.4f} | {reg:.4f} | {auc:.4f} +/- {auc_se:.4f} | {cos:.4f} | {var:.6f} |".format(
                    method=row["method"],
                    ret=float(row["final_true_return_mean"]),
                    ret_se=float(row["final_true_return_se"]),
                    reg=float(row["final_regret_mean"]),
                    auc=float(row["auc_return_mean"]),
                    auc_se=float(row["auc_return_se"]),
                    cos=float(row["gradient_cosine_mean"]),
                    var=float(row["posterior_dir_var_mean"]),
                )
            )
        return "\n".join(table_lines)

    setting_sections: List[str] = []
    sweep_lines = [
        "| beta | l2 | Best AUC | Best final return | Lowest dir. variance | PerfAL-greedy AUC rank | PerfAL-greedy beats random-v AUC? |",
        "|---:|---:|---|---|---|---:|---|",
    ]
    for beta, l2 in settings:
        aggregate = aggregate_summary(rows, final_round=args.rounds, beta=beta, l2=l2)
        best_auc = aggregate[0]
        best_return = max(aggregate, key=lambda r: float(r["final_true_return_mean"]))
        best_var = min(aggregate, key=lambda r: float(r["posterior_dir_var_mean"]))
        auc_order = [str(row["method"]) for row in aggregate]
        greedy_rank = auc_order.index("perfal_greedy") + 1
        greedy_auc = next(float(row["auc_return_mean"]) for row in aggregate if row["method"] == "perfal_greedy")
        random_auc = next(float(row["auc_return_mean"]) for row in aggregate if row["method"] == "perfal_random_v")
        sweep_lines.append(
            "| {beta:g} | {l2:g} | `{auc}` | `{ret}` | `{var}` | {rank} | {beats} |".format(
                beta=beta,
                l2=l2,
                auc=best_auc["method"],
                ret=best_return["method"],
                var=best_var["method"],
                rank=greedy_rank,
                beats="yes" if greedy_auc > random_auc else "no",
            )
        )
        setting_sections.append(
            "### beta={beta:g}, l2={l2:g}\n\n{table}\n\n"
            "- Best return AUC: `{best_auc}`.\n"
            "- Best final true return: `{best_return}`.\n"
            "- Lowest policy-direction posterior variance: `{best_var}`.\n".format(
                beta=beta,
                l2=l2,
                table=method_table(aggregate),
                best_auc=best_auc["method"],
                best_return=best_return["method"],
                best_var=best_var["method"],
            )
        )

    quick_note = (
        "This is a preliminary quick smoke run, intended to validate the implementation "
        "and produce a first artifact rather than a paper-grade comparison."
        if args.preset == "quick"
        else "This summary was generated from the configured CLI sweep."
    )
    text = f"""# PerfAL Toy Experiment Result Summary

Generated: {datetime.now().isoformat(timespec="seconds")}

{quick_note}

## Configuration

```json
{json.dumps(config, indent=2)}
```

## Aggregate Results

{chr(10).join(sweep_lines)}

## Per-Setting Results

{chr(10).join(setting_sections)}

## Readout

- Non-oracle PerfAL directions use the proxy gradient only; if the proxy gradient is zero, they use a random unit policy direction, not the true gradient.
- All methods within a seed share the same initial dataset and deterministic per-query Bradley-Terry label noise.
- Self-pairs are excluded from initial data, on-policy random queries, and candidate pools.
- The `perfal_random_v` baseline is included as the randomized-direction sanity check; it should be interpreted as a control for whether the PerfAL direction is doing useful work.

## Output Files

- `results/{args.output_name}/metrics.csv`: per-seed, per-round metrics.
- Single-setting runs put plots directly in `results/{args.output_name}/`.
- Multi-setting sweeps put plots under `results/{args.output_name}/beta*_l2*/`.
"""
    path.write_text(text)


def write_config(path: Path, config: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("quick", "custom", "full_sweep"), default="custom")
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--init-labels", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--candidate-pairs", type=int, default=2000)
    parser.add_argument("--beta", type=float, default=3.0)
    parser.add_argument("--betas", nargs="+", type=float, default=None)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--l2s", nargs="+", type=float, default=None)
    parser.add_argument("--policy-lr", type=float, default=0.8)
    args = parser.parse_args()
    if args.preset == "quick":
        args.seeds = [0, 1, 2]
        args.rounds = 10
        args.init_labels = 10
        args.batch_size = 10
        args.candidate_pairs = 2000
        args.beta = 3.0
        args.l2 = 1e-3
        args.policy_lr = 0.8
        if args.output_name is None:
            args.output_name = "quick"
    elif args.preset == "full_sweep":
        args.seeds = list(range(50))
        args.rounds = 30
        args.init_labels = 10
        args.batch_size = 10
        args.candidate_pairs = 10000
        args.betas = [1.0, 3.0, 10.0]
        args.l2s = [1e-4, 1e-3, 1e-2]
        args.policy_lr = 0.8
        if args.output_name is None:
            args.output_name = "full_sweep"
    if args.output_name is None:
        args.output_name = "custom"
    return args


def configured_settings(args: argparse.Namespace) -> List[Tuple[float, float]]:
    betas = args.betas if args.betas is not None else [args.beta]
    l2s = args.l2s if args.l2s is not None else [args.l2]
    return [(float(beta), float(l2)) for beta in betas for l2 in l2s]


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    out_dir = root / "results" / args.output_name
    out_dir.mkdir(parents=True, exist_ok=True)

    traj = enumerate_trajectories()
    ref_eval = evaluate_policy(initial_theta(), traj)
    explore_eval = evaluate_policy(np.zeros((N_STATES, N_ACTIONS), dtype=np.float64), traj)
    rows: List[Dict[str, float | int | str]] = []

    settings = configured_settings(args)
    for beta, l2 in settings:
        setting_args = copy.copy(args)
        setting_args.beta = beta
        setting_args.l2 = l2
        setting_rows: List[Dict[str, float | int | str]] = []
        for seed in args.seeds:
            init_rng = np.random.default_rng(seed * 1009 + 1337)
            initial_data = initial_dataset(
                init_rng,
                label_seed=seed,
                init_labels=args.init_labels,
                ref_probs=ref_eval.probs,
                explore_probs=explore_eval.probs,
                true_rewards=traj["true_rewards"],
                beta=beta,
            )
            for method in METHODS:
                setting_rows.extend(
                    run_one_method(
                        method=method,
                        seed=seed,
                        args=setting_args,
                        traj=traj,
                        ref_probs=ref_eval.probs,
                        explore_probs=explore_eval.probs,
                        initial_data=initial_data,
                    )
                )
        rows.extend(setting_rows)

    config = {
        "preset": args.preset,
        "output_name": args.output_name,
        "seeds": args.seeds,
        "rounds": args.rounds,
        "init_labels": args.init_labels,
        "batch_size": args.batch_size,
        "candidate_pairs": args.candidate_pairs,
        "beta": args.beta,
        "betas": sorted({beta for beta, _ in settings}),
        "l2": args.l2,
        "l2s": sorted({l2 for _, l2 in settings}),
        "policy_lr": args.policy_lr,
        "grid": [GRID, GRID],
        "horizon": HORIZON,
        "actions": list(ACTIONS),
        "true_reward": TRUE_W.tolist(),
        "goal": GOAL,
        "trap": TRAP,
        "side": SIDE,
        "methods": list(METHODS),
    }

    write_metrics_csv(out_dir / "metrics.csv", rows)
    write_config(out_dir / "config.json", config)
    for beta, l2 in settings:
        setting_rows = [row for row in rows if row_setting(row) == (beta, l2)]
        plot_dir = out_dir if len(settings) == 1 else out_dir / setting_name(beta, l2)
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_metric(setting_rows, "true_return", "Exact true return", plot_dir / "true_return.png")
        plot_metric(
            setting_rows,
            "posterior_directional_variance",
            "Posterior directional variance",
            plot_dir / "posterior_directional_variance.png",
        )
        plot_metric(setting_rows, "gradient_cosine", "Gradient cosine", plot_dir / "gradient_cosine.png")
    write_summary(root / "result_summary.md", rows, args, config)
    print(f"Wrote {out_dir / 'metrics.csv'}")
    print(f"Wrote {root / 'result_summary.md'}")


if __name__ == "__main__":
    main()
