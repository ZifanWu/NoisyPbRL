#!/usr/bin/env python3
import argparse
import itertools
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import yaml

try:
    import seaborn as sns
except ImportError:
    sns = None


GAUGES = ("none", "mean_buf", "mean_ref", "no_tanh", "no_tanh_mean_buf", "no_tanh_mean_ref")
GAUGE_FAMILIES = {
    "tanh": ("none", "mean_buf", "mean_ref"),
    "identity": ("no_tanh", "no_tanh_mean_buf", "no_tanh_mean_ref"),
}
FAMILY_BASE_GAUGE = {"tanh": "none", "identity": "no_tanh"}


def find_eval_csvs(root):
    for dirpath, _dirnames, filenames in os.walk(root):
        if "eval.csv" in filenames:
            yield os.path.join(dirpath, "eval.csv")


def read_hydra_config(run_dir):
    path = os.path.join(run_dir, ".hydra", "config.yaml")
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def metadata_from_path(path):
    run_dir = os.path.dirname(path)
    cfg = read_hydra_config(run_dir)
    gauge_cfg = cfg.get("gauge", {}) if isinstance(cfg, dict) else {}
    meta = {
        "run_dir": run_dir,
        "env": cfg.get("env", "unknown") if isinstance(cfg, dict) else "unknown",
        "seed": cfg.get("seed", np.nan) if isinstance(cfg, dict) else np.nan,
        "method": gauge_cfg.get("method", "unknown"),
        "alpha_mode": gauge_cfg.get("alpha_mode", "unknown"),
        "active_gauge": gauge_cfg.get("active", "unknown"),
    }
    if meta["active_gauge"] == "unknown":
        text = run_dir
        for gauge in GAUGES:
            if re.search(r"(^|[/_]){}($|[/_])".format(re.escape(gauge)), text):
                meta["active_gauge"] = gauge
                break
    meta["final_activation"] = "identity" if str(meta["active_gauge"]).startswith("no_tanh") else "tanh"
    return meta


def load_eval_frames(root):
    frames = []
    for path in find_eval_csvs(root):
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        meta = metadata_from_path(path)
        for key, value in meta.items():
            frame[key] = value
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def final_window_means(eval_df, step_window):
    if eval_df.empty:
        return pd.DataFrame()
    rows = []
    group_cols = ["run_dir", "env", "seed", "method", "alpha_mode", "active_gauge", "final_activation"]
    carried_prefixes = ("proxy_return_", "proxy_gap_", "gauge_shift_")
    for keys, group in eval_df.groupby(group_cols, dropna=False):
        if "step" in group.columns:
            max_step = group["step"].max()
            group = group[group["step"] >= max_step - step_window]
        value_col = "true_episode_reward"
        if value_col not in group.columns:
            value_col = "eval/true_episode_reward" if "eval/true_episode_reward" in group.columns else None
        if value_col is None:
            continue
        row = dict(zip(group_cols, keys))
        row["mean_true_return"] = group[value_col].mean()
        row["max_step"] = group["step"].max() if "step" in group.columns else np.nan
        for col in group.columns:
            if col == "episode_length" or col.startswith(carried_prefixes):
                row[col] = pd.to_numeric(group[col], errors="coerce").mean()
        rows.append(row)
    return pd.DataFrame(rows)


def cohens_d(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    if pooled == 0:
        return 0.0
    return (a.mean() - b.mean()) / pooled


def pairwise_stats(final_df):
    rows = []
    if final_df.empty:
        return pd.DataFrame()
    for (env, alpha_mode, method), group in final_df.groupby(["env", "alpha_mode", "method"], dropna=False):
        gauges = sorted(group["active_gauge"].dropna().unique())
        pairs = list(itertools.combinations(gauges, 2))
        n_comparisons = max(len(pairs), 1)
        for g1, g2 in pairs:
            a = group[group["active_gauge"] == g1]["mean_true_return"].dropna().to_numpy()
            b = group[group["active_gauge"] == g2]["mean_true_return"].dropna().to_numpy()
            if len(a) == 0 or len(b) == 0:
                continue
            t_stat, p_val = st.ttest_ind(a, b, equal_var=False)
            diff = a.mean() - b.mean()
            se = np.sqrt(a.var(ddof=1) / max(len(a), 1) + b.var(ddof=1) / max(len(b), 1)) if len(a) > 1 and len(b) > 1 else np.nan
            ci_low, ci_high = (np.nan, np.nan)
            if np.isfinite(se) and se > 0:
                df_num = (a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)) ** 2
                df_den = ((a.var(ddof=1) / len(a)) ** 2 / (len(a) - 1)) + ((b.var(ddof=1) / len(b)) ** 2 / (len(b) - 1))
                dof = df_num / df_den if df_den > 0 else len(a) + len(b) - 2
                delta = st.t.ppf(0.975, dof) * se
                ci_low, ci_high = diff - delta, diff + delta
            rows.append(
                {
                    "env": env,
                    "alpha_mode": alpha_mode,
                    "method": method,
                    "gauge_a": g1,
                    "gauge_b": g2,
                    "n_a": len(a),
                    "n_b": len(b),
                    "mean_a": a.mean(),
                    "mean_b": b.mean(),
                    "diff_a_minus_b": diff,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "t_stat": t_stat,
                    "p_value": p_val,
                    "n_comparisons": n_comparisons,
                    "cohens_d": cohens_d(a, b),
                }
            )
    stats_df = pd.DataFrame(rows)
    if stats_df.empty:
        return stats_df
    stats_df["p_bonferroni"] = np.minimum(stats_df["p_value"] * stats_df["n_comparisons"], 1.0)
    return stats_df


def rrm_proxy_summary(final_df):
    """Summarize RRM proxy reward offsets within each final-activation family."""
    if final_df.empty:
        return pd.DataFrame()
    rows = []
    rrm_df = final_df[final_df["method"] == "rrm"]
    for _idx, row in rrm_df.iterrows():
        family = row.get("final_activation", "identity" if str(row.get("active_gauge", "")).startswith("no_tanh") else "tanh")
        gauges = GAUGE_FAMILIES.get(family, ())
        base_gauge = FAMILY_BASE_GAUGE.get(family)
        base_col = "proxy_return_{}".format(base_gauge)
        if not base_gauge or base_col not in final_df.columns:
            continue
        base_proxy = row.get(base_col, np.nan)
        episode_length = row.get("episode_length", np.nan)
        for gauge in gauges:
            proxy_col = "proxy_return_{}".format(gauge)
            shift_col = "gauge_shift_{}".format(gauge)
            if proxy_col not in final_df.columns:
                continue
            proxy_return = row.get(proxy_col, np.nan)
            shift = row.get(shift_col, np.nan) if shift_col in final_df.columns else np.nan
            actual_delta = base_proxy - proxy_return
            expected_delta = shift * episode_length if np.isfinite(shift) and np.isfinite(episode_length) else np.nan
            rows.append(
                {
                    "run_dir": row.get("run_dir", ""),
                    "env": row.get("env", "unknown"),
                    "seed": row.get("seed", np.nan),
                    "alpha_mode": row.get("alpha_mode", "unknown"),
                    "final_activation": family,
                    "base_gauge": base_gauge,
                    "proxy_gauge": gauge,
                    "base_proxy_return": base_proxy,
                    "proxy_return": proxy_return,
                    "actual_base_minus_proxy": actual_delta,
                    "gauge_shift": shift,
                    "episode_length": episode_length,
                    "expected_base_minus_proxy": expected_delta,
                    "offset_error": actual_delta - expected_delta if np.isfinite(expected_delta) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def write_plots(eval_df, final_df, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    if sns is not None:
        sns.set_theme(style="whitegrid")
    if not eval_df.empty and "step" in eval_df.columns and "true_episode_reward" in eval_df.columns:
        plt.figure(figsize=(12, 6))
        if sns is not None:
            sns.lineplot(
                data=eval_df,
                x="step",
                y="true_episode_reward",
                hue="active_gauge",
                style="alpha_mode",
                errorbar="se",
            )
        else:
            for (gauge, alpha), group in eval_df.groupby(["active_gauge", "alpha_mode"], dropna=False):
                curve = group.groupby("step")["true_episode_reward"].mean()
                plt.plot(curve.index, curve.values, label="{} / {}".format(gauge, alpha))
            plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "true_return_curves.png"), dpi=200)
        plt.close()

    if not final_df.empty:
        plt.figure(figsize=(12, 6))
        if sns is not None:
            sns.boxplot(data=final_df, x="active_gauge", y="mean_true_return", hue="alpha_mode")
        else:
            groups = [g["mean_true_return"].dropna().to_numpy() for _, g in final_df.groupby("active_gauge")]
            labels = [str(k) for k, _ in final_df.groupby("active_gauge")]
            if groups:
                plt.boxplot(groups, labels=labels)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "final_true_return_box.png"), dpi=200)
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze NoisyPbRL gauge runs.")
    parser.add_argument("--root", default="exp")
    parser.add_argument("--out", default="results/gauge_analysis")
    parser.add_argument("--step-window", type=int, default=200000)
    args = parser.parse_args()

    eval_df = load_eval_frames(args.root)
    os.makedirs(args.out, exist_ok=True)
    eval_df.to_csv(os.path.join(args.out, "eval_merged.csv"), index=False)

    final_df = final_window_means(eval_df, args.step_window)
    final_df.to_csv(os.path.join(args.out, "final_window_means.csv"), index=False)

    stats_df = pairwise_stats(final_df)
    stats_df.to_csv(os.path.join(args.out, "welch_pairwise.csv"), index=False)

    proxy_df = rrm_proxy_summary(final_df)
    proxy_df.to_csv(os.path.join(args.out, "rrm_proxy_summary.csv"), index=False)

    write_plots(eval_df, final_df, args.out)
    print("runs: {}".format(final_df["run_dir"].nunique() if not final_df.empty else 0))
    print("wrote {}".format(args.out))


if __name__ == "__main__":
    main()
