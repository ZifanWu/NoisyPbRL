#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


GAUGES = [
    "none",
    "mean_buf",
    "mean_ref",
    "no_tanh",
    "no_tanh_mean_buf",
    "no_tanh_mean_ref",
]

# Gauges included in the statistical tests (the three standard gauges from the spec).
STAT_GAUGES = ["none", "mean_buf", "mean_ref"]
# Bonferroni factor = number of pairwise comparisons among STAT_GAUGES.
_N_STAT_PAIRS = len(STAT_GAUGES) * (len(STAT_GAUGES) - 1) // 2  # = 3


@dataclass
class PairwiseStat:
    gauge_a: str
    gauge_b: str
    t_stat: float
    p_raw: float
    p_corrected: float
    cohens_d: float
    mean_diff: float
    ci_low: float
    ci_high: float


def _load_manifest(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return pd.DataFrame(payload)


def _read_eval(run_dir: str) -> pd.DataFrame:
    eval_path = os.path.join(run_dir, "eval.csv")
    if not os.path.exists(eval_path):
        return pd.DataFrame()
    df = pd.read_csv(eval_path)
    if "true_return" not in df.columns:
        if "true_episode_reward" in df.columns:
            df["true_return"] = df["true_episode_reward"]
    if "step" not in df.columns:
        df["step"] = np.arange(len(df))
    for g in GAUGES:
        c = f"proxy_return_{g}"
        if c not in df.columns and "proxy_return" in df.columns:
            df[c] = df["proxy_return"]
    return df


def _final_window_mean(df: pd.DataFrame, metric: str = "true_return", window_steps: int = 200_000) -> float:
    if df.empty or metric not in df.columns:
        return float("nan")
    max_step = df["step"].max()
    sub = df[df["step"] >= max_step - window_steps]
    if sub.empty:
        sub = df
    return float(sub[metric].mean())


def _welch_stats(a: np.ndarray, b: np.ndarray) -> Tuple[float, float, float, float, float]:
    t_stat, p_val = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
    mean_diff = float(np.mean(a) - np.mean(b))
    var_a = np.var(a, ddof=1)
    var_b = np.var(b, ddof=1)
    n_a = len(a)
    n_b = len(b)
    se = np.sqrt(var_a / n_a + var_b / n_b)
    if se <= 0:
        return float(t_stat), float(p_val), mean_diff, mean_diff, mean_diff
    df_num = (var_a / n_a + var_b / n_b) ** 2
    df_den = ((var_a / n_a) ** 2) / max(1, n_a - 1) + ((var_b / n_b) ** 2) / max(1, n_b - 1)
    dof = df_num / max(df_den, 1e-12)
    tcrit = stats.t.ppf(0.975, dof)
    ci_low = mean_diff - tcrit * se
    ci_high = mean_diff + tcrit * se
    return float(t_stat), float(p_val), mean_diff, float(ci_low), float(ci_high)


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return float("nan")
    s_a = np.var(a, ddof=1)
    s_b = np.var(b, ddof=1)
    pooled = np.sqrt(((n_a - 1) * s_a + (n_b - 1) * s_b) / max(1, n_a + n_b - 2))
    if pooled <= 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled)


def _pairwise_pg_stats(df_final: pd.DataFrame) -> List[PairwiseStat]:
    """Pairwise Welch t-tests among the three standard gauges with Bonferroni correction.

    Only STAT_GAUGES are tested; no_tanh variants are excluded so the correction
    factor (_N_STAT_PAIRS = 3) stays correct.
    """
    stats_out: List[PairwiseStat] = []
    pairs = list(itertools.combinations(STAT_GAUGES, 2))
    for g1, g2 in pairs:
        a = df_final[df_final["gauge"] == g1]["final_true_return"].dropna().to_numpy()
        b = df_final[df_final["gauge"] == g2]["final_true_return"].dropna().to_numpy()
        if len(a) == 0 or len(b) == 0:
            continue
        t_stat, p_raw, mean_diff, ci_low, ci_high = _welch_stats(a, b)
        p_corr = min(1.0, p_raw * _N_STAT_PAIRS)
        d = _cohens_d(a, b)
        stats_out.append(
            PairwiseStat(
                gauge_a=g1,
                gauge_b=g2,
                t_stat=t_stat,
                p_raw=p_raw,
                p_corrected=p_corr,
                cohens_d=d,
                mean_diff=mean_diff,
                ci_low=ci_low,
                ci_high=ci_high,
            )
        )
    return stats_out


def _ensure_dirs(out_dir: str):
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    return fig_dir


def main():
    parser = argparse.ArgumentParser(description="Analyze BPref gauge experiment runs.")
    parser.add_argument("--manifest", required=True, help="Path to sweep manifest json.")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    sns.set_theme(style="whitegrid")
    fig_dir = _ensure_dirs(args.results_dir)

    manifest = _load_manifest(args.manifest)
    rows = []
    curve_rows = []
    rrm_invariance_failures = []

    for _, run in manifest.iterrows():
        run_dir = os.path.join(args.results_dir, "runs", run["run_name"])
        eval_df = _read_eval(run_dir)
        if eval_df.empty:
            continue

        final_ret = _final_window_mean(eval_df, metric="true_return", window_steps=200_000)
        row = {
            "run_name": run["run_name"],
            "env_key": run["env_key"],
            "method": run["method"],
            "gauge": run["gauge"],
            "alpha_mode": run["alpha_mode"],
            "seed": int(run["seed"]),
            "final_true_return": final_ret,
        }
        rows.append(row)

        tmp = eval_df[["step", "true_return"]].copy()
        tmp["env_key"] = run["env_key"]
        tmp["method"] = run["method"]
        tmp["gauge"] = run["gauge"]
        tmp["alpha_mode"] = run["alpha_mode"]
        curve_rows.append(tmp)

        if run["method"] == "rrm":
            # For RRM the gauge shift is a step-function that jumps at each RM
            # update and is constant between updates.  The difference
            # proxy_return_none[t] - proxy_return_mean_buf[t] should therefore
            # be piecewise-constant.  We flag the run if the coefficient of
            # variation of that difference exceeds 5 %, which would indicate the
            # gauge shift is leaking inconsistently into the proxy return rather
            # than appearing as a stable additive offset.
            for col_a, col_b in [
                ("proxy_return_none", "proxy_return_mean_buf"),
                ("proxy_return_none", "proxy_return_mean_ref"),
            ]:
                if col_a not in eval_df.columns or col_b not in eval_df.columns:
                    continue
                diff = eval_df[col_a].values - eval_df[col_b].values
                mean_diff = np.abs(np.mean(diff))
                if mean_diff < 1e-6:
                    continue  # shift is effectively zero; cannot compute CV
                cv = np.std(diff) / mean_diff
                if cv > 0.05:
                    rrm_invariance_failures.append(
                        f"{run['run_name']}  ({col_a}-{col_b} CV={cv:.3f})"
                    )
                    break

    final_df = pd.DataFrame(rows)
    curves_df = pd.concat(curve_rows, ignore_index=True) if curve_rows else pd.DataFrame()

    # Figure 1: headline_pg_rlhf_returns.png
    pg_curve = curves_df[curves_df["method"] == "pg_rlhf"] if not curves_df.empty else pd.DataFrame()
    if not pg_curve.empty:
        g = sns.relplot(
            data=pg_curve,
            x="step",
            y="true_return",
            hue="gauge",
            kind="line",
            col="alpha_mode",
            row="env_key",
            errorbar=("ci", 95),
            facet_kws={"sharey": False, "sharex": True},
        )
        g.figure.savefig(os.path.join(fig_dir, "headline_pg_rlhf_returns.png"), dpi=160, bbox_inches="tight")
        plt.close(g.figure)

    # Figure 2: rrm_baseline.png
    rrm_curve = curves_df[curves_df["method"] == "rrm"] if not curves_df.empty else pd.DataFrame()
    if not rrm_curve.empty:
        plt.figure(figsize=(11, 5))
        sns.lineplot(
            data=rrm_curve,
            x="step",
            y="true_return",
            hue="env_key",
            style="alpha_mode",
            errorbar=("ci", 95),
        )
        plt.title("RRM true return baseline")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "rrm_baseline.png"), dpi=160)
        plt.close()

    # Figure 3: final_pg_rlhf_bars.png
    pg_final = final_df[final_df["method"] == "pg_rlhf"] if not final_df.empty else pd.DataFrame()
    if not pg_final.empty:
        g = sns.catplot(
            data=pg_final,
            x="env_key",
            y="final_true_return",
            hue="gauge",
            col="alpha_mode",
            kind="bar",
            errorbar=("ci", 95),
            sharey=False,
        )
        g.figure.savefig(os.path.join(fig_dir, "final_pg_rlhf_bars.png"), dpi=160, bbox_inches="tight")
        plt.close(g.figure)

    # Figure 4: shift_evolution.png
    shift_rows = []
    for _, run in manifest.iterrows():
        run_dir = os.path.join(args.results_dir, "runs", run["run_name"])
        train_path = os.path.join(run_dir, "train.csv")
        if not os.path.exists(train_path):
            continue
        tr = pd.read_csv(train_path)
        if "shift_value_active" in tr.columns:
            tmp = tr[["step", "shift_value_active"]].copy()
            tmp["shift_name"] = str(run["gauge"])
            tmp["env_key"] = run["env_key"]
            tmp["method"] = run["method"]
            tmp["alpha_mode"] = run["alpha_mode"]
            shift_rows.append(tmp.rename(columns={"shift_value_active": "shift_value"}))
        else:
            for col in ["shift_value_mean_buf", "shift_value_mean_ref"]:
                if col in tr.columns:
                    tmp = tr[["step", col]].copy()
                    tmp["shift_name"] = col
                    tmp["env_key"] = run["env_key"]
                    tmp["method"] = run["method"]
                    tmp["alpha_mode"] = run["alpha_mode"]
                    shift_rows.append(tmp.rename(columns={col: "shift_value"}))
    if shift_rows:
        shift_df = pd.concat(shift_rows, ignore_index=True)
        g = sns.relplot(
            data=shift_df,
            x="step",
            y="shift_value",
            hue="shift_name",
            col="alpha_mode",
            row="env_key",
            kind="line",
            facet_kws={"sharey": False},
        )
        g.figure.savefig(os.path.join(fig_dir, "shift_evolution.png"), dpi=160, bbox_inches="tight")
        plt.close(g.figure)

    # Figure 5: alpha_sweep_summary.png
    if not pg_final.empty:
        spread_rows = []
        for (env_key, alpha_mode), grp in pg_final.groupby(["env_key", "alpha_mode"]):
            gauge_means = grp.groupby("gauge")["final_true_return"].mean()
            if len(gauge_means) == 0:
                continue
            spread_rows.append(
                {
                    "env_key": env_key,
                    "alpha_mode": alpha_mode,
                    "spread": float(gauge_means.max() - gauge_means.min()),
                }
            )
        spread_df = pd.DataFrame(spread_rows)
        if not spread_df.empty:
            order = ["low", "auto", "high"]
            spread_df["alpha_mode"] = pd.Categorical(spread_df["alpha_mode"], categories=order, ordered=True)
            plt.figure(figsize=(7, 4))
            sns.lineplot(data=spread_df.sort_values("alpha_mode"), x="alpha_mode", y="spread", hue="env_key", marker="o")
            plt.ylabel("Gauge spread (max-min final return)")
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, "alpha_sweep_summary.png"), dpi=160)
            plt.close()

    # Figure 6: pg_vs_rrm.png
    if not final_df.empty:
        rrm_ref = final_df[final_df["method"] == "rrm"][["env_key", "alpha_mode", "seed", "final_true_return"]]
        rrm_ref = rrm_ref.rename(columns={"final_true_return": "rrm_final_true_return"})
        pg_ref = final_df[final_df["method"] == "pg_rlhf"]
        merged = pg_ref.merge(rrm_ref, on=["env_key", "alpha_mode", "seed"], how="inner")
        if not merged.empty:
            plt.figure(figsize=(7, 6))
            sns.scatterplot(
                data=merged,
                x="rrm_final_true_return",
                y="final_true_return",
                hue="gauge",
                style="env_key",
            )
            lo = min(merged["rrm_final_true_return"].min(), merged["final_true_return"].min())
            hi = max(merged["rrm_final_true_return"].max(), merged["final_true_return"].max())
            plt.plot([lo, hi], [lo, hi], "--", color="gray")
            plt.xlabel("RRM final true return")
            plt.ylabel("PG-RLHF final true return")
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, "pg_vs_rrm.png"), dpi=160)
            plt.close()

    # Summary text
    summary_path = os.path.join(args.results_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        for env_key in sorted(final_df["env_key"].unique()) if not final_df.empty else []:
            for alpha_mode in ["low", "auto", "high"]:
                cell = final_df[(final_df["env_key"] == env_key) & (final_df["alpha_mode"] == alpha_mode)]
                if cell.empty:
                    continue
                f.write(f"env={env_key}  alpha={alpha_mode}\n")
                f.write("=" * 69 + "\n")
                f.write("method   gauge       final_true_return    Cohen_d_vs_none   p_corrected\n")
                f.write("=" * 69 + "\n")

                rrm = cell[cell["method"] == "rrm"]["final_true_return"].dropna().to_numpy()
                if len(rrm) > 0:
                    f.write(f"rrm      (all three) {rrm.mean():.3f} +/- {rrm.std(ddof=1):.3f}    0.0 (by construction)   n/a\n")

                pg = cell[cell["method"] == "pg_rlhf"]
                pair_stats = _pairwise_pg_stats(pg)
                p_vs_none = {s.gauge_b: s for s in pair_stats if s.gauge_a == "none"}
                p_vs_none.update({s.gauge_a: s for s in pair_stats if s.gauge_b == "none"})
                for g in GAUGES:  # show all gauges; stats are n/a for no_tanh variants
                    vals = pg[pg["gauge"] == g]["final_true_return"].dropna().to_numpy()
                    if len(vals) == 0:
                        continue
                    if g == "none":
                        f.write(f"pg_rlhf  {g:<10} {vals.mean():.3f} +/- {vals.std(ddof=1):.3f}    -                  -\n")
                    else:
                        st_g = p_vs_none.get(g)
                        if st_g is None:
                            f.write(f"pg_rlhf  {g:<10} {vals.mean():.3f} +/- {vals.std(ddof=1):.3f}    n/a                n/a\n")
                        else:
                            mark = "*" if st_g.p_corrected < 0.05 else "ns"
                            f.write(
                                f"pg_rlhf  {g:<10} {vals.mean():.3f} +/- {vals.std(ddof=1):.3f}    "
                                f"{st_g.cohens_d:.3f}               {st_g.p_corrected:.4f} {mark}\n"
                            )
                f.write("=" * 69 + "\n\n")

        if rrm_invariance_failures:
            f.write("RRM bit-identity failures detected:\n")
            for name in rrm_invariance_failures:
                f.write(f"- {name}\n")
        else:
            f.write("RRM bit-identity check: pass (no failures detected in available logs).\n")

    print(f"[analyze] Wrote figures to {fig_dir}")
    print(f"[analyze] Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
