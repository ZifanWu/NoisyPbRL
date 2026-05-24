#!/usr/bin/env python3
"""Analyze gauge-ambiguity experiment results from W&B.

Usage (after all runs are finished):
    python analyze_gauge.py \\
        --project pbrl_gauge_ambiguity \\
        --envs walker_walk cheetah_run \\
        --seeds 1 2 3 4 5 \\
        --output results/

Produces:
  results/track1/  (num_interact=5000, ~11 RM updates — standard regime)
  results/track2/  (num_interact=1000, ~50 RM updates — stress test, if runs exist)
Each track directory contains:
  plot1_true_return.png   Plot 2: rm/gauge_gap over time
  plot2_gauge_gap.png     Plot 3: rm/param_norm over time
  plot3_param_norm.png    Plot 4: eval/return_gap over time
  plot4_return_gap.png
  results.md              Verdict + embedded plots + statistics table
"""

import argparse
import os
import textwrap
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# ── Optional imports (fail gracefully so the file is importable without them) ──
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _MATPLOTLIB = True
except ImportError:
    _MATPLOTLIB = False
    print("[warn] matplotlib not found — plots will be skipped")

try:
    import wandb
    _WANDB = True
except ImportError:
    _WANDB = False
    print("[warn] wandb not found — run `pip install wandb`")


# ── Constants ──────────────────────────────────────────────────────────────────

GAUGE_MODES  = ["l2", "none", "zero_mean_ref"]
GAUGE_COLORS = {"l2": "#1f77b4", "none": "#ff7f0e", "zero_mean_ref": "#2ca02c"}
GAUGE_LABELS = {
    "l2":           "Mode A – L2 (wd=1e-4)",
    "none":         "Mode None (wd=0, no proj)",
    "zero_mean_ref":"Mode B – Zero-mean ref",
}

# Metrics loaded from W&B history
_METRICS = [
    "eval/true_episode_reward",
    "eval/proxy_return",
    "eval/return_gap",
    "rm/gauge_gap",
    "rm/param_norm",
    "rm/mean_on_ref",
    "rm/mean_on_policy",
    "_step",
]


# ── Data loading ───────────────────────────────────────────────────────────────

def load_runs(project: str, envs: List[str], seeds: List[int]) -> pd.DataFrame:
    """Download all finished runs from W&B, return a flat DataFrame.

    Args:
        project: W&B project name.
        envs:    environment filter (empty list = accept all).
        seeds:   seed filter (empty list = accept all).

    Returns:
        DataFrame with columns: env, gauge_mode, seed, num_interact, + metric columns.
        Each row corresponds to one logged step from one run.
    """
    if not _WANDB:
        raise RuntimeError("wandb is required: pip install wandb")

    api   = wandb.Api(timeout=60)
    runs  = api.runs(project)
    parts: List[pd.DataFrame] = []

    for run in runs:
        if run.state not in ("finished", "running"):
            continue
        cfg          = run.config
        env          = cfg.get("env", "unknown")
        gauge_mode   = cfg.get("gauge_mode", "unknown")
        seed         = int(cfg.get("seed", -1))
        num_interact = int(cfg.get("num_interact", 5000))

        if envs and env not in envs:
            continue
        if seeds and seed not in seeds:
            continue

        history = run.history(samples=2000, keys=_METRICS, x_axis="_step")
        if history.empty:
            continue

        history["env"]          = env
        history["gauge_mode"]   = gauge_mode
        history["seed"]         = seed
        history["num_interact"] = num_interact
        parts.append(history)

    if not parts:
        print("[warn] No matching runs found in project:", project)
        return pd.DataFrame()

    df = pd.concat(parts, ignore_index=True)
    df = df.sort_values(["env", "gauge_mode", "seed", "_step"]).reset_index(drop=True)
    return df


# ── Statistics helpers ─────────────────────────────────────────────────────────

def _bootstrap_ci(
    values: np.ndarray,
    n_boot: int = 2000,
    ci: float = 0.95,
) -> Tuple[float, float, float]:
    """Return (mean, lower_ci, upper_ci) using percentile bootstrap."""
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    boot_means = np.array(
        [np.random.choice(values, size=len(values), replace=True).mean()
         for _ in range(n_boot)]
    )
    alpha = (1.0 - ci) / 2.0
    return (
        float(values.mean()),
        float(np.percentile(boot_means, 100 * alpha)),
        float(np.percentile(boot_means, 100 * (1 - alpha))),
    )


def _step_bin(df: pd.DataFrame, step_col: str = "_step", bin_size: int = 10_000) -> pd.DataFrame:
    """Round steps to nearest bin_size so runs align for aggregation."""
    df = df.copy()
    df["step_bin"] = (df[step_col] / bin_size).round().astype(int) * bin_size
    return df


def compute_band_stats(
    df: pd.DataFrame,
    metric: str,
    group_keys: List[str],
    bin_size: int = 10_000,
) -> pd.DataFrame:
    """For each (group × step_bin), compute mean and 95% bootstrap CI over seeds.

    Returns a DataFrame with columns: *group_keys, step_bin, mean, lo, hi.
    """
    binned = _step_bin(df, bin_size=bin_size)
    records = []
    for group_vals, grp in binned.groupby(group_keys + ["step_bin"]):
        vals = grp[metric].dropna().values
        if len(vals) == 0:
            continue
        mean, lo, hi = _bootstrap_ci(vals)
        row = dict(zip(group_keys + ["step_bin"], group_vals
                       if isinstance(group_vals, tuple) else (group_vals,)))
        row.update(mean=mean, lo=lo, hi=hi)
        records.append(row)
    return pd.DataFrame(records)


def _auc(steps: np.ndarray, values: np.ndarray) -> float:
    """Trapezoidal AUC, normalised by the step range."""
    order  = np.argsort(steps)
    xs, ys = steps[order], values[order]
    if len(xs) < 2:
        return float("nan")
    raw = float(np.trapz(ys, xs))
    span = xs[-1] - xs[0]
    return raw / span if span > 0 else float("nan")


def run_statistical_tests(
    df: pd.DataFrame,
    envs: List[str],
    pairs: List[Tuple[str, str]],
) -> pd.DataFrame:
    """Paired t-test (final step + AUC) for each gauge_mode pair.

    Pairing is by (env, seed) — each seed across both envs contributes one pair.

    Args:
        df:    full run DataFrame.
        envs:  list of environments used.
        pairs: gauge_mode pairs to test, e.g. [("l2","zero_mean_ref")].

    Returns:
        DataFrame with columns: pair, test, p_value, effect_size_d, significant.
    """
    records = []
    binned  = _step_bin(df)

    for g_a, g_b in pairs:
        for test_name in ("final_step", "auc"):
            scores_a, scores_b = [], []

            for env in envs:
                for seed in df["seed"].unique():
                    sub_a = df[(df["env"] == env) & (df["gauge_mode"] == g_a)
                               & (df["seed"] == seed)]
                    sub_b = df[(df["env"] == env) & (df["gauge_mode"] == g_b)
                               & (df["seed"] == seed)]
                    if sub_a.empty or sub_b.empty:
                        continue

                    if test_name == "final_step":
                        val_a = sub_a["eval/true_episode_reward"].dropna().iloc[-1] \
                                if not sub_a["eval/true_episode_reward"].dropna().empty else np.nan
                        val_b = sub_b["eval/true_episode_reward"].dropna().iloc[-1] \
                                if not sub_b["eval/true_episode_reward"].dropna().empty else np.nan
                    else:
                        col = "eval/true_episode_reward"
                        val_a = _auc(sub_a["_step"].values, sub_a[col].fillna(0).values)
                        val_b = _auc(sub_b["_step"].values, sub_b[col].fillna(0).values)

                    if np.isfinite(val_a) and np.isfinite(val_b):
                        scores_a.append(val_a)
                        scores_b.append(val_b)

            if len(scores_a) < 2:
                continue

            a_arr, b_arr = np.array(scores_a), np.array(scores_b)
            _, p_val     = stats.ttest_rel(a_arr, b_arr)
            diff         = a_arr - b_arr
            d            = diff.mean() / (diff.std(ddof=1) + 1e-12)

            records.append(dict(
                pair=f"{g_a} vs {g_b}",
                test=test_name,
                n_pairs=len(scores_a),
                mean_a=float(a_arr.mean()),
                mean_b=float(b_arr.mean()),
                mean_diff=float(diff.mean()),
                p_value=float(p_val),
                effect_size_d=float(d),
                significant=bool(p_val < 0.05),
            ))

    return pd.DataFrame(records)


# ── Plotting ───────────────────────────────────────────────────────────────────

def _plot_metric(
    df: pd.DataFrame,
    metric: str,
    envs: List[str],
    output_path: str,
    ylabel: str,
    title: str,
    hline: Optional[float] = None,
) -> None:
    """Generic band-plot for one metric, one subplot per env."""
    if not _MATPLOTLIB:
        return

    stats_df = compute_band_stats(df, metric, ["env", "gauge_mode"])
    if stats_df.empty:
        print(f"[warn] No data for metric {metric!r} — skipping plot")
        return

    n_envs = len(envs)
    fig, axes = plt.subplots(1, n_envs, figsize=(6 * n_envs, 4), squeeze=False)

    for col, env in enumerate(envs):
        ax = axes[0][col]
        env_df = stats_df[stats_df["env"] == env]
        for gm in GAUGE_MODES:
            gdf = env_df[env_df["gauge_mode"] == gm].sort_values("step_bin")
            if gdf.empty:
                continue
            xs = gdf["step_bin"].values
            ax.plot(xs, gdf["mean"].values,
                    color=GAUGE_COLORS[gm], label=GAUGE_LABELS[gm])
            ax.fill_between(xs, gdf["lo"].values, gdf["hi"].values,
                             color=GAUGE_COLORS[gm], alpha=0.2)
        if hline is not None:
            ax.axhline(hline, color="k", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_title(f"{env}")
        ax.set_xlabel("Environment steps")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  saved: {output_path}")


def plot_true_return(df: pd.DataFrame, envs: List[str], output_dir: str) -> str:
    path = os.path.join(output_dir, "plot1_true_return.png")
    _plot_metric(df, "eval/true_episode_reward", envs, path,
                 ylabel="True episode return",
                 title="Plot 1 — True environment return (primary metric)")
    return path


def plot_gauge_gap(df: pd.DataFrame, envs: List[str], output_dir: str) -> str:
    path = os.path.join(output_dir, "plot2_gauge_gap.png")
    _plot_metric(df, "rm/gauge_gap", envs, path,
                 ylabel="rm/gauge_gap  (E[r|π_curr] – E[r|π_ref])",
                 title="Plot 2 — Gauge gap over training (key diagnostic)\n"
                       "Mode B stays near zero by construction; Mode A can drift",
                 hline=0.0)
    return path


def plot_param_norm(df: pd.DataFrame, envs: List[str], output_dir: str) -> str:
    path = os.path.join(output_dir, "plot3_param_norm.png")
    _plot_metric(df, "rm/param_norm", envs, path,
                 ylabel="RM parameter ℓ2-norm (all members)",
                 title="Plot 3 — RM parameter norm\n"
                       "Mode A (L2) should stay bounded; Mode B may grow")
    return path


def plot_return_gap(df: pd.DataFrame, envs: List[str], output_dir: str) -> str:
    path = os.path.join(output_dir, "plot4_return_gap.png")
    _plot_metric(df, "eval/return_gap", envs, path,
                 ylabel="eval/return_gap  (proxy – true return)",
                 title="Plot 4 — Goodhart indicator (proxy – true return)\n"
                       "NOTE: proxy and true rewards are on different scales",
                 hline=0.0)
    return path


# ── results.md writer ─────────────────────────────────────────────────────────

def write_results_md(
    output_dir: str,
    track_label: str,
    envs: List[str],
    n_seeds: int,
    df: pd.DataFrame,
    stats_df: pd.DataFrame,
    plot_paths: Dict[str, str],
) -> None:
    """Write results.md with description, embedded plots, and stats table."""

    n_rm_updates = "~11" if "5000" in track_label else "~50"

    # Determine verdict from stats
    verdict_lines = []
    for _, row in stats_df.iterrows():
        sig = "significant (p={:.3f})".format(row["p_value"]) \
              if row["significant"] else "not significant (p={:.3f})".format(row["p_value"])
        verdict_lines.append(
            f"- **{row['pair']}** ({row['test']}): {sig}, "
            f"Cohen's d={row['effect_size_d']:.3f}, "
            f"mean diff={row['mean_diff']:.3f}"
        )

    if stats_df.empty:
        verdict = ("Insufficient data for statistical testing. "
                   "Collect all runs and rerun this script.")
    elif stats_df["significant"].any():
        verdict = (
            "**The hypothesis is SUPPORTED** by the data. "
            "At least one gauge comparison yields a statistically significant "
            "difference in true environment return (p < 0.05). "
            "See the statistics table below for details."
        )
    else:
        verdict = (
            "**The hypothesis is NOT SUPPORTED** by the data. "
            "No statistically significant difference in true environment return "
            "was found between any gauge modes (all p > 0.05). "
            "The gauge effect is theoretically real but appears practically "
            "negligible in this experimental regime."
        )

    def rel_path(p: str) -> str:
        return os.path.relpath(p, output_dir)

    stats_table = stats_df.to_markdown(index=False, floatfmt=".4f") \
                  if not stats_df.empty else "_No runs loaded._"

    md = textwrap.dedent(f"""\
    # Gauge-Ambiguity Experiment Results — {track_label}

    ## What was run

    PEBBLE was trained on {len(envs)} DMControl environments ({', '.join(envs)})
    across 3 gauge modes (`l2`, `none`, `zero_mean_ref`) with {n_seeds} random seeds
    each, for a total of {len(envs) * 3 * n_seeds} runs per track.
    The RM is retrained **{n_rm_updates}** times across 1 M environment steps in this
    regime (feedback budget `max_feedback=1400`, query interval `num_interact`
    = {track_label.split('=')[-1] if '=' in track_label else '5000'}).

    Metrics logged at every RM training event (≈ every `num_interact` env steps):
    `rm/gauge_gap`, `rm/param_norm`, `rm/mean_on_ref`, `rm/mean_on_policy`.
    Eval metrics logged every 10 000 steps: `eval/true_episode_reward`,
    `eval/proxy_return`, `eval/return_gap`.

    > **Note on proxy vs true return scale**: `eval/proxy_return` is the sum of
    > RM-predicted per-step rewards.  The RM is not normalised relative to the
    > environment reward, so proxy and true returns are on **different scales** and
    > cannot be compared in magnitude.  `eval/return_gap` = proxy – true is shown
    > as a Goodhart-drift diagnostic only.

    ## Verdict

    {verdict}

    ---

    ## Plot 1 — True environment return (primary metric)

    ![True return]({rel_path(plot_paths.get('plot1', ''))})

    ## Plot 2 — Gauge gap over training

    ![Gauge gap]({rel_path(plot_paths.get('plot2', ''))})

    `rm/gauge_gap = E_{{π_curr}}[r] − E_{{π_ref}}[r]`.  Mode B keeps this near zero
    by construction; Mode A can drift freely.

    ## Plot 3 — RM parameter norm

    ![Param norm]({rel_path(plot_paths.get('plot3', ''))})

    Mode A (L2 weight decay) should remain bounded; Mode B may grow since it has no
    L2 penalty.

    ## Plot 4 — Goodhart indicator (proxy − true return)

    ![Return gap]({rel_path(plot_paths.get('plot4', ''))})

    ## Statistical tests

    Paired t-test on `eval/true_episode_reward`, pairing by (env, seed) across both
    environments.  Two tests per pair: final-step value and AUC over training.

    {stats_table}

    ### Per-comparison details

    {chr(10).join(verdict_lines) if verdict_lines else '_No pairs tested._'}

    ---
    *Generated by `analyze_gauge.py`.*
    """)

    out_path = os.path.join(output_dir, "results.md")
    with open(out_path, "w") as f:
        f.write(md)
    print(f"  saved: {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def _process_track(
    df: pd.DataFrame,
    track_label: str,
    envs: List[str],
    seeds: List[int],
    output_dir: str,
    test_pairs: List[Tuple[str, str]],
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n=== {track_label} ({len(df)} logged steps from {df.groupby(['env','gauge_mode','seed']).ngroups} run-slices) ===")

    plot_paths: Dict[str, str] = {}
    plot_paths["plot1"] = plot_true_return(df, envs, output_dir)
    plot_paths["plot2"] = plot_gauge_gap(df, envs, output_dir)
    plot_paths["plot3"] = plot_param_norm(df, envs, output_dir)
    plot_paths["plot4"] = plot_return_gap(df, envs, output_dir)

    stats_df = run_statistical_tests(df, envs, test_pairs)
    if not stats_df.empty:
        print(stats_df[["pair", "test", "p_value", "effect_size_d", "significant"]].to_string())

    write_results_md(
        output_dir=output_dir,
        track_label=track_label,
        envs=envs,
        n_seeds=len(seeds) if seeds else df["seed"].nunique(),
        df=df,
        stats_df=stats_df,
        plot_paths=plot_paths,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze gauge-ambiguity experiment results from W&B."
    )
    parser.add_argument("--project",    default="pbrl_gauge_ambiguity",
                        help="W&B project name")
    parser.add_argument("--envs",       nargs="+",
                        default=["walker_walk", "cheetah_run"],
                        help="Environments to include")
    parser.add_argument("--seeds",      nargs="*", type=int, default=[],
                        help="Seeds to include (empty = all seeds)")
    parser.add_argument("--output-dir", default="results/",
                        help="Root directory for output files")
    parser.add_argument("--track",      choices=["1", "2", "both"], default="both",
                        help="Which track(s) to analyse: "
                             "1=standard (num_interact=5000), "
                             "2=stress (num_interact=1000), both=both")
    args = parser.parse_args()

    np.random.seed(42)  # reproducible bootstrap

    print(f"Loading runs from W&B project: {args.project!r}")
    df = load_runs(args.project, args.envs, args.seeds)
    if df.empty:
        print("No data loaded. Exiting.")
        return

    test_pairs: List[Tuple[str, str]] = [
        ("l2",   "zero_mean_ref"),
        ("none", "zero_mean_ref"),
        ("l2",   "none"),
    ]

    os.makedirs(args.output_dir, exist_ok=True)

    # Track 1: standard regime (num_interact=5000, ~11 RM updates)
    if args.track in ("1", "both"):
        t1 = df[df["num_interact"] == 5000]
        if not t1.empty:
            _process_track(t1, "Track 1 — standard (num_interact=5000)",
                           args.envs, args.seeds,
                           os.path.join(args.output_dir, "track1"),
                           test_pairs)
        else:
            print("[Track 1] No runs found with num_interact=5000.")

    # Track 2: stress test (num_interact=1000, ~50 RM updates)
    if args.track in ("2", "both"):
        t2 = df[df["num_interact"] == 1000]
        if not t2.empty:
            _process_track(t2, "Track 2 — stress test (num_interact=1000)",
                           args.envs, args.seeds,
                           os.path.join(args.output_dir, "track2"),
                           test_pairs)
        else:
            print("[Track 2] No runs found with num_interact=1000 (stress test). "
                  "Launch with num_interact=1000 if you want this track.")

    print(f"\nAll outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
