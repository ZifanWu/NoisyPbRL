#!/usr/bin/env python3
"""Analyze performativity of gauge ambiguity: sigma_iter / sigma_oneshot ratio.

Reads local CSV files produced by train_PEBBLE.py (iterative) and
train_PEBBLE_oneshot.py (one-shot control).  Produces:

  results/
    plot_A_bar.png           Bar chart: sigma per condition per env
    plot_B_bootstrap.png     Bootstrap distribution of sigma_iter/sigma_oneshot
    plot_C_trajectories.png  True-return curves overlaid (iterative vs one-shot)
    plot_D_gauge_gap.png     rm/gauge_gap over training steps
    performativity_summary.md  Verdict + statistics table

Usage::

    python analyze_performativity.py \\
        --exp_dir /your/exp \\
        --envs walker_walk cheetah_run \\
        --seeds 1 2 3 4 5 \\
        --output results/

The script discovers runs by walking exp_dir; the env/gauge_mode/seed/condition
are parsed from the directory structure written by Hydra.
"""

import argparse
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _MPL = True
except ImportError:
    _MPL = False
    print("[warn] matplotlib not found — plots will be skipped")

# ── Constants ──────────────────────────────────────────────────────────────────

GAUGE_MODES  = ["none", "l2", "zero_mean_ref"]
GAUGE_COLORS = {"none": "#ff7f0e", "l2": "#1f77b4", "zero_mean_ref": "#2ca02c"}
GAUGE_LABELS = {
    "none":          "none (wd=0)",
    "l2":            "l2 (wd=1e-4)",
    "zero_mean_ref": "zero_mean_ref",
}

N_BOOTSTRAP = 5000
RATIO_AMPLIFICATION_THRESHOLD = 1.5
RATIO_NO_EFFECT_THRESHOLD     = 1.2


# ── Path parser ────────────────────────────────────────────────────────────────

_GAUGE_RE  = re.compile(r'/gauge_([^/]+)/')
_SEED_RE   = re.compile(r'_seed(\d+)(?:/|$)')
_ENV_PARTS = re.compile(r'eval\.csv$')


def _parse_run_path(csv_path: str) -> Optional[Dict]:
    """Extract (env, gauge_mode, seed, is_oneshot) from an eval.csv path.

    Expected path fragment:
      {exp_dir}/{env}/H.../gauge_{gauge_mode}/{PEBBLE[_oneshot]}_..._seed{seed}/eval.csv
    """
    gm_match   = _GAUGE_RE.search(csv_path)
    seed_match = _SEED_RE.search(csv_path)
    if gm_match is None or seed_match is None:
        return None

    gauge_mode = gm_match.group(1)
    seed       = int(seed_match.group(1))
    is_oneshot = 'PEBBLE_oneshot' in csv_path or 'oneshot' in os.path.basename(
        os.path.dirname(csv_path)).lower()

    # env is the path component right after exp_dir (second non-empty component
    # of the relative path from exp_dir)
    parts = [p for p in csv_path.split(os.sep) if p]
    # find 'gauge_{gm}' component to anchor backwards
    gauge_idx = next(
        (i for i, p in enumerate(parts) if p.startswith('gauge_')), None)
    if gauge_idx is None or gauge_idx < 2:
        return None
    env = parts[gauge_idx - 1].split('/')[0]
    # env is usually two levels above the gauge component:
    # {exp_dir}/{env}/H...  so we want the token *right after* exp_dir
    # Fall back: use the component immediately after an "H" or "L" prefixed one
    # Just use the Hydra-written structure: env is gauge_idx - 2 or similar.
    # More robust: find the component that doesn't start with H/L/teacher/label/schedule/tandem/gauge
    for i in range(len(parts) - 1, -1, -1):
        p = parts[i]
        if p.startswith('gauge_'):
            # env is several steps back; walk until we find a non-config component
            for j in range(i - 1, -1, -1):
                candidate = parts[j]
                if not any(candidate.startswith(pref) for pref in (
                        'H', 'L', 'teacher', 'label', 'schedule', 'tandem',
                        'gauge', 'PEBBLE')):
                    env = candidate
                    break
            break

    return dict(env=env, gauge_mode=gauge_mode, seed=seed, is_oneshot=is_oneshot)


# ── CSV discovery and loading ──────────────────────────────────────────────────

def discover_runs(exp_dir: str,
                  envs: List[str],
                  seeds: List[int]) -> pd.DataFrame:
    """Walk exp_dir, load eval.csv files, return a flat DataFrame.

    Columns: env, gauge_mode, seed, condition (iterative|oneshot), step,
             true_episode_reward, (optional) proxy_return, return_gap,
             rm_gauge_gap, rm_param_norm.
    """
    rows: List[dict] = []

    for root, _dirs, files in os.walk(exp_dir):
        if 'eval.csv' not in files:
            continue
        csv_path = os.path.join(root, 'eval.csv')
        meta = _parse_run_path(csv_path)
        if meta is None:
            continue
        if envs and meta['env'] not in envs:
            continue
        if seeds and meta['seed'] not in seeds:
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[warn] could not read {csv_path}: {e}")
            continue

        if 'true_episode_reward' not in df.columns:
            print(f"[warn] missing true_episode_reward in {csv_path}")
            continue

        df['env']       = meta['env']
        df['gauge_mode'] = meta['gauge_mode']
        df['seed']       = meta['seed']
        df['condition']  = 'oneshot' if meta['is_oneshot'] else 'iterative'
        rows.append(df)

    if not rows:
        print(f"[error] no eval.csv files found under {exp_dir}")
        sys.exit(1)

    data = pd.concat(rows, ignore_index=True)

    # Also pull in gauge_gap from train.csv when available
    data = _merge_train_metrics(data, exp_dir, envs, seeds)
    return data


def _merge_train_metrics(data: pd.DataFrame,
                          exp_dir: str,
                          envs: List[str],
                          seeds: List[int]) -> pd.DataFrame:
    """Best-effort merge of rm_gauge_gap / rm_param_norm from train.csv."""
    train_rows: List[dict] = []
    for root, _dirs, files in os.walk(exp_dir):
        if 'train.csv' not in files:
            continue
        csv_path = os.path.join(root, 'train.csv')
        meta = _parse_run_path(csv_path)
        if meta is None:
            continue
        if envs and meta['env'] not in envs:
            continue
        if seeds and meta['seed'] not in seeds:
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        df['env']        = meta['env']
        df['gauge_mode'] = meta['gauge_mode']
        df['seed']       = meta['seed']
        df['condition']  = 'oneshot' if meta['is_oneshot'] else 'iterative'
        train_rows.append(df)

    if not train_rows:
        return data

    train_df = pd.concat(train_rows, ignore_index=True)
    keep_cols = ['env', 'gauge_mode', 'seed', 'condition', 'step']
    for col in ('rm_gauge_gap', 'rm_param_norm', 'rm_mean_on_ref',
                'rm_mean_on_policy'):
        if col in train_df.columns:
            keep_cols.append(col)

    train_df = train_df[[c for c in keep_cols if c in train_df.columns]]
    if 'step' not in data.columns or 'step' not in train_df.columns:
        return data

    return pd.merge(data, train_df,
                    on=['env', 'gauge_mode', 'seed', 'condition', 'step'],
                    how='left')


# ── Core statistics ────────────────────────────────────────────────────────────

def _final_return(data: pd.DataFrame,
                  env: str,
                  gauge_mode: str,
                  condition: str,
                  seeds: List[int],
                  last_frac: float = 0.1) -> np.ndarray:
    """Mean true return over the last `last_frac` of training, per seed.

    Returns array of shape (n_seeds,).  NaN for missing seeds.
    """
    sub = data[
        (data['env']        == env) &
        (data['gauge_mode'] == gauge_mode) &
        (data['condition']  == condition)
    ]
    results = []
    for seed in seeds:
        s = sub[sub['seed'] == seed]
        if s.empty:
            results.append(float('nan'))
            continue
        step_max = s['step'].max()
        cutoff   = step_max * (1.0 - last_frac)
        tail     = s[s['step'] >= cutoff]['true_episode_reward']
        results.append(float(tail.mean()) if len(tail) > 0 else float('nan'))
    return np.array(results)


def compute_sigma(per_gauge_means: Dict[str, float]) -> float:
    """Std of per-gauge mean final returns (across gauge modes)."""
    vals = [v for v in per_gauge_means.values() if not np.isnan(v)]
    return float(np.std(vals, ddof=0)) if len(vals) > 1 else float('nan')


def bootstrap_ratio(sigma_iter_samples: np.ndarray,
                    sigma_oneshot_samples: np.ndarray,
                    n_bootstrap: int = N_BOOTSTRAP,
                    rng: Optional[np.random.Generator] = None
                    ) -> Tuple[float, float, float, np.ndarray]:
    """Bootstrap the sigma_iter / sigma_oneshot ratio.

    Args:
        sigma_iter_samples:    per-seed final returns for iterative, shape (n_gauges, n_seeds)
        sigma_oneshot_samples: per-seed final returns for one-shot, shape (n_gauges, n_seeds)
        n_bootstrap: number of bootstrap resamples
        rng: optional numpy Generator

    Returns:
        (median, ci_low, ci_high, ratio_dist) where ratio_dist is shape (n_bootstrap,).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n_gauges, n_seeds = sigma_iter_samples.shape
    ratio_dist = np.zeros(n_bootstrap)

    for b in range(n_bootstrap):
        # Resample seeds with replacement
        seed_idx = rng.integers(0, n_seeds, size=n_seeds)
        it_means  = sigma_iter_samples[:, seed_idx].mean(axis=1)   # (n_gauges,)
        os_means  = sigma_oneshot_samples[:, seed_idx].mean(axis=1)

        sig_it = float(np.std(it_means, ddof=0))
        sig_os = float(np.std(os_means, ddof=0))

        if sig_os < 1e-9:
            ratio_dist[b] = float('nan')
        else:
            ratio_dist[b] = sig_it / sig_os

    valid = ratio_dist[~np.isnan(ratio_dist)]
    if len(valid) == 0:
        return float('nan'), float('nan'), float('nan'), ratio_dist

    median = float(np.median(valid))
    ci_low  = float(np.percentile(valid, 2.5))
    ci_high = float(np.percentile(valid, 97.5))
    return median, ci_low, ci_high, ratio_dist


def verdict(ci_low: float, ci_high: float) -> str:
    if np.isnan(ci_low) or np.isnan(ci_high):
        return "INSUFFICIENT_DATA"
    if ci_low > RATIO_AMPLIFICATION_THRESHOLD:
        return "PERFORMATIVE_AMPLIFICATION"
    if ci_high < RATIO_NO_EFFECT_THRESHOLD:
        return "NO_PERFORMATIVE_AMPLIFICATION"
    return "INCONCLUSIVE"


# ── Plots ──────────────────────────────────────────────────────────────────────

def _plot_bar(results_by_env: Dict, out_dir: str) -> None:
    if not _MPL:
        return
    envs = list(results_by_env.keys())
    n    = len(envs)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    for ax, env in zip(axes[0], envs):
        R = results_by_env[env]
        for cond, marker in [('iterative', 'o'), ('oneshot', 's')]:
            sigmas = {gm: R[cond].get(gm, float('nan'))
                      for gm in GAUGE_MODES if gm in R[cond]}
            x = np.arange(len(sigmas))
            means = [np.nanmean(v) for v in sigmas.values()]
            sems  = [np.nanstd(v) / max(np.sqrt(np.sum(~np.isnan(v))), 1)
                     for v in sigmas.values()]
            ax.bar(x + (0.2 if cond == 'oneshot' else -0.2),
                   means, 0.35, yerr=sems, label=cond, alpha=0.75,
                   color=['#2196F3' if cond == 'iterative' else '#FF9800'][0])
        ax.set_xticks(np.arange(len(GAUGE_MODES)))
        ax.set_xticklabels([GAUGE_LABELS.get(g, g) for g in GAUGE_MODES],
                           rotation=15, ha='right', fontsize=8)
        ax.set_title(env)
        ax.set_ylabel('Mean final J(π)')
        ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'plot_A_bar.png'), dpi=150)
    plt.close(fig)
    print(f"[plot A] saved to {out_dir}/plot_A_bar.png")


def _plot_bootstrap(ratio_dist: np.ndarray, median: float,
                    ci_low: float, ci_high: float,
                    env: str, out_dir: str) -> None:
    if not _MPL:
        return
    valid = ratio_dist[~np.isnan(ratio_dist)]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(valid, bins=60, color='steelblue', alpha=0.8, edgecolor='white')
    ax.axvline(median, color='k',    linestyle='-',  label=f'Median {median:.2f}')
    ax.axvline(ci_low, color='grey', linestyle='--', label=f'95% CI [{ci_low:.2f}, {ci_high:.2f}]')
    ax.axvline(ci_high,color='grey', linestyle='--')
    ax.axvline(RATIO_AMPLIFICATION_THRESHOLD, color='red',   linestyle=':',
               label=f'Amplif. threshold ({RATIO_AMPLIFICATION_THRESHOLD})')
    ax.axvline(RATIO_NO_EFFECT_THRESHOLD, color='green', linestyle=':',
               label=f'No-effect threshold ({RATIO_NO_EFFECT_THRESHOLD})')
    ax.set_xlabel(r'$\sigma_\mathrm{iter} / \sigma_\mathrm{oneshot}$')
    ax.set_ylabel('Bootstrap count')
    ax.set_title(f'Bootstrap ratio distribution — {env}')
    ax.legend(fontsize=8)
    plt.tight_layout()
    fname = os.path.join(out_dir, f'plot_B_bootstrap_{env}.png')
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"[plot B] saved to {fname}")


def _plot_trajectories(data: pd.DataFrame, env: str, out_dir: str) -> None:
    if not _MPL:
        return
    sub  = data[data['env'] == env]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, cond in zip(axes, ['iterative', 'oneshot']):
        csub = sub[sub['condition'] == cond]
        for gm in GAUGE_MODES:
            gsub = csub[csub['gauge_mode'] == gm]
            if gsub.empty:
                continue
            grouped = gsub.groupby('step')['true_episode_reward']
            mean = grouped.mean()
            sem  = grouped.sem()
            steps = mean.index.values
            ax.plot(steps, mean.values, color=GAUGE_COLORS.get(gm, 'k'),
                    label=GAUGE_LABELS.get(gm, gm))
            ax.fill_between(steps,
                            (mean - sem).values, (mean + sem).values,
                            color=GAUGE_COLORS.get(gm, 'k'), alpha=0.15)
        ax.set_title(f'{env} — {cond}')
        ax.set_xlabel('Env steps')
        ax.set_ylabel('True episode return')
        ax.legend(fontsize=8)
    plt.tight_layout()
    fname = os.path.join(out_dir, f'plot_C_trajectories_{env}.png')
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"[plot C] saved to {fname}")


def _plot_gauge_gap(data: pd.DataFrame, env: str, out_dir: str) -> None:
    if not _MPL or 'rm_gauge_gap' not in data.columns:
        return
    sub = data[(data['env'] == env) & data['rm_gauge_gap'].notna()]
    if sub.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, cond in zip(axes, ['iterative', 'oneshot']):
        csub = sub[sub['condition'] == cond]
        for gm in GAUGE_MODES:
            gsub = csub[csub['gauge_mode'] == gm]
            if gsub.empty:
                continue
            grouped = gsub.groupby('step')['rm_gauge_gap']
            mean = grouped.mean()
            ax.plot(mean.index.values, mean.values,
                    color=GAUGE_COLORS.get(gm, 'k'),
                    label=GAUGE_LABELS.get(gm, gm))
        ax.axhline(0, color='k', linestyle='--', linewidth=0.7)
        ax.set_title(f'{env} — {cond}')
        ax.set_xlabel('Env steps')
        ax.set_ylabel('rm/gauge_gap')
        ax.legend(fontsize=8)
    plt.tight_layout()
    fname = os.path.join(out_dir, f'plot_D_gauge_gap_{env}.png')
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"[plot D] saved to {fname}")


# ── Report ─────────────────────────────────────────────────────────────────────

def write_report(rows: List[dict], out_path: str) -> None:
    lines = [
        "# Performativity analysis: sigma_iter / sigma_oneshot\n",
        "## Per-environment results\n",
        "| Env | sigma_iter | sigma_oneshot | Ratio (median) | 95% CI | Verdict |",
        "|-----|-----------|--------------|----------------|--------|---------|",
    ]
    for r in rows:
        ci = f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}]"
        lines.append(
            f"| {r['env']} | {r['sigma_iter']:.4f} | {r['sigma_oneshot']:.4f} "
            f"| {r['ratio_median']:.3f} | {ci} | **{r['verdict']}** |"
        )
    lines += [
        "\n## Thresholds",
        f"- Performative amplification: ratio CI low > {RATIO_AMPLIFICATION_THRESHOLD}",
        f"- No performative amplification: ratio CI high < {RATIO_NO_EFFECT_THRESHOLD}",
        "- Otherwise: INCONCLUSIVE",
        "\n## Plots",
        "- `plot_A_bar.png` — per-gauge mean final return (bar chart)",
        "- `plot_B_bootstrap_*.png` — bootstrap ratio distribution per env",
        "- `plot_C_trajectories_*.png` — true-return learning curves",
        "- `plot_D_gauge_gap_*.png` — rm/gauge_gap over training",
    ]
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"[report] written to {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exp_dir', required=True,
                        help='Root experiment directory (contains env subdirs)')
    parser.add_argument('--envs', nargs='*', default=[],
                        help='Environment filter (empty = all)')
    parser.add_argument('--seeds', nargs='*', type=int, default=[],
                        help='Seed filter (empty = all)')
    parser.add_argument('--output', default='results/',
                        help='Output directory for plots and summary')
    parser.add_argument('--last_frac', type=float, default=0.1,
                        help='Fraction of training used to compute "final" return')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"Loading runs from {args.exp_dir} ...")
    data = discover_runs(args.exp_dir, args.envs, args.seeds)

    envs_found = sorted(data['env'].unique())
    seeds_used = sorted(data['seed'].unique())
    print(f"Found envs: {envs_found}, seeds: {seeds_used}, "
          f"conditions: {sorted(data['condition'].unique())}")

    report_rows = []

    for env in envs_found:
        print(f"\n── {env} ──")
        gauges_present = sorted(data[data['env'] == env]['gauge_mode'].unique())

        # Collect per-seed returns: shape (n_gauges, n_seeds) for each condition
        it_mat = np.full((len(gauges_present), len(seeds_used)), float('nan'))
        os_mat = np.full((len(gauges_present), len(seeds_used)), float('nan'))

        results_by_cond: Dict = {'iterative': {}, 'oneshot': {}}

        for gi, gm in enumerate(gauges_present):
            it_arr = _final_return(data, env, gm, 'iterative', seeds_used,
                                   args.last_frac)
            os_arr = _final_return(data, env, gm, 'oneshot',   seeds_used,
                                   args.last_frac)
            it_mat[gi] = it_arr
            os_mat[gi] = os_arr
            results_by_cond['iterative'][gm] = it_arr
            results_by_cond['oneshot'][gm]   = os_arr

            print(f"  {gm:20s}  iter {np.nanmean(it_arr):.2f}±{np.nanstd(it_arr):.2f}"
                  f"  oneshot {np.nanmean(os_arr):.2f}±{np.nanstd(os_arr):.2f}")

        it_means = {gm: float(np.nanmean(it_mat[gi]))
                    for gi, gm in enumerate(gauges_present)}
        os_means = {gm: float(np.nanmean(os_mat[gi]))
                    for gi, gm in enumerate(gauges_present)}
        sig_iter    = compute_sigma(it_means)
        sig_oneshot = compute_sigma(os_means)

        rng = np.random.default_rng(seed=42)
        valid_cols = ~np.all(np.isnan(it_mat), axis=0) & \
                     ~np.all(np.isnan(os_mat), axis=0)
        if valid_cols.sum() < 2:
            print("  [warn] fewer than 2 valid seeds — skipping bootstrap")
            median, ci_low, ci_high, ratio_dist = float('nan'), float('nan'), float('nan'), np.array([float('nan')])
        else:
            median, ci_low, ci_high, ratio_dist = bootstrap_ratio(
                it_mat[:, valid_cols], os_mat[:, valid_cols],
                n_bootstrap=N_BOOTSTRAP, rng=rng)

        v = verdict(ci_low, ci_high)
        print(f"  sigma_iter={sig_iter:.4f}  sigma_oneshot={sig_oneshot:.4f}")
        print(f"  ratio median={median:.3f}  95%CI=[{ci_low:.3f},{ci_high:.3f}]")
        print(f"  VERDICT: {v}")

        report_rows.append(dict(
            env=env, sigma_iter=sig_iter, sigma_oneshot=sig_oneshot,
            ratio_median=median, ci_low=ci_low, ci_high=ci_high, verdict=v))

        _plot_bootstrap(ratio_dist, median, ci_low, ci_high, env, args.output)
        _plot_trajectories(data, env, args.output)
        _plot_gauge_gap(data, env, args.output)

    # Plot A spans all envs
    all_results = {env: {'iterative': {}, 'oneshot': {}} for env in envs_found}
    for gi_env_cond in data.groupby(['env', 'gauge_mode', 'condition']):
        pass  # built above per-env; reconstruct for plot A
    for env in envs_found:
        for gm in sorted(data[data['env'] == env]['gauge_mode'].unique()):
            for cond in ('iterative', 'oneshot'):
                arr = _final_return(data, env, gm, cond, seeds_used, args.last_frac)
                all_results[env][cond][gm] = arr
    _plot_bar(all_results, args.output)

    write_report(report_rows, os.path.join(args.output, 'performativity_summary.md'))
    print("\nDone.")


if __name__ == '__main__':
    main()
