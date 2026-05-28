#!/usr/bin/env python3
"""Analyze performativity of gauge ambiguity: sigma_iter / sigma_oneshot ratio.

Reads local CSV files produced by train_PEBBLE.py (iterative) and
train_PEBBLE_oneshot.py (one-shot control).  Produces per-(env, teacher, feed_type)
results so that teacher noise and query strategy are not mixed into the gauge sigma.

  results/
    plot_A_bar.png              Bar chart: sigma per condition per env
    plot_B_bootstrap_*.png      Bootstrap distribution of sigma_iter/sigma_oneshot
    plot_C_trajectories_*.png   True-return curves overlaid
    plot_D_gauge_gap_*.png      rm/gauge_gap over training
    performativity_summary.md   Verdict + statistics table

Usage::

    # All teacher/feed_type groups:
    python analyze_performativity.py \\
        --exp_dir /your/exp \\
        --envs metaworld_drawer-open-v2 metaworld_window-close-v2 \\
        --seeds 12345 23451 34512 \\
        --output results/

    # Oracle teacher, feed_type=0 only:
    python analyze_performativity.py \\
        --exp_dir /your/exp \\
        --teacher_dirs teacher_b-1_g1.0_m0_s0_e0 \\
        --feed_types 0 \\
        --output results/oracle_f0/

teacher_dirs are the literal path component strings (teacher_b..._g..._m..._s..._e...).
If omitted, all discovered teacher directories are analyzed as separate groups.
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

_GAUGE_RE   = re.compile(r'/gauge_([^/]+)/')
_SEED_RE    = re.compile(r'_seed(\d+)(?:/|$)')
_TEACHER_RE = re.compile(r'/(teacher_b[^/]+)/')
_FEED_RE    = re.compile(r'_sample(\d+)_')


def _parse_run_path(csv_path: str) -> Optional[Dict]:
    """Extract metadata from an eval.csv or train.csv path.

    Expected path fragment:
      {exp_dir}/{env}/H.../{teacher_dir}/label_smooth.../tandem_.../gauge_{gm}/
          {PEBBLE[_oneshot]}_..._sample{feed_type}_..._seed{seed}/eval.csv
    """
    gm_match      = _GAUGE_RE.search(csv_path)
    seed_match    = _SEED_RE.search(csv_path)
    teacher_match = _TEACHER_RE.search(csv_path)
    feed_match    = _FEED_RE.search(csv_path)

    if gm_match is None or seed_match is None:
        return None

    gauge_mode   = gm_match.group(1)
    seed         = int(seed_match.group(1))
    teacher_dir  = teacher_match.group(1) if teacher_match else 'unknown'
    feed_type    = int(feed_match.group(1)) if feed_match else -1
    is_oneshot   = 'PEBBLE_oneshot' in csv_path

    # env: walk path components backwards from gauge_ to find non-config token
    parts = [p for p in csv_path.split(os.sep) if p]
    env = 'unknown'
    for i, p in enumerate(parts):
        if p.startswith('gauge_'):
            for j in range(i - 1, -1, -1):
                candidate = parts[j]
                if not any(candidate.startswith(pref) for pref in (
                        'H', 'L', 'teacher', 'label', 'schedule', 'tandem',
                        'gauge', 'PEBBLE')):
                    env = candidate
                    break
            break

    return dict(env=env, gauge_mode=gauge_mode, seed=seed,
                teacher_dir=teacher_dir, feed_type=feed_type,
                is_oneshot=is_oneshot)


# ── CSV discovery and loading ──────────────────────────────────────────────────

def discover_runs(exp_dir: str,
                  envs: List[str],
                  seeds: List[int],
                  teacher_dirs: List[str],
                  feed_types: List[int]) -> pd.DataFrame:
    """Walk exp_dir, load eval.csv files, return a flat DataFrame.

    Columns: env, gauge_mode, seed, teacher_dir, feed_type,
             condition (iterative|oneshot), step, true_episode_reward, ...
    """
    rows: List[dict] = []

    for root, _dirs, files in os.walk(exp_dir):
        if 'eval.csv' not in files:
            continue
        csv_path = os.path.join(root, 'eval.csv')
        meta = _parse_run_path(csv_path)
        if meta is None:
            continue
        if envs        and meta['env']         not in envs:
            continue
        if seeds       and meta['seed']        not in seeds:
            continue
        if teacher_dirs and meta['teacher_dir'] not in teacher_dirs:
            continue
        if feed_types  and meta['feed_type']   not in feed_types:
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[warn] could not read {csv_path}: {e}")
            continue

        if 'true_episode_reward' not in df.columns:
            print(f"[warn] missing true_episode_reward in {csv_path}")
            continue

        df['env']         = meta['env']
        df['gauge_mode']  = meta['gauge_mode']
        df['seed']        = meta['seed']
        df['teacher_dir'] = meta['teacher_dir']
        df['feed_type']   = meta['feed_type']
        df['condition']   = 'oneshot' if meta['is_oneshot'] else 'iterative'
        rows.append(df)

    if not rows:
        print(f"[error] no eval.csv files found under {exp_dir}")
        sys.exit(1)

    data = pd.concat(rows, ignore_index=True)
    data = _merge_train_metrics(data, exp_dir, envs, seeds, teacher_dirs, feed_types)
    return data


def _merge_train_metrics(data: pd.DataFrame,
                          exp_dir: str,
                          envs: List[str],
                          seeds: List[int],
                          teacher_dirs: List[str],
                          feed_types: List[int]) -> pd.DataFrame:
    """Best-effort merge of rm_gauge_gap / rm_param_norm from train.csv."""
    train_rows: List[dict] = []
    for root, _dirs, files in os.walk(exp_dir):
        if 'train.csv' not in files:
            continue
        csv_path = os.path.join(root, 'train.csv')
        meta = _parse_run_path(csv_path)
        if meta is None:
            continue
        if envs        and meta['env']         not in envs:
            continue
        if seeds       and meta['seed']        not in seeds:
            continue
        if teacher_dirs and meta['teacher_dir'] not in teacher_dirs:
            continue
        if feed_types  and meta['feed_type']   not in feed_types:
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        df['env']         = meta['env']
        df['gauge_mode']  = meta['gauge_mode']
        df['seed']        = meta['seed']
        df['teacher_dir'] = meta['teacher_dir']
        df['feed_type']   = meta['feed_type']
        df['condition']   = 'oneshot' if meta['is_oneshot'] else 'iterative'
        train_rows.append(df)

    if not train_rows:
        return data

    train_df = pd.concat(train_rows, ignore_index=True)
    merge_keys = ['env', 'gauge_mode', 'seed', 'teacher_dir', 'feed_type', 'condition', 'step']
    keep_cols  = [k for k in merge_keys if k in train_df.columns]
    for col in ('rm_gauge_gap', 'rm_param_norm', 'rm_mean_on_ref', 'rm_mean_on_policy'):
        if col in train_df.columns:
            keep_cols.append(col)
    train_df = train_df[[c for c in keep_cols if c in train_df.columns]]

    on_cols = [k for k in merge_keys if k in data.columns and k in train_df.columns]
    if not on_cols:
        return data
    return pd.merge(data, train_df, on=on_cols, how='left')


# ── Core statistics ────────────────────────────────────────────────────────────

def _final_return(data: pd.DataFrame,
                  env: str,
                  teacher_dir: str,
                  feed_type: int,
                  gauge_mode: str,
                  condition: str,
                  seeds: List[int],
                  last_frac: float = 0.1) -> np.ndarray:
    """Mean true return over the last `last_frac` of training, per seed."""
    sub = data[
        (data['env']         == env) &
        (data['teacher_dir'] == teacher_dir) &
        (data['feed_type']   == feed_type) &
        (data['gauge_mode']  == gauge_mode) &
        (data['condition']   == condition)
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
    vals = [v for v in per_gauge_means.values() if not np.isnan(v)]
    return float(np.std(vals, ddof=0)) if len(vals) > 1 else float('nan')


def bootstrap_ratio(sigma_iter_samples: np.ndarray,
                    sigma_oneshot_samples: np.ndarray,
                    n_bootstrap: int = N_BOOTSTRAP,
                    rng: Optional[np.random.Generator] = None
                    ) -> Tuple[float, float, float, np.ndarray]:
    if rng is None:
        rng = np.random.default_rng(42)
    n_gauges, n_seeds = sigma_iter_samples.shape
    ratio_dist = np.zeros(n_bootstrap)
    for b in range(n_bootstrap):
        seed_idx  = rng.integers(0, n_seeds, size=n_seeds)
        it_means  = sigma_iter_samples[:, seed_idx].mean(axis=1)
        os_means  = sigma_oneshot_samples[:, seed_idx].mean(axis=1)
        sig_it = float(np.std(it_means, ddof=0))
        sig_os = float(np.std(os_means, ddof=0))
        ratio_dist[b] = float('nan') if sig_os < 1e-9 else sig_it / sig_os

    valid = ratio_dist[~np.isnan(ratio_dist)]
    if len(valid) == 0:
        return float('nan'), float('nan'), float('nan'), ratio_dist
    return (float(np.median(valid)),
            float(np.percentile(valid, 2.5)),
            float(np.percentile(valid, 97.5)),
            ratio_dist)


def verdict(ci_low: float, ci_high: float) -> str:
    if np.isnan(ci_low) or np.isnan(ci_high):
        return "INSUFFICIENT_DATA"
    if ci_low > RATIO_AMPLIFICATION_THRESHOLD:
        return "PERFORMATIVE_AMPLIFICATION"
    if ci_high < RATIO_NO_EFFECT_THRESHOLD:
        return "NO_PERFORMATIVE_AMPLIFICATION"
    return "INCONCLUSIVE"


# ── Plots ──────────────────────────────────────────────────────────────────────

def _plot_bar(all_results: Dict, out_dir: str, tag: str) -> None:
    if not _MPL:
        return
    envs = list(all_results.keys())
    n    = len(envs)
    fig, axes = plt.subplots(1, max(n, 1), figsize=(5 * max(n, 1), 4), squeeze=False)
    for ax, env in zip(axes[0], envs):
        for cond, offset in [('iterative', -0.2), ('oneshot', 0.2)]:
            cond_data = all_results[env][cond]
            x     = np.arange(len(cond_data))
            means = [np.nanmean(v) for v in cond_data.values()]
            sems  = [np.nanstd(v) / max(np.sqrt(np.sum(~np.isnan(v))), 1)
                     for v in cond_data.values()]
            color = '#2196F3' if cond == 'iterative' else '#FF9800'
            ax.bar(x + offset, means, 0.35, yerr=sems, label=cond,
                   alpha=0.75, color=color)
        ax.set_xticks(np.arange(len(GAUGE_MODES)))
        ax.set_xticklabels([GAUGE_LABELS.get(g, g) for g in GAUGE_MODES],
                           rotation=15, ha='right', fontsize=8)
        ax.set_title(env)
        ax.set_ylabel('Mean final J(π)')
        ax.legend()
    plt.tight_layout()
    fname = os.path.join(out_dir, f'plot_A_bar_{tag}.png')
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"[plot A] {fname}")


def _plot_bootstrap(ratio_dist: np.ndarray, median: float,
                    ci_low: float, ci_high: float,
                    tag: str, out_dir: str) -> None:
    if not _MPL:
        return
    valid = ratio_dist[~np.isnan(ratio_dist)]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(valid, bins=60, color='steelblue', alpha=0.8, edgecolor='white')
    ax.axvline(median, color='k',     linestyle='-',  label=f'Median {median:.2f}')
    ax.axvline(ci_low, color='grey',  linestyle='--', label=f'95% CI [{ci_low:.2f}, {ci_high:.2f}]')
    ax.axvline(ci_high,color='grey',  linestyle='--')
    ax.axvline(RATIO_AMPLIFICATION_THRESHOLD, color='red',   linestyle=':',
               label=f'Amplif. ({RATIO_AMPLIFICATION_THRESHOLD})')
    ax.axvline(RATIO_NO_EFFECT_THRESHOLD,     color='green', linestyle=':',
               label=f'No-effect ({RATIO_NO_EFFECT_THRESHOLD})')
    ax.set_xlabel(r'$\sigma_\mathrm{iter} / \sigma_\mathrm{oneshot}$')
    ax.set_ylabel('Bootstrap count')
    ax.set_title(f'Bootstrap ratio — {tag}')
    ax.legend(fontsize=8)
    plt.tight_layout()
    fname = os.path.join(out_dir, f'plot_B_bootstrap_{tag}.png')
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"[plot B] {fname}")


def _plot_trajectories(sub: pd.DataFrame, tag: str, out_dir: str) -> None:
    if not _MPL:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, cond in zip(axes, ['iterative', 'oneshot']):
        csub = sub[sub['condition'] == cond]
        for gm in GAUGE_MODES:
            gsub = csub[csub['gauge_mode'] == gm]
            if gsub.empty:
                continue
            grouped = gsub.groupby('step')['true_episode_reward']
            mean, sem = grouped.mean(), grouped.sem()
            ax.plot(mean.index.values, mean.values,
                    color=GAUGE_COLORS.get(gm, 'k'), label=GAUGE_LABELS.get(gm, gm))
            ax.fill_between(mean.index.values,
                            (mean - sem).values, (mean + sem).values,
                            color=GAUGE_COLORS.get(gm, 'k'), alpha=0.15)
        ax.set_title(f'{cond}')
        ax.set_xlabel('Env steps')
        ax.set_ylabel('True episode return')
        ax.legend(fontsize=8)
    fig.suptitle(tag, fontsize=10)
    plt.tight_layout()
    fname = os.path.join(out_dir, f'plot_C_trajectories_{tag}.png')
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"[plot C] {fname}")


def _plot_gauge_gap(sub: pd.DataFrame, tag: str, out_dir: str) -> None:
    if not _MPL or 'rm_gauge_gap' not in sub.columns:
        return
    s = sub[sub['rm_gauge_gap'].notna()]
    if s.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, cond in zip(axes, ['iterative', 'oneshot']):
        csub = s[s['condition'] == cond]
        for gm in GAUGE_MODES:
            gsub = csub[csub['gauge_mode'] == gm]
            if gsub.empty:
                continue
            mean = gsub.groupby('step')['rm_gauge_gap'].mean()
            ax.plot(mean.index.values, mean.values,
                    color=GAUGE_COLORS.get(gm, 'k'), label=GAUGE_LABELS.get(gm, gm))
        ax.axhline(0, color='k', linestyle='--', linewidth=0.7)
        ax.set_title(f'{cond}')
        ax.set_xlabel('Env steps')
        ax.set_ylabel('rm/gauge_gap')
        ax.legend(fontsize=8)
    fig.suptitle(tag, fontsize=10)
    plt.tight_layout()
    fname = os.path.join(out_dir, f'plot_D_gauge_gap_{tag}.png')
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"[plot D] {fname}")


# ── Report ─────────────────────────────────────────────────────────────────────

def write_report(rows: List[dict], out_path: str) -> None:
    lines = [
        "# Performativity analysis: sigma_iter / sigma_oneshot\n",
        "## Results per (env, teacher, feed_type)\n",
        "| Env | Teacher | Feed | sigma_iter | sigma_oneshot | Ratio (median) | 95% CI | Verdict |",
        "|-----|---------|------|-----------|--------------|----------------|--------|---------|",
    ]
    for r in rows:
        ci = f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}]"
        lines.append(
            f"| {r['env']} | {r['teacher_dir']} | {r['feed_type']} "
            f"| {r['sigma_iter']:.4f} | {r['sigma_oneshot']:.4f} "
            f"| {r['ratio_median']:.3f} | {ci} | **{r['verdict']}** |"
        )
    lines += [
        "\n## Thresholds",
        f"- Performative amplification: ratio CI low > {RATIO_AMPLIFICATION_THRESHOLD}",
        f"- No performative amplification: ratio CI high < {RATIO_NO_EFFECT_THRESHOLD}",
        "- Otherwise: INCONCLUSIVE",
    ]
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"[report] written to {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--exp_dir', required=True,
                        help='Root experiment directory (contains env subdirs)')
    parser.add_argument('--envs', nargs='*', default=[],
                        help='Environment filter (empty = all)')
    parser.add_argument('--seeds', nargs='*', type=int, default=[],
                        help='Seed filter (empty = all)')
    parser.add_argument('--teacher_dirs', nargs='*', default=[],
                        help='Teacher-dir filter, e.g. teacher_b-1_g1.0_m0_s0_e0 '
                             '(empty = all, analyzed as separate groups)')
    parser.add_argument('--feed_types', nargs='*', type=int, default=[],
                        help='Feed-type filter, e.g. 0 1 (empty = all groups)')
    parser.add_argument('--output', default='results/',
                        help='Output directory for plots and summary')
    parser.add_argument('--last_frac', type=float, default=0.1,
                        help='Fraction of training used to compute "final" return')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"Loading runs from {args.exp_dir} ...")
    data = discover_runs(args.exp_dir, args.envs, args.seeds,
                         args.teacher_dirs, args.feed_types)

    envs_found     = sorted(data['env'].unique())
    seeds_used     = sorted(data['seed'].unique())
    teachers_found = sorted(data['teacher_dir'].unique())
    feeds_found    = sorted(data['feed_type'].unique())
    print(f"Found: envs={envs_found}")
    print(f"       seeds={seeds_used}")
    print(f"       teacher_dirs={teachers_found}")
    print(f"       feed_types={feeds_found}")
    print(f"       conditions={sorted(data['condition'].unique())}")

    if len(teachers_found) > 1 or len(feeds_found) > 1:
        print("\n[info] Multiple teacher_dirs or feed_types found — "
              "results will be reported per group.  "
              "Use --teacher_dirs / --feed_types to restrict to one group.")

    report_rows = []

    for env in envs_found:
        for teacher_dir in teachers_found:
            for feed_type in feeds_found:
                sub = data[
                    (data['env']         == env) &
                    (data['teacher_dir'] == teacher_dir) &
                    (data['feed_type']   == feed_type)
                ]
                if sub.empty:
                    continue

                gauges_present = sorted(sub['gauge_mode'].unique())
                tag = f"{env}__{teacher_dir}__f{feed_type}"
                tag_short = f"{env[:12]}_f{feed_type}_{teacher_dir.split('_')[1]}"
                print(f"\n── {tag} ──")

                it_mat = np.full((len(gauges_present), len(seeds_used)), float('nan'))
                os_mat = np.full((len(gauges_present), len(seeds_used)), float('nan'))
                bar_data: Dict = {'iterative': {}, 'oneshot': {}}

                for gi, gm in enumerate(gauges_present):
                    it_arr = _final_return(sub, env, teacher_dir, feed_type,
                                           gm, 'iterative', seeds_used, args.last_frac)
                    os_arr = _final_return(sub, env, teacher_dir, feed_type,
                                           gm, 'oneshot',   seeds_used, args.last_frac)
                    it_mat[gi] = it_arr
                    os_mat[gi] = os_arr
                    bar_data['iterative'][gm] = it_arr
                    bar_data['oneshot'][gm]   = os_arr
                    print(f"  {gm:20s}  iter {np.nanmean(it_arr):.2f}±{np.nanstd(it_arr):.2f}"
                          f"  oneshot {np.nanmean(os_arr):.2f}±{np.nanstd(os_arr):.2f}")

                it_means = {gm: float(np.nanmean(it_mat[gi]))
                            for gi, gm in enumerate(gauges_present)}
                os_means = {gm: float(np.nanmean(os_mat[gi]))
                            for gi, gm in enumerate(gauges_present)}
                sig_iter    = compute_sigma(it_means)
                sig_oneshot = compute_sigma(os_means)

                rng = np.random.default_rng(seed=42)
                valid_cols = (~np.all(np.isnan(it_mat), axis=0) &
                              ~np.all(np.isnan(os_mat), axis=0))
                if valid_cols.sum() < 2:
                    print("  [warn] fewer than 2 valid seeds — skipping bootstrap")
                    median = ci_low = ci_high = float('nan')
                    ratio_dist = np.array([float('nan')])
                else:
                    median, ci_low, ci_high, ratio_dist = bootstrap_ratio(
                        it_mat[:, valid_cols], os_mat[:, valid_cols],
                        n_bootstrap=N_BOOTSTRAP, rng=rng)

                v = verdict(ci_low, ci_high)
                print(f"  sigma_iter={sig_iter:.4f}  sigma_oneshot={sig_oneshot:.4f}")
                print(f"  ratio median={median:.3f}  95%CI=[{ci_low:.3f},{ci_high:.3f}]")
                print(f"  VERDICT: {v}")

                report_rows.append(dict(
                    env=env, teacher_dir=teacher_dir, feed_type=feed_type,
                    sigma_iter=sig_iter, sigma_oneshot=sig_oneshot,
                    ratio_median=median, ci_low=ci_low, ci_high=ci_high, verdict=v))

                _plot_bar({env: bar_data}, args.output, tag_short)
                _plot_bootstrap(ratio_dist, median, ci_low, ci_high,
                                tag_short, args.output)
                _plot_trajectories(sub, tag_short, args.output)
                _plot_gauge_gap(sub, tag_short, args.output)

    write_report(report_rows, os.path.join(args.output, 'performativity_summary.md'))
    print("\nDone.")


if __name__ == '__main__':
    main()
