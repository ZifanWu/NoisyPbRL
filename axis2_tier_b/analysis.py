"""
Axis 2 Tier B — Analysis.

Reads axis2_metrics.csv files from all conditions/tasks/seeds, computes:
  - 3×3 Spearman correlation matrix (instruments × error components)
  - 3×3 partial Spearman correlation matrix (controlling for the other two components)
  - Bootstrap 95% CIs for both matrices
  - 9-panel scatter plots (3 instruments × 3 components, with LOWESS curves)
  - Condition summary table (mean ± std per condition)
  - H2.1–H2.5 verdicts → results.md

Usage:
    python -m axis2_tier_b.analysis --results_dir /path/to/results/axis2_tier_b
    python -m axis2_tier_b.analysis --results_dir exp/axis2_tier_b --out_dir my_out

Primary 3×3 matrix uses gof_deployed (held-out residual of the deployed RM) as
the GoF instrument.  gof_inf is logged only as the deployed-class floor.
"""

import os
import sys
import glob
import json
import argparse
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.stats as stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── correlation utilities ─────────────────────────────────────────────────────

def spearman(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Return (rho, p-value) for complete-case Spearman correlation."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5:
        return float('nan'), float('nan')
    rho, p = stats.spearmanr(x[mask], y[mask])
    return float(rho), float(p)


def partial_spearman(x: np.ndarray, y: np.ndarray,
                     controls: List[np.ndarray]) -> Tuple[float, float]:
    """
    Partial Spearman correlation of x and y, controlling for all variables in controls.

    Implements the Heckman–Olsson approach: compute Spearman residuals of x and y
    after regression on controls (using rank-transformation + OLS), then correlate.
    """
    all_vars = [x, y] + controls
    mask = np.ones(len(x), dtype=bool)
    for v in all_vars:
        mask &= np.isfinite(v)
    n = mask.sum()
    if n < 10:
        return float('nan'), float('nan')

    def rank_resid(target: np.ndarray, covs: List[np.ndarray]) -> np.ndarray:
        """Regress rank(target) on rank(covs), return residuals."""
        rk_t = stats.rankdata(target[mask])
        C = np.column_stack([stats.rankdata(c[mask]) for c in covs] +
                            [np.ones(n)])
        try:
            coef, _, _, _ = np.linalg.lstsq(C, rk_t, rcond=None)
            return rk_t - C @ coef
        except np.linalg.LinAlgError:
            return rk_t

    r_x = rank_resid(x, controls)
    r_y = rank_resid(y, controls)
    rho, p = stats.pearsonr(r_x, r_y)
    return float(rho), float(p)


def bootstrap_ci(x: np.ndarray, y: np.ndarray,
                 fn, n_boot: int = 1000, alpha: float = 0.05,
                 controls: Optional[List[np.ndarray]] = None) -> Tuple[float, float]:
    """Bootstrap percentile CI for a correlation function fn(x, y[, controls])."""
    mask = np.isfinite(x) & np.isfinite(y)
    if controls is not None:
        for c in controls:
            mask &= np.isfinite(c)
    n = mask.sum()
    if n < 5:
        return float('nan'), float('nan')

    x_m = x[mask]; y_m = y[mask]
    ctrl_m = [c[mask] for c in controls] if controls is not None else None

    rhos = []
    rng = np.random.default_rng(0)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        xi = x_m[idx]; yi = y_m[idx]
        if controls is not None:
            ci_list = [c[idx] for c in ctrl_m]
            rho, _ = fn(xi, yi, ci_list)
        else:
            rho, _ = fn(xi, yi)
        rhos.append(rho)

    rhos = np.array(rhos)
    lo = np.nanpercentile(rhos, 100 * alpha / 2)
    hi = np.nanpercentile(rhos, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


# ── data loading ──────────────────────────────────────────────────────────────

def load_all_metrics(results_dir: str) -> pd.DataFrame:
    """
    Recursively find all axis2_metrics.csv files under results_dir and concatenate.
    Infers condition, task, seed from directory path.
    """
    files = glob.glob(os.path.join(results_dir, '**/axis2_metrics.csv'), recursive=True)
    if not files:
        raise FileNotFoundError(
            f"No axis2_metrics.csv found under {results_dir}.\n"
            "Run train_PEBBLE_axis2.py first.")

    root_name = os.path.basename(os.path.normpath(results_dir))
    default_task = root_name if (
        'walker' in root_name or 'cheetah' in root_name or root_name.startswith('metaworld_')
    ) else 'unknown'

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        parts = f.replace(results_dir, '').split(os.sep)
        # expected path: .../axis2_tier_b/<task>/<condition>/seed<N>/axis2_metrics.csv
        condition = 'unknown'
        task = default_task
        seed = -1
        for p in parts:
            if p in ('control', 'shift', 'epi') or p.startswith('mis'):
                condition = p.split('_h')[0] if '_h' in p else p
            elif 'walker' in p or 'cheetah' in p or p.startswith('metaworld_'):
                task = p
            elif p.startswith('seed'):
                try:
                    seed = int(p.replace('seed', ''))
                except ValueError:
                    pass
        df['condition'] = condition
        df['task'] = task
        df['seed'] = seed
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df_all)} monitoring rows from {len(files)} runs.")
    print(f"Conditions: {df_all['condition'].unique()}")
    print(f"Tasks: {df_all['task'].unique()}")
    return df_all


# ── 3×3 matrix computation ────────────────────────────────────────────────────

# GoF instrument is the DEPLOYED RM's held-out CE residual above floor — that is what
# practitioners observe and what diagnoses the deployed model's misspecification + epi.
# gof_inf is the deployed-class infinite-label floor; it's logged for the chained
# decomposition (gof_epi = gof_deployed − gof_inf) but NOT used as the headline instrument.
INSTRUMENTS = ['kappa_hat', 'ensemble_spread', 'gof_deployed']
INSTR_LABELS = ['κ̂', 'Ensemble spread', 'GoF (deployed)']
COMPONENTS = ['e_shift', 'e_epi', 'e_mis']
COMP_LABELS = ['e_shift', 'e_epi', 'e_mis']


def compute_3x3_matrix(df: pd.DataFrame, n_boot: int = 1000) -> Dict:
    """
    Compute Spearman and partial-Spearman 3×3 matrices with bootstrap CIs.

    Returns dict with keys 'spearman' and 'partial', each a dict:
      matrix[instr][comp] = {'rho': ..., 'p': ..., 'ci_lo': ..., 'ci_hi': ...}
    """
    results = {'spearman': {}, 'partial': {}}

    for instr in INSTRUMENTS:
        results['spearman'][instr] = {}
        results['partial'][instr] = {}
        x = df[instr].values

        for comp in COMPONENTS:
            y = df[comp].values

            # Spearman
            rho, p = spearman(x, y)
            ci_lo, ci_hi = bootstrap_ci(x, y, spearman, n_boot=n_boot)
            results['spearman'][instr][comp] = dict(rho=rho, p=p, ci_lo=ci_lo, ci_hi=ci_hi)

            # Partial Spearman (control for the other two components)
            other_comps = [df[c].values for c in COMPONENTS if c != comp]
            rho_p, p_p = partial_spearman(x, y, other_comps)
            ci_lo_p, ci_hi_p = bootstrap_ci(x, y, partial_spearman, n_boot=n_boot,
                                             controls=other_comps)
            results['partial'][instr][comp] = dict(rho=rho_p, p=p_p,
                                                   ci_lo=ci_lo_p, ci_hi=ci_hi_p)

    return results


# ── hypothesis verdicts ───────────────────────────────────────────────────────

def _is_strong(entry: dict, threshold: float = 0.3) -> bool:
    """Correlation is 'strong' if |rho| > threshold and CI excludes 0."""
    rho = entry['rho']
    lo, hi = entry['ci_lo'], entry['ci_hi']
    if any(v != v for v in [rho, lo, hi]):
        return False
    return abs(rho) > threshold and not (lo < 0 < hi)


def _is_weak(entry: dict, threshold: float = 0.2) -> bool:
    """Correlation is 'weak' if |rho| < threshold."""
    rho = entry['rho']
    if rho != rho:
        return False
    return abs(rho) < threshold


def evaluate_hypotheses(matrix_results: dict, df: pd.DataFrame) -> str:
    """Return a results.md string with H2.1–H2.4 verdicts."""
    S = matrix_results['partial']  # use partial correlations for hypothesis tests

    h21 = (_is_strong(S['kappa_hat']['e_shift']) and
            _is_weak(S['kappa_hat']['e_epi']) and
            _is_weak(S['kappa_hat']['e_mis']))
    h22 = (_is_strong(S['ensemble_spread']['e_epi']) and
            _is_weak(S['ensemble_spread']['e_shift']) and
            _is_weak(S['ensemble_spread']['e_mis']))
    h23 = (_is_strong(S['gof_deployed']['e_mis']) and
            _is_weak(S['gof_deployed']['e_shift']) and
            _is_weak(S['gof_deployed']['e_epi']))
    h24 = h21 and h22 and h23

    def fmt(e):
        if e['rho'] != e['rho']:
            return 'NaN'
        sign = '*' if not (e['ci_lo'] < 0 < e['ci_hi']) else ''
        return f"{e['rho']:+.3f}{sign} [{e['ci_lo']:+.3f}, {e['ci_hi']:+.3f}]"

    lines = [
        "# Axis 2 Tier B Results\n",
        "## 3×3 Partial Spearman Correlations (instrument × component)\n",
        "Partial correlations control for the other two error components.\n",
        "(*) = CI excludes 0.\n\n",
        "```",
        f"{'':22} {'e_shift':>20} {'e_epi':>20} {'e_mis':>20}",
    ]
    for instr, il in zip(INSTRUMENTS, INSTR_LABELS):
        row = f"{il:22}"
        for comp in COMPONENTS:
            row += f" {fmt(S[instr][comp]):>20}"
        lines.append(row)
    lines += ["```\n"]

    lines += [
        "## Hypothesis Verdicts\n",
        f"- **H2.1** (κ̂ ↔ e_shift, blind to e_epi/e_mis): {'SUPPORTED' if h21 else 'NOT SUPPORTED'}",
        f"  partial ρ(κ̂, e_shift) = {fmt(S['kappa_hat']['e_shift'])}",
        f"- **H2.2** (spread ↔ e_epi, blind to e_shift/e_mis): {'SUPPORTED' if h22 else 'NOT SUPPORTED'}",
        f"  partial ρ(spread, e_epi) = {fmt(S['ensemble_spread']['e_epi'])}",
        f"- **H2.3** (GoF ↔ e_mis — the BLIND SPOT): {'SUPPORTED' if h23 else 'NOT SUPPORTED'}",
        f"  partial ρ(GoF_deployed, e_mis) = {fmt(S['gof_deployed']['e_mis'])}",
        f"  GoF off-diagonal: ρ(GoF_deployed, e_shift)={fmt(S['gof_deployed']['e_shift'])}, "
        f"ρ(GoF_deployed, e_epi)={fmt(S['gof_deployed']['e_epi'])}",
        f"- **H2.4** (3×3 diagonal dominance via partials): {'SUPPORTED' if h24 else 'NOT SUPPORTED'}",
        "",
        "## Condition Summary",
    ]

    # Condition-level means
    summary = df.groupby('condition')[
        INSTRUMENTS + COMPONENTS
    ].mean().round(4)
    lines.append(summary.to_string())
    lines.append("")

    # GoF deployed vs inf cross-check
    if 'gof_deployed' in df.columns and 'gof_inf' in df.columns:
        lines.append("## GoF Decomposition (deployed vs ψ_inf)\n")
        gof_cross = df.groupby('condition')[
            ['gof_deployed', 'gof_inf', 'gof_epi']
        ].mean().round(4)
        lines.append(gof_cross.to_string())
        lines.append("")
        lines.append("gof_epi should be large in epi condition and near 0 in mis condition.")
        lines.append("")

    return "\n".join(lines)


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_3x3_scatter(df: pd.DataFrame, out_dir: str):
    """9-panel scatter plot: rows=instruments, cols=components, with LOWESS curves."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from statsmodels.nonparametric.smoothers_lowess import lowess
    except ImportError:
        warnings.warn("matplotlib/statsmodels not available — skipping scatter plots")
        return

    cond_colors = {'control': 'C0', 'shift': 'C1', 'epi': 'C2', 'mis': 'C3', 'unknown': 'C4'}
    fig, axes = plt.subplots(3, 3, figsize=(13, 11))

    for ii, (instr, il) in enumerate(zip(INSTRUMENTS, INSTR_LABELS)):
        for jj, (comp, cl) in enumerate(zip(COMPONENTS, COMP_LABELS)):
            ax = axes[ii, jj]
            for cond, grp in df.groupby('condition'):
                x = grp[instr].dropna()
                y = grp[comp].dropna()
                idx = x.index.intersection(y.index)
                if len(idx) < 2:
                    continue
                ax.scatter(x[idx], y[idx], s=8, alpha=0.4,
                           color=cond_colors.get(cond, 'grey'), label=cond)

            # LOWESS over all conditions combined
            xall = df[instr].values
            yall = df[comp].values
            mask = np.isfinite(xall) & np.isfinite(yall)
            if mask.sum() > 10:
                order = np.argsort(xall[mask])
                lw = lowess(yall[mask][order], xall[mask][order], frac=0.4)
                ax.plot(lw[:, 0], lw[:, 1], 'k-', lw=1.5, alpha=0.8)

            rho, _ = spearman(df[instr].values, df[comp].values)
            ax.set_xlabel(il if ii == 2 else '')
            ax.set_ylabel(cl if jj == 0 else '')
            ax.set_title(f'ρ={rho:+.2f}' if rho == rho else 'NaN', fontsize=9)
            if ii == 0 and jj == 2:
                ax.legend(fontsize=7, markerscale=2)

    plt.suptitle('Axis 2 Tier B — 3×3 Instrument × Component Scatter\n'
                 '(LOWESS curve = all conditions combined)', fontsize=12)
    plt.tight_layout()
    out = os.path.join(out_dir, 'scatter_3x3.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved scatter_3x3.png to {out_dir}")


def plot_matrix_heatmap(matrix_results: dict, out_dir: str):
    """Heatmap of Spearman and partial-Spearman 3×3 matrices."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (mtype, title) in zip(axes, [
        ('spearman', 'Spearman ρ'),
        ('partial', 'Partial Spearman ρ\n(controlling other 2 components)')
    ]):
        mat = np.full((3, 3), float('nan'))
        for ii, instr in enumerate(INSTRUMENTS):
            for jj, comp in enumerate(COMPONENTS):
                mat[ii, jj] = matrix_results[mtype][instr][comp]['rho']
        im = ax.imshow(mat, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        ax.set_xticks(range(3)); ax.set_xticklabels(COMP_LABELS, fontsize=9)
        ax.set_yticks(range(3)); ax.set_yticklabels(INSTR_LABELS, fontsize=9)
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046)
        for ii in range(3):
            for jj in range(3):
                v = mat[ii, jj]
                txt = f'{v:+.2f}' if v == v else 'NaN'
                ax.text(jj, ii, txt, ha='center', va='center',
                        fontsize=10, color='white' if abs(v) > 0.5 else 'black')

    plt.suptitle('Axis 2 Tier B — 3×3 Correlation Matrices', fontsize=13)
    plt.tight_layout()
    out = os.path.join(out_dir, 'matrix_heatmap.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved matrix_heatmap.png to {out_dir}")


# ── main ─────────────────────────────────────────────────────────────────────

def run_analysis(results_dir: str, out_dir: Optional[str] = None, n_boot: int = 1000):
    if out_dir is None:
        out_dir = os.path.join(results_dir, 'analysis')
    os.makedirs(out_dir, exist_ok=True)

    df = load_all_metrics(results_dir)

    # Drop rows with all-NaN instruments (incomplete runs)
    key_cols = INSTRUMENTS + COMPONENTS
    df = df.dropna(subset=key_cols, how='all')
    print(f"After dropping all-NaN rows: {len(df)} rows")

    print("\nComputing 3×3 correlation matrices...")
    matrix_results = compute_3x3_matrix(df, n_boot=n_boot)

    # Save raw matrix data
    with open(os.path.join(out_dir, 'matrix_results.json'), 'w') as f:
        json.dump(matrix_results, f, indent=2, default=lambda x: float(x))

    # Write results.md
    verdict_text = evaluate_hypotheses(matrix_results, df)
    results_md = os.path.join(out_dir, 'results.md')
    with open(results_md, 'w') as f:
        f.write(verdict_text)
    print(f"\nWrote {results_md}")
    print(verdict_text[:2000])

    # Plots
    plot_3x3_scatter(df, out_dir)
    plot_matrix_heatmap(matrix_results, out_dir)

    return matrix_results, df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Axis 2 Tier B Analysis')
    parser.add_argument('--results_dir', required=True,
                        help='Root directory containing axis2_metrics.csv files')
    parser.add_argument('--out_dir', default=None,
                        help='Output directory (default: results_dir/analysis)')
    parser.add_argument('--n_boot', type=int, default=1000,
                        help='Bootstrap samples for CIs')
    args = parser.parse_args()
    run_analysis(args.results_dir, args.out_dir, args.n_boot)
