"""Post-hoc analysis for the dynamics and PG-RLHF experiments.

Reads exp/axis2_dynamics_pg/dynamics/<env>/.../seed*/dynamics_fd.csv plus the
TB-or-CSV true_episode_reward stream, and produces:

  dynamics_summary.md  — per-seed mean(J_true) in P0/P1/P2 phases, plus the
                         ρ_FD and κ_FD trajectory (with phase markers).
  pg_rlhf_summary.md   — per-seed mean(J_true) at end-of-training across
                         method={standard, pg_fd, pg_hvp}, with Δ(method−std).
  *.png                — trajectory plots.

Run:
  conda run -n bpref python -m axis2_dynamics_pg.analysis \
      --root /home/zifan/NoisyPbRL/exp/axis2_dynamics_pg \
      --kind dynamics
  conda run -n bpref python -m axis2_dynamics_pg.analysis \
      --root /home/zifan/NoisyPbRL/exp/axis2_dynamics_pg \
      --kind pg_rlhf

The script is intentionally lightweight (pandas + matplotlib) and tolerant
of partial runs.
"""
import argparse
import csv
import json
import os
from glob import glob
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Tiny helpers (no pandas dependency, more portable in this codebase)
# ---------------------------------------------------------------------------

def read_csv(path: str) -> Tuple[List[str], List[Dict[str, str]]]:
    if not os.path.exists(path):
        return [], []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    return fieldnames, rows


def cols_as_float(rows: List[Dict[str, str]], key: str) -> np.ndarray:
    vals: List[float] = []
    for r in rows:
        v = r.get(key, '')
        try:
            vals.append(float(v))
        except (TypeError, ValueError):
            vals.append(float('nan'))
    return np.asarray(vals)


def read_train_log(seed_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (steps, J_true, J_RM) from train.log if present, else empty.

    The PEBBLE logger writes line-delimited JSON to train.log under each work_dir.
    """
    log_path = os.path.join(seed_dir, 'train.log')
    if not os.path.exists(log_path):
        return np.array([]), np.array([]), np.array([])
    steps, jt, jrm = [], [], []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if 'true_episode_reward' in row and 'step' in row:
                steps.append(int(row['step']))
                jt.append(float(row['true_episode_reward']))
                jrm.append(float(row.get('episode_reward', float('nan'))))
    return np.asarray(steps), np.asarray(jt), np.asarray(jrm)


# ---------------------------------------------------------------------------
# Dynamics analysis
# ---------------------------------------------------------------------------

def find_dynamics_seeds(root: str) -> List[str]:
    """seed dirs under root/dynamics/<env>/.../seed*."""
    return sorted(glob(os.path.join(root, 'dynamics', '*', '*', 'seed*')))


def analyze_dynamics(root: str, out_path: Optional[str] = None) -> None:
    seed_dirs = find_dynamics_seeds(root)
    if not seed_dirs:
        print(f"No dynamics seeds found under {root}/dynamics/.")
        return
    print(f"Found {len(seed_dirs)} dynamics seed dirs.")

    lines = [f"# Dynamics experiment summary\n", f"Root: {root}", ""]
    rows: List[Dict[str, float]] = []

    for sd in seed_dirs:
        fd_path = os.path.join(sd, 'dynamics_fd.csv')
        fields, fd_rows = read_csv(fd_path)
        if not fd_rows:
            print(f"  skip {sd}: no dynamics_fd.csv")
            continue

        steps_fd = cols_as_float(fd_rows, 'step')
        rho_fd = cols_as_float(fd_rows, 'rho_fd')
        kappa_fd = cols_as_float(fd_rows, 'kappa_fd')
        phases = [r.get('phase', '') for r in fd_rows]

        # Average rho/kappa per phase
        p1_mask = np.array([p == 'p1_frozen' for p in phases])
        p2_mask = np.array([p == 'p2_resumed' for p in phases])
        p0_mask = ~p1_mask & ~p2_mask
        agg = {
            'seed_dir': sd,
            'rho_p0': float(np.nanmean(rho_fd[p0_mask])) if p0_mask.any() else float('nan'),
            'rho_p1': float(np.nanmean(rho_fd[p1_mask])) if p1_mask.any() else float('nan'),
            'rho_p2': float(np.nanmean(rho_fd[p2_mask])) if p2_mask.any() else float('nan'),
            'kappa_p0': float(np.nanmean(kappa_fd[p0_mask])) if p0_mask.any() else float('nan'),
            'kappa_p1': float(np.nanmean(kappa_fd[p1_mask])) if p1_mask.any() else float('nan'),
            'kappa_p2': float(np.nanmean(kappa_fd[p2_mask])) if p2_mask.any() else float('nan'),
        }

        # J_true at end-of-each-phase (from train.log)
        steps_log, jt, jrm = read_train_log(sd)
        if steps_log.size > 0:
            # Get phase boundaries from phase_log.csv (if available)
            p1_boundary, p2_boundary = _phase_boundaries(sd)
            agg['j_true_p0_mean'] = float(np.nanmean(
                jt[steps_log < p1_boundary])) if p1_boundary else float('nan')
            agg['j_true_p1_mean'] = float(np.nanmean(
                jt[(steps_log >= p1_boundary) & (steps_log < p2_boundary)])) \
                if p1_boundary and p2_boundary else float('nan')
            agg['j_true_p2_mean'] = float(np.nanmean(
                jt[steps_log >= p2_boundary])) if p2_boundary else float('nan')
        rows.append(agg)
        print(f"  {sd}: "
              f"rho_p1={agg['rho_p1']:.3f} rho_p2={agg['rho_p2']:.3f} "
              f"kappa_p1={agg['kappa_p1']:.3f} kappa_p2={agg['kappa_p2']:.3f}")

        # Plot trajectory for this seed
        _plot_dynamics_seed(sd, steps_fd, rho_fd, kappa_fd,
                             steps_log, jt, jrm)

    # Summary table
    lines.append("## Per-seed summary (ρ_FD, κ_FD by phase)\n")
    lines.append("| seed_dir | ρ_p1 | ρ_p2 | κ_p1 | κ_p2 | J_true_p0 | J_true_p1 | J_true_p2 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(f"| {os.path.basename(r['seed_dir'])} | "
                     f"{r['rho_p1']:.3f} | {r['rho_p2']:.3f} | "
                     f"{r['kappa_p1']:.3f} | {r['kappa_p2']:.3f} | "
                     f"{r.get('j_true_p0_mean', float('nan')):.4g} | "
                     f"{r.get('j_true_p1_mean', float('nan')):.4g} | "
                     f"{r.get('j_true_p2_mean', float('nan')):.4g} |")

    summary = "\n".join(lines)
    if out_path is None:
        out_path = os.path.join(root, 'dynamics_summary.md')
    with open(out_path, 'w') as f:
        f.write(summary)
    print(f"\nWrote {out_path}")


def _phase_boundaries(seed_dir: str) -> Tuple[Optional[int], Optional[int]]:
    """Return (phase1_freeze_step, phase2_resume_step) read from phase_log.csv
    or from the hydra config dump if available."""
    pl = os.path.join(seed_dir, 'phase_log.csv')
    _, rows = read_csv(pl)
    p1, p2 = None, None
    for r in rows:
        if r.get('phase') == 'p1_frozen' and p1 is None:
            p1 = int(r.get('step', 0))
        if r.get('phase') == 'p2_resumed' and p2 is None:
            p2 = int(r.get('step', 0))
    return p1, p2


def _plot_dynamics_seed(sd: str, steps_fd, rho_fd, kappa_fd,
                         steps_log, jt, jrm):
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    p1, p2 = _phase_boundaries(sd)

    ax = axes[0]
    if steps_log.size:
        ax.plot(steps_log, jt, color='tab:blue', label='J_true (episode)')
        ax.plot(steps_log, jrm, color='tab:orange', alpha=0.5, label='J_RM (episode)')
    if p1: ax.axvline(p1, ls='--', color='k', alpha=0.5, label=f'P1 freeze ({p1})')
    if p2: ax.axvline(p2, ls='--', color='r', alpha=0.5, label=f'P2 resume ({p2})')
    ax.set_ylabel('episode reward')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(steps_fd, rho_fd, color='tab:red')
    ax.axhline(0, color='k', alpha=0.5, lw=0.5)
    if p1: ax.axvline(p1, ls='--', color='k', alpha=0.5)
    if p2: ax.axvline(p2, ls='--', color='r', alpha=0.5)
    ax.set_ylabel('ρ_FD')
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(steps_fd, kappa_fd, color='tab:green')
    if p1: ax.axvline(p1, ls='--', color='k', alpha=0.5)
    if p2: ax.axvline(p2, ls='--', color='r', alpha=0.5)
    ax.set_xlabel('step'); ax.set_ylabel('κ_FD')
    ax.grid(True, alpha=0.3)

    out = os.path.join(sd, 'dynamics_plot.png')
    plt.tight_layout()
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# PG-RLHF analysis
# ---------------------------------------------------------------------------

def find_pg_seeds(root: str) -> List[Tuple[str, str]]:
    """List of (method, seed_dir) for runs under root/pg_rlhf/<env>/<method>/seed*."""
    out = []
    for sd in sorted(glob(os.path.join(root, 'pg_rlhf', '*', '*', 'seed*'))):
        parts = sd.split(os.sep)
        if len(parts) < 4:
            continue
        method = parts[-2]
        out.append((method, sd))
    return out


def analyze_pg_rlhf(root: str, out_path: Optional[str] = None) -> None:
    seeds = find_pg_seeds(root)
    if not seeds:
        print(f"No PG-RLHF seeds found under {root}/pg_rlhf/.")
        return

    by_method: Dict[str, List[float]] = {}
    rows: List[Dict[str, float]] = []
    for method, sd in seeds:
        steps_log, jt, jrm = read_train_log(sd)
        if steps_log.size == 0:
            print(f"  skip {sd}: no train.log")
            continue
        n = max(10, steps_log.size // 20)  # last 5%
        jt_end = float(np.nanmean(jt[-n:]))
        jrm_end = float(np.nanmean(jrm[-n:]))
        by_method.setdefault(method, []).append(jt_end)
        rows.append({'seed_dir': sd, 'method': method,
                     'jt_end': jt_end, 'jrm_end': jrm_end})
        print(f"  {sd}: method={method} J_true_end={jt_end:.4g} "
              f"J_RM_end={jrm_end:.4g}")

    # Build summary
    lines = ["# PG-RLHF experiment summary\n", f"Root: {root}", ""]
    lines.append("## Per-seed end-of-training (last 5% mean)\n")
    lines.append("| method | seed | J_true_end | J_RM_end |")
    lines.append("|---|---|---|---|")
    for r in rows:
        lines.append(f"| {r['method']} | {os.path.basename(r['seed_dir'])} | "
                     f"{r['jt_end']:.4g} | {r['jrm_end']:.4g} |")

    lines.append("\n## Method aggregate")
    lines.append("| method | n | mean J_true_end | std |")
    lines.append("|---|---|---|---|")
    method_mean: Dict[str, float] = {}
    for m in sorted(by_method.keys()):
        vals = np.asarray(by_method[m])
        method_mean[m] = float(np.nanmean(vals))
        lines.append(f"| {m} | {len(vals)} | {method_mean[m]:.4g} | "
                     f"{float(np.nanstd(vals)):.4g} |")

    if 'standard' in method_mean:
        lines.append("\n## Δ J_true_end vs standard (positive = better)")
        for m in sorted(method_mean.keys()):
            if m == 'standard':
                continue
            d = method_mean[m] - method_mean['standard']
            lines.append(f"  {m} − standard = {d:+.4g}")

        std_mean = method_mean['standard']
        verdict = "INCONCLUSIVE"
        for m, v in method_mean.items():
            if m == 'standard':
                continue
            if v < std_mean - 0.02 * max(abs(std_mean), 1.0):
                verdict = f"PG-RLHF ({m}) WORSE than Standard → hacking-consistent"
                break
            if v > std_mean + 0.02 * max(abs(std_mean), 1.0):
                verdict = f"PG-RLHF ({m}) BETTER than Standard → no hacking"
        lines.append(f"\n**Verdict:** {verdict}")

    summary = "\n".join(lines)
    if out_path is None:
        out_path = os.path.join(root, 'pg_rlhf_summary.md')
    with open(out_path, 'w') as f:
        f.write(summary)
    print(f"\nWrote {out_path}")

    # Plot J_true trajectories overlaid by method
    _plot_pg_trajectories(seeds, root)


def _plot_pg_trajectories(seeds: List[Tuple[str, str]], root: str):
    colors = {'standard': 'tab:blue', 'pg_fd': 'tab:red', 'pg_hvp': 'tab:green'}
    fig, ax = plt.subplots(figsize=(9, 5))
    for method, sd in seeds:
        steps, jt, _ = read_train_log(sd)
        if steps.size == 0:
            continue
        ax.plot(steps, jt, color=colors.get(method, 'gray'), alpha=0.5, lw=0.8,
                label=f'{method} ({os.path.basename(sd)})')
    ax.set_xlabel('step'); ax.set_ylabel('J_true (episode)')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    ax.set_title('PG-RLHF: J_true trajectories by method')
    out = os.path.join(root, 'pg_rlhf_trajectories.png')
    plt.tight_layout()
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  wrote {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True,
                        help='Root dir under which dynamics/<env>/.../seed* or '
                             'pg_rlhf/<env>/<method>/seed* live')
    parser.add_argument('--kind', choices=['dynamics', 'pg_rlhf', 'both'],
                        default='both')
    args = parser.parse_args()

    if args.kind in ('dynamics', 'both'):
        analyze_dynamics(args.root)
    if args.kind in ('pg_rlhf', 'both'):
        analyze_pg_rlhf(args.root)


if __name__ == '__main__':
    main()
