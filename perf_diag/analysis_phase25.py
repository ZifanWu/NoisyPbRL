"""Phase 2.5 go/no-go analysis.

Compares K-resample decomposition between full_rm and small_rm, per env. Tests the
variance hypothesis: big RM has more scatter (variance), small RM is more
direction-stable (systematic). Per the addendum review:

  - Filter each run to κ̂₀ probes (low steps_since_relabel, bottom 20% quantile),
    so we read the variance-only regime, not staleness.
  - Use scatter² / total² (scale-free) — not raw scatter_norm — to compare across
    capacities (g0 norm differs by orders of magnitude).
  - Report ortho_cos_max distribution: if it's mostly > 0.5, the squared-norm
    decomposition is biased and the variance interpretation needs the caveat.
  - Bootstrap CIs over seeds for the (full − small) gap.

Run:
    python -m perf_diag.analysis_phase25
"""

from __future__ import annotations
import json
import os
import sys
import numpy as np
from collections import defaultdict

_THIS = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.environ.get("PD_OUT_DIR", os.path.join(_THIS, "runs"))


def load_trace(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def parse_name(fname: str):
    """Parse a JSONL filename. Supports three layouts (newest first):
        v3: task__capacity__regime__teacher_tag__steps{X}__seed{N}.jsonl
        v2: task__capacity__regime__teacher_tag__seed{N}.jsonl
        v1: task__capacity__regime__seed{N}.jsonl
    Detect by looking at the second-from-last token:
        starts with "steps" → v3
        starts with bt/eps/rational → v2
        else → v1
    Returns (task, capacity, regime, teacher_tag, steps_tag, seed).
    teacher_tag and/or steps_tag are None for older formats.
    """
    base = fname.replace(".jsonl", "")
    parts = base.split("__")
    seed = int(parts[-1].replace("seed", ""))
    second_last = parts[-2] if len(parts) >= 2 else ""
    if second_last.startswith("steps"):
        # v3 layout
        steps_tag = parts[-2][len("steps"):]
        teacher_tag = parts[-3]
        regime = parts[-4]
        capacity = parts[-5]
        task = "__".join(parts[:-5])
    elif second_last.startswith("bt") or second_last.startswith("eps") or second_last == "rational":
        # v2 layout
        steps_tag = None
        teacher_tag = parts[-2]
        regime = parts[-3]
        capacity = parts[-4]
        task = "__".join(parts[:-4])
    else:
        # v1 layout
        steps_tag = None
        teacher_tag = None
        regime = parts[-2]
        capacity = parts[-3]
        task = "__".join(parts[:-3])
    return task, capacity, regime, teacher_tag, steps_tag, seed


def per_run_summary(rows: list[dict], staleness_quantile: float = 0.20) -> dict | None:
    """Filter to κ̂₀ probes, return per-run scatter_frac / pair_cos / ortho_cos_max medians,
    plus the final-gold-eval and the total number of κ̂₀ probes (heldout sample size).
    """
    if not rows:
        return None
    # Phase 3 rows are the ones with kresample_systematic_norm present.
    kr_rows = [r for r in rows if "kresample_systematic_norm" in r]
    nan = float("nan")
    if not kr_rows:
        # Phase 2-only runs (PD_KRESAMPLE_EVERY=0): no kresample, but heldout_acc /
        # teacher_disagreement / final_gold are still computable. Fill kresample fields
        # with NaN so the downstream prints don't KeyError.
        hg = np.array([r.get("heldout_acc_gold", np.nan) for r in rows], dtype=np.float64)
        hg = hg[np.isfinite(hg)]
        ht = np.array([r.get("heldout_acc_teacher", np.nan) for r in rows], dtype=np.float64)
        ht = ht[np.isfinite(ht)]
        tdg = np.array([r.get("teacher_gold_disagreement", np.nan) for r in rows], dtype=np.float64)
        tdg = tdg[np.isfinite(tdg)]
        all_gold = np.array([r.get("gold_eval", np.nan) for r in rows], dtype=np.float64)
        all_gold = all_gold[np.isfinite(all_gold)]
        final_gold = float(np.mean(all_gold[-5:])) if all_gold.size >= 5 else (
            float(all_gold[-1]) if all_gold.size else nan
        )
        if hg.size and ht.size:
            n_min = min(hg.size, ht.size)
            gap = float(np.median(ht[:n_min] - hg[:n_min]))
        else:
            gap = nan
        return dict(
            n_kresample=0,
            median_scatter_frac=nan, median_syst_frac=nan, median_pair_cos=nan,
            median_ortho_cos_max=nan, median_g0_norm=nan, median_gperp_norm=nan,
            median_scatter_norm=nan, median_syst_norm=nan,
            median_heldout_acc_gold=float(np.median(hg)) if hg.size else nan,
            median_heldout_acc_teacher=float(np.median(ht)) if ht.size else nan,
            median_fitting_noise_gap=gap,
            teacher_disagreement_mean=float(np.mean(tdg)) if tdg.size else nan,
            final_gold=final_gold,
        )

    # Build steps_since_relabel array from kr_rows (some early rows can have NaN).
    sss = np.array([r.get("steps_since_relabel", np.nan) for r in kr_rows], dtype=np.float64)
    finite = np.isfinite(sss)
    if finite.sum() < 2:
        kappa0_mask = np.ones(len(kr_rows), dtype=bool)
    else:
        # Bottom quantile within this run's distribution of staleness.
        q = float(np.quantile(sss[finite], staleness_quantile))
        kappa0_mask = (sss <= q) | ~np.isfinite(sss)
    kr_rows_k0 = [r for r, m in zip(kr_rows, kappa0_mask) if m]
    if not kr_rows_k0:
        kr_rows_k0 = kr_rows

    def col(rs, k):
        a = np.array([r.get(k, np.nan) for r in rs], dtype=np.float64)
        return a[np.isfinite(a)]

    syst = col(kr_rows_k0, "kresample_systematic_norm")
    scat = col(kr_rows_k0, "kresample_scatter_norm")
    tot = col(kr_rows_k0, "kresample_total_sq_mean")
    pc = col(kr_rows_k0, "kresample_pairwise_cos")
    ocm = col(kr_rows_k0, "kresample_ortho_cos_max")
    g0 = col(kr_rows_k0, "g0_norm")
    gperp = col(kr_rows_k0, "gperp_norm")
    # Phase 2 (corrected): heldout accuracy vs gold (primary), vs teacher (legacy), and gap.
    hg = col(kr_rows_k0, "heldout_acc_gold")
    ht = col(kr_rows_k0, "heldout_acc_teacher")
    # Protocol §1: teacher's empirical disagreement with gold, run-wide (from all probes, not just k0).
    tdg = np.array([r.get("teacher_gold_disagreement", np.nan) for r in rows], dtype=np.float64)
    tdg = tdg[np.isfinite(tdg)]

    # scatter_frac per probe row = scatter² / total²
    pair_total_sq = tot  # already squared (mean of ‖ĝ⊥‖² over K)
    scatter_sq = scat ** 2
    syst_sq = syst ** 2
    scatter_frac = scatter_sq / np.maximum(pair_total_sq, 1e-30)
    syst_frac = syst_sq / np.maximum(pair_total_sq, 1e-30)

    # Final gold from ALL rows (not just kresample probes), EMA-smoothed last-5.
    all_gold = np.array([r.get("gold_eval", np.nan) for r in rows], dtype=np.float64)
    all_gold = all_gold[np.isfinite(all_gold)]
    final_gold = float(np.mean(all_gold[-5:])) if all_gold.size >= 5 else (float(all_gold[-1]) if all_gold.size else float("nan"))

    # Phase 2 (corrected) summaries
    median_heldout_gold = float(np.median(hg)) if hg.size else float("nan")
    median_heldout_teacher = float(np.median(ht)) if ht.size else float("nan")
    # Fitting-noise gap: how much higher accuracy is on teacher labels vs gold labels.
    # Large gap = RM is fitting noise (matches teacher's specific noise pattern but
    # not gold). Should be larger for big_rm under uniform teacher.
    if hg.size and ht.size:
        gap_arr = ht[: min(hg.size, ht.size)] - hg[: min(hg.size, ht.size)]
        median_fitting_noise_gap = float(np.median(gap_arr))
    else:
        median_fitting_noise_gap = float("nan")

    # Teacher disagreement-with-gold (run-level: mean over all probes that logged it).
    teacher_disagreement_mean = float(np.mean(tdg)) if tdg.size else float("nan")

    return dict(
        n_kresample=int(len(kr_rows_k0)),
        median_scatter_frac=float(np.median(scatter_frac)) if scatter_frac.size else float("nan"),
        median_syst_frac=float(np.median(syst_frac)) if syst_frac.size else float("nan"),
        median_pair_cos=float(np.median(pc)) if pc.size else float("nan"),
        median_ortho_cos_max=float(np.median(ocm)) if ocm.size else float("nan"),
        median_g0_norm=float(np.median(g0)) if g0.size else float("nan"),
        median_gperp_norm=float(np.median(gperp)) if gperp.size else float("nan"),
        median_scatter_norm=float(np.median(scat)) if scat.size else float("nan"),
        median_syst_norm=float(np.median(syst)) if syst.size else float("nan"),
        median_heldout_acc_gold=median_heldout_gold,
        median_heldout_acc_teacher=median_heldout_teacher,
        median_fitting_noise_gap=median_fitting_noise_gap,
        teacher_disagreement_mean=teacher_disagreement_mean,
        final_gold=final_gold,
    )


def bootstrap_mean_diff(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, alpha: float = 0.05) -> tuple[float, float, float]:
    """Bootstrap mean(a) − mean(b) over their own seeds. Returns (mean_diff, ci_lo, ci_hi)."""
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(0)
    diffs = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        diffs[i] = rng.choice(a, size=a.size, replace=True).mean() - rng.choice(b, size=b.size, replace=True).mean()
    return float(diffs.mean()), float(np.quantile(diffs, alpha / 2)), float(np.quantile(diffs, 1 - alpha / 2))


def main():
    # Gather per-run summaries by (task, capacity).
    by_cell = defaultdict(list)  # (task, capacity) -> list of per-run dict (one per seed)
    for fname in sorted(os.listdir(RUNS_DIR)):
        if not fname.endswith(".jsonl"):
            continue
        if fname.startswith("smoke_"):
            continue
        try:
            task, capacity, regime, teacher_tag, steps_tag, seed = parse_name(fname)
        except Exception:
            continue
        rows = load_trace(os.path.join(RUNS_DIR, fname))
        summary = per_run_summary(rows)
        if summary is None:
            continue
        summary["seed"] = seed
        summary["regime"] = regime
        summary["teacher_tag"] = teacher_tag or "unknown"
        summary["steps_tag"] = steps_tag or "unknown"
        # Group by (task, capacity, teacher_tag, steps_tag) so neither cross-teacher
        # nor cross-horizon sweeps are pooled.
        by_cell[(task, capacity, teacher_tag or "unknown", steps_tag or "unknown")].append(summary)

    if not by_cell:
        print("No Phase 2.5 JSONLs found.")
        return

    # Print per-cell tables.
    print("=" * 100)
    print("Phase 2.5 — per-cell summaries (κ̂₀ regime, bottom 20% quantile of steps_since_relabel)")
    print("=" * 100)
    for (task, capacity, teacher, steps), runs in sorted(by_cell.items()):
        print(f"\n[{task} / {capacity} / teacher={teacher} / steps={steps}]  n_seeds={len(runs)}")
        print(f"  {'seed':<5} {'n_kr':<5} {'scat_frac':<10} {'syst_frac':<10} {'pair_cos':<9} "
              f"{'ortho_max':<10} {'h_gold':<8} {'h_tea':<8} {'fit_gap':<8} {'tea_dis':<8} "
              f"{'final_gold':<10}")
        for r in sorted(runs, key=lambda x: x["seed"]):
            print(f"  {r['seed']:<5} {r['n_kresample']:<5} "
                  f"{r['median_scatter_frac']:<10.3f} {r['median_syst_frac']:<10.3f} "
                  f"{r['median_pair_cos']:<9.3f} {r['median_ortho_cos_max']:<10.3f} "
                  f"{r['median_heldout_acc_gold']:<8.3f} {r['median_heldout_acc_teacher']:<8.3f} "
                  f"{r['median_fitting_noise_gap']:<+8.3f} "
                  f"{r['teacher_disagreement_mean']:<8.3f} "
                  f"{r['final_gold']:<10.1f}")

    # Compute per-(task, teacher) gap full_rm − small_rm with bootstrap CIs.
    # Cross-teacher pooling is NOT done (per protocol §6 guard).
    print("\n" + "=" * 100)
    print("Phase 2.5 — go/no-go gap (full_rm − small_rm), bootstrap 95% CI over seeds,")
    print("            grouped per (task, teacher, steps)")
    print("=" * 100)
    task_teacher_steps = sorted({(t, tc, s) for (t, _, tc, s) in by_cell})
    for task, teacher, steps in task_teacher_steps:
        full = by_cell.get((task, "full_rm", teacher, steps))
        small = by_cell.get((task, "small_rm", teacher, steps))
        if not full or not small:
            print(f"\n[{task} / teacher={teacher} / steps={steps}]  missing one of (full_rm, small_rm)")
            continue
        print(f"\n[{task} / teacher={teacher} / steps={steps}]  full_rm n_seeds={len(full)}  small_rm n_seeds={len(small)}")

        for key, label in [
            ("median_scatter_frac",        "scatter_frac (variance share)"),
            ("median_syst_frac",           "syst_frac    (systematic share)"),
            ("median_pair_cos",            "pair_cos"),
            ("median_ortho_cos_max",       "ortho_cos_max"),
            ("median_heldout_acc_gold",    "heldout_acc_gold (primary)"),
            ("median_heldout_acc_teacher", "heldout_acc_teacher (legacy)"),
            ("median_fitting_noise_gap",   "fitting_noise_gap (teacher − gold)"),
            ("teacher_disagreement_mean",  "teacher_disagreement_with_gold"),
            ("final_gold",                 "final_gold"),
        ]:
            a = np.array([r[key] for r in full], dtype=np.float64)
            b = np.array([r[key] for r in small], dtype=np.float64)
            mean_a = float(np.nanmean(a))
            mean_b = float(np.nanmean(b))
            diff, lo, hi = bootstrap_mean_diff(a, b)
            sig = "***" if (lo > 0 and hi > 0) or (lo < 0 and hi < 0) else ""
            print(f"  {label:<40}  full={mean_a:>8.3f}  small={mean_b:>8.3f}  "
                  f"diff={diff:>+7.3f}  [{lo:>+7.3f}, {hi:>+7.3f}]  {sig}")

    # Decision summary.
    print("\n" + "=" * 100)
    print("Phase 2.5 — DECISION ROW")
    print("=" * 100)
    print("Variance hypothesis predicts: scatter_frac[full] > scatter_frac[small]  AND")
    print("                              pair_cos[small] > pair_cos[full]  AND")
    print("                              final_gold[small] > final_gold[full]  (capacity anomaly)")
    print("Orthogonality assertion:      ortho_cos_max consistently < 0.5  (else decomp biased)")
    print("Per protocol §1: VALID only if teacher_disagreement_with_gold is matched across teachers.")
    print("                If you only ran one teacher type, look at the raw disagreement column to")
    print("                know what noise level you actually had.")
    for task, teacher, steps in task_teacher_steps:
        full = by_cell.get((task, "full_rm", teacher, steps))
        small = by_cell.get((task, "small_rm", teacher, steps))
        if not full or not small:
            continue
        sf_full = float(np.nanmean([r["median_scatter_frac"] for r in full]))
        sf_small = float(np.nanmean([r["median_scatter_frac"] for r in small]))
        pc_full = float(np.nanmean([r["median_pair_cos"] for r in full]))
        pc_small = float(np.nanmean([r["median_pair_cos"] for r in small]))
        gold_full = float(np.nanmean([r["final_gold"] for r in full]))
        gold_small = float(np.nanmean([r["final_gold"] for r in small]))
        ocm_full = float(np.nanmean([r["median_ortho_cos_max"] for r in full]))
        ocm_small = float(np.nanmean([r["median_ortho_cos_max"] for r in small]))
        tdg_full = float(np.nanmean([r["teacher_disagreement_mean"] for r in full]))
        tdg_small = float(np.nanmean([r["teacher_disagreement_mean"] for r in small]))

        var_signal = "✓" if sf_full > sf_small else "✗"
        cos_signal = "✓" if pc_small > pc_full else "✗"
        anomaly    = "✓" if gold_small > gold_full else "✗"
        ortho_ok   = "✓" if max(ocm_full, ocm_small) < 0.5 else "⚠"
        tdg = (tdg_full + tdg_small) / 2
        print(f"\n[{task} / teacher={teacher} / steps={steps}]  (mean teacher_disagreement_with_gold ≈ {tdg:.3f})")
        print(f"  variance hypothesis (scatter_frac full > small):  {var_signal}")
        print(f"  direction stability (pair_cos small > full):       {cos_signal}")
        print(f"  capacity anomaly (final_gold small > full):        {anomaly}")
        print(f"  orthogonality (max ortho_cos_max < 0.5):           {ortho_ok}")


if __name__ == "__main__":
    main()
