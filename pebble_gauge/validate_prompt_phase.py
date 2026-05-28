#!/usr/bin/env python3
import argparse
import csv
import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

try:
    from scipy import stats as scipy_stats
except Exception:
    scipy_stats = None


def read_csv_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    try:
        with path.open(newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def as_float(value, default=math.nan):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def as_int(value, default=-1):
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def gauge_family(gauge):
    return "identity" if str(gauge).startswith("no_tanh") else "tanh"


def latest_long_metric(rows, key, split=None):
    best_step = -1
    best_value = math.nan
    for row in rows:
        if row.get("key") != key:
            continue
        if split is not None and row.get("split") != split:
            continue
        step = as_int(row.get("step"), -1)
        if step >= best_step:
            best_step = step
            best_value = as_float(row.get("value"))
    return best_value, best_step


def metric_values(rows, prefix=None, split=None):
    values = defaultdict(list)
    for row in rows:
        key = row.get("key", "")
        if prefix is not None and not key.startswith(prefix):
            continue
        if split is not None and row.get("split") != split:
            continue
        value = as_float(row.get("value"))
        if math.isfinite(value):
            values[key].append((as_int(row.get("step"), -1), value))
    return values


def final_eval_summary(eval_rows, metric_rows, step_window):
    max_step = -1
    for row in eval_rows:
        max_step = max(max_step, as_int(row.get("step"), -1))
    for row in metric_rows:
        max_step = max(max_step, as_int(row.get("step"), -1))

    values = []
    if eval_rows:
        for row in eval_rows:
            step = as_int(row.get("step"), -1)
            if max_step >= 0 and step < max_step - step_window:
                continue
            value = as_float(row.get("true_episode_reward"))
            if math.isfinite(value):
                values.append(value)

    if not values:
        for row in metric_rows:
            if row.get("split") != "eval" or row.get("key") != "true_episode_reward":
                continue
            step = as_int(row.get("step"), -1)
            if max_step >= 0 and step < max_step - step_window:
                continue
            value = as_float(row.get("value"))
            if math.isfinite(value):
                values.append(value)

    mean_true = float(np.mean(values)) if values else math.nan
    return max_step, mean_true, len(values)


def cohen_d(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return math.nan
    denom = len(a) + len(b) - 2
    if denom <= 0:
        return math.nan
    pooled = math.sqrt(((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1)) / denom)
    if pooled == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled)


def welch_p(a, b):
    if scipy_stats is None or len(a) < 2 or len(b) < 2:
        return math.nan
    return float(scipy_stats.ttest_ind(a, b, equal_var=False, nan_policy="omit").pvalue)


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_pairwise(final_rows):
    groups = defaultdict(list)
    for row in final_rows:
        if row["status"] != "complete" or not math.isfinite(row["mean_true_return"]):
            continue
        key = (row["env"], row["alpha_mode"], row["method"], row["family"])
        groups[key].append(row)

    out = []
    for (env, alpha, method, family), rows in sorted(groups.items()):
        by_gauge = defaultdict(list)
        for row in rows:
            by_gauge[row["gauge"]].append(row["mean_true_return"])
        for g1, g2 in combinations(sorted(by_gauge), 2):
            a = [x for x in by_gauge[g1] if math.isfinite(x)]
            b = [x for x in by_gauge[g2] if math.isfinite(x)]
            if not a or not b:
                continue
            out.append({
                "env": env,
                "alpha_mode": alpha,
                "method": method,
                "family": family,
                "gauge_a": g1,
                "gauge_b": g2,
                "n_a": len(a),
                "n_b": len(b),
                "mean_a": float(np.mean(a)),
                "mean_b": float(np.mean(b)),
                "diff_a_minus_b": float(np.mean(a) - np.mean(b)),
                "cohens_d": cohen_d(a, b),
                "p_value": welch_p(a, b),
            })
    return out


def recommend(phase, run_rows, pairwise_rows):
    expected = len(run_rows)
    complete = sum(1 for r in run_rows if r["status"] == "complete")
    incomplete = expected - complete
    rrm_fail = sum(1 for r in run_rows if r["method"] == "rrm" and r.get("rrm_bit_identity_passed") == 0.0)
    perfg_runs = [r for r in run_rows if r["method"] == "perfg" and r["status"] == "complete"]
    perfg_applied = sum(1 for r in perfg_runs if r.get("perf_applied_count", 0) > 0)
    nonzero_shift_runs = sum(1 for r in run_rows if r.get("nonzero_shift_count", 0) > 0)

    effect_rows = [
        r for r in pairwise_rows
        if r["method"] == "perfg" and math.isfinite(r.get("cohens_d", math.nan)) and abs(r["cohens_d"]) >= 0.5
    ]
    p_rows = [
        r for r in pairwise_rows
        if r["method"] == "perfg" and math.isfinite(r.get("p_value", math.nan)) and r["p_value"] < 0.1
    ]

    if incomplete > 0 or rrm_fail > 0:
        return "rerun_or_debug", (
            f"Do not advance yet: {incomplete}/{expected} runs are incomplete and "
            f"{rrm_fail} RRM runs failed the bit-identity check. Rerun missing cells or inspect logs first."
        )

    if phase == "phase1":
        if effect_rows or p_rows:
            return "advance_phase2", (
                "Advance to phase2: the pipeline completed and at least one PerfG within-family gauge comparison "
                "shows a moderate effect size or p<0.1."
            )
        if perfg_runs and perfg_applied < max(1, len(perfg_runs) // 2):
            return "inspect_perfg", (
                "Hold before phase2: runs completed, but PerfG corrections were rarely applied. Check recent-sample cache, "
                "reward-buffer fill, and perf diagnostics."
            )
        if nonzero_shift_runs == 0:
            return "inspect_gauge", "Hold before phase2: runs completed, but gauge shifts look identically zero."
        return "advance_phase2_low_signal", (
            "Phase1 completed with no strong gauge effect. Phase2 is still reasonable if the goal is to test alpha/env "
            "interactions, but the current phase does not yet show a clear signal."
        )

    if phase == "phase2":
        if effect_rows or p_rows:
            return "advance_phase3", (
                "Advance to phase3: phase2 found at least one moderate/significant PerfG gauge comparison. "
                "Phase3 should increase seed count for the affected env/alpha/family cells."
            )
        return "stop_or_redesign", (
            "Do not spend full phase3 by default: phase2 completed but did not show a clear PerfG gauge effect. "
            "Consider teacher/query/coverage-control variants before a six-seed full sweep."
        )

    return "finalize", "Phase3 completed. Use the pairwise table and final-window means as the main report inputs."


def main():
    parser = argparse.ArgumentParser(description="Validate prompt-gauge phase results and recommend the next phase.")
    parser.add_argument("--phase", required=True, choices=["phase1", "phase2", "phase3"])
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--done-step", type=int, default=990000)
    parser.add_argument("--step-window", type=int, default=200000)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.load(open(args.manifest, "r", encoding="utf-8"))

    run_rows = []
    for cell in manifest:
        run_dir = Path(cell.get("run_dir") or Path(args.root) / cell["run_name"])
        eval_rows = read_csv_rows(run_dir / "eval.csv")
        train_rows = read_csv_rows(run_dir / "train.csv")
        metric_rows = read_csv_rows(run_dir / "gauge_metrics.csv")
        max_step, mean_true, n_eval_points = final_eval_summary(eval_rows, metric_rows, args.step_window)
        train_max_step = max([as_int(r.get("step"), -1) for r in train_rows] + [-1])
        max_step = max(max_step, train_max_step)
        status = "complete" if max_step >= args.done_step else "incomplete"
        if not run_dir.exists():
            status = "missing"

        rrm_pass, _ = latest_long_metric(metric_rows, "rrm_bit_identity_passed", split="eval")
        perf_applied = metric_values(metric_rows, prefix="perf_applied")
        shifts = metric_values(metric_rows, prefix="gauge_shift_")
        nonzero_shift_count = 0
        for key, vals in shifts.items():
            if key == "gauge_shift_buf_ref_gap":
                continue
            if vals and abs(vals[-1][1]) > 1e-6:
                nonzero_shift_count += 1

        run_rows.append({
            "phase": args.phase,
            "env": cell["env"],
            "method": cell["method"],
            "gauge": cell["gauge"],
            "family": gauge_family(cell["gauge"]),
            "alpha_mode": cell["alpha_mode"],
            "seed": int(cell["seed"]),
            "run_name": cell["run_name"],
            "run_dir": str(run_dir),
            "status": status,
            "max_step": max_step,
            "mean_true_return": mean_true,
            "n_eval_points": n_eval_points,
            "rrm_bit_identity_passed": rrm_pass,
            "perf_applied_count": sum(len(v) for v in perf_applied.values()),
            "nonzero_shift_count": nonzero_shift_count,
        })

    pairwise_rows = build_pairwise(run_rows)
    rec_code, rec_text = recommend(args.phase, run_rows, pairwise_rows)

    write_csv(out_dir / "run_status.csv", run_rows, [
        "phase", "env", "method", "gauge", "family", "alpha_mode", "seed", "run_name", "run_dir",
        "status", "max_step", "mean_true_return", "n_eval_points", "rrm_bit_identity_passed",
        "perf_applied_count", "nonzero_shift_count",
    ])
    write_csv(out_dir / "pairwise_gauge_effects.csv", pairwise_rows, [
        "env", "alpha_mode", "method", "family", "gauge_a", "gauge_b", "n_a", "n_b",
        "mean_a", "mean_b", "diff_a_minus_b", "cohens_d", "p_value",
    ])

    summary = {
        "phase": args.phase,
        "expected_runs": len(run_rows),
        "complete_runs": sum(1 for r in run_rows if r["status"] == "complete"),
        "missing_runs": sum(1 for r in run_rows if r["status"] == "missing"),
        "incomplete_runs": sum(1 for r in run_rows if r["status"] == "incomplete"),
        "rrm_bit_identity_failures": sum(1 for r in run_rows if r["method"] == "rrm" and r.get("rrm_bit_identity_passed") == 0.0),
        "perfg_runs_with_correction": sum(1 for r in run_rows if r["method"] == "perfg" and r.get("perf_applied_count", 0) > 0),
        "runs_with_nonzero_shift": sum(1 for r in run_rows if r.get("nonzero_shift_count", 0) > 0),
        "recommendation": rec_code,
        "recommendation_text": rec_text,
    }
    json.dump(summary, open(out_dir / "summary.json", "w", encoding="utf-8"), indent=2)

    top = sorted(
        [r for r in pairwise_rows if r["method"] == "perfg"],
        key=lambda r: abs(r.get("cohens_d", 0.0)) if math.isfinite(r.get("cohens_d", math.nan)) else -1,
        reverse=True,
    )[:10]

    lines = [
        f"# {args.phase} prompt-gauge validation",
        "",
        f"- Expected runs: {summary['expected_runs']}",
        f"- Complete runs: {summary['complete_runs']}",
        f"- Missing runs: {summary['missing_runs']}",
        f"- Incomplete runs: {summary['incomplete_runs']}",
        f"- RRM bit-identity failures: {summary['rrm_bit_identity_failures']}",
        f"- PerfG runs with at least one correction: {summary['perfg_runs_with_correction']}",
        f"- Runs with nonzero gauge shifts: {summary['runs_with_nonzero_shift']}",
        "",
        f"## Recommendation: {rec_code}",
        rec_text,
        "",
        "## Top PerfG Gauge Comparisons",
    ]
    if top:
        lines.append("| env | alpha | family | gauge_a | gauge_b | diff | d | p |")
        lines.append("| --- | --- | --- | --- | --- | ---: | ---: | ---: |")
        for row in top:
            p_value = row.get("p_value", math.nan)
            lines.append(
                f"| {row['env']} | {row['alpha_mode']} | {row['family']} | {row['gauge_a']} | {row['gauge_b']} | "
                f"{row['diff_a_minus_b']:.4g} | {row['cohens_d']:.3g} | {p_value:.3g} |"
            )
    else:
        lines.append("No complete PerfG within-family comparisons yet.")
    lines.append("")
    lines.append("Generated files: `run_status.csv`, `pairwise_gauge_effects.csv`, `summary.json`.")
    (out_dir / "phase_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"wrote {out_dir / 'phase_report.md'}")


if __name__ == "__main__":
    main()
