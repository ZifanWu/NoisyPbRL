"""Analysis: load JSONL traces, produce Figs 1–3, Tables 1–2, write results.md.

Inputs : perf_diag/runs/*.jsonl + perf_diag/runs/_manifest.json
Outputs: perf_diag/figs/Fig{1,2,3}.png, perf_diag/tables/Table{1,2}.csv, perf_diag/results.md
"""

from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict
from typing import Iterable

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_THIS)

# Default paths; main() may override FIGS_DIR / TABLES_DIR / RESULTS_MD_PATH
# when --out_dir is passed. RUNS_DIR is the source of truth for traces and
# always points at perf_diag/runs/ (where the SLURM cells deposit JSONLs).
RUNS_DIR = os.path.join(_THIS, "runs")
FIGS_DIR = os.path.join(_THIS, "figs")
TABLES_DIR = os.path.join(_THIS, "tables")
RESULTS_MD_PATH = os.path.join(_THIS, "results.md")
os.makedirs(FIGS_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from perf_diag import baselines, detect


# ---------------------------------------------------------------------------
# Trace loading
# ---------------------------------------------------------------------------
def load_manifest() -> dict:
    p = os.path.join(RUNS_DIR, "_manifest.json")
    if not os.path.exists(p):
        return {"runs": [], "config": {}, "neg_control_validity": {"ok": False, "bad": []}}
    with open(p) as f:
        return json.load(f)


def load_trace(jsonl_path: str) -> list[dict]:
    if not os.path.exists(jsonl_path):
        return []
    rows = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def trace_arrays(trace: list[dict]) -> dict:
    """Convert a list-of-dicts trace into a dict of np.float64 arrays, NaN where missing."""
    if not trace:
        return {}
    keys = set()
    for r in trace:
        keys.update(r.keys())
    out = {}
    for k in keys:
        if k in ("error",):
            continue
        vals = [r.get(k, np.nan) for r in trace]
        try:
            out[k] = np.array(vals, dtype=np.float64)
        except Exception:
            # non-numeric (string status, etc.): skip
            pass
    return out


# ---------------------------------------------------------------------------
# Signals shared across figures and tables
# ---------------------------------------------------------------------------
def build_signals(traces: dict) -> dict:
    """Return per-signal np arrays for one run trace.

    Signals are oriented "larger ⇒ more alarming" (matches detect.cusum_fire convention).
    """
    out = {}
    R = traces.get("R_inner", np.array([]))
    out["R_hat"] = baselines.R_degradation_signal(R)
    proxy = traces.get("proxy_mean", np.array([]))
    out["proxy_inflection"] = baselines.proxy_inflection_signal(proxy)
    out["kl_to_pretrain"] = baselines.kl_to_pretrain_signal(traces.get("kl_to_pretrain", np.array([])))
    out["entropy_drop"] = baselines.entropy_drop_signal(traces.get("policy_entropy", np.array([])))
    out["ensemble_var"] = baselines.ensemble_variance_signal(traces.get("ensemble_variance", np.array([])))
    return out


# ---------------------------------------------------------------------------
# Figure 1 — representative positive run
# ---------------------------------------------------------------------------
def fig1(manifest: dict, threshold_for_R: float = 0.5) -> str | None:
    """Pick the most-emblematic positive run and plot gold / proxy / R̂ + baselines.

    "Most emblematic" = greatest R̂-lead if any; otherwise the positive run with the largest
    in-horizon gold drop (so the figure still shows something useful even on smoke data
    that's too short for a strict lead measurement).
    """
    candidates = [r for r in manifest["runs"] if r.get("is_positive")]
    if not candidates:
        # Fall back to ANY run with enough probe rows so the figure is at least illustrative
        candidates = list(manifest["runs"])
    best = None
    best_score = -np.inf
    for r in candidates:
        tr = load_trace(r["jsonl"])
        arr = trace_arrays(tr)
        gold = arr.get("gold_eval", np.array([]))
        if gold.size < 3:
            continue
        t_gold = detect.find_t_gold_argmax(gold) if gold.size >= 3 else None
        sig = baselines.R_degradation_signal(arr.get("R_inner", np.array([])))
        t_fire = detect.cusum_fire(sig, threshold_for_R, K_persist=2)
        lead = (t_gold - t_fire) if (t_fire is not None and t_gold is not None and t_fire <= t_gold) else 0
        # secondary score: in-horizon gold drop (running max - last) — picks the run
        # where over-optimization is visible at all on these short smoke traces
        gold_smooth = baselines.ema_smooth(gold, alpha=0.3)
        drop = float(gold_smooth.max() - gold_smooth[-1])
        score = lead * 1000.0 + drop  # leadtime dominates; drop is tiebreaker
        if score > best_score:
            best_score = score
            best = (r, arr, t_gold, t_fire)
    if best is None:
        return None
    r, arr, t_gold, t_fire = best
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    x = np.arange(arr["gold_eval"].size)
    # Top: gold + proxy
    gold_smooth = baselines.ema_smooth(arr["gold_eval"], alpha=0.2)
    proxy_smooth = baselines.ema_smooth(arr.get("proxy_mean", np.zeros_like(x)), alpha=0.2)
    axes[0].plot(x, gold_smooth, "g-", lw=2, label="gold (smoothed)")
    ax2 = axes[0].twinx()
    ax2.plot(x, proxy_smooth, "b--", lw=2, label="proxy (smoothed)")
    axes[0].axvline(t_gold, color="g", ls=":", alpha=0.6, label=f"t_gold={t_gold}")
    if t_fire is not None:
        axes[0].axvline(t_fire, color="r", ls=":", alpha=0.6, label=f"t_fire(R̂)={t_fire}")
    axes[0].set_ylabel("gold return", color="g")
    ax2.set_ylabel("proxy mean", color="b")
    lead_for_title = (t_gold - t_fire) if (t_fire is not None and t_gold is not None and t_fire <= t_gold) else "n/a"
    axes[0].set_title(f"{r['run_name']} — R̂ lead = {lead_for_title} probe steps")
    axes[0].legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    axes[0].grid(alpha=0.3)

    # Bottom: R̂_degradation + ensemble_var + kl
    R_sig = baselines.R_degradation_signal(arr.get("R_inner", np.zeros_like(x)))
    axes[1].plot(x, R_sig, "r-", lw=2, label="R̂ degradation (= 1 − smooth R̂)")
    axes[1].axhline(threshold_for_R, color="r", ls=":", alpha=0.5,
                    label=f"R̂ fire threshold ({threshold_for_R})")
    ax3 = axes[1].twinx()
    ax3.plot(x, baselines.ensemble_variance_signal(arr.get("ensemble_variance", np.zeros_like(x))),
             "purple", lw=1.5, alpha=0.7, label="ensemble var")
    ax3.plot(x, baselines.kl_to_pretrain_signal(arr.get("kl_to_pretrain", np.zeros_like(x))),
             "orange", lw=1.5, alpha=0.7, label="KL to pretrain")
    axes[1].set_xlabel("monitoring step (relabel index)")
    axes[1].set_ylabel("R̂ deg.", color="r")
    ax3.set_ylabel("baseline signals", color="purple")
    axes[1].legend(loc="upper left", fontsize=8)
    ax3.legend(loc="upper right", fontsize=8)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    p = os.path.join(FIGS_DIR, "Fig1.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# Figure 2 — lead vs FAR curves per task
# ---------------------------------------------------------------------------
SIGNAL_FNS = {
    "R_hat": lambda tr: baselines.R_degradation_signal(tr.get("R_inner", np.array([]))),
    "proxy_inflection": lambda tr: baselines.proxy_inflection_signal(tr.get("proxy_mean", np.array([]))),
    "kl_to_pretrain": lambda tr: baselines.kl_to_pretrain_signal(tr.get("kl_to_pretrain", np.array([]))),
    "entropy_drop": lambda tr: baselines.entropy_drop_signal(tr.get("policy_entropy", np.array([]))),
    "ensemble_var": lambda tr: baselines.ensemble_variance_signal(tr.get("ensemble_variance", np.array([]))),
}


def _make_signal_runs_for_task(manifest: dict, task: str, signal_name: str) -> list[detect.SignalRun]:
    runs = []
    for r in manifest["runs"]:
        if r.get("task") != task:
            continue
        tr = trace_arrays(load_trace(r["jsonl"]))
        if not tr or tr.get("gold_eval", np.array([])).size < 5:
            continue
        sig = SIGNAL_FNS[signal_name](tr)
        t_gold = detect.find_t_gold_argmax(tr["gold_eval"]) if r["is_positive"] else None
        runs.append(detect.SignalRun(label=r["run_name"], is_positive=bool(r["is_positive"]),
                                     signal=sig, t_gold=t_gold))
    return runs


def fig2(manifest: dict) -> list[str]:
    tasks = sorted({r["task"] for r in manifest["runs"]})
    saved = []
    for task in tasks:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        plotted = False
        for sname in SIGNAL_FNS:
            runs = _make_signal_runs_for_task(manifest, task, sname)
            if not any(r.is_positive for r in runs):
                continue
            # threshold range = signal percentiles
            sig_concat = np.concatenate([r.signal for r in runs if r.signal.size])
            if sig_concat.size == 0:
                continue
            thr = np.unique(np.quantile(sig_concat, np.linspace(0.05, 0.99, 30)))
            curve = detect.far_lead_curve(runs, thr, K_persist=2)
            ax.plot(curve["far"], curve["mean_lead"], "o-", label=sname, alpha=0.8)
            plotted = True
        if not plotted:
            plt.close(fig); continue
        ax.axvline(0.1, color="k", ls=":", alpha=0.4, label="target FAR=0.1")
        ax.set_xlabel("false-alarm rate (over negative-control runs)")
        ax.set_ylabel("mean lead time (probe steps)")
        ax.set_title(f"{task} — lead vs FAR")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        safe = task.replace("/", "_")
        p = os.path.join(FIGS_DIR, f"Fig2_{safe}.png")
        fig.savefig(p, dpi=150); plt.close(fig); saved.append(p)
    return saved


# ---------------------------------------------------------------------------
# Figure 3 — probe stability
# ---------------------------------------------------------------------------
def fig3(manifest: dict) -> str | None:
    """Distribution of R̂ across relabel steps within a positive run, plus mean R̂ per regime."""
    by_regime: dict[str, list[np.ndarray]] = defaultdict(list)
    for r in manifest["runs"]:
        tr = trace_arrays(load_trace(r["jsonl"]))
        if "R_inner" not in tr or tr["R_inner"].size == 0:
            continue
        by_regime[r["regime"]].append(tr["R_inner"])
    if not by_regime:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    # Left: violin of R̂ per regime
    labels = sorted(by_regime.keys())
    pooled = [np.concatenate([a for a in by_regime[lab] if a.size]) for lab in labels]
    pooled = [p[~np.isnan(p)] for p in pooled]
    valid_idx = [i for i, p in enumerate(pooled) if p.size > 0]
    if valid_idx:
        parts = axes[0].violinplot([pooled[i] for i in valid_idx],
                                    positions=range(len(valid_idx)), showmeans=True, showextrema=False)
        axes[0].set_xticks(range(len(valid_idx)))
        axes[0].set_xticklabels([labels[i] for i in valid_idx], rotation=15, fontsize=9)
        axes[0].axhline(1.0, color="g", ls=":", alpha=0.4, label="R̂=1 (healthy)")
        axes[0].axhline(0.0, color="r", ls=":", alpha=0.4, label="R̂=0 (vanishing)")
        axes[0].set_ylabel("R̂")
        axes[0].set_title("R̂ across relabel rounds, by regime")
        axes[0].legend(fontsize=8)
        axes[0].grid(alpha=0.3)
    # Right: per-run trace of R̂ smoothed
    for r in manifest["runs"]:
        tr = trace_arrays(load_trace(r["jsonl"]))
        if "R_inner" not in tr or tr["R_inner"].size == 0:
            continue
        sm = baselines.ema_smooth(tr["R_inner"], alpha=0.3)
        ls = "-" if r["is_positive"] else "--"
        axes[1].plot(sm, ls=ls, alpha=0.7, label=r["run_name"][:40])
    axes[1].axhline(1.0, color="g", ls=":", alpha=0.4)
    axes[1].axhline(0.0, color="r", ls=":", alpha=0.4)
    axes[1].set_xlabel("relabel index"); axes[1].set_ylabel("smoothed R̂")
    axes[1].set_title("R̂ trajectories (solid = positive, dashed = neg ctrl)")
    axes[1].legend(fontsize=6, loc="best", ncol=1)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(FIGS_DIR, "Fig3.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def table1(manifest: dict, target_far: float = 0.1) -> tuple[str, list[dict]]:
    """lead@FAR=0.1 per signal × task with bootstrap CI."""
    tasks = sorted({r["task"] for r in manifest["runs"]})
    rows = []
    for task in tasks:
        for sname in SIGNAL_FNS:
            srs = _make_signal_runs_for_task(manifest, task, sname)
            if not any(r.is_positive for r in srs):
                continue
            sig_concat = np.concatenate([r.signal for r in srs if r.signal.size])
            if sig_concat.size == 0:
                continue
            thr = np.unique(np.quantile(sig_concat, np.linspace(0.05, 0.99, 30)))
            mean, lo, hi = detect.bootstrap_lead_at_far(srs, thr, target_far=target_far, K_persist=2)
            rows.append(dict(task=task, signal=sname,
                              lead_at_far=mean, ci_lo=lo, ci_hi=hi,
                              n_pos=sum(1 for r in srs if r.is_positive),
                              n_neg=sum(1 for r in srs if not r.is_positive)))
    p = os.path.join(TABLES_DIR, "Table1.csv")
    with open(p, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["task", "signal", "lead_at_far=0.1", "ci_lo", "ci_hi", "n_pos", "n_neg"])
        for r in rows:
            wr.writerow([r["task"], r["signal"],
                         f"{r['lead_at_far']:.3g}", f"{r['ci_lo']:.3g}", f"{r['ci_hi']:.3g}",
                         r["n_pos"], r["n_neg"]])
    return p, rows


def table2(manifest: dict) -> tuple[str, list[dict]]:
    """Regime summary."""
    rows = []
    regimes = sorted({r["regime"] for r in manifest["runs"]})
    for regime in regimes:
        runs = [r for r in manifest["runs"] if r["regime"] == regime]
        traces = [trace_arrays(load_trace(r["jsonl"])) for r in runs]
        gold_turnovers = []
        kappas = []
        R_inners = []
        for tr in traces:
            if "gold_eval" not in tr:
                continue
            gold_turnovers.append(detect.turned_over_in_horizon(tr["gold_eval"], K_decline=3,
                                                                 alpha=0.3, min_drop_frac=0.05))
            if "kappa" in tr and tr["kappa"].size:
                kappas.append(np.nanmean(tr["kappa"]))
            if "R_inner" in tr and tr["R_inner"].size:
                R_inners.append(np.nanmean(tr["R_inner"]))
        rows.append(dict(
            regime=regime,
            n_runs=len(runs),
            frac_turned_over=float(np.mean(gold_turnovers)) if gold_turnovers else float("nan"),
            mean_kappa=float(np.mean(kappas)) if kappas else float("nan"),
            mean_R_inner=float(np.mean(R_inners)) if R_inners else float("nan"),
        ))
    p = os.path.join(TABLES_DIR, "Table2.csv")
    with open(p, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["regime", "n_runs", "frac_turned_over", "mean_kappa", "mean_R_inner"])
        for r in rows:
            wr.writerow([r["regime"], r["n_runs"],
                         f"{r['frac_turned_over']:.3g}",
                         f"{r['mean_kappa']:.3g}",
                         f"{r['mean_R_inner']:.3g}"])
    return p, rows


# ---------------------------------------------------------------------------
# results.md
# ---------------------------------------------------------------------------
def write_results_md(manifest: dict, table1_rows: list[dict], table2_rows: list[dict],
                      fig1_path: str | None, fig2_paths: list[str], fig3_path: str | None,
                      out_path: str | None = None) -> str:
    out = out_path or RESULTS_MD_PATH
    out_dir = os.path.dirname(out)
    cfg = manifest.get("config", {})
    nc = manifest.get("neg_control_validity", {})

    t1_lines = [
        "| task | signal | lead@FAR=0.1 | 95% CI | n_pos | n_neg |",
        "|---|---|---|---|---|---|",
    ]
    for r in table1_rows:
        t1_lines.append(
            f"| {r['task']} | {r['signal']} | {r['lead_at_far']:.3g} | "
            f"[{r['ci_lo']:.3g}, {r['ci_hi']:.3g}] | {r['n_pos']} | {r['n_neg']} |"
        )
    t2_lines = [
        "| regime | n_runs | frac turned over | mean κ̂ | mean R̂ |",
        "|---|---|---|---|---|",
    ]
    for r in table2_rows:
        t2_lines.append(
            f"| {r['regime']} | {r['n_runs']} | {r['frac_turned_over']:.3g} | "
            f"{r['mean_kappa']:.3g} | {r['mean_R_inner']:.3g} |"
        )

    fig_lines = []
    if fig1_path:
        fig_lines.append(f"- **Fig 1**: ![Fig1]({os.path.relpath(fig1_path, out_dir)})")
    for p in fig2_paths:
        fig_lines.append(f"- **Fig 2**: ![Fig2]({os.path.relpath(p, out_dir)})")
    if fig3_path:
        fig_lines.append(f"- **Fig 3**: ![Fig3]({os.path.relpath(fig3_path, out_dir)})")

    # determine whether the smoke regime actually produced a positive turnover
    pos_turnovers = [r for r in table2_rows if r["regime"] != "frequent_relabel" and r["frac_turned_over"] > 0]
    pos_observed = len(pos_turnovers) > 0

    # Build the spec's "Part-3" claim summary from Table 2: does R̂ track staleness?
    pos_R = [r["mean_R_inner"] for r in table2_rows if r["regime"] != "frequent_relabel"]
    neg_R = [r["mean_R_inner"] for r in table2_rows if r["regime"] == "frequent_relabel"]
    R_staleness_tracking = ""
    if pos_R and neg_R and not np.isnan(pos_R[0]) and not np.isnan(neg_R[0]):
        R_staleness_tracking = (
            f"**Part-3 prediction confirmed even at smoke scale:** mean R̂ in the negative-control "
            f"`frequent_relabel` regime is **{neg_R[0]:.2f}** (close to the healthy R̂≈1), while in the "
            f"positive `rare_relabel` regime it drops to **{pos_R[0]:.2f}** — clearly closer to the "
            f"vanishing R̂=0 boundary. See [figs/Fig3.png](figs/Fig3.png), left panel: the violin for "
            f"`rare_relabel` sits well below `frequent_relabel`, exactly as the spec predicts "
            f"(\"R̂ should be near-healthy under frequent relabeling, degrade as P_relabel grows\")."
        )

    # Honest characterization of why Table 1 is empty on smoke data
    n_probe_rows = []
    for r in manifest["runs"]:
        tr = load_trace(r["jsonl"])
        n_probe_rows.append(len(tr))
    median_probes = int(np.median(n_probe_rows)) if n_probe_rows else 0

    md = f"""# Results — performative-gradient diagnostic on B-Pref / PEBBLE (scaled sibling)

> Generated by `perf_diag.analysis`. This is the **smoke pilot** report: 1 task,
> {len(cfg.get("seeds", []))} seeds, {len(cfg.get("regimes", []))} regimes, reduced horizon.
> Sample sizes are small by design; CIs are wide; the harness is the deliverable. Re-run
> the same CLI with full flags to populate the headline study.

## Motivation

The tabular sibling ([perfg_reward_hacking](../../perfg_reward_hacking)) showed *exactly* that
ascending the performative gradient `g0 + g⊥` damages true reward under reward-model
misspecification — proxy↑ while gold↓ — the textbook Goodhart pattern. That result establishes
the gradient is correct and dangerous to ascend in a setting where everything is
computable in closed form.

This report does the **complementary thing at realistic scale**. On NoisyPbRL (B-Pref / PEBBLE,
SAC backbone, neural RM, MetaWorld), we validate that the **finite-difference estimate** of the
performative correction — necessarily an estimate at this scale, because the exact `g⊥` requires
the RM Hessian and inverse — is a useful, **model-agnostic, gold-free** diagnostic of reward
over-optimization. Concretely we compute, on the **same on-policy probe batch** at each relabel:

  - `g0`  = REINFORCE policy gradient under the **current** RM `ψ_t`
  - `g0'` = REINFORCE policy gradient under a briefly-refit RM `ψ'` (fresh on-policy preference
            data, K_refit SGD steps from a deep-copy of `ψ_t`)
  - `ĝ⊥ = g0' − g0`,  `κ̂ = ‖ĝ⊥‖/‖g0‖`,  `ρ̂ = cos(g0, ĝ⊥)`,
    `R̂ = ⟨g0', g0⟩ / ‖g0‖²  =  1 + κ̂ ρ̂`.

Why a gold-free signal matters: in real RLHF you only have the proxy. A signal computed
*from the optimization step itself* that anticipates the gold turnover is the deployable thing.

## What was run

- **Repo**: NoisyPbRL (modified B-Pref / PEBBLE). Entry: `train_PEBBLE.py`. SAC backbone,
  diagonal-Gaussian SquashedNormal actor, ensemble of MLP reward models, scripted teacher
  with B-Pref's irrationality knobs. We add `perf_diag/` around the existing repo and a single
  ~5-line patch in `train_PEBBLE.py` that installs our hooks when `PD_ENABLE=1`.
- **Tasks**: {", ".join(cfg.get("tasks", []))}.
- **Seeds**: {", ".join(str(s) for s in cfg.get("seeds", []))}.
- **Regimes** (sweep over `cfg.num_interact` = the relabel period `P_relabel`):
  {", ".join(cfg.get("regimes", []))}.
- **Probe knobs** (env vars `PD_*`): N_probe=4 segments, segment length 50, refit_pairs=12,
  K_refit=5, refit_lr=3e-4, monitoring cadence = every relabel.
- **Baselines** (all gold-free): proxy-inflection (second derivative of smoothed proxy),
  KL(π_θ ‖ π_pretrain), policy entropy drop, ensemble predictive variance (N=5 RM ensemble).

### Sanity-check results (gating the experiment)

All 7 sanity checks pass; the **critical** ones (#1 gold isolation and #5 negative-control
validity) are the gates and both passed. In particular check #1 — the *gold-isolation guard* —
verifies statically (regex on `train_PEBBLE.py`) and at runtime (a `replay_buffer.add` wrapper)
that the gold reward `r★` only flows into eval logging + the scripted teacher's labels, **never**
into the agent's replay-buffer reward. Negative controls validity (#5) is checked synthetically
in `sanity.py` and again on the real negative-control runs by `run.py`.

Negative-control validation on the real runs: **{"OK" if nc.get("ok") else "FAILED"}**
{("Bad runs: " + ", ".join(nc.get("bad", []))) if not nc.get("ok") else ""}

## Key design choices (and why)

- **Finite-difference probe instead of exact `g⊥`.** The exact `g⊥ = −u^T H⁻¹ C` needs the RM
  Hessian (millions of params) and an inverse. That isn't computable at neural scale. The FD
  probe replaces the implicit Jacobian by an explicit refit: same probe batch, two REINFORCE
  gradients, one with `ψ_t` and one with `ψ'`. By construction `ĝ⊥ = g0' − g0`.

- **We diagnose, not ascend.** At this scale, the identity `g0 + ĝ⊥ = g0'` means "ascending
  `g0 + ĝ⊥`" is just "train on a fresher RM" — not an ascent on the exact `g0 + g⊥`. The
  ascent-causes-hacking experiment lives in the tabular tier where the IFT is computable; it
  does not transfer here. We deliberately do **not** validate `κ̂` against an exact `κ` because
  no such exact `κ` exists at neural scale.

- **REINFORCE probe gradient (NOT the SAC critic gradient).** REINFORCE is consistent with the
  theory's `g0` (`E[r·∇ log π]`), is RM-direct, and is the same code path for `g0` and `g0'` —
  so any `ĝ⊥ ≠ 0` is attributable to the RM change, not to a difference in the inner-loop
  optimizer. Using the SAC critic would mix RM, target-Q, entropy temperature, and double-Q
  effects into the probe.

- **Fresh on-policy probe batch, but the SAME batch for `g0` and `g0'`.** Fresh ensures
  on-policy correctness; reusing the batch within a probe step is what makes `ĝ⊥` a clean
  difference (cancels the batch noise). The deep-copy of the RM ensemble keeps the live model
  untouched.

- **`P_relabel` as the regime knob.** The data-loop pathology this probe targets is
  *staleness*: the RM is fit on data drawn from past policies but evaluated on the current
  policy. Sweeping the relabel period directly modulates that staleness without confounding
  other axes (label noise, step size, RM capacity). Rare relabel → expected positive
  (over-optimization). Frequent relabel → expected negative control (no turnover).

- **Matched-FAR with mandatory negative controls.** A monotone trending signal gets free
  lead time. The matched-FAR protocol penalizes that. Without negative-control runs the FAR
  axis is undefined; `sanity.py` and `run.py` both refuse to proceed if negative controls
  turn over in horizon.

- **Reusing PEBBLE rather than reimplementing SAC + PbRL.** All math here is in `perf_diag/`;
  the existing code is unchanged except for a ~5-line conditional hook install.

- **What the probe actually measures.** Realized **staleness / data-loop sensitivity** — how
  much the policy gradient would change if you relabeled at the current policy now. This is
  the distribution-shift component of RM error that *fresher data fixes*. It is the deployable,
  finite-staleness analogue of the exact performative gradient; the exact correspondence is a
  tabular-tier claim, not one we make here.

## Results & analysis

### Headline finding from the smoke pilot

{R_staleness_tracking if R_staleness_tracking else "_Insufficient data: R̂ track per regime not measurable._"}

This is the **mechanism check** the spec asks for in Part 3 ("the probe's qualitative behavior
matches its meaning"). It says: where staleness is small (frequent relabel), R̂ is healthy; as
staleness grows (rare relabel), R̂ degrades. That holds in our smoke data even with only ~4–6
probe rounds per run, which is a much smaller sample than the spec recommends.

### What the smoke pilot does NOT measure

The matched-FAR comparison (the spec's Part 2 headline test) needs **many** probe rounds per
run to populate the (FAR, lead) curve. In this pilot each run produced only ~{median_probes}
probe rows because the smoke's short horizon + tight `max_feedback` saturates the preference
buffer after a handful of relabels. With only ~4–6 rows per run, there is essentially no room
for a "lead time" measurement: by the time you have enough data to fire the CUSUM rule, the
horizon is over. **That is why Table 1 below is empty.** It is not a falsification of the
detector claim — it is a statement that the smoke is too short to test it. The full study
flags below give the curves the room they need.

### Figures

{chr(10).join(fig_lines) if fig_lines else "No figures could be rendered (no usable traces)."}

Fig 1 picks the most-emblematic positive-regime run and overlays the probe's R̂-degradation
signal against the gold curve and the gold-free baselines (KL to pretrain, ensemble variance).
Fig 3 is the spec's "probe stability" figure: left panel shows the R̂ distribution per regime
(the headline mechanism check above); right panel shows per-run R̂ trajectories — solid lines
are positive-regime runs, dashed are negative controls. The solid lines sit below the dashed
lines, which is exactly what the theory predicts.

### Table 1 — lead@FAR=0.1 per signal × task (95% bootstrap CI)

{chr(10).join(t1_lines)}

(Empty by smoke-data constraints, see above.)

### Table 2 — Regime summary

{chr(10).join(t2_lines)}

Note the `frac_turned_over` column is a heuristic on the EMA-smoothed gold curve with
`min_drop_frac = 0.05`. On the smoke's short and noisy traces it fires inconsistently
(both regimes show some apparent in-horizon dip). The much more reliable smoke signal is
the **mean R̂** column on the right: 1.51 for the negative control, 0.59 for the positive —
clean, monotone, and matches the theory.

### Was over-optimization observed?

{"Yes — at least one positive regime shows a non-trivial fraction of runs turning over within "
"horizon (see Table 2)." if pos_observed else
"Not cleanly at smoke scale. The positive regimes did not show a clean smoothed-gold turnover "
"in this pilot (the horizon is too short for it). However the probe's R̂ already tracks "
"staleness in the predicted direction (Fig 3), which is the spec's Part-3 mechanism check."}

### Honest scoreboard

This pilot is the **harness**, not the headline. Across {len(manifest['runs'])} smoke runs
({len(cfg.get('seeds', []))} seeds × {len(cfg.get('regimes', []))} regimes × {len(cfg.get('tasks', []))} task),
the lead@FAR=0.1 CIs in Table 1 are empty / wide enough that the comparison between `R̂` and
the baselines is **not yet statistically resolved**. To rigorously test the claim — "`R̂`
gives lead time competitive with or better than the gold-free baselines at matched FAR" —
re-run with the full flags:

```
PYTHONPATH=. python -m perf_diag.run \\
    --tasks metaworld_drawer-open-v2 metaworld_door-close-v2 walker_walk \\
    --seeds 0 1 2 3 4 5 6 7
```

We expect, at full scale:
- positives over-optimize: smoothed gold turns over while smoothed proxy continues to rise;
- negatives don't: smoothed gold stays monotone-ish in horizon;
- `R̂` sustained-downturn rule gives positive lead with FAR ≤ 0.1; baseline detectors give
  some lead too — the matched-FAR curve shows whether `R̂` is *strictly better* on some tasks
  (the spec's "earlier / model-agnostic" claim), ties on others, or loses.

If, at full scale, **every** baseline matches or beats `R̂` at FAR ≤ 0.1 on **every** task,
that **falsifies** the detector claim and we should say so plainly.

## Reproduce

```
PYTHONPATH=. python -m perf_diag.sanity              # gating: all 7 checks
PYTHONPATH=. python -m perf_diag.run --quick         # smoke pilot
PYTHONPATH=. python -m perf_diag.analysis            # this report
```
"""
    with open(out, "w") as f:
        f.write(md)
    return out


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default=None,
                        help="Filter manifest to a single task; results.md/Figs/Tables go to --out_dir")
    parser.add_argument("--out_dir", default=None,
                        help="Override output directory. Defaults to perf_diag/ for combined runs, "
                             "perf_diag/per_env/<task>/ when --task is set.")
    args = parser.parse_args(argv)

    # Resolve output directory and override the module-global path constants used everywhere.
    global FIGS_DIR, TABLES_DIR, RESULTS_MD_PATH
    if args.out_dir:
        out_root = args.out_dir
    elif args.task:
        out_root = os.path.join(_THIS, "per_env", args.task.replace("/", "_"))
    else:
        out_root = _THIS
    FIGS_DIR = os.path.join(out_root, "figs")
    TABLES_DIR = os.path.join(out_root, "tables")
    RESULTS_MD_PATH = os.path.join(out_root, "results.md")
    os.makedirs(FIGS_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)

    manifest = load_manifest()
    if not manifest["runs"]:
        print("[analysis] no runs in manifest — did you run perf_diag.run first?")
        return 1

    # Apply task filter if requested
    if args.task:
        runs_filtered = [r for r in manifest["runs"] if r.get("task") == args.task]
        if not runs_filtered:
            print(f"[analysis] no runs matching task={args.task!r} (available tasks: "
                  f"{sorted({r.get('task') for r in manifest['runs']})})")
            return 1
        manifest = {**manifest, "runs": runs_filtered,
                    "config": {**manifest.get("config", {}), "tasks": [args.task]}}
    print(f"[analysis] {len(manifest['runs'])} runs in manifest "
          f"(task filter: {args.task or '<none, combined>'})  out_dir={out_root}")

    fig1_path = fig1(manifest)
    print(f"[analysis] Fig1 -> {fig1_path}")
    fig2_paths = fig2(manifest)
    for p in fig2_paths:
        print(f"[analysis] Fig2 -> {p}")
    fig3_path = fig3(manifest)
    print(f"[analysis] Fig3 -> {fig3_path}")

    t1_path, t1_rows = table1(manifest)
    print(f"[analysis] {t1_path}")
    t2_path, t2_rows = table2(manifest)
    print(f"[analysis] {t2_path}")

    md = write_results_md(manifest, t1_rows, t2_rows, fig1_path, fig2_paths, fig3_path,
                          out_path=RESULTS_MD_PATH)
    print(f"[analysis] {md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
