#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from dataclasses import dataclass, asdict
from typing import Dict, List

import yaml


@dataclass
class SweepCell:
    env_key: str
    train_env: str
    method: str
    gauge: str
    alpha_mode: str
    seed: int
    horizon: int
    use_perf_correction: bool
    learnable_temperature: bool
    init_temperature: float

    @property
    def run_name(self) -> str:
        return (
            f"{self.env_key}__{self.method}__{self.gauge}"
            f"__{self.alpha_mode}__seed{self.seed}"
        )


def _load_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _make_cells(base_cfg: Dict, phase_cfg: Dict) -> List[SweepCell]:
    cells: List[SweepCell] = []
    for env_key in phase_cfg["envs"]:
        env_meta = base_cfg["envs"][env_key]
        for alpha_mode in phase_cfg["alpha_modes"]:
            alpha_meta = base_cfg["alpha_modes"][alpha_mode]
            for seed in phase_cfg["seeds"]:
                for method in phase_cfg["methods"]:
                    method_meta = base_cfg["methods"][method]
                    for gauge in method_meta["gauges"]:
                        cells.append(
                            SweepCell(
                                env_key=env_key,
                                train_env=env_meta["train_env"],
                                method=method,
                                gauge=gauge,
                                alpha_mode=alpha_mode,
                                seed=int(seed),
                                horizon=int(env_meta["horizon"]),
                                use_perf_correction=bool(method_meta["use_perf_correction"]),
                                learnable_temperature=bool(alpha_meta["learnable_temperature"]),
                                init_temperature=float(alpha_meta["init_temperature"]),
                            )
                        )
    return cells


def _build_overrides(base_cfg: Dict, cell: SweepCell) -> List[str]:
    base_overrides = base_cfg.get("default_overrides", {})
    overrides = [f"{k}={v}" for k, v in base_overrides.items()]
    overrides.extend(
        [
            f"env={cell.train_env}",
            f"seed={cell.seed}",
            f"method={cell.method}",
            f"gauge={cell.gauge}",
            f"alpha_mode={cell.alpha_mode}",
            f"use_perf_correction={'true' if cell.use_perf_correction else 'false'}",
            f"h_horizon={cell.horizon}",
            f"agent.params.learnable_temperature={'true' if cell.learnable_temperature else 'false'}",
            f"agent.params.init_temperature={cell.init_temperature}",
            f"exp_dir={os.path.abspath(base_cfg.get('results_dir', 'results'))}/runs",
            f"reference_dataset_dir={os.path.abspath(base_cfg.get('reference_dataset_dir', 'reference_dataset'))}",
            f"hydra.run.dir={os.path.abspath(base_cfg.get('results_dir', 'results'))}/runs/{cell.run_name}",
        ]
    )
    return overrides


def _parse_set_overrides(items: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --set value {item!r}; expected key=value")
        k, v = item.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k:
            raise ValueError(f"Invalid --set value {item!r}; empty key")
        out[k] = v
    return out


def _write_manifest(cells: List[SweepCell], out_dir: str, phase: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, f"manifest_{phase}.json")
    csv_path = os.path.join(out_dir, f"manifest_{phase}.csv")
    payload = [asdict(c) for c in cells]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(payload[0].keys()) if payload else [])
        if payload:
            writer.writeheader()
            writer.writerows(payload)
    return json_path


def main():
    parser = argparse.ArgumentParser(description="Run gauge experiment sweep.")
    parser.add_argument("--phase", default="phase1", choices=["phase1", "phase2", "phase3", "full"])
    parser.add_argument("--config-dir", default="configs/gauge_experiment")
    parser.add_argument("--execute", action="store_true", help="Execute runs; otherwise print commands only.")
    parser.add_argument("--max-runs", type=int, default=0, help="Optional cap for quick smoke tests.")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra Hydra overrides forwarded to every run; repeatable.",
    )
    args = parser.parse_args()

    base_cfg = _load_yaml(os.path.join(args.config_dir, "base.yaml"))
    phase_cfg = _load_yaml(os.path.join(args.config_dir, f"{args.phase}.yaml"))
    cells = _make_cells(base_cfg, phase_cfg)
    if args.max_runs > 0:
        cells = cells[: args.max_runs]

    manifest_path = _write_manifest(cells, os.path.join(base_cfg.get("results_dir", "results"), "manifests"), args.phase)
    print(f"[run_sweep] phase={args.phase} cells={len(cells)} manifest={manifest_path}")

    script = base_cfg.get("script", "train_PEBBLE.py")
    extra_overrides = _parse_set_overrides(args.set)
    for idx, cell in enumerate(cells, 1):
        cmd = ["python", script] + _build_overrides(base_cfg, cell)
        cmd += [f"{k}={v}" for k, v in extra_overrides.items()]
        print(f"[{idx:04d}/{len(cells):04d}] {cell.run_name}")
        print(" ".join(cmd))
        if args.execute:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
