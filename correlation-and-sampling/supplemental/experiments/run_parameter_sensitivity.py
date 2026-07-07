"""supplemental.experiments.run_parameter_sensitivity

Reproduce the parameter-sensitivity study of paper §9.3 (Table 5):
fix every parameter except the normalized support δ̂, sweep δ̂ over
{1e-3, 2e-3, 3e-3, 4e-3, 5e-3}, and record the F1 of each algorithm.

For each (dataset, δ̂) pair this script invokes ``run_pipeline`` once per
algorithm variant and parses the resulting F1.  Results are written to a CSV
whose rows mirror Table 5 (one row per dataset × δ̂ × algorithm).

Usage (from the correlation-and-sampling directory):
    python -m supplemental.experiments.run_parameter_sensitivity \
        --datasets datasets/RT.csv datasets/Adult.csv \
        --truth-cfds  truth/RT.txt truth/Adult.txt \
        --algorithms SCFDM_all SCFDM_part CFDMiner CTANE \
        --out parameter_sensitivity.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

# supplemental lives inside correlation-and-sampling; subprocess calls to
# `python -m supplemental.xxx` must run with cwd = correlation-and-sampling.
_CAS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# δ̂ values used in Table 5 / Appendix H.2
DELTA_HAT_VALUES = [1e-3, 2e-3, 3e-3, 4e-3, 5e-3]

# map user-facing algorithm names to the C++ variant + mining strategy
_ALGO_VARIANT = {
    "SCFDM_all": ("SCFDM_all", "FD-First-DFS-dfs"),
    "SCFDM_part": ("SCFDM_part", "FD-First-DFS-dfs"),
    "CFDMiner": ("SCFDM_part", "Integrated-DFS"),
    "CTANE": ("SCFDM_all", "Integrated-BFS"),
    "Itemset-First": ("SCFDM_all", "Itemset-First-DFS-dfs"),
    "FD-First": ("SCFDM_all", "FD-First-DFS-dfs"),
}


def _row_count(dataset: str) -> int:
    with open(dataset, "r", encoding="utf-8", errors="ignore") as f:
        return max(0, sum(1 for _ in f) - 1)


def _run_one(dataset: str, truth: Optional[str], delta_hat: float,
             algorithm: str, base: Dict, workdir: str) -> Tuple[float, Optional[float]]:
    """Run the pipeline once and return (runtime, f1)."""
    variant, strategy = _ALGO_VARIANT.get(algorithm, ("SCFDM_all", "FD-First-DFS-dfs"))
    support = max(1, int(delta_hat * _row_count(dataset)))
    args = [
        sys.executable, "-m", "supplemental.experiments.run_pipeline",
        "--dataset", dataset,
        "--support", str(support),
        "--confidence", str(base.get("confidence", 0.95)),
        "--max-size", str(base.get("max_size", 3)),
        "--cores", str(base.get("cores", 8)),
        "--variant", variant,
        "--delta-hat", str(delta_hat),
        "--eps", str(base.get("eps", 5e-4)),
        "--eta", str(base.get("eta", 0.9)),
        "--lam", str(base.get("lam", 0.05)),
        "--workdir", workdir,
    ]
    if truth:
        args += ["--truth-cfds", truth]
    print(f"  $ {' '.join(args)}")
    subprocess.run(args, cwd=_CAS_DIR, capture_output=True, text=True)
    report_path = os.path.join(workdir, "pipeline_report.json")
    if not os.path.exists(report_path):
        return (0.0, None)
    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)
    runtime = report.get("total_elapsed", 0.0)
    f1 = None
    for stage in report.get("stages", []):
        out = stage.get("output", "")
        if "F1=" in out:
            try:
                f1 = float(out.split("F1=")[1].split()[0])
            except (IndexError, ValueError):
                pass
    return (runtime, f1)


def run(datasets: List[str], truths: List[Optional[str]],
        algorithms: List[str], base: Dict, out_csv: str) -> None:
    rows: List[Dict] = []
    for ds, truth in zip(datasets, truths):
        ds_name = os.path.splitext(os.path.basename(ds))[0]
        for delta_hat in DELTA_HAT_VALUES:
            for algo in algorithms:
                workdir = os.path.join(
                    base.get("workdir", "runs/sensitivity"),
                    ds_name, f"d{delta_hat}", algo)
                os.makedirs(workdir, exist_ok=True)
                print(f"\n--- {ds_name} | δ̂={delta_hat} | {algo} ---")
                runtime, f1 = _run_one(ds, truth, delta_hat, algo, base, workdir)
                rows.append({
                    "dataset": ds_name,
                    "delta_hat": delta_hat,
                    "algorithm": algo,
                    "runtime_s": runtime,
                    "f1": f1,
                })
                print(f"    -> runtime={runtime:.2f}s  F1={f1}")

    if rows:
        keys = list(rows[0].keys())
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"\nParameter-sensitivity results written to {out_csv}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="δ̂ parameter sensitivity (paper Table 5)")
    ap.add_argument("--datasets", nargs="+", required=True, help="input CSVs")
    ap.add_argument("--truth-cfds", nargs="+", default=None,
                    help="ground-truth CFD files (one per dataset, same order)")
    ap.add_argument("--algorithms", nargs="+", default=["SCFDM_all", "SCFDM_part"],
                    choices=list(_ALGO_VARIANT.keys()))
    ap.add_argument("--out", default="parameter_sensitivity.csv")
    ap.add_argument("--workdir", default="runs/sensitivity")
    ap.add_argument("--confidence", type=float, default=0.95)
    ap.add_argument("--max-size", type=int, default=3)
    ap.add_argument("--cores", type=int, default=8)
    ap.add_argument("--eps", type=float, default=5e-4)
    ap.add_argument("--eta", type=float, default=0.9)
    ap.add_argument("--lam", type=float, default=0.05)
    args = ap.parse_args(argv)

    truths = args.truth_cfds or [None] * len(args.datasets)
    if len(truths) != len(args.datasets):
        # broadcast a single truth file to all datasets, or pad with None
        if len(truths) == 1:
            truths = truths * len(args.datasets)
        else:
            truths = truths + [None] * (len(args.datasets) - len(truths))

    base = {k: getattr(args, k) for k in
            ("confidence", "max_size", "cores", "eps", "eta", "lam", "workdir")}
    run(args.datasets, truths, args.algorithms, base, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
