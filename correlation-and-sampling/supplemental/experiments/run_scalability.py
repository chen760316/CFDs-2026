"""supplemental.experiments.run_scalability

Reproduce the 6-dimension scalability study of paper §9.2 (Fig. 4) by
repeatedly invoking ``run_pipeline`` with one parameter swept at a time.

Dimensions (mirroring Fig. 4 a–l):
  * |ρ|  -- tuple sampling ratio  (0.1 – 1.0)
  * |m|  -- number of attributes  (column truncation)
  * |δ̂|  -- normalized support    (1e-3 – 5e-3)
  * |θ|  -- confidence threshold  (0.80 – 1.00)
  * |ℓ|  -- max CFD size          (2 – 5)
  * |c|  -- CPU cores             (28 – 112)

Each run records runtime and (when a ground-truth file is supplied) F1.
Results are written to a CSV for easy plotting.

Usage (from the correlation-and-sampling directory):
    python -m supplemental.experiments.run_scalability \
        --dataset datasets/adult.csv \
        --dimension m \
        --truth-cfds ground_truth.txt \
        --out scalability_m.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import tempfile
from typing import Dict, List

import pandas as pd

# supplemental lives inside correlation-and-sampling; subprocess calls to
# `python -m supplemental.xxx` must run with cwd = correlation-and-sampling.
_CAS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _column_truncated_csv(src: str, m: int, dst: str) -> str:
    """Write a copy of ``src`` keeping only the first ``m`` columns."""
    df = pd.read_csv(src, nrows=1)
    cols = df.columns.tolist()[:m]
    pd.read_csv(src, usecols=cols).to_csv(dst, index=False)
    return dst


def _run_pipeline(args_list: List[str]) -> Dict:
    """Invoke run_pipeline.py and parse its JSON report."""
    cmd = [sys.executable, "-m", "supplemental.experiments.run_pipeline"] + args_list
    print(f"  $ {' '.join(cmd)}")
    cp = subprocess.run(cmd, capture_output=True, text=True, cwd=_CAS_DIR)
    # the report is written to <workdir>/pipeline_report.json
    return cp


def _extract_metrics(workdir: str) -> Dict:
    import json
    report_path = os.path.join(workdir, "pipeline_report.json")
    if not os.path.exists(report_path):
        return {"runtime": None, "f1": None}
    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)
    runtime = report.get("total_elapsed")
    f1 = None
    for stage in report.get("stages", []):
        out = stage.get("output", "")
        if "F1=" in out:
            try:
                f1 = float(out.split("F1=")[1].split()[0])
            except (IndexError, ValueError):
                pass
    return {"runtime": runtime, "f1": f1}


# --------------------------------------------------------------------------- #
# Per-dimension sweeps
# --------------------------------------------------------------------------- #
def vary_rho(dataset, base_args, truth, out_csv):
    points = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    rows = []
    for rho in points:
        # sampling ratio is realised via the N bound multiplier
        eps = 5e-4 / rho if rho > 0 else 5e-4
        workdir = os.path.join(base_args["workdir"], f"rho_{rho}")
        args = _build_args(dataset, base_args, truth, eps=eps, workdir=workdir)
        _run_pipeline(args)
        m = _extract_metrics(workdir)
        rows.append({"dimension": "rho", "value": rho, **m})
    _write_csv(out_csv, rows)


def vary_m(dataset, base_args, truth, out_csv):
    points = [6, 8, 10, 12, 14]  # Adult-style; adjust per dataset
    rows = []
    for m in points:
        tmp = tempfile.mktemp(suffix=".csv")
        _column_truncated_csv(dataset, m, tmp)
        workdir = os.path.join(base_args["workdir"], f"m_{m}")
        args = _build_args(tmp, base_args, truth, workdir=workdir)
        _run_pipeline(args)
        met = _extract_metrics(workdir)
        rows.append({"dimension": "m", "value": m, **met})
    _write_csv(out_csv, rows)


def vary_delta(dataset, base_args, truth, out_csv):
    points = [1e-3, 2e-3, 3e-3, 4e-3, 5e-3]
    rows = []
    for d in points:
        support = max(1, int(d * _row_count(dataset)))
        workdir = os.path.join(base_args["workdir"], f"delta_{d}")
        args = _build_args(dataset, base_args, truth, support=support,
                           delta_hat=d, workdir=workdir)
        _run_pipeline(args)
        met = _extract_metrics(workdir)
        rows.append({"dimension": "delta_hat", "value": d, **met})
    _write_csv(out_csv, rows)


def vary_theta(dataset, base_args, truth, out_csv):
    points = [0.80, 0.85, 0.90, 0.95, 1.00]
    rows = []
    for th in points:
        workdir = os.path.join(base_args["workdir"], f"theta_{th}")
        args = _build_args(dataset, base_args, truth, confidence=th, workdir=workdir)
        _run_pipeline(args)
        met = _extract_metrics(workdir)
        rows.append({"dimension": "theta", "value": th, **met})
    _write_csv(out_csv, rows)


def vary_ell(dataset, base_args, truth, out_csv):
    points = [2, 3, 4, 5]
    rows = []
    for ell in points:
        workdir = os.path.join(base_args["workdir"], f"ell_{ell}")
        args = _build_args(dataset, base_args, truth, max_size=ell, workdir=workdir)
        _run_pipeline(args)
        met = _extract_metrics(workdir)
        rows.append({"dimension": "ell", "value": ell, **met})
    _write_csv(out_csv, rows)


def vary_c(dataset, base_args, truth, out_csv):
    points = [28, 56, 84, 112]
    rows = []
    for c in points:
        workdir = os.path.join(base_args["workdir"], f"c_{c}")
        args = _build_args(dataset, base_args, truth, cores=c, workdir=workdir)
        _run_pipeline(args)
        met = _extract_metrics(workdir)
        rows.append({"dimension": "cores", "value": c, **met})
    _write_csv(out_csv, rows)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _row_count(dataset: str) -> int:
    with open(dataset, "r", encoding="utf-8", errors="ignore") as f:
        return max(0, sum(1 for _ in f) - 1)


def _build_args(dataset, base, truth, **overrides) -> List[str]:
    args = [
        "--dataset", dataset,
        "--support", str(base.get("support", 200)),
        "--confidence", str(base.get("confidence", 0.95)),
        "--max-size", str(base.get("max_size", 3)),
        "--cores", str(base.get("cores", 8)),
        "--variant", base.get("variant", "SCFDM_all"),
        "--delta-hat", str(base.get("delta_hat", 2e-3)),
        "--eps", str(base.get("eps", 5e-4)),
        "--eta", str(base.get("eta", 0.9)),
        "--lam", str(base.get("lam", 0.05)),
        "--workdir", base.get("workdir", "runs/scalability"),
    ]
    if truth:
        args += ["--truth-cfds", truth]
    for k, v in overrides.items():
        flag = "--" + k.replace("_", "-")
        # replace existing flag value
        if flag in args:
            args[args.index(flag) + 1] = str(v)
        else:
            args += [flag, str(v)]
    return args


def _write_csv(path: str, rows: List[Dict]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\nScalability results written to {path}")


_SWEEPS = {
    "rho": vary_rho,
    "m": vary_m,
    "delta": vary_delta,
    "theta": vary_theta,
    "ell": vary_ell,
    "c": vary_c,
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Scalability sweep (paper Fig. 4)")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--dimension", required=True, choices=list(_SWEEPS.keys()),
                    help="which parameter to sweep")
    ap.add_argument("--truth-cfds", default=None)
    ap.add_argument("--out", default="scalability.csv", help="output CSV path")
    ap.add_argument("--workdir", default="runs/scalability")
    ap.add_argument("--support", type=int, default=200)
    ap.add_argument("--confidence", type=float, default=0.95)
    ap.add_argument("--max-size", type=int, default=3)
    ap.add_argument("--cores", type=int, default=8)
    ap.add_argument("--variant", default="SCFDM_all")
    ap.add_argument("--delta-hat", type=float, default=2e-3)
    ap.add_argument("--eps", type=float, default=5e-4)
    ap.add_argument("--eta", type=float, default=0.9)
    ap.add_argument("--lam", type=float, default=0.05)
    args = ap.parse_args(argv)

    base = {k: getattr(args, k) for k in
            ("support", "confidence", "max_size", "cores", "variant",
             "delta_hat", "eps", "eta", "lam", "workdir")}
    _SWEEPS[args.dimension](args.dataset, base, args.truth_cfds, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
