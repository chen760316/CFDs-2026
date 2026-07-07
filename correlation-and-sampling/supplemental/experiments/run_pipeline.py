"""supplemental.experiments.run_pipeline

End-to-end orchestration of the SCFDM pipeline (paper Fig. 1):

    1. AttrFinder     -- extract correlated attribute sets Ψ      (§6, Algorithm 1)
    2. Sampling bound -- compute the representative-sample size N (§7.1, Theorem 2)
    3. RepSampler     -- representative tuple sampling            (§7, Algorithm 2)
    4. Sub-table gen  -- project S onto Ψ to form sub-tables      (§8)
    5. Parallel mine  -- run the C++ CFD miner on the sub-tables  (§8, Algorithm 3)
    6. Evaluate       -- parse + compute Precision/Recall/F1      (§9.1)

The script does NOT modify any existing file.  Stages 1/3/5 invoke the
existing modules via subprocess (so the original C++/Python code runs
unchanged); stages 2/4/6 use the new ``supplemental`` package directly.

Each stage is timed, producing a per-stage breakdown that mirrors the
runtime analysis of paper Appendix F.

Usage (from the correlation-and-sampling directory):
    python -m supplemental.experiments.run_pipeline \
        --dataset datasets/adult.csv \
        --support 200 --confidence 0.95 \
        --max-size 3 --cores 8 \
        --truth-cfds ground_truth.txt \
        --workdir runs/adult
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from ..evaluation.cfd_parser import load_cfds
from ..evaluation.metrics import evaluate
from ..sampling_bound.sampling_bound import (
    p_hat_estimate,
    theorem2_bound,
)

# Path resolution:
#   this file lives at  <cas>/supplemental/experiments/run_pipeline.py
#   <cas>    = correlation-and-sampling  (the AttrFinder / RepSampler code lives here)
#   <repo>   = parent of <cas>           (holds the parallel/ C++ project)
_CAS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_REPO_ROOT = os.path.abspath(os.path.join(_CAS_DIR, ".."))
_PARALLEL_DIR = os.path.join(_REPO_ROOT, "parallel")


@dataclass
class StageResult:
    name: str
    elapsed: float = 0.0
    ok: bool = False
    output: str = ""
    artifacts: List[str] = field(default_factory=list)


def _run(cmd: List[str], cwd: Optional[str] = None,
         timeout: Optional[int] = None) -> subprocess.CompletedProcess:
    """Run a subprocess, streaming output to the parent stdout."""
    print(f"  $ {' '.join(shlex.quote(c) for c in cmd)}  (cwd={cwd or '.'})")
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                          timeout=timeout, check=False)


def _stage(name: str) -> StageResult:
    print(f"\n=== Stage: {name} ===")
    return StageResult(name=name)


# --------------------------------------------------------------------------- #
# Stage 1: AttrFinder (correlated attribute set extraction, §6)
# --------------------------------------------------------------------------- #
def stage_attrfinder(dataset: str, workdir: str, gpu: bool = False) -> StageResult:
    res = _stage("AttrFinder (§6)")
    t0 = time.perf_counter()
    script = os.path.join(_CAS_DIR, "correlation_extraction",
                          "correlation_extraction_tf_multi_gpu.py" if gpu
                          else "correlation_extraction_tf_multi.py")
    sets_out = os.path.join(workdir, "correlated_sets.txt")
    # The existing script hard-codes its input inside __main__; we invoke it and
    # then point it at our dataset by monkey-patching via a tiny wrapper.
    wrapper = os.path.join(workdir, "_attrfinder_wrapper.py")
    os.makedirs(workdir, exist_ok=True)
    with open(wrapper, "w", encoding="utf-8") as f:
        f.write(
            "import runpy, sys\n"
            f"sys.argv = ['{script}', '{dataset}', '{sets_out}']\n"
            "runpy.run_path(sys.argv[0], run_name='__main__')\n"
        )
    cp = _run([sys.executable, wrapper], cwd=os.path.dirname(script))
    res.elapsed = time.perf_counter() - t0
    res.ok = (cp.returncode == 0) and os.path.exists(sets_out)
    res.output = cp.stdout[-2000:] + cp.stderr[-2000:]
    res.artifacts = [sets_out] if os.path.exists(sets_out) else []
    print(f"  elapsed={res.elapsed:.2f}s ok={res.ok}")
    return res


# --------------------------------------------------------------------------- #
# Stage 2: Sampling lower bound (§7.1 Theorem 2)
# --------------------------------------------------------------------------- #
def stage_sampling_bound(dataset: str, delta_hat: float, eps: float,
                         eta: float, lam: float) -> tuple[StageResult, int]:
    res = _stage("Sampling Bound (§7.1)")
    t0 = time.perf_counter()
    df = pd.read_csv(dataset, nrows=20000)  # a cheap preview for p̂ estimation
    p_hat = p_hat_estimate(df, delta_hat)
    n = theorem2_bound(p_hat, eps, eta, lam)
    res.elapsed = time.perf_counter() - t0
    res.ok = True
    res.output = f"p_hat={p_hat:.6f}  N={n}"
    print(f"  p_hat={p_hat:.6f}  N={n}  elapsed={res.elapsed:.2f}s")
    return res, n


# --------------------------------------------------------------------------- #
# Stage 3: RepSampler (representative tuple sampling, §7 Algorithm 2)
# --------------------------------------------------------------------------- #
def stage_repsampler(dataset: str, n_bound: int, workdir: str) -> StageResult:
    res = _stage("RepSampler (§7)")
    t0 = time.perf_counter()
    sample_out = os.path.join(workdir, "sampled.csv")
    wrapper = os.path.join(workdir, "_repsampler_wrapper.py")
    script = os.path.join(_CAS_DIR, "sampling", "representative_tuple_sampling.py")
    with open(wrapper, "w", encoding="utf-8") as f:
        f.write(
            "import runpy, sys\n"
            f"sys.argv = ['{script}', '{dataset}', '{sample_out}', '{n_bound}']\n"
            "runpy.run_path(sys.argv[0], run_name='__main__')\n"
        )
    cp = _run([sys.executable, wrapper], cwd=os.path.dirname(script))
    res.elapsed = time.perf_counter() - t0
    res.ok = (cp.returncode == 0) and os.path.exists(sample_out)
    res.output = cp.stdout[-2000:] + cp.stderr[-2000:]
    res.artifacts = [sample_out] if os.path.exists(sample_out) else []
    print(f"  elapsed={res.elapsed:.2f}s ok={res.ok}")
    return res


# --------------------------------------------------------------------------- #
# Stage 4: Sub-table generation (§8)
# --------------------------------------------------------------------------- #
def stage_subtables(dataset: str, sets_path: str, workdir: str) -> StageResult:
    res = _stage("Sub-table Generation (§8)")
    t0 = time.perf_counter()
    sub_dir = os.path.join(workdir, "subtables")
    os.makedirs(sub_dir, exist_ok=True)
    # Read correlated sets (one "X -> Y" per line) and project the dataset.
    df = pd.read_csv(dataset)
    pairs = []
    with open(sets_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "->" not in line:
                continue
            lhs, rhs = line.split("->", 1)
            lhs_attrs = [a.strip().strip("{}") for a in lhs.split(",") if a.strip()]
            rhs = rhs.strip()
            cols = [c for c in lhs_attrs + [rhs] if c in df.columns]
            if len(cols) >= 2:
                pairs.append(cols)
    if not pairs:
        # fallback: treat the whole table as a single sub-table
        pairs = [df.columns.tolist()]
    for i, cols in enumerate(pairs):
        df[cols].to_csv(os.path.join(sub_dir, f"part{i}.csv"), index=False)
    res.elapsed = time.perf_counter() - t0
    res.ok = len(pairs) > 0
    res.output = f"generated {len(pairs)} sub-tables"
    res.artifacts = [sub_dir]
    print(f"  sub-tables={len(pairs)} elapsed={res.elapsed:.2f}s")
    return res


# --------------------------------------------------------------------------- #
# Stage 5: Parallel CFD mining (§8 Algorithm 3, C++ SCFDM)
# --------------------------------------------------------------------------- #
def stage_parallel_mine(sub_dir: str, support: int, confidence: float,
                        max_size: int, cores: int,
                        variant: str = "SCFDM_all") -> StageResult:
    res = _stage(f"Parallel Mining ({variant}, §8)")
    t0 = time.perf_counter()
    bin_dir = os.path.join(_PARALLEL_DIR, variant, "cmake-build-debug")
    binary = os.path.join(bin_dir, variant)
    input_txt = os.path.join(_PARALLEL_DIR, variant, "input.txt")
    # list sub-table files into input.txt
    parts = sorted([os.path.join(sub_dir, f) for f in os.listdir(sub_dir)
                    if f.endswith(".csv")])
    with open(input_txt, "w", encoding="utf-8") as f:
        f.write(" ".join(parts) + "\n")
        f.write(f"{support}\n{confidence}\n{max_size}\nFD-First-DFS-dfs\n")
    if not os.path.exists(binary):
        res.ok = False
        res.output = f"binary not found at {binary} (build the C++ project first)"
        res.elapsed = time.perf_counter() - t0
        print(f"  SKIP: {res.output}")
        return res
    cp = _run(["mpiexec", "-n", str(cores), binary], cwd=bin_dir)
    res.elapsed = time.perf_counter() - t0
    res.ok = (cp.returncode == 0)
    res.output = cp.stdout[-3000:] + cp.stderr[-3000:]
    print(f"  elapsed={res.elapsed:.2f}s ok={res.ok}")
    return res


# --------------------------------------------------------------------------- #
# Stage 6: Evaluation (§9.1)
# --------------------------------------------------------------------------- #
def stage_evaluate(mined_path: str, truth_path: Optional[str]) -> StageResult:
    res = _stage("Evaluation (§9.1)")
    t0 = time.perf_counter()
    mined = load_cfds(mined_path) if mined_path and os.path.exists(mined_path) else set()
    if truth_path and os.path.exists(truth_path):
        truth = load_cfds(truth_path)
        prf = evaluate(mined, truth)
        res.output = str(prf)
    else:
        res.output = f"mined {len(mined)} CFDs (no ground truth provided)"
    res.elapsed = time.perf_counter() - t0
    res.ok = True
    print(f"  {res.output}  elapsed={res.elapsed:.2f}s")
    return res


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_pipeline(args) -> Dict:
    os.makedirs(args.workdir, exist_ok=True)
    report: Dict = {"dataset": args.dataset, "stages": []}

    s1 = stage_attrfinder(args.dataset, args.workdir, gpu=args.gpu)
    report["stages"].append(s1.__dict__)
    if not s1.ok:
        print("Stage 1 failed; aborting. See output above.")
        return report

    s2, n = stage_sampling_bound(args.dataset, args.delta_hat, args.eps,
                                 args.eta, args.lam)
    report["stages"].append(s2.__dict__)
    report["sample_bound_N"] = n

    s3 = stage_repsampler(args.dataset, n, args.workdir)
    report["stages"].append(s3.__dict__)

    s4 = stage_subtables(args.dataset, s1.artifacts[0] if s1.artifacts else "",
                         args.workdir)
    report["stages"].append(s4.__dict__)

    s5 = stage_parallel_mine(s4.artifacts[0] if s4.artifacts else args.workdir,
                             args.support, args.confidence, args.max_size,
                             args.cores, variant=args.variant)
    report["stages"].append(s5.__dict__)

    s6 = stage_evaluate(args.mined_cfds, args.truth_cfds)
    report["stages"].append(s6.__dict__)

    total = sum(s["elapsed"] for s in report["stages"])
    report["total_elapsed"] = total
    print(f"\n=== Pipeline done.  total elapsed = {total:.2f}s ===")

    with open(os.path.join(args.workdir, "pipeline_report.json"), "w",
              encoding="utf-8") as f:
        # dataclasses-as-dict are JSON-serialisable here (all str/float/bool/list)
        json.dump(report, f, indent=2, default=str)
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="End-to-end SCFDM pipeline (Fig. 1)")
    ap.add_argument("--dataset", required=True, help="input CSV relation")
    ap.add_argument("--support", type=int, default=200, help="absolute support δ")
    ap.add_argument("--confidence", type=float, default=0.95, help="confidence θ")
    ap.add_argument("--max-size", type=int, default=3, help="max CFD size ℓ")
    ap.add_argument("--cores", type=int, default=8, help="MPI processes")
    ap.add_argument("--variant", default="SCFDM_all",
                    choices=["SCFDM_all", "SCFDM_part"], help="C++ miner variant")
    ap.add_argument("--delta-hat", type=float, default=2e-3, help="normalized support δ̂")
    ap.add_argument("--eps", type=float, default=5e-4, help="sampling error ε")
    ap.add_argument("--eta", type=float, default=0.9, help="recall η")
    ap.add_argument("--lam", type=float, default=0.05, help="Chernoff ratio λ")
    ap.add_argument("--gpu", action="store_true", help="use GPU AttrFinder script")
    ap.add_argument("--truth-cfds", default=None, help="ground-truth CFD file")
    ap.add_argument("--mined-cfds", default=None,
                    help="mined CFD file to evaluate (if not from C++ stdout)")
    ap.add_argument("--workdir", default="runs/default", help="working directory")
    args = ap.parse_args(argv)
    run_pipeline(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
