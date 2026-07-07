"""supplemental.evaluation.evaluate_cfds

CLI: compute Precision / Recall / F1 between a mined CFD file and a
ground-truth CFD file (paper §9.1).

Usage (from the correlation-and-sampling directory):
    python -m supplemental.evaluation.evaluate_cfds \
        --mined  path/to/mined_cfds.txt \
        --truth  path/to/ground_truth_cfds.txt
"""
from __future__ import annotations

import argparse
import json
import sys

from .cfd_parser import load_cfds
from .metrics import evaluate


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="CFD Precision/Recall/F1 (§9.1)")
    ap.add_argument("--mined", required=True, help="path to mined CFD file")
    ap.add_argument("--truth", required=True, help="path to ground-truth CFD file")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of text")
    args = ap.parse_args(argv)

    mined = load_cfds(args.mined)
    truth = load_cfds(args.truth)
    result = evaluate(mined, truth)

    if args.json:
        json.dump(result.as_dict(), sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        print("=" * 56)
        print("CFD Discovery Evaluation (paper §9.1)")
        print("=" * 56)
        print(f"  mined CFDs (de-duplicated): {result.mined_count}")
        print(f"  ground-truth CFDs         : {result.truth_count}")
        print(f"  overlap                   : {result.overlap_count}")
        print("-" * 56)
        print(f"  Precision : {result.precision:.4f}")
        print(f"  Recall    : {result.recall:.4f}")
        print(f"  F1        : {result.f1:.4f}")
        print("=" * 56)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
