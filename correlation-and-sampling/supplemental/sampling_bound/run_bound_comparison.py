"""Reproduce the sampling-bound comparison of paper §7.1 (Example 5).

Example 5 parameters: δ̂=0.01, ε=0.001, ζ=0.999 (i.e. η=0.999), λ=0.02,
which yields p̂₁≈0.142 and the proposed bound N≈37,624 vs. the hybrid
Chernoff bound N≈456,055 (>91.7% reduction).

Run (from the correlation-and-sampling directory):
    python -m supplemental.sampling_bound.run_bound_comparison
"""
from __future__ import annotations

import argparse

from .sampling_bound import (
    bennett_bound,
    chebyshev_bound,
    compare_bounds,
    hybrid_chernoff_bound,
    p_hat_estimate,
    theorem2_bound,
    variance_augmented_bound,
    _regime_boundaries,
)


def _reduction(prop: int, baseline: int) -> float:
    if baseline <= 0:
        return 0.0
    return 1.0 - prop / baseline


def run(p_hat: float, eps: float, eta: float, lam: float) -> None:
    bounds = compare_bounds(p_hat, eps, eta, lam)
    p1, p2 = _regime_boundaries(lam)

    print("=" * 64)
    print("Sampling Lower-Bound Comparison (paper §7.1)")
    print("=" * 64)
    print(f"Parameters: p̂={p_hat}, ε={eps}, η={eta}, λ={lam}")
    print(f"Regime boundaries: p̂₁={p1:.4f}, p̂₂={p2:.4f}")
    if p1 <= p_hat <= p2:
        print(f"p̂ lies in (p̂₁, p̂₂] -> Theorem 2 selects the hybrid Chernoff regime.")
    else:
        print(f"p̂ lies outside (p̂₁, p̂₂] -> Theorem 2 selects the variance-augmented regime.")
    print("-" * 64)
    for name, n in bounds.items():
        print(f"  {name:<22s}: N = {n:>10d}")
    print("-" * 64)
    prop = bounds["variance_augmented"]
    for name in ("chebyshev", "bennett", "hybrid_chernoff"):
        r = _reduction(prop, bounds[name])
        print(f"  reduction vs {name:<16s}: {r*100:6.2f}%")
    print("=" * 64)


def main() -> None:
    ap = argparse.ArgumentParser(description="Sampling bound comparison (§7.1)")
    ap.add_argument("--p_hat", type=float, default=0.142,
                    help="estimated normalized support (default: Example 5 p̂≈0.142)")
    ap.add_argument("--eps", type=float, default=1e-3, help="additive error ε")
    ap.add_argument("--eta", type=float, default=0.999, help="recall η")
    ap.add_argument("--lam", type=float, default=0.02, help="Chernoff ratio λ")
    args = ap.parse_args()
    run(args.p_hat, args.eps, args.eta, args.lam)


if __name__ == "__main__":
    # Default reproduces Example 5: N≈37,624 vs Chernoff 456,055 (>91.7% reduction)
    main()
