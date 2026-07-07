"""
supplemental.sampling_bound -- sampling lower-bound computation (paper §7.1).

Implements the four sampling lower bounds compared in the paper:

  * variance-augmented Hoeffding-Bernstein bound (Theorem 1, the proposed one)
  * Chebyshev bound
  * Bennett bound
  * hybrid Chernoff bound [16]

plus the piecewise Theorem 2 that selects the tightest bound per p̂ regime.
"""
from .sampling_bound import (
    variance_augmented_bound,
    chebyshev_bound,
    bennett_bound,
    hybrid_chernoff_bound,
    theorem2_bound,
    p_hat_estimate,
    compare_bounds,
)

__all__ = [
    "variance_augmented_bound",
    "chebyshev_bound",
    "bennett_bound",
    "hybrid_chernoff_bound",
    "theorem2_bound",
    "p_hat_estimate",
    "compare_bounds",
]
