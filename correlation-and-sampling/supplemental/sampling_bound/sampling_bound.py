"""supplemental.sampling_bound.sampling_bound

Sampling lower bounds for representative tuple sampling (paper §7.1,
Theorems 1 & 2).

Notation (mirrors the paper):
  * ``p_hat``  -- estimated normalized support frequency of the rarest
                  candidate pattern, p̂ = δ̂ (or a data-driven estimate).
  * ``eps``    -- additive error bound ε.
  * ``eta``    -- recall η (confidence that the sample is representative).
  * ``lam``    -- Chernoff ratio λ.
  * ``delta_hat`` -- normalized support threshold δ̂.

All bounds return the minimum sample size N (an integer >= 1).
"""
from __future__ import annotations

import math
from typing import Dict

import numpy as np


# --------------------------------------------------------------------------- #
# Individual bounds (§7.1)
# --------------------------------------------------------------------------- #
def variance_augmented_bound(p_hat: float, eps: float, eta: float) -> int:
    """The proposed variance-augmented Hoeffding-Bernstein lower bound.

    N = p̂(1-p̂) · ln(2/(1-η)) / (2 ε²)

    (Theorem 1; tighter than both Chebyshev and Bennett.)
    """
    if p_hat <= 0.0 or p_hat >= 1.0:
        # degenerate: a deterministic attribute needs just 1 sample to confirm
        return 1
    num = p_hat * (1.0 - p_hat) * math.log(2.0 / (1.0 - eta))
    den = 2.0 * eps * eps
    return max(1, int(math.ceil(num / den)))


def chebyshev_bound(p_hat: float, eps: float, eta: float) -> int:
    """Chebyshev lower bound (comparison baseline).

    N = p̂(1-p̂) / (ε² (1-η))
    """
    if p_hat <= 0.0 or p_hat >= 1.0:
        return 1
    num = p_hat * (1.0 - p_hat)
    den = eps * eps * (1.0 - eta)
    return max(1, int(math.ceil(num / den)))


def bennett_bound(p_hat: float, eps: float, eta: float) -> int:
    """Bennett lower bound (comparison baseline).

    N = ( (p̂+ε) · ln(1 + ε/p̂) − ε ) / ln(2/(1-η))
    """
    if p_hat <= 0.0:
        return 1
    num = (p_hat + eps) * math.log1p(eps / p_hat) - eps
    den = math.log(2.0 / (1.0 - eta))
    if num <= 0.0 or den <= 0.0:
        return 1
    return max(1, int(math.ceil(num / den)))


def hybrid_chernoff_bound(eps: float, eta: float) -> int:
    """Hybrid Chernoff bound [16] (comparison baseline).

    N = (3/2) · ln(2/(1-η)) / ε²
    """
    num = 1.5 * math.log(2.0 / (1.0 - eta))
    den = eps * eps
    return max(1, int(math.ceil(num / den)))


# --------------------------------------------------------------------------- #
# Theorem 2 -- the piecewise tightest bound
# --------------------------------------------------------------------------- #
def _regime_boundaries(lam: float) -> tuple[float, float]:
    """Regime boundaries p̂₁, p̂₂ (paper §7.1).

      p̂₁ = (1 − √(1 − 24λ)) / 2
      p̂₂ = (1 + √(1 − 24λ)) / 2

    Valid only when 1 − 24λ ≥ 0, i.e. λ ≤ 1/24.  For larger λ the quadratic has
    no real roots and the whole [0,1] interval falls into the variance regime.
    """
    disc = 1.0 - 24.0 * lam
    if disc < 0.0:
        return float("inf"), float("inf")  # no real boundaries
    sqrt_disc = math.sqrt(disc)
    return (1.0 - sqrt_disc) / 2.0, (1.0 + sqrt_disc) / 2.0


def theorem2_bound(p_hat: float, eps: float, eta: float, lam: float) -> int:
    """Theorem 2: the piecewise sampling lower bound actually used by RepSampler.

    Three regimes depending on p̂ relative to (p̂₁, p̂₂) and on 1/(24λ):

      * regime A (variance-augmented):
            0 ≤ p̂ ≤ p̂₁  or  p̂₂ < p̂ ≤ 1,  with 1/(24λ) ≥ 1
            N = p̂(1-p̂)·ln(2/(1-η)) / (2 ε²)
      * regime B (hybrid Chernoff):
            p̂ ∈ (p̂₁, p̂₂]
            N = (3/2)·ln(2/(1-η)) / ε²
      * regime C (variance-augmented, scaled):
            0 ≤ p̂ ≤ 1,  with 1/(24λ) < 1
            N = p̂(1-p̂)·ln(2/(1-η)) / (2 ε²·(1 − 1/(24λ)))
    """
    if p_hat <= 0.0 or p_hat >= 1.0:
        return 1
    inv_term = 1.0 / (24.0 * lam) if lam > 0 else float("inf")
    p1, p2 = _regime_boundaries(lam)

    # regime C: 1/(24λ) < 1  -> variance-augmented scaled up
    if inv_term < 1.0:
        num = p_hat * (1.0 - p_hat) * math.log(2.0 / (1.0 - eta))
        den = 2.0 * eps * eps * (1.0 - inv_term)
        return max(1, int(math.ceil(num / den)))

    # regime B: p̂ inside (p̂₁, p̂₂]
    if p1 < p_hat <= p2:
        return hybrid_chernoff_bound(eps, eta)

    # regime A: otherwise -> plain variance-augmented
    return variance_augmented_bound(p_hat, eps, eta)


# --------------------------------------------------------------------------- #
# Data-driven p̂ estimation
# --------------------------------------------------------------------------- #
def p_hat_estimate(df, delta_hat: float) -> float:
    """Estimate p̂ from a relation as the normalized support of the rarest
    *frequent* attribute-value combination.

    Concretely, for each column we compute the relative frequency of its least
    frequent value that still meets the absolute support implied by δ̂, and take
    the minimum across columns.  This is the quantity the variance-augmented
    bound is most sensitive to (the worst-case pattern).

    Parameters
    ----------
    df : pandas.DataFrame
        The (optionally sampled) relation.
    delta_hat : float
        Normalized support threshold δ̂ in (0, 1].
    """
    n = len(df)
    if n == 0:
        return delta_hat
    abs_support = max(1, int(math.ceil(delta_hat * n)))
    worst = delta_hat  # fall back to δ̂ itself
    for col in df.columns:
        counts = df[col].astype(str).value_counts()
        # frequencies that clear the absolute support threshold
        freq = counts[counts >= abs_support]
        if freq.empty:
            continue
        rel = freq.min() / n
        if rel < worst:
            worst = rel
    # clamp into (0, 0.5] -- the bound is symmetric and meaningless at 0/1
    return float(min(0.5, max(1e-6, worst)))


# --------------------------------------------------------------------------- #
# One-shot comparison (used by run_bound_comparison.py and the experiment scripts)
# --------------------------------------------------------------------------- #
def compare_bounds(p_hat: float, eps: float, eta: float, lam: float) -> Dict[str, int]:
    """Compute all four bounds plus the Theorem-2 selection at once."""
    return {
        "variance_augmented": variance_augmented_bound(p_hat, eps, eta),
        "chebyshev": chebyshev_bound(p_hat, eps, eta),
        "bennett": bennett_bound(p_hat, eps, eta),
        "hybrid_chernoff": hybrid_chernoff_bound(eps, eta),
        "theorem2": theorem2_bound(p_hat, eps, eta, lam),
    }
