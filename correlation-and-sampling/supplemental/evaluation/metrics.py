"""supplemental.evaluation.metrics

Precision / Recall / F1 for CFD discovery (paper §9.1).

  P(Σ̂, Σ) = |Σ̂ ∩ Σ| / |Σ̂|     -- proportion of discovered CFDs that are valid
  R(Σ̂, Σ) = |Σ̂ ∩ Σ| / |Σ|      -- fraction of ground-truth CFDs recovered
  F1       = 2 P R / (P + R)

CFDs are compared by their normalised key (see ``cfd_parser.cfd_key``) so that
order-invariant duplicates count once.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .cfd_parser import CFD, cfd_set_keys


@dataclass
class PRF1:
    precision: float
    recall: float
    f1: float
    mined_count: int
    truth_count: int
    overlap_count: int

    def as_dict(self) -> dict:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "mined_count": self.mined_count,
            "truth_count": self.truth_count,
            "overlap_count": self.overlap_count,
        }

    def __str__(self) -> str:
        return (f"Precision={self.precision:.4f}  Recall={self.recall:.4f}  "
                f"F1={self.f1:.4f}  (mined={self.mined_count}, "
                f"truth={self.truth_count}, overlap={self.overlap_count})")


def precision(mined: Iterable[CFD], truth: Iterable[CFD]) -> float:
    """P = |Σ̂ ∩ Σ| / |Σ̂|."""
    mined_keys = cfd_set_keys(mined)
    truth_keys = cfd_set_keys(truth)
    if not mined_keys:
        return 0.0
    overlap = mined_keys & truth_keys
    return len(overlap) / len(mined_keys)


def recall(mined: Iterable[CFD], truth: Iterable[CFD]) -> float:
    """R = |Σ̂ ∩ Σ| / |Σ|."""
    mined_keys = cfd_set_keys(mined)
    truth_keys = cfd_set_keys(truth)
    if not truth_keys:
        return 0.0
    overlap = mined_keys & truth_keys
    return len(overlap) / len(truth_keys)


def f1(p: float, r: float) -> float:
    """F1 = 2 P R / (P + R)."""
    if p + r <= 0.0:
        return 0.0
    return 2.0 * p * r / (p + r)


def evaluate(mined: Iterable[CFD], truth: Iterable[CFD]) -> PRF1:
    """Compute Precision, Recall and F1 in one pass."""
    mined_keys = cfd_set_keys(mined)
    truth_keys = cfd_set_keys(truth)
    overlap = mined_keys & truth_keys
    p = len(overlap) / len(mined_keys) if mined_keys else 0.0
    r = len(overlap) / len(truth_keys) if truth_keys else 0.0
    return PRF1(
        precision=p,
        recall=r,
        f1=f1(p, r),
        mined_count=len(mined_keys),
        truth_count=len(truth_keys),
        overlap_count=len(overlap),
    )
