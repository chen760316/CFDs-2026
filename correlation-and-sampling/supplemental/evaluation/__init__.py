"""
supplemental.evaluation -- CFD evaluation (paper §9.1).

Parses the CFD output format produced by the existing SCFDM C++ miners and the
verify.py scripts, de-duplicates rules, and computes Precision / Recall / F1.
"""
from .cfd_parser import (
    CFD,
    parse_cfd_line,
    load_cfds,
    cfd_key,
)
from .metrics import precision, recall, f1, evaluate

__all__ = [
    "CFD", "parse_cfd_line", "load_cfds", "cfd_key",
    "precision", "recall", "f1", "evaluate",
]
