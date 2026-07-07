"""supplemental.evaluation.cfd_parser

Parse, normalise and de-duplicate CFDs from the output format used across the
existing codebase:

    [A, B] => C, (a1, b1 || c1)

This is the format emitted by the SCFDM C++ miners (see ``parallel/SCFDM_all``
``Output::printCFD``) and consumed by ``correlation-and-sampling/sampling/verify.py``
via the regex ``\\[(.*?)\\] => (.*?), \\((.*?)\\|\\|(.*?)\\)``.

A CFD is represented as the dataclass ``CFD(lhs_attrs, rhs_attr, lhs_vals, rhs_val)``
where ``lhs_attrs``/``rhs_attr`` are attribute-name lists and ``lhs_vals``/
``rhs_val`` are the corresponding pattern values (strings).  Variable CFDs
(wildcard ``_``) are supported.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import FrozenSet, Iterable, List, Optional, Set

# The regex used by the existing verify.py — kept identical for compatibility.
_CFD_PATTERN = re.compile(r"\[(.*?)\]\s*=>\s*(.*?),\s*\((.*?)\|\|(.*?)\)")


@dataclass(frozen=True)
class CFD:
    """A single conditional functional dependency.

    Attributes
    ----------
    lhs_attrs : tuple[str, ...]
        Left-hand-side attribute names, e.g. ``("A", "B")``.
    rhs_attr : str
        Right-hand-side attribute name, e.g. ``"C"``.
    lhs_vals : tuple[str, ...]
        Pattern values for the LHS (``"_"`` for variable wildcards).
    rhs_val : str
        Pattern value for the RHS (``"_"`` for a variable CFD).
    """
    lhs_attrs: tuple
    rhs_attr: str
    lhs_vals: tuple
    rhs_val: str

    def key(self) -> tuple:
        """A hashable, order-invariant key for set comparison.

        LHS attribute-value pairs are sorted so that ``[A,B]=>C,(a,b||c)`` and
        ``[B,A]=>C,(b,a||c)`` compare equal (they are the same CFD).
        """
        pairs = tuple(sorted(zip(self.lhs_attrs, self.lhs_vals)))
        return (pairs, self.rhs_attr, self.rhs_val)


def _split_list(raw: str) -> List[str]:
    """Split a comma-separated attribute/value list, stripping whitespace."""
    return [item.strip() for item in raw.split(",") if item.strip() != ""]


def parse_cfd_line(line: str) -> Optional[CFD]:
    """Parse a single CFD line.  Returns ``None`` if the line does not match."""
    line = line.strip()
    if not line:
        return None
    match = _CFD_PATTERN.search(line)
    if not match:
        return None
    lhs_attrs = tuple(_split_list(match.group(1)))
    rhs_attr = match.group(2).strip()
    lhs_vals = tuple(_split_list(match.group(3)))
    rhs_val = match.group(4).strip()
    if not lhs_attrs or not rhs_attr:
        return None
    # pad/truncate lhs_vals to match lhs_attrs length (defensive)
    if len(lhs_vals) < len(lhs_attrs):
        lhs_vals = lhs_vals + ("_",) * (len(lhs_attrs) - len(lhs_vals))
    return CFD(lhs_attrs=lhs_attrs, rhs_attr=rhs_attr, lhs_vals=lhs_vals, rhs_val=rhs_val)


def cfd_key(cfd: CFD) -> tuple:
    """Backward-compatible accessor for the comparison key."""
    return cfd.key()


def load_cfds(path: str) -> Set[CFD]:
    """Load a CFD file, returning a de-duplicated set.

    Blank lines and unparseable lines are silently skipped, matching the
    tolerant behaviour of the existing ``verify.py``.
    """
    out: Set[CFD] = set()
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            cfd = parse_cfd_line(line)
            if cfd is not None:
                out.add(cfd)
    return out


def cfd_set_keys(cfds: Iterable[CFD]) -> FrozenSet[tuple]:
    """Convert an iterable of CFDs to a frozen set of comparison keys."""
    return frozenset(c.key() for c in cfds)
