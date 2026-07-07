"""
supplemental -- supplementary modules completing the SCFDM codebase.

This package adds the pieces required by the paper that were not present in
the original repository.  Nothing here modifies existing files; all modules
call into the existing code (under ``correlation-and-sampling/`` and
``parallel/``) via import/subprocess.

Run from the ``correlation-and-sampling`` directory so that ``supplemental``
is importable as a top-level package, e.g.::

    python -m supplemental.sampling_bound.run_bound_comparison

Sub-packages
------------
- ``sampling_bound`` : Theorem 1 & 2 sampling lower-bound computation (§7.1).
- ``evaluation``     : CFD parsing, de-duplication, and Precision/Recall/F1 (§9.1).
- ``experiments``    : end-to-end orchestration and scalability scripts (§9.2–9.3).
"""
