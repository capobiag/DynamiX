"""Symmetric positive definite banded solves (lower storage, LAPACK ``?pbsv`` layout).

``ab[d, c]`` holds ``G[c + d, c]`` for ``0 <= d <= kd``. SciPy's banded Cholesky is used when it
is installed; otherwise the matrix is expanded and solved densely with NumPy.
"""

from __future__ import annotations

import numpy as np

try:
    from scipy.linalg import solveh_banded
except ImportError:  # pragma: no cover - exercised by tests through monkeypatching
    solveh_banded = None


def banded_to_dense(ab: np.ndarray) -> np.ndarray:
    """Expand lower banded storage into the full symmetric matrix."""
    rows, n = ab.shape
    dense = np.zeros((n, n))
    for d in range(rows):
        idx = np.arange(n - d)
        dense[idx + d, idx] = ab[d, : n - d]
        dense[idx, idx + d] = ab[d, : n - d]
    return dense


def solve_spd_banded(ab: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve ``G x = rhs`` for symmetric positive definite G in lower banded storage."""
    if solveh_banded is not None:
        return solveh_banded(ab, rhs, lower=True)
    return np.linalg.solve(banded_to_dense(ab), rhs)
