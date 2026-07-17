"""Correlation helpers used by evaluation."""

from __future__ import annotations

import numpy as np


def spearman(x, y) -> float:
    """Spearman rank correlation; returns nan if undefined (constant input)."""
    from scipy.stats import spearmanr

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    rho, _ = spearmanr(x, y)
    return float(rho)
