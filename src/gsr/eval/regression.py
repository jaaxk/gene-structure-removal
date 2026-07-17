"""Supervised per-assay regression baseline.

Trains a simple cross-validated Ridge regressor per DMS assay to predict DMS_score
from embeddings, and reports the mean held-out Spearman across assays. This is the
supervised upper-reference for how much selection-relevant signal the embeddings
carry (contrast with the zero-shot centroid metric). Works on any embeddings.
"""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np

from gsr.utils.spearman import spearman


def evaluate_regression(embeddings: np.ndarray, meta,
                        project_fn: Callable[[np.ndarray], np.ndarray] | None = None,
                        n_splits: int = 5, min_variants: int = 30,
                        alpha: float = 1.0, seed: int = 0) -> Dict[str, float]:
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold

    Z = embeddings if project_fn is None else project_fn(embeddings)
    meta = meta.reset_index(drop=True)
    per_type: Dict[str, list] = {}
    per_assay_rho = []

    for dms_id, g in meta.groupby("dms_id"):
        if len(g) < min_variants:
            continue
        idx = g.index.to_numpy()
        X = Z[idx]
        y = g["DMS_score"].to_numpy()
        stype = g["selection_type"].iloc[0]
        kf = KFold(n_splits=min(n_splits, len(idx)), shuffle=True, random_state=seed)
        preds = np.zeros(len(idx))
        for tr, te in kf.split(X):
            model = Ridge(alpha=alpha)
            model.fit(X[tr], y[tr])
            preds[te] = model.predict(X[te])
        rho = spearman(preds, y)
        if rho == rho:
            per_assay_rho.append(rho)
            per_type.setdefault(stype, []).append(rho)

    out = {}
    for stype, rhos in per_type.items():
        out[f"regression/spearman/{stype}"] = float(np.mean(rhos))
    out["regression/spearman_mean"] = (
        float(np.mean(per_assay_rho)) if per_assay_rho else float("nan"))
    out["regression/n_assays"] = float(len(per_assay_rho))
    return out
