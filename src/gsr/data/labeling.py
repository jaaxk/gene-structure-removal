"""Assign contrastive labels from per-gene quartiles of |LL_mut - LL_wt|.

Within each gene, variants in the bottom ``quartile_low`` fraction of |delta| are
labeled ``same`` (likelihood close to WT) and those in the top ``quartile_high``
fraction are labeled ``different``; the middle band is labeled ``middle`` and
dropped from the contrastive loss (kept in the store for later use). The WT row
itself has delta 0 and is always ``same``.

Labeling is per-gene so that genes with different overall likelihood scales are
each split relative to their own distribution.
"""

from __future__ import annotations

import pandas as pd


def assign_labels(
    df: pd.DataFrame,
    quartile_low: float = 0.25,
    quartile_high: float = 0.25,
    min_variants: int = 10,
) -> pd.DataFrame:
    """Return ``df`` with a ``label`` column; drop genes with too few variants.

    Requires columns: ``gene_id``, ``abs_delta``, ``is_wt``.
    """
    df = df.copy()
    df["label"] = "middle"

    n_total_genes = df["gene_id"].nunique()
    keep_genes = []
    for gene, g in df.groupby("gene_id"):
        mut = g[~g["is_wt"]]
        if len(mut) < min_variants:
            continue
        keep_genes.append(gene)
        low_thr = mut["abs_delta"].quantile(quartile_low)
        high_thr = mut["abs_delta"].quantile(1.0 - quartile_high)
        idx = mut.index
        same = idx[mut["abs_delta"] <= low_thr]
        diff = idx[mut["abs_delta"] >= high_thr]
        df.loc[same, "label"] = "same"
        df.loc[diff, "label"] = "different"

    df = df[df["gene_id"].isin(keep_genes)].copy()
    df.loc[df["is_wt"], "label"] = "same"

    n_dropped = n_total_genes - len(keep_genes)
    if n_dropped:
        print(f"[labeling] dropped {n_dropped} genes with < {min_variants} variants")
    return df
