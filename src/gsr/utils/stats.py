"""Dataset statistics printed as a sanity check at the start of every run.

Per the project rules, build/train/eval entrypoints all call
``print_dataset_stats`` so a glance at the logs confirms the data looks right
before any expensive compute proceeds. The returned dict is also logged to wandb.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def dataset_stats(df: pd.DataFrame) -> Dict[str, float]:
    """Compute summary stats over a per-variant scores DataFrame."""
    mut = df[~df["is_wt"]] if "is_wt" in df.columns else df
    stats: Dict[str, float] = {
        "n_rows": int(len(df)),
        "n_genes": int(df["gene_id"].nunique()),
        "n_variants": int(len(mut)),
        "n_wt": int(df["is_wt"].sum()) if "is_wt" in df.columns else 0,
    }
    if "label" in df.columns:
        counts = mut["label"].value_counts().to_dict()
        for lab in ("same", "different", "middle"):
            stats[f"n_{lab}"] = int(counts.get(lab, 0))
    if len(mut):
        stats["variants_per_gene_mean"] = float(
            mut.groupby("gene_id").size().mean()
        )
    if "seq_len" in df.columns and len(df):
        stats["seq_len_min"] = int(df["seq_len"].min())
        stats["seq_len_median"] = float(df["seq_len"].median())
        stats["seq_len_max"] = int(df["seq_len"].max())
    if "abs_delta" in mut.columns and len(mut):
        stats["abs_delta_median"] = float(mut["abs_delta"].median())
    return stats


def print_dataset_stats(df: pd.DataFrame, title: str = "dataset") -> Dict[str, float]:
    """Print a compact human-readable summary and return the stats dict."""
    s = dataset_stats(df)
    print(f"\n===== {title} stats =====")
    print(f"  genes         : {s['n_genes']}")
    print(f"  variants      : {s['n_variants']}  (+ {s.get('n_wt', 0)} WT rows)")
    if "n_same" in s:
        tot = max(s["n_variants"], 1)
        print(
            f"  labels        : same={s['n_same']} ({100*s['n_same']/tot:.1f}%)  "
            f"different={s['n_different']} ({100*s['n_different']/tot:.1f}%)  "
            f"middle(dropped)={s.get('n_middle', 0)}"
        )
    if "variants_per_gene_mean" in s:
        print(f"  variants/gene : {s['variants_per_gene_mean']:.1f} mean")
    if "seq_len_max" in s:
        print(
            f"  seq_len       : min={s['seq_len_min']} "
            f"median={s['seq_len_median']:.0f} max={s['seq_len_max']}"
        )
    if "abs_delta_median" in s:
        print(f"  |delta| median: {s['abs_delta_median']:.4f}")
    print("==========================\n")
    return s
