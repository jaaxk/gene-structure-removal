"""Load DMS (deep mutational scanning) data for evaluation.

We use ONLY the per-selection-type CSVs under /scratch/jv2807/dms_data/datasets
(Activity.csv, Binding.csv, ...). The CSV file stem IS the (coarse) selection
type -- this is the authoritative selection-type label; no join is needed. The
existing precomputed h5 embedding caches there are intentionally not used; we
recompute embeddings in our own pipeline.

Only single-amino-acid substitutions are kept (multi-mutants, marked with ':' in
``mutant``, are skipped). To bound embedding cost, variants are subsampled per
assay (``max_per_assay``). Gene = ``uniprot_id``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from gsr import paths

_USECOLS = ["mutant", "mutated_sequence", "DMS_score", "uniprot_id", "dms_id"]
_POS_RE = re.compile(r"^[A-Z](\d+)[A-Z]$")


def _parse_pos(mutant: str):
    m = _POS_RE.match(mutant)
    return int(m.group(1)) if m else None


def load_dms(
    selection_types: List[str],
    max_per_assay: int = 200,
    csv_dir: Path = paths.DMS_DATASETS_DIR,
    seed: int = 0,
) -> pd.DataFrame:
    """Return a DataFrame of single-aa DMS variants across selection types.

    Columns: selection_type, dms_id, uniprot_id, mutant, mutated_sequence,
    DMS_score, pos.
    """
    csv_dir = Path(csv_dir)
    rng = np.random.default_rng(seed)
    frames = []
    for stype in selection_types:
        path = csv_dir / f"{stype}.csv"
        if not path.exists():
            print(f"[dms] WARNING: {path} not found, skipping {stype}")
            continue
        df = pd.read_csv(path, usecols=_USECOLS)
        df = df[~df["mutant"].astype(str).str.contains(":")]  # singles only
        df["pos"] = df["mutant"].astype(str).map(_parse_pos)
        df = df.dropna(subset=["pos", "DMS_score"])
        df["pos"] = df["pos"].astype(int)
        # Subsample per assay to bound embedding cost.
        sampled = []
        for dms_id, g in df.groupby("dms_id"):
            if len(g) > max_per_assay:
                g = g.iloc[rng.choice(len(g), max_per_assay, replace=False)]
            sampled.append(g)
        df = pd.concat(sampled, ignore_index=True)
        df["selection_type"] = stype
        frames.append(df)
        print(f"[dms] {stype}: {df['dms_id'].nunique()} assays, {len(df)} variants")
    if not frames:
        raise FileNotFoundError(f"No DMS CSVs found in {csv_dir}")
    out = pd.concat(frames, ignore_index=True)
    return out


def gene_level_split(genes: List[str], query_frac: float, seed: int = 0):
    """Split genes into (centroid_genes, query_genes) at the gene level."""
    genes = sorted(set(genes))
    rng = np.random.default_rng(seed)
    rng.shuffle(genes)
    n_query = max(1, int(round(len(genes) * query_frac)))
    if len(genes) - n_query < 1:  # keep at least one centroid gene
        n_query = len(genes) - 1
    query = set(genes[:n_query])
    centroid = set(genes[n_query:])
    return centroid, query


def load_wt_reference() -> pd.DataFrame:
    """DMS_id -> (target_seq, seq_len) reference table: the WT sequence per
    ProteinGym assay. Shared by anything needing a WT sequence for DMS
    variants (the LLR-projection metric, and the WT-mean-aware pooling
    modes' eval-side embedding), so it lives here rather than duplicated per
    consumer."""
    csv_path = paths.DMS_DATASETS_DIR / "DMS_substitutions.csv"
    ref = pd.read_csv(csv_path, usecols=["DMS_id", "target_seq", "seq_len"])
    return ref.rename(columns={"DMS_id": "dms_id"})
