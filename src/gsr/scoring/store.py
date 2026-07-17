"""On-disk store for per-variant scores + embeddings (our own format).

Layout (under a single ``base_dir`` in scratch):

    base_dir/
      scores/<shard>.parquet     # per-variant scalars + labels (columnar, filterable)
      embeddings/<shard>.h5       # dense float32 embeddings: dataset "X" (N,D) + "variant_id"
      manifest/<shard>.json       # {shard, embedding_dim, n, genes, variant_ids}

Rationale:
- **Parquet** for scalars: cheap to load/filter, trivial to compute dataset stats
  and per-gene quartiles from.
- **HDF5** for dense embeddings: chunked+compressed, fast row slicing.
- **Sharding + per-shard files** (no shared writers): a SLURM array job writes one
  shard per task with zero lock contention. A shard = a group of genes.

Scores and embeddings within a shard are written together and kept row-aligned by
``variant_id`` so they can never drift apart.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np
import pandas as pd

# Canonical scalar columns for a variant row (WT rows use mutant="WT", pos=0).
# `mutated_sequence` holds the variant sequence (for WT rows, the WT sequence) so
# the LoRA live-embedding path can serve sequences without the original FASTA.
SCORE_COLUMNS = [
    "gene_id", "variant_id", "mutant", "pos", "wt_aa", "mut_aa", "seq_len",
    "is_wt", "wt_score", "mut_score", "delta", "abs_delta", "label",
    "mutated_sequence",
]


class VariantStore:
    def __init__(self, base_dir: Path):
        self.base = Path(base_dir)
        self.scores_dir = self.base / "scores"
        self.emb_dir = self.base / "embeddings"
        self.manifest_dir = self.base / "manifest"

    # --- writing --------------------------------------------------------
    def _ensure(self) -> None:
        for d in (self.scores_dir, self.emb_dir, self.manifest_dir):
            d.mkdir(parents=True, exist_ok=True)

    def write_part(
        self, shard: str, df: pd.DataFrame,
        mut_embeddings: np.ndarray, wt_embeddings: np.ndarray,
    ) -> None:
        """Write one shard's scores + (mut, wt) embeddings + manifest.

        ``df`` must contain a ``variant_id`` column; ``mut_embeddings[i]`` and
        ``wt_embeddings[i]`` correspond to ``df.iloc[i]``. The WT embedding is the
        wild-type pooled at the SAME mutated position, so WT-anchored losses
        compare a variant to its own WT like-for-like.
        """
        assert "variant_id" in df.columns, "df must have a variant_id column"
        mut = np.asarray(mut_embeddings, dtype=np.float32)
        wt = np.asarray(wt_embeddings, dtype=np.float32)
        assert len(df) == len(mut) == len(wt), (
            f"length mismatch: df={len(df)} mut={len(mut)} wt={len(wt)}")
        assert mut.ndim == 2 and mut.shape == wt.shape, "embeddings must be (N,D)"
        self._ensure()

        df.to_parquet(self.scores_dir / f"{shard}.parquet", index=False)

        variant_ids = df["variant_id"].tolist()
        chunks = (min(1024, len(mut)), mut.shape[1])
        with h5py.File(self.emb_dir / f"{shard}.h5", "w") as h5:
            for name, arr in (("X_mut", mut), ("X_wt", wt)):
                h5.create_dataset(name, data=arr, dtype="float32", chunks=chunks,
                                  compression="gzip", compression_opts=4)
            dt = h5py.string_dtype(encoding="utf-8")
            h5.create_dataset("variant_id", data=np.array(variant_ids, dtype=object),
                              dtype=dt)

        manifest = {
            "shard": shard,
            "embedding_dim": int(mut.shape[1]),
            "n": int(len(df)),
            "genes": sorted(df["gene_id"].unique().tolist()),
            "variant_ids": variant_ids,
        }
        with open(self.manifest_dir / f"{shard}.json", "w") as fh:
            json.dump(manifest, fh)

    # --- reading --------------------------------------------------------
    def load_scores(self, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Concatenate all shard parquet parts into one DataFrame."""
        parts = sorted(self.scores_dir.glob("*.parquet"))
        if not parts:
            raise FileNotFoundError(f"No score shards under {self.scores_dir}")
        frames = [pd.read_parquet(p, columns=columns) for p in parts]
        return pd.concat(frames, ignore_index=True)

    def _manifest_parts(self) -> List[dict]:
        parts = sorted(self.manifest_dir.glob("*.json"))
        if not parts:
            raise FileNotFoundError(f"No manifest parts under {self.manifest_dir}")
        out = []
        for p in parts:
            with open(p) as fh:
                out.append(json.load(fh))
        return out

    def embedding_dim(self) -> int:
        dims = {m["embedding_dim"] for m in self._manifest_parts()}
        assert len(dims) == 1, f"inconsistent embedding dims across shards: {dims}"
        return dims.pop()

    def _build_index(self) -> Dict[str, tuple]:
        """variant_id -> (shard, row) from manifest parts."""
        index: Dict[str, tuple] = {}
        for m in self._manifest_parts():
            for row, vid in enumerate(m["variant_ids"]):
                index[vid] = (m["shard"], row)
        return index

    def load_embeddings(self, variant_ids: List[str], which: str = "mut") -> np.ndarray:
        """Load embeddings for the given variant_ids, in the requested order.

        ``which`` selects the mutant ('mut' -> dataset X_mut) or the position-
        matched wild-type ('wt' -> X_wt) embedding.
        """
        dset = {"mut": "X_mut", "wt": "X_wt"}[which]
        index = self._build_index()
        missing = [v for v in variant_ids if v not in index]
        if missing:
            raise KeyError(
                f"{len(missing)} variant_ids not in store (e.g. {missing[:3]})"
            )
        by_shard: Dict[str, List[tuple]] = {}
        for out_i, vid in enumerate(variant_ids):
            shard, row = index[vid]
            by_shard.setdefault(shard, []).append((out_i, row))

        D = self.embedding_dim()
        out = np.empty((len(variant_ids), D), dtype=np.float32)
        for shard, pairs in by_shard.items():
            out_idx = np.array([p[0] for p in pairs])
            rows = np.array([p[1] for p in pairs])
            order = np.argsort(rows)  # h5 fancy-indexing requires increasing order
            with h5py.File(self.emb_dir / f"{shard}.h5", "r") as h5:
                data = h5[dset][rows[order]]
            out[out_idx[order]] = data
        return out
