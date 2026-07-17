"""Streaming variant dataset (frozen path, bounded memory).

Reads per-variant (mut, wt) embeddings from the shared EmbeddingCache H5 on
demand, so training never holds the whole set in RAM. Used for datasets larger
than the resident-RAM budget. Each DataLoader worker opens its own read-only H5
handle lazily; with gene-diverse batching a batch is one gene's (contiguous) rows,
so reads are localized. Requires the cache to be fully warmed first (a CPU run
with misses errors upstream; a GPU run self-warms).
"""

from __future__ import annotations

import os

import pandas as pd
import torch
from torch.utils.data import Dataset

from gsr.data.dataset import GroupableMixin


class StreamingVariantDataset(Dataset, GroupableMixin):
    def __init__(self, df: pd.DataFrame, emb_cache):
        self.df = df.reset_index(drop=True)
        vids = self.df["variant_id"].tolist()
        rows = emb_cache.row_index(vids)
        if rows is None:
            raise ValueError(
                "Streaming requires all embeddings cached, but some are missing. "
                "Warm the cache on a GPU node first (run train on GPU, or "
                "--warm_only).")
        self.rows = rows
        self.h5_path = str(emb_cache.h5_path)
        self.labels = torch.tensor(self.df["label_id"].to_numpy(), dtype=torch.long)
        self.gene_codes = torch.tensor(
            pd.factorize(self.df["gene_id"])[0], dtype=torch.long)
        # Read embedding dim once (own handle, closed immediately) so no live
        # handle is inherited across the fork to workers.
        with emb_cache.open_readonly() as h5:
            self.input_dim = int(h5["X_mut"].shape[1])
        self._h5 = None  # opened lazily per worker process

    def _handle(self):
        if self._h5 is None:
            import h5py
            os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        h5 = self._handle()
        r = self.rows[idx]
        mut = torch.from_numpy(h5["X_mut"][r].astype("float32"))
        wt = torch.from_numpy(h5["X_wt"][r].astype("float32"))
        return mut, wt, self.labels[idx], self.gene_codes[idx]


def stream_worker_init(worker_id: int) -> None:
    """Ensure each worker opens its own H5 handle (not an inherited one)."""
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    info = torch.utils.data.get_worker_info()
    if info is not None:
        info.dataset._h5 = None
