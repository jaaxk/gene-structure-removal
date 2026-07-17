"""Dataset over a built VariantStore (frozen-embedding training path).

Loads per-variant scalars (parquet) and their precomputed embeddings (h5) into
memory. Middle-labeled variants are dropped from training (they carry no
contrastive signal); WT rows are kept as 'same' anchors. Each item is
``(embedding, label_id, gene_code)``.

When LoRA finetuning is added, a sequence-serving variant of this dataset will
replace the cached embeddings; the interface (label_id + gene_code per item)
stays the same so the sampler/trainer are unaffected.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from gsr.losses.base import LABEL_TO_ID, MIDDLE
from gsr.scoring.store import VariantStore


class GroupableMixin:
    """Grouping helpers used by the sampler; needs self.labels + self.gene_codes."""

    def gene_to_indices(self) -> dict:
        out: dict = {}
        for i, g in enumerate(self.gene_codes.numpy()):
            out.setdefault(int(g), []).append(i)
        return out

    def indices_by_gene_and_label(self) -> dict:
        """{gene_code: {label_id: [row indices]}}."""
        out: dict = {}
        genes = self.gene_codes.numpy()
        labels = self.labels.numpy()
        for i in range(len(self.labels)):
            out.setdefault(int(genes[i]), {}).setdefault(
                int(labels[i]), []).append(i)
        return out


class VariantDataset(Dataset, GroupableMixin):
    def __init__(self, store: VariantStore, drop_middle: bool = True,
                 gene_ids: List[str] | None = None):
        df = store.load_scores()
        if gene_ids is not None:
            df = df[df["gene_id"].isin(set(gene_ids))].copy()
        df = df[~df["is_wt"]].copy()  # WT anchoring uses the per-variant wt embedding
        df["label_id"] = df["label"].map(LABEL_TO_ID).astype(int)
        if drop_middle:
            df = df[df["label_id"] != MIDDLE].copy()
        df = df.reset_index(drop=True)
        if len(df) == 0:
            raise ValueError("VariantDataset is empty after filtering.")

        self.df = df
        vids = df["variant_id"].tolist()
        self.mut_emb = torch.from_numpy(store.load_embeddings(vids, "mut")).float()
        self.wt_emb = torch.from_numpy(store.load_embeddings(vids, "wt")).float()
        assert self.mut_emb.shape == self.wt_emb.shape, (
            f"mut/wt embedding shape mismatch: {tuple(self.mut_emb.shape)} vs "
            f"{tuple(self.wt_emb.shape)}")
        self.labels = torch.tensor(df["label_id"].to_numpy(), dtype=torch.long)
        self.gene_codes = torch.tensor(
            pd.factorize(df["gene_id"])[0], dtype=torch.long
        )
        self.input_dim = self.mut_emb.shape[1]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        return (self.mut_emb[idx], self.wt_emb[idx],
                self.labels[idx], self.gene_codes[idx])
