"""In-memory datasets for training.

Both datasets are built from a metadata DataFrame (columns include gene_id and
label_id, one row per training variant, WT rows and middle-labeled rows already
removed). Embeddings are held RESIDENT (loaded/computed once, never read from disk
per batch) so the training loop does no I/O and the GPU is not starved.

- VariantDataset: frozen path -- serves precomputed (mut, wt) embedding tensors.
- (LoRA path uses SequenceVariantDataset in seq_dataset.py.)
"""

from __future__ import annotations

import pandas as pd
import torch
from torch.utils.data import Dataset


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
    def __init__(self, df: pd.DataFrame, mut_emb: torch.Tensor,
                 wt_emb: torch.Tensor):
        assert len(df) == len(mut_emb) == len(wt_emb)
        assert mut_emb.shape == wt_emb.shape, (
            f"mut/wt embedding shape mismatch: {tuple(mut_emb.shape)} vs "
            f"{tuple(wt_emb.shape)}")
        self.df = df.reset_index(drop=True)
        self.mut_emb = mut_emb.float()
        self.wt_emb = wt_emb.float()
        self.labels = torch.tensor(self.df["label_id"].to_numpy(), dtype=torch.long)
        self.gene_codes = torch.tensor(
            pd.factorize(self.df["gene_id"])[0], dtype=torch.long)
        self.input_dim = self.mut_emb.shape[1]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        return (self.mut_emb[idx], self.wt_emb[idx],
                self.labels[idx], self.gene_codes[idx])
