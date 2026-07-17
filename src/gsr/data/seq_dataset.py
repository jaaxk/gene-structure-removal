"""Sequence-serving dataset for the LoRA live-embedding training path.

When the backbone is LoRA-finetuned, embeddings cannot be precomputed (the
backbone changes during training), so the dataset serves the mutant sequence, its
wild-type sequence, and the mutated position; the trainer embeds them on the fly
through the LoRA backbone each step. Labels/positions come from the same built
store as the frozen path (WT sequences are recovered from the stored is_wt rows).
"""

from __future__ import annotations

from typing import List

import pandas as pd
import torch
from torch.utils.data import Dataset

from gsr.data.dataset import GroupableMixin
from gsr.losses.base import LABEL_TO_ID, MIDDLE
from gsr.scoring.store import VariantStore


class SequenceVariantDataset(Dataset, GroupableMixin):
    def __init__(self, store: VariantStore, drop_middle: bool = True,
                 gene_ids: List[str] | None = None):
        df = store.load_scores()
        if gene_ids is not None:
            df = df[df["gene_id"].isin(set(gene_ids))].copy()
        # WT sequence per gene from the stored is_wt rows.
        wt_rows = df[df["is_wt"]]
        self.gene_to_wt = dict(zip(wt_rows["gene_id"], wt_rows["mutated_sequence"]))

        df = df[~df["is_wt"]].copy()
        df["label_id"] = df["label"].map(LABEL_TO_ID).astype(int)
        if drop_middle:
            df = df[df["label_id"] != MIDDLE].copy()
        df = df.reset_index(drop=True)
        if len(df) == 0:
            raise ValueError("SequenceVariantDataset is empty after filtering.")

        self.df = df
        self.mut_seqs = df["mutated_sequence"].tolist()
        self.wt_seqs = [self.gene_to_wt[g] for g in df["gene_id"]]
        self.positions = df["pos"].astype(int).tolist()
        self.labels = torch.tensor(df["label_id"].to_numpy(), dtype=torch.long)
        self.gene_codes = torch.tensor(
            pd.factorize(df["gene_id"])[0], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        return (self.mut_seqs[idx], self.wt_seqs[idx], self.positions[idx],
                int(self.labels[idx]), int(self.gene_codes[idx]))


def collate_sequences(batch):
    """Collate into lists of sequences/positions + label/gene tensors."""
    mut_seqs, wt_seqs, positions, labels, genes = zip(*batch)
    return {
        "mut_seqs": list(mut_seqs),
        "wt_seqs": list(wt_seqs),
        "positions": list(positions),
        "labels": torch.tensor(labels, dtype=torch.long),
        "genes": torch.tensor(genes, dtype=torch.long),
    }
