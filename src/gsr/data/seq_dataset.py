"""Sequence-serving dataset for the LoRA live-embedding path.

Built from the same training metadata DataFrame as the frozen path (columns:
mutated_sequence, wt_seq, pos, gene_id, label_id). The trainer embeds mut + WT (at
the mutated position) on the fly through the LoRA backbone each step, since a
finetuned backbone's embeddings cannot be cached.
"""

from __future__ import annotations

import pandas as pd
import torch
from torch.utils.data import Dataset

from gsr.data.dataset import GroupableMixin


class SequenceVariantDataset(Dataset, GroupableMixin):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        self.mut_seqs = self.df["mutated_sequence"].tolist()
        self.wt_seqs = self.df["wt_seq"].tolist()
        self.positions = self.df["pos"].astype(int).tolist()
        self.labels = torch.tensor(self.df["label_id"].to_numpy(), dtype=torch.long)
        self.gene_codes = torch.tensor(
            pd.factorize(self.df["gene_id"])[0], dtype=torch.long)

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
