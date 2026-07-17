"""Batch samplers that control gene diversity.

Default ``gene_diverse``: each batch is drawn from a *single* gene, and one epoch
visits every gene exactly once (maximum gene diversity per epoch). Because genes
are re-shuffled and variant subsets re-sampled each epoch, repeated epochs show a
gene with different variants. Optionally the within-gene batch is balanced 50/50
between 'same' and 'different'.

``cross_gene`` (off by default): each batch mixes ``genes_per_batch`` different
genes, ``batch_size // genes_per_batch`` variants each -- a harder task for later
experiments.

Both yield lists of dataset row indices and are used as a ``batch_sampler``.
"""

from __future__ import annotations

from typing import Iterator, List

import numpy as np

from gsr.losses.base import DIFFERENT, SAME


class GeneBatchSampler:
    def __init__(self, dataset, batch_size: int, batch_mode: str = "gene_diverse",
                 genes_per_batch: int = 2, balance_labels: bool = True,
                 seed: int = 0):
        self.batch_size = batch_size
        self.batch_mode = batch_mode
        self.genes_per_batch = genes_per_batch
        self.balance_labels = balance_labels
        self.by_gl = dataset.indices_by_gene_and_label()
        self.genes = sorted(self.by_gl.keys())
        self.epoch = 0
        self._seed = seed

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _rng(self) -> np.random.Generator:
        return np.random.default_rng(self._seed + self.epoch)

    def _pick_from_gene(self, gene: int, k: int, rng: np.random.Generator) -> List[int]:
        by_label = self.by_gl[gene]
        same = by_label.get(SAME, [])
        diff = by_label.get(DIFFERENT, [])
        if self.balance_labels and same and diff:
            half = k // 2
            picks = list(rng.choice(same, size=min(half, len(same)), replace=False))
            picks += list(rng.choice(diff, size=min(k - len(picks), len(diff)),
                                     replace=False))
        else:
            pool = same + diff
            picks = list(rng.choice(pool, size=min(k, len(pool)), replace=False))
        return picks

    def __iter__(self) -> Iterator[List[int]]:
        rng = self._rng()
        order = list(self.genes)
        rng.shuffle(order)
        if self.batch_mode == "gene_diverse":
            for gene in order:
                batch = self._pick_from_gene(gene, self.batch_size, rng)
                if len(batch) >= 2:
                    yield batch
        elif self.batch_mode == "cross_gene":
            per_gene = self.batch_size // self.genes_per_batch
            for i in range(0, len(order) - self.genes_per_batch + 1,
                           self.genes_per_batch):
                batch: List[int] = []
                for gene in order[i:i + self.genes_per_batch]:
                    batch += self._pick_from_gene(gene, per_gene, rng)
                if len(batch) >= 2:
                    yield batch
        else:
            raise ValueError(f"Unknown batch_mode {self.batch_mode!r}")

    def __len__(self) -> int:
        if self.batch_mode == "gene_diverse":
            return len(self.genes)
        return len(self.genes) // self.genes_per_batch
