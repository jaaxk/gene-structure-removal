"""WT-anchored cross-entropy contrastive loss (project default).

Each mutant is compared to ITS OWN wild-type: the predicted probability that the
pair is 'same' (likelihood close) is ``sigmoid(alpha * sim(z_mut, z_wt) + beta)``
with a learnable scale/bias, and the BCE target is 1 for 'same', 0 for
'different'. This directly realizes the design intent -- pull the mutant toward
its WT when their ESM likelihoods are similar, push apart when dissimilar.

For mutated_position/concat pooling the WT embedding was pooled at the same
mutated residue as the variant, so the comparison isolates the substitution.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from gsr.losses.base import BaseLoss, MIDDLE, SAME


class WTAnchoredBCELoss(BaseLoss):
    def __init__(self, distance_metric: str = "cosine",
                 use_learnable_scale: bool = True):
        super().__init__()
        self.distance_metric = distance_metric
        self.use_learnable_scale = use_learnable_scale
        init_alpha = 10.0 if distance_metric == "cosine" else -1.0
        self.alpha = nn.Parameter(torch.tensor(init_alpha))
        self.beta = nn.Parameter(torch.tensor(0.0))
        self.bce = nn.BCEWithLogitsLoss()

    def _pair_similarity(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self.distance_metric == "cosine":
            return F.cosine_similarity(a, b, dim=-1)
        if self.distance_metric == "euclidean":
            return -torch.linalg.vector_norm(a - b, dim=-1)  # larger = closer
        raise ValueError(f"Unknown metric {self.distance_metric!r}")

    def forward(self, z_mut: torch.Tensor, z_wt: torch.Tensor, y: torch.Tensor
                ) -> Tuple[torch.Tensor, Dict[str, float]]:
        valid = y != MIDDLE
        if valid.sum() == 0:
            zero = z_mut.sum() * 0.0
            return zero, {"loss": 0.0, "n_items": 0.0}
        zm, zw, yy = z_mut[valid], z_wt[valid], y[valid]
        sim = self._pair_similarity(zm, zw)
        logits = self.alpha * sim + self.beta if self.use_learnable_scale else sim
        target = (yy == SAME).float()      # 'same' -> pull together (prob 1)
        loss = self.bce(logits, target)
        with torch.no_grad():
            acc = ((torch.sigmoid(logits) > 0.5).float() == target).float().mean()
        return loss, {
            "loss": float(loss.detach()),
            "pair_acc": float(acc),
            "n_items": float(valid.sum()),
            "same_frac": float(target.mean()),
            "alpha": float(self.alpha.detach()),
            "beta": float(self.beta.detach()),
        }
