"""Cross-entropy (BCE) contrastive loss over in-batch pairs.

Mirrors the sibling repo's ``ContrastiveLoss`` (dms_contrastive/src/models.py):
for every valid upper-triangle pair, the target is 1 if the two items share a
label and 0 otherwise, and the predicted probability is a sigmoid of a learnable
affine transform of the pair similarity (``alpha`` scale + ``beta`` bias) -- i.e.
a learnable "temperature" rather than a fixed margin. Middle-labeled items are
excluded from all pairs.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from gsr.losses.base import BaseLoss, MIDDLE, pairwise_similarity


class ContrastiveCELoss(BaseLoss):
    def __init__(self, distance_metric: str = "cosine",
                 use_learnable_scale: bool = True):
        super().__init__()
        self.distance_metric = distance_metric
        self.use_learnable_scale = use_learnable_scale
        # Init sign matches the metric's "similar => large" convention.
        init_alpha = 10.0 if distance_metric == "cosine" else 1.0
        self.alpha = nn.Parameter(torch.tensor(init_alpha))
        self.beta = nn.Parameter(torch.tensor(0.0))
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, z: torch.Tensor, y: torch.Tensor
                ) -> Tuple[torch.Tensor, Dict[str, float]]:
        sim = pairwise_similarity(z, self.distance_metric)          # (B,B)
        same_label = (y.unsqueeze(0) == y.unsqueeze(1)).float()     # (B,B)
        valid = (y != MIDDLE)
        valid_pair = valid.unsqueeze(0) & valid.unsqueeze(1)
        upper = torch.triu(torch.ones_like(same_label), diagonal=1).bool()
        mask = upper & valid_pair
        if mask.sum() == 0:
            zero = z.sum() * 0.0
            return zero, {"loss": 0.0, "n_pairs": 0.0}

        if self.use_learnable_scale:
            logits = self.alpha * sim + self.beta
        else:
            logits = sim
        logits = logits[mask]
        targets = same_label[mask]
        loss = self.bce(logits, targets)
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            acc = ((probs > 0.5).float() == targets).float().mean()
        return loss, {
            "loss": float(loss.detach()),
            "pair_acc": float(acc),
            "n_pairs": float(mask.sum()),
            "pos_frac": float(targets.mean()),
            "alpha": float(self.alpha.detach()),
            "beta": float(self.beta.detach()),
        }
