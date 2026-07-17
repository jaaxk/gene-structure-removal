"""Triplet-margin contrastive loss over in-batch label groups.

Alternative to the BCE loss. For each anchor we form (anchor, positive, negative)
triplets where the positive shares the anchor's label and the negative does not,
and minimize ``max(0, d(a,p) - d(a,n) + margin)`` with euclidean distance on
L2-normalized embeddings. Selected via ``--loss_type triplet``.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from gsr.losses.base import BaseLoss, MIDDLE


class TripletLoss(BaseLoss):
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, z: torch.Tensor, y: torch.Tensor
                ) -> Tuple[torch.Tensor, Dict[str, float]]:
        valid = y != MIDDLE
        if valid.sum() < 3:
            zero = z.sum() * 0.0
            return zero, {"loss": 0.0, "n_triplets": 0.0}
        z = F.normalize(z[valid], dim=-1)
        y = y[valid]
        dist = torch.cdist(z, z, p=2)                         # (B,B)
        same = y.unsqueeze(0) == y.unsqueeze(1)
        eye = torch.eye(len(y), dtype=torch.bool, device=z.device)
        pos_mask = same & ~eye
        neg_mask = ~same
        # Batch-hard: hardest positive (farthest) and hardest negative (closest).
        anchors = pos_mask.any(1) & neg_mask.any(1)
        if anchors.sum() == 0:
            zero = z.sum() * 0.0
            return zero, {"loss": 0.0, "n_triplets": 0.0}
        hardest_pos = (dist.masked_fill(~pos_mask, float("-inf"))).max(1).values
        hardest_neg = (dist.masked_fill(~neg_mask, float("inf"))).min(1).values
        losses = F.relu(hardest_pos - hardest_neg + self.margin)[anchors]
        loss = losses.mean()
        return loss, {"loss": float(loss.detach()),
                      "n_triplets": float(anchors.sum())}
