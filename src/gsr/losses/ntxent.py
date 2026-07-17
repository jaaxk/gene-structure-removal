"""NT-Xent (InfoNCE) contrastive loss over in-batch label groups.

Alternative to the BCE loss. For each anchor, the positives are the other items
sharing its (non-middle) label; the loss is the standard normalized-temperature
cross-entropy with cosine similarity. Anchors with no in-batch positive are
skipped. Selected via ``--loss_type ntxent``.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from gsr.losses.base import BaseLoss, MIDDLE


class NTXentLoss(BaseLoss):
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z: torch.Tensor, y: torch.Tensor
                ) -> Tuple[torch.Tensor, Dict[str, float]]:
        valid = y != MIDDLE
        if valid.sum() < 2:
            zero = z.sum() * 0.0
            return zero, {"loss": 0.0, "n_anchors": 0.0}
        z = F.normalize(z[valid], dim=-1)
        y = y[valid]
        sim = (z @ z.t()) / self.temperature                # (B,B)
        B = z.shape[0]
        self_mask = torch.eye(B, dtype=torch.bool, device=z.device)
        sim.masked_fill_(self_mask, float("-inf"))
        pos_mask = (y.unsqueeze(0) == y.unsqueeze(1)) & ~self_mask
        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        has_pos = pos_mask.any(dim=1)
        if has_pos.sum() == 0:
            zero = z.sum() * 0.0
            return zero, {"loss": 0.0, "n_anchors": 0.0}
        pos_log_prob = (log_prob * pos_mask).sum(1)[has_pos] / \
            pos_mask.sum(1)[has_pos].clamp(min=1)
        loss = -pos_log_prob.mean()
        return loss, {"loss": float(loss.detach()),
                      "n_anchors": float(has_pos.sum())}
