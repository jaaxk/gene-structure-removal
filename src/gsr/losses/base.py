"""Loss interface + shared label conventions.

Every loss takes projected mutant embeddings ``z_mut`` (B, D), the projected
embedding of each mutant's own wild-type ``z_wt`` (B, D), and integer labels
``y`` (B,), returning ``(loss, metrics)``. Label ids:

    SAME=0, DIFFERENT=1, MIDDLE=-1  (MIDDLE rows are excluded from every loss)

Two loss families share this interface:
- WT-anchored (``wt_anchored_bce``): compares each mutant to ITS wild-type
  (uses z_wt) -- pull together when the likelihood is close ('same'), apart when
  far ('different').
- In-batch (``contrastive_ce``, ``ntxent``, ``triplet``): compares mutants to each
  other by shared label (uses only z_mut; z_wt is ignored).
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

SAME = 0
DIFFERENT = 1
MIDDLE = -1

LABEL_TO_ID = {"same": SAME, "different": DIFFERENT, "middle": MIDDLE}


class BaseLoss(nn.Module):
    def forward(self, z_mut: torch.Tensor, z_wt: torch.Tensor, y: torch.Tensor
                ) -> Tuple[torch.Tensor, Dict[str, float]]:
        raise NotImplementedError


def pairwise_similarity(z: torch.Tensor, metric: str) -> torch.Tensor:
    """(B,B) similarity matrix; higher = more similar for both metrics."""
    if metric == "cosine":
        zn = torch.nn.functional.normalize(z, dim=-1)
        return zn @ zn.t()
    if metric == "euclidean":
        # negative distance so larger = more similar
        return -torch.cdist(z, z, p=2)
    raise ValueError(f"Unknown distance metric {metric!r}")
