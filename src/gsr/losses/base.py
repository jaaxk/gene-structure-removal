"""Loss interface + shared label conventions.

Every loss takes projected embeddings ``z`` (B, D) and integer labels ``y`` (B,)
and returns ``(loss, metrics)`` where ``metrics`` is a dict of scalar floats for
logging. Label ids:

    SAME=0, DIFFERENT=1, MIDDLE=-1  (MIDDLE rows are excluded from every loss)

A pair is a *positive* iff both items share the same non-middle label (both SAME
or both DIFFERENT); this pulls likelihood-similar variants together and pushes
likelihood-dissimilar ones apart, regardless of gene.
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
    def forward(self, z: torch.Tensor, y: torch.Tensor
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
