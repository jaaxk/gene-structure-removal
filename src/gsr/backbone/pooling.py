"""Pooling strategies that turn per-residue ESM embeddings into one vector.

Three strategies, all config-driven:
- ``mean``             : attention-masked mean over residues (position-independent).
- ``mutated_position`` : the embedding at a specific residue position.
- ``concat``           : concat of mean and the position embedding (2*D).

Position convention: the tokenizer prepends one special token, so a 1-indexed
residue position ``p`` is at token index ``p``. For a WILD-TYPE sequence pooled
relative to a mutant, the *same* position ``p`` (the mutated site) is used, so the
mutant and its WT are compared like-for-like at that residue.

``position=None`` (e.g. when a strategy needs no position) falls back to the mean
vector for the positional component.
"""

from __future__ import annotations

from typing import List, Optional

import torch


def mean_pool(hidden: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
    """(B,L,D),(B,L) -> (B,D) attention-masked mean."""
    mask = attn.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)


def pool_batch(hidden: torch.Tensor, attn: torch.Tensor,
               positions: Optional[List[Optional[int]]], pooling: str) -> torch.Tensor:
    """Pool a batch of sequences to (B, out_dim) per the chosen strategy."""
    B, L, D = hidden.shape
    mean = mean_pool(hidden, attn)
    if pooling == "mean":
        return mean
    if positions is None:
        positions = [None] * B
    pos_vec = mean.clone()
    for i, p in enumerate(positions):
        if p is not None and 0 <= p < L:
            pos_vec[i] = hidden[i, p]
    if pooling == "mutated_position":
        return pos_vec
    if pooling == "concat":
        return torch.cat([mean, pos_vec], dim=-1)
    raise ValueError(f"Unknown pooling {pooling!r}")


def output_dim(hidden_dim: int, pooling: str) -> int:
    return hidden_dim * (2 if pooling == "concat" else 1)
