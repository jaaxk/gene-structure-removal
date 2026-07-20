"""Pooling strategies that turn per-residue ESM embeddings into one vector.

Five strategies, all config-driven:
- ``mean``               : attention-masked mean over residues (position-independent).
- ``mutated_position``   : the embedding at a specific residue position.
- ``concat``             : concat of mean and the position embedding (2*D).
- ``wt_mut_mean_concat`` : concat(mean, mutated_position, wt_mean) (3*D) -- bakes
  the WT gene's mean-pooled embedding into the vector the head sees, so
  variant-to-variant losses (which never otherwise look at the WT side) get
  gene/WT context too.
- ``wt_subtracted_mean`` : mean - wt_mean (1*D, mut_mean minus wt_mean --
  matches the sibling ``dms_contrastive`` repo's ``normalize_to_wt`` sign).

Position convention: the tokenizer prepends one special token, so a 1-indexed
residue position ``p`` is at token index ``p``. For a WILD-TYPE sequence pooled
relative to a mutant, the *same* position ``p`` (the mutated site) is used, so the
mutant and its WT are compared like-for-like at that residue.

``position=None`` (e.g. when a strategy needs no position) falls back to the mean
vector for the positional component.

``wt_mean`` (only consulted by the two WT-mean-aware modes above) is the WT
gene's own mean-pooled vector -- the SAME value for every variant of that gene,
since mean pooling doesn't depend on position. If omitted, it defaults to this
call's own ``mean`` -- i.e. pooling a sequence "relative to itself". This is
exactly the WT side's natural case (a WT embedded relative to its own mean),
so ``pool_batch`` needs no branching on which side (mutant or WT) is being
pooled: the same call handles both, and callers only need to supply an
explicit ``wt_mean`` when pooling the MUTANT side relative to a *different*
sequence's mean.
"""

from __future__ import annotations

from typing import List, Optional

import torch

WT_MEAN_POOLINGS = ("wt_mut_mean_concat", "wt_subtracted_mean")


def mean_pool(hidden: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
    """(B,L,D),(B,L) -> (B,D) attention-masked mean."""
    mask = attn.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)


def pool_batch(hidden: torch.Tensor, attn: torch.Tensor,
               positions: Optional[List[Optional[int]]], pooling: str,
               wt_mean: Optional[torch.Tensor] = None) -> torch.Tensor:
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
    if pooling in WT_MEAN_POOLINGS:
        if wt_mean is None:
            wt_mean = mean  # self-referential: see module docstring.
        else:
            # Externally supplied wt_mean (e.g. from WtMeanCache, which reads
            # numpy/CPU data) may not already be on hidden's device/dtype.
            wt_mean = wt_mean.to(device=mean.device, dtype=mean.dtype)
        if pooling == "wt_mut_mean_concat":
            return torch.cat([mean, pos_vec, wt_mean], dim=-1)
        return mean - wt_mean  # wt_subtracted_mean: mut_mean - wt_mean
    raise ValueError(f"Unknown pooling {pooling!r}")


def output_dim(hidden_dim: int, pooling: str) -> int:
    if pooling == "concat":
        return hidden_dim * 2
    if pooling == "wt_mut_mean_concat":
        return hidden_dim * 3
    return hidden_dim  # mean, mutated_position, wt_subtracted_mean
