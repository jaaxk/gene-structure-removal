"""Frozen-ESM likelihood scorers that produce contrastive labels.

For each wild-type gene we score its sampled single-aa variants and return, per
variant, ``(wt_score, mut_score, delta)`` where ``delta = LL_mut - LL_wt`` is the
signed likelihood change used for labeling. The absolute delta's per-gene
quartiles decide 'same' vs 'different' (see data/labeling.py).

Three scorers, from cheapest to most faithful:

- ``wt_marginal``  : one unmasked forward over the WT; delta at position p is the
  log-likelihood ratio ``logp(mut|wt-context) - logp(wt|wt-context)``. 1 pass/gene.
- ``masked_marginal`` (default): mask each mutated position in the WT, read the
  ratio there. ~L passes/gene, scores every variant at every masked position; the
  standard ESM-1v/ProteinGym choice.
- ``pll`` : full pseudo-log-likelihood of each sequence
  ``PLL(seq)=sum_i logp(seq_i | seq masked at i)``; delta = PLL(mut) - PLL(wt).
  ~L passes PER sequence -- expensive, opt-in, only over sampled variants.

Note: log-softmax is over the full vocabulary; the normalization constant cancels
in ``delta`` for the marginal scorers, so restricting to canonical AAs is unneeded.
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F
from tqdm import tqdm

from gsr.data.mutagenesis import Variant


def _aa2id(backbone) -> Dict[str, int]:
    return {aa: int(backbone.aa_token_ids[i]) for aa, i in backbone.aa_to_col.items()}


def _masked_logp_rows(backbone, base_ids, base_attn, positions, batch_size):
    """Return log-softmax rows at each masked token position.

    For each token index p in ``positions``, mask position p of ``base_ids`` and
    return the (vocab,) log-softmax at p. Result: (len(positions), vocab).
    """
    rows = []
    for start in range(0, len(positions), batch_size):
        chunk = positions[start:start + batch_size]
        ids = base_ids.repeat(len(chunk), 1).clone()
        attn = base_attn.repeat(len(chunk), 1)
        for i, p in enumerate(chunk):
            ids[i, p] = backbone.mask_token_id
        logits = backbone.forward_logits(ids, attn)     # (b, L, V)
        logp = F.log_softmax(logits.float(), dim=-1)
        for i, p in enumerate(chunk):
            rows.append(logp[i, p].cpu())
    return torch.stack(rows) if rows else torch.empty(0)


def _score_wt_marginal(backbone, wt_seq, variants, aa2id, batch_size):
    ids, attn = backbone.tokenize([wt_seq])
    logits = backbone.forward_logits(ids, attn)[0]       # (L, V)
    logp = F.log_softmax(logits.float(), dim=-1).cpu()
    out = []
    for v in variants:
        ws = float(logp[v.pos, aa2id[v.wt_aa]])
        ms = float(logp[v.pos, aa2id[v.mut_aa]])
        out.append((ws, ms, ms - ws))
    return out


def _score_masked_marginal(backbone, wt_seq, variants, aa2id, batch_size):
    ids, attn = backbone.tokenize([wt_seq])
    positions = sorted({v.pos for v in variants})
    rows = _masked_logp_rows(backbone, ids, attn, positions, batch_size)
    pos_to_row = {p: rows[i] for i, p in enumerate(positions)}
    out = []
    for v in variants:
        logp = pos_to_row[v.pos]
        ws = float(logp[aa2id[v.wt_aa]])
        ms = float(logp[aa2id[v.mut_aa]])
        out.append((ws, ms, ms - ws))
    return out


def _pll_of_sequence(backbone, seq, batch_size):
    ids, attn = backbone.tokenize([seq])
    L = ids.shape[1]
    positions = list(range(1, L - 1))  # skip leading/trailing special tokens
    rows = _masked_logp_rows(backbone, ids, attn, positions, batch_size)
    true_ids = ids[0, positions].cpu()
    return float(rows[torch.arange(len(positions)), true_ids].sum())


def _score_pll(backbone, wt_seq, variants, aa2id, batch_size):
    pll_wt = _pll_of_sequence(backbone, wt_seq, batch_size)
    out = []
    for v in tqdm(variants, desc="pll variants", leave=False):
        pll_mut = _pll_of_sequence(backbone, v.sequence, batch_size)
        out.append((pll_wt, pll_mut, pll_mut - pll_wt))
    return out


_SCORERS = {
    "wt_marginal": _score_wt_marginal,
    "masked_marginal": _score_masked_marginal,
    "pll": _score_pll,
}


def score_gene(
    backbone,
    wt_seq: str,
    variants: List[Variant],
    scorer: str = "masked_marginal",
    batch_size: int = 8,
):
    """Score a gene's variants; returns list of (wt_score, mut_score, delta)."""
    if scorer not in _SCORERS:
        raise ValueError(f"Unknown scorer {scorer!r}; choices: {list(_SCORERS)}")
    if not variants:
        return []
    aa2id = _aa2id(backbone)
    return _SCORERS[scorer](backbone, wt_seq, variants, aa2id, batch_size)
