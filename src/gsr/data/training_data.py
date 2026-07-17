"""Assemble training data from the score + embedding caches (lazy hybrid flow).

Steps:
1. Per gene, get all-variant scores from the ScoreCache (compute+cache on miss).
2. Label over the FULL variant distribution (stable quartiles), keep same/different.
3. Sample ``variants_per_gene`` of those as the gene's training pool.
4. (Frozen only) fill (mut, wt) embeddings for the pooled variants from the
   EmbeddingCache, computing misses live and caching them; hold them RESIDENT so
   the training loop does no per-batch I/O.

The metadata DataFrame it returns feeds either VariantDataset (frozen, with the
resident embeddings) or SequenceVariantDataset (LoRA, sequences embedded live).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from gsr.data.labeling import assign_labels
from gsr.data.uniref import WTRecord
from gsr.losses.base import LABEL_TO_ID


def build_metadata(records: List[WTRecord], args, backbone, score_cache
                   ) -> pd.DataFrame:
    """Score (cached), label over all variants, and sample the training pool."""
    rng = np.random.default_rng(args.seed)
    # pll is per-variant expensive -> score only a capped candidate set; marginal
    # scorers score every variant for free (one sweep) -> no cap.
    candidate_cap = max(args.variants_per_gene * 4, 200) if args.scorer == "pll" else 0
    if candidate_cap:
        print(f"[score] pll scorer: capping candidates to {candidate_cap}/gene "
              f"(quartiles computed over that subset).")
    frames = []
    for rec in tqdm(records, desc="score+sample"):
        df_all = score_cache.get_or_compute(
            rec.gene_id, rec.sequence, backbone,
            score_batch_size=getattr(args, "score_batch_size", 16),
            candidate_cap=candidate_cap, seed=args.seed)
        labeled = assign_labels(df_all, quartile_low=args.quartile_low,
                                quartile_high=args.quartile_high,
                                min_variants=args.min_variants_per_gene)
        if rec.gene_id not in set(labeled["gene_id"]):
            continue
        sd = labeled[(~labeled["is_wt"]) &
                     labeled["label"].isin(["same", "different"])].copy()
        if len(sd) < 2:
            continue
        if len(sd) > args.variants_per_gene:
            sd = sd.iloc[rng.choice(len(sd), args.variants_per_gene, replace=False)]
        sd["wt_seq"] = rec.sequence
        frames.append(sd)

    if not frames:
        raise ValueError("No training variants produced; check data/quartiles.")
    df = pd.concat(frames, ignore_index=True)
    df["label_id"] = df["label"].map(LABEL_TO_ID).astype(int)
    return df


def fill_embeddings(df: pd.DataFrame, args, backbone, emb_cache
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Bulk-load (mut, wt) embeddings from cache; compute+cache misses live.

    Returns resident CPU tensors aligned to ``df`` rows. Missing embeddings never
    raise -- they are computed live; they are persisted unless another writer holds
    the cache lock, in which case a warning is printed and they stay in memory only.
    """
    vids = df["variant_id"].tolist()
    mut_c, wt_c, missing = emb_cache.get(vids)
    D = backbone.output_dim(args.pooling)
    mut = np.zeros((len(vids), D), np.float32)
    wt = np.zeros((len(vids), D), np.float32)
    if mut_c.size:                       # some were cached
        found = [i for i in range(len(vids)) if i not in set(missing)]
        mut[found] = mut_c[found]
        wt[found] = wt_c[found]
    print(f"[emb] {len(vids)-len(missing)}/{len(vids)} cached, "
          f"{len(missing)} to compute ({args.esm_model} {args.pooling})")

    if missing:
        bs = getattr(args, "score_batch_size", 16)
        flush_every = 4096  # persist incrementally so long fills are resumable
        mseqs = df["mutated_sequence"].tolist()
        wseqs = df["wt_seq"].tolist()
        pos = df["pos"].astype(int).tolist()
        pending, any_unsaved = [], False

        def _flush(rows):
            nonlocal any_unsaved
            if not rows:
                return
            ok = emb_cache.put([vids[i] for i in rows],
                               mut[rows], wt[rows])
            any_unsaved = any_unsaved or (not ok)

        for s in tqdm(range(0, len(missing), bs), desc="embed misses"):
            idx = missing[s:s + bs]
            p = [pos[i] for i in idx]
            m = backbone.embed([mseqs[i] for i in idx], layer=args.embedding_layer,
                               pooling=args.pooling, positions=p).float().cpu().numpy()
            w = backbone.embed([wseqs[i] for i in idx], layer=args.embedding_layer,
                               pooling=args.pooling, positions=p).float().cpu().numpy()
            for k, i in enumerate(idx):
                mut[i] = m[k]
                wt[i] = w[k]
            pending.extend(idx)
            if len(pending) >= flush_every:
                _flush(pending)
                pending = []
        _flush(pending)
        if any_unsaved:
            print("[emb] WARNING: cache locked by another writer for some flushes; "
                  "those embeddings were NOT persisted this run (kept in memory).")
    return torch.from_numpy(mut), torch.from_numpy(wt)
