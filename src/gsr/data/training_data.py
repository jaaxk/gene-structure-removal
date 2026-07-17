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


def select_embeddings_mode(n_items: int, dim: int, max_resident_gb: float,
                           mode: str) -> str:
    """Resolve 'auto' to 'ram' (fits budget) or 'stream' by pool size (mut+wt)."""
    if mode != "auto":
        return mode
    pool_gb = n_items * dim * 4 * 2 / 1e9
    return "ram" if pool_gb <= max_resident_gb else "stream"


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


def _compute_misses(df, args, backbone, emb_cache, missing_positions,
                    mut=None, wt=None):
    """Compute embeddings for ``missing_positions`` (indices into df), append them
    to the cache incrementally (resumable), and optionally scatter into resident
    ``mut``/``wt`` arrays. Returns True if all flushes persisted."""
    bs = getattr(args, "score_batch_size", 16)
    flush_every = 4096
    mseqs = df["mutated_sequence"].tolist()
    wseqs = df["wt_seq"].tolist()
    pos = df["pos"].astype(int).tolist()
    vids = df["variant_id"].tolist()
    pending_ids, pending_mut, pending_wt, all_saved = [], [], [], True

    def _flush():
        nonlocal all_saved, pending_ids, pending_mut, pending_wt
        if not pending_ids:
            return
        ok = emb_cache.put(pending_ids, np.stack(pending_mut), np.stack(pending_wt))
        all_saved = all_saved and ok
        pending_ids, pending_mut, pending_wt = [], [], []

    for s in tqdm(range(0, len(missing_positions), bs), desc="embed misses"):
        idx = missing_positions[s:s + bs]
        p = [pos[i] for i in idx]
        m = backbone.embed([mseqs[i] for i in idx], layer=args.embedding_layer,
                           pooling=args.pooling, positions=p).float().cpu().numpy()
        w = backbone.embed([wseqs[i] for i in idx], layer=args.embedding_layer,
                           pooling=args.pooling, positions=p).float().cpu().numpy()
        for k, i in enumerate(idx):
            if mut is not None:
                mut[i] = m[k]
                wt[i] = w[k]
            pending_ids.append(vids[i])
            pending_mut.append(m[k])
            pending_wt.append(w[k])
        if len(pending_ids) >= flush_every:
            _flush()
    _flush()
    return all_saved


def fill_embeddings(df: pd.DataFrame, args, backbone, emb_cache,
                    resident: bool = True):
    """Ensure (mut, wt) embeddings for df are cached; optionally return them resident.

    resident=True  -> bulk-load cached + compute misses, return (mut, wt) tensors
                      aligned to df (used by the in-RAM VariantDataset).
    resident=False -> only compute+append missing embeddings to the cache in
                      O(batch) memory (used to warm the cache for streaming). Returns
                      (None, None).

    Missing embeddings never raise; they are persisted unless another writer holds
    the cache lock, in which case a warning is printed (kept for this run only).
    """
    vids = df["variant_id"].tolist()
    D = backbone.output_dim(args.pooling)

    if not resident:
        missing_ids = set(emb_cache.missing_ids(vids))
        missing_pos = [i for i, v in enumerate(vids) if v in missing_ids]
        print(f"[emb] warm: {len(vids)-len(missing_pos)}/{len(vids)} cached, "
              f"{len(missing_pos)} to compute ({args.esm_model} {args.pooling})")
        if missing_pos and not _compute_misses(df, args, backbone, emb_cache,
                                               missing_pos):
            print("[emb] WARNING: cache locked by another writer; some embeddings "
                  "were NOT persisted this run.")
        return None, None

    # Determine misses cheaply from the sidecar (no bulk H5 read / NaN alloc) so
    # progress is visible immediately and a mostly-cold cache doesn't stall on a
    # giant eager allocation.
    missing_set = set(emb_cache.missing_ids(vids))
    missing = [i for i, v in enumerate(vids) if v in missing_set]
    print(f"[emb] {len(vids)-len(missing)}/{len(vids)} cached, {len(missing)} to "
          f"compute ({args.esm_model} {args.pooling}); resident {len(vids)}x{D}",
          flush=True)
    mut = np.zeros((len(vids), D), np.float32)  # lazy calloc, not eager NaN
    wt = np.zeros((len(vids), D), np.float32)
    if len(missing) < len(vids):                # read only the cached rows
        cached_pos = [i for i, v in enumerate(vids) if v not in missing_set]
        cmut, cwt, _ = emb_cache.get([vids[i] for i in cached_pos])
        mut[cached_pos] = cmut
        wt[cached_pos] = cwt
    if missing and not _compute_misses(df, args, backbone, emb_cache, missing,
                                       mut=mut, wt=wt):
        print("[emb] WARNING: cache locked by another writer; some embeddings "
              "were NOT persisted this run (kept in memory).")
    return torch.from_numpy(mut), torch.from_numpy(wt)
