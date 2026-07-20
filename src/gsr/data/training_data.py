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

from gsr.backbone.pooling import WT_MEAN_POOLINGS
from gsr.cache.wt_mean_cache import WtMeanCache, ensure_and_broadcast
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
                    mut=None, wt=None, wt_mean_cache=None):
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

    # Dedup WT means across ALL misses up front (not per bs-sized chunk) --
    # this is what makes wt_mean a once-per-gene cost rather than
    # once-per-variant, since the same WT sequence repeats across every
    # variant of that gene.
    wt_mean_by_seq = None
    if wt_mean_cache is not None:
        unique_wt = sorted({wseqs[i] for i in missing_positions})
        wt_mean_unique = ensure_and_broadcast(wt_mean_cache, backbone, unique_wt, bs)
        wt_mean_by_seq = dict(zip(unique_wt, wt_mean_unique))

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
        wm = (torch.stack([wt_mean_by_seq[wseqs[i]] for i in idx])
              if wt_mean_by_seq is not None else None)
        m, w = backbone.embed_pair(
            [mseqs[i] for i in idx], [wseqs[i] for i in idx],
            layer=args.embedding_layer, pooling=args.pooling, positions=p,
            wt_mean=wm)
        m, w = m.float().cpu().numpy(), w.float().cpu().numpy()
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
    wt_mean_cache = (WtMeanCache(args.esm_model, args.embedding_layer)
                     if args.pooling in WT_MEAN_POOLINGS else None)

    if not resident:
        missing_ids = set(emb_cache.missing_ids(vids))
        missing_pos = [i for i, v in enumerate(vids) if v in missing_ids]
        print(f"[emb] warm: {len(vids)-len(missing_pos)}/{len(vids)} cached, "
              f"{len(missing_pos)} to compute ({args.esm_model} {args.pooling})")
        if missing_pos and not _compute_misses(df, args, backbone, emb_cache,
                                               missing_pos,
                                               wt_mean_cache=wt_mean_cache):
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
                                       mut=mut, wt=wt,
                                       wt_mean_cache=wt_mean_cache):
        print("[emb] WARNING: cache locked by another writer; some embeddings "
              "were NOT persisted this run (kept in memory).")
    return torch.from_numpy(mut), torch.from_numpy(wt)
