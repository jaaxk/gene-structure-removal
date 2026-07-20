"""Build / load a cache of frozen-ESM embeddings for DMS eval variants.

The backbone is frozen, so DMS variant embeddings are computed once and cached to
scratch; every eval (during training or standalone) then just applies the current
projection head to these cached embeddings. Cache key encodes model/layer/pooling
/max_per_assay so different configs never collide.

For the two WT-mean-aware pooling modes (see gsr.backbone.pooling), the
projection head's input_dim was fixed at training time to include WT context,
so these DMS embeddings must be constructed the identical way: this module
joins the DMS_substitutions.csv WT reference (gsr.data.dms.load_wt_reference,
the same one gsr.eval.llr_projection uses) to get each row's WT sequence, and
computes/caches a per-dms_id WT mean (gsr.cache.wt_mean_cache) exactly as the
training pipeline does -- one shared implementation (pool_batch/embed), not
duplicated logic. Old pooling modes are completely unaffected (no WT join, no
wt_mean, byte-for-byte identical cache).
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from gsr import paths
from gsr.backbone.pooling import WT_MEAN_POOLINGS
from gsr.cache.wt_mean_cache import WtMeanCache, ensure_and_broadcast
from gsr.data.dms import load_dms, load_wt_reference


def _cache_key(args) -> str:
    types = "-".join(sorted(args.dms_selection_types))
    return (f"{args.esm_model}_layer{args.embedding_layer}_{args.pooling}"
            f"_max{args.dms_max_per_assay}_{types}")


def dms_cache_exists(args) -> bool:
    """Whether the DMS eval embedding cache for this config is already built."""
    cache_dir = paths.EVAL_DIR / "dms_cache" / _cache_key(args)
    return (cache_dir / "meta.parquet").exists() and \
        (cache_dir / "embeddings.h5").exists()


def build_or_load_dms_cache(args, backbone=None):
    """Return (embeddings (N,D) float32, meta DataFrame). Builds + caches if absent."""
    key = _cache_key(args)
    cache_dir = paths.EVAL_DIR / "dms_cache" / key
    meta_path = cache_dir / "meta.parquet"
    emb_path = cache_dir / "embeddings.h5"

    if meta_path.exists() and emb_path.exists():
        meta = pd.read_parquet(meta_path)
        with h5py.File(emb_path, "r") as h5:
            emb = h5["X"][:]
        print(f"[dms_cache] loaded {len(meta)} cached DMS embeddings ({key})")
        return emb, meta

    print(f"[dms_cache] building DMS embedding cache: {key}")
    meta = load_dms(args.dms_selection_types, max_per_assay=args.dms_max_per_assay,
                    seed=args.seed)
    if backbone is None:
        from gsr.backbone.esm import ESMBackbone
        device = args.device if torch.cuda.is_available() else "cpu"
        backbone = ESMBackbone(args.esm_model, device=device)

    bs = getattr(args, "score_batch_size", 16)
    wt_mean_tensor = None
    if args.pooling in WT_MEAN_POOLINGS:
        ref = load_wt_reference()
        meta = meta.merge(ref[["dms_id", "target_seq"]], on="dms_id", how="left",
                          validate="many_to_one")
        n_missing = int(meta["target_seq"].isna().sum())
        if n_missing:
            print(f"[dms_cache] WARNING: {n_missing} variants have no target_seq "
                  "match in DMS_substitutions.csv; dropping")
            meta = meta.dropna(subset=["target_seq"]).reset_index(drop=True)
        wt_mean_cache = WtMeanCache(args.esm_model, args.embedding_layer)
        wt_mean_tensor = ensure_and_broadcast(wt_mean_cache, backbone,
                                              meta["target_seq"].tolist(), bs)

    seqs = meta["mutated_sequence"].tolist()
    positions = meta["pos"].tolist()
    embs = []
    for start in tqdm(range(0, len(seqs), bs), desc="embed DMS"):
        sl = slice(start, start + bs)
        wm = wt_mean_tensor[sl] if wt_mean_tensor is not None else None
        e = backbone.embed(seqs[sl], layer=args.embedding_layer,
                           pooling=args.pooling, positions=positions[sl],
                           wt_mean=wm)
        embs.append(e.float().cpu().numpy())
    emb = np.concatenate(embs, axis=0).astype(np.float32)

    cache_dir.mkdir(parents=True, exist_ok=True)
    meta.to_parquet(meta_path, index=False)
    with h5py.File(emb_path, "w") as h5:
        h5.create_dataset("X", data=emb, compression="gzip", compression_opts=4)
    print(f"[dms_cache] cached {emb.shape} -> {cache_dir}")
    return emb, meta
