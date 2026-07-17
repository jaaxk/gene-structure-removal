"""Build / load a cache of frozen-ESM embeddings for DMS eval variants.

The backbone is frozen, so DMS variant embeddings are computed once and cached to
scratch; every eval (during training or standalone) then just applies the current
projection head to these cached embeddings. Cache key encodes model/layer/pooling
/max_per_assay so different configs never collide.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from gsr import paths
from gsr.data.dms import load_dms


def _cache_key(args) -> str:
    types = "-".join(sorted(args.dms_selection_types))
    return (f"{args.esm_model}_layer{args.embedding_layer}_{args.pooling}"
            f"_max{args.dms_max_per_assay}_{types}")


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
    seqs = meta["mutated_sequence"].tolist()
    positions = meta["pos"].tolist()
    embs = []
    for start in tqdm(range(0, len(seqs), bs), desc="embed DMS"):
        e = backbone.embed(seqs[start:start + bs], layer=args.embedding_layer,
                           pooling=args.pooling,
                           positions=positions[start:start + bs])
        embs.append(e.float().cpu().numpy())
    emb = np.concatenate(embs, axis=0).astype(np.float32)

    cache_dir.mkdir(parents=True, exist_ok=True)
    meta.to_parquet(meta_path, index=False)
    with h5py.File(emb_path, "w") as h5:
        h5.create_dataset("X", data=emb, compression="gzip", compression_opts=4)
    print(f"[dms_cache] cached {emb.shape} -> {cache_dir}")
    return emb, meta
