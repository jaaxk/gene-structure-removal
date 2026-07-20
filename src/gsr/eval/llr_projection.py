"""Zero-shot LLR-projection metric.

For every DMS variant across ALL 186 ProteinGym genes (no held-out split --
this is a zero-shot correlation, each gene randomly subsampled to a fixed
variant count so no single large gene dominates), correlate a "projection
effect score" (cosine distance between the projected WT and mutant
embeddings) against LLR (``delta = LL_mut - LL_wt``, the frozen-ESM
log-likelihood ratio already used by ``gsr.scoring.scorer`` to label UniRef90
training data -- computed here for the first time on DMS data). A single
pooled Spearman across every gene/selection-type is the point: if the
projection has actually removed gene identity signal, effect magnitude should
sit on the same scale across genes, so it correlates with LLR even when
everything is pooled together.

Two things are genuinely new here (never computed before anywhere in this
repo): the LLR scores themselves, and WT embeddings (``dms_cache.py`` only
ever caches MUTANT embeddings -- ``CentroidDMSEvaluator`` compares a variant
to a centroid of other variants, never to its own WT). Mutant embeddings for
this exact config are reused directly from the existing
``gsr.eval.dms_cache`` cache (built once, shared with ``CentroidDMSEvaluator``)
rather than recomputed -- ``build_or_load_dms_cache`` is called with the same
args here and there, so whichever evaluator runs first warms it for the other.

Cached artifacts under ``paths.EVAL_DIR / "llr_projection"`` (parquet + HDF5,
same convention as ``dms_cache.py``):
  - the per-gene-subsampled variant table (also carries ``_dms_cache_row``,
    each row's position in the shared dms_cache meta/embeddings -- how mutant
    embeddings are looked up without recomputing them)
  - LLR values for that table (depends on esm_model + scorer only)
  - WT embeddings, deduped by (dms_id, pos) since multiple substitutions at
    the same site share one WT-at-that-position embedding (depends on
    esm_model + layer + pooling)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Dict, Optional

import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from gsr import paths
from gsr.data.dms import load_dms, load_wt_reference
from gsr.data.mutagenesis import Variant
from gsr.utils.spearman import spearman

_MUTANT_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")


def _key(args) -> str:
    types = "-".join(sorted(args.dms_selection_types))
    return f"pergene{args.dms_per_gene_subsample}_seed{args.seed}_{types}"


def _cache_dir() -> Path:
    return paths.EVAL_DIR / "llr_projection"


def _meta_path(args) -> Path:
    return _cache_dir() / f"meta_{_key(args)}.parquet"


def _llr_path(args) -> Path:
    return _cache_dir() / f"llr_{args.esm_model}_{args.scorer}_{_key(args)}.parquet"


def _wt_emb_path(args) -> Path:
    """WT embeddings only -- mutant embeddings are reused from dms_cache.py."""
    return (_cache_dir() / f"wt_emb_{args.esm_model}_layer{args.embedding_layer}"
            f"_{args.pooling}_{_key(args)}.h5")


def llr_projection_cache_exists(args) -> bool:
    """Whether this evaluator's own cached artifacts exist (meta, LLR, WT
    embeddings). Does NOT check dms_cache.py's mutant-embedding cache --
    scripts/train.py's CPU guard checks that separately via dms_cache_exists."""
    return (_meta_path(args).exists() and _llr_path(args).exists()
            and _wt_emb_path(args).exists())


def _build_meta(args) -> pd.DataFrame:
    m = load_dms(args.dms_selection_types, max_per_assay=args.dms_max_per_assay,
                 seed=args.seed)
    # dms_cache.py builds its embedding cache from this SAME load_dms(...) call
    # (identical args), in this same row order, and never reorders it before
    # caching -- so this row's position IS its position in that cache's
    # embeddings array. Recorded now, before any merge/dropna/subsampling, so
    # mutant embeddings can be looked up later instead of recomputed.
    m["_dms_cache_row"] = np.arange(len(m))

    # Per-assay hi/lo quartile membership, computed on the FULL per-assay pool
    # before subsampling -- same convention as CentroidDMSEvaluator._prepare.
    m["_hi"] = False
    m["_lo"] = False
    for _dms, g in m.groupby("dms_id"):
        hi = g["DMS_score"] >= g["DMS_score"].quantile(0.75)
        lo = g["DMS_score"] <= g["DMS_score"].quantile(0.25)
        m.loc[g.index[hi], "_hi"] = True
        m.loc[g.index[lo], "_lo"] = True

    ref = load_wt_reference()
    # how="left" + validate="many_to_one" preserves row count/order, so
    # _dms_cache_row set above stays correct through this merge.
    m = m.merge(ref, on="dms_id", how="left", validate="many_to_one")
    missing = int(m["target_seq"].isna().sum())
    if missing:
        print(f"[llr_projection] WARNING: {missing} variants have no target_seq "
              "match in DMS_substitutions.csv; dropping")
        m = m.dropna(subset=["target_seq"])
    m["relative_pos"] = m["pos"] / m["seq_len"]

    # Per-GENE (not per-assay) reproducible subsample: multi-assay genes are
    # pooled across their assays before capping, so big genes don't dominate
    # the pooled Spearman.
    rng = np.random.default_rng(args.seed)
    kept = []
    for _gene, g in m.groupby("uniprot_id"):
        if len(g) > args.dms_per_gene_subsample:
            idx = rng.choice(g.index.to_numpy(), args.dms_per_gene_subsample,
                              replace=False)
            kept.append(g.loc[idx])
        else:
            kept.append(g)
    out = pd.concat(kept, ignore_index=True)
    print(f"[llr_projection] {out['uniprot_id'].nunique()} genes, {len(out)} "
          f"variants (per-gene cap {args.dms_per_gene_subsample})")
    return out


def _build_or_load_meta(args) -> pd.DataFrame:
    path = _meta_path(args)
    if path.exists():
        meta = pd.read_parquet(path)
        print(f"[llr_projection] loaded {len(meta)} cached variants ({path.name})")
        return meta
    meta = _build_meta(args)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta.to_parquet(path, index=False)
    return meta


def _build_llr(args, meta: pd.DataFrame, backbone) -> np.ndarray:
    """Per-row LLR (delta = LL_mut - LL_wt), one scorer.score_gene call per assay."""
    from gsr.scoring.scorer import score_gene

    meta = meta.reset_index(drop=True)
    delta = np.full(len(meta), np.nan, dtype=np.float32)
    groups = list(meta.groupby("dms_id"))
    for dms_id, g in tqdm(groups, desc=f"LLR ({args.scorer})"):
        wt_seq = g["target_seq"].iloc[0]
        variants, row_positions = [], []
        for row_pos, row in g.iterrows():
            match = _MUTANT_RE.match(str(row["mutant"]))
            if not match:
                continue
            wt_aa, _, mut_aa = match.groups()
            variants.append(Variant(gene_id=dms_id, mutant=row["mutant"],
                                     pos=int(row["pos"]), wt_aa=wt_aa, mut_aa=mut_aa,
                                     sequence=row["mutated_sequence"]))
            row_positions.append(row_pos)
        if not variants:
            continue
        scores = score_gene(backbone, wt_seq, variants, scorer=args.scorer,
                             batch_size=args.score_batch_size)
        for row_pos, (_ws, _ms, d) in zip(row_positions, scores):
            delta[row_pos] = d
    n_nan = int(np.isnan(delta).sum())
    if n_nan:
        print(f"[llr_projection] WARNING: {n_nan}/{len(delta)} variants have no "
              "parseable LLR (unparsed mutant string)")
    return delta


def _build_or_load_llr(args, meta: pd.DataFrame, backbone) -> pd.DataFrame:
    path = _llr_path(args)
    if path.exists():
        llr_df = pd.read_parquet(path)
        print(f"[llr_projection] loaded cached LLR ({path.name})")
        return llr_df
    print(f"[llr_projection] computing LLR for {meta['dms_id'].nunique()} assays "
          f"via scorer={args.scorer}")
    delta = _build_llr(args, meta, backbone)
    out = meta[["dms_id", "mutant"]].copy()
    out["llr"] = delta
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return out


def _mutant_embeddings_from_dms_cache(args, meta: pd.DataFrame, backbone) -> np.ndarray:
    """Reuse the existing dms_cache.py mutant-embedding cache (same config,
    built once and shared with CentroidDMSEvaluator) instead of recomputing."""
    from gsr.eval.dms_cache import build_or_load_dms_cache
    full_X, _full_meta = build_or_load_dms_cache(args, backbone=backbone)
    return full_X[meta["_dms_cache_row"].to_numpy()]


def _embed_wt_deduped(args, meta: pd.DataFrame, backbone, layer, pooling,
                      batch_size) -> np.ndarray:
    """Embed each unique (dms_id, pos) WT-at-position exactly once (multiple
    substitutions at the same site share the same WT embedding under
    position-matched pooling), then broadcast back to meta's row order.

    No ``wt_mean`` is passed to the embed() call below: this WT embedding is
    the metric's own "WT relative to ITSELF" quantity used as the
    cosine-distance's WT-side reference (a separate concept from the new
    pooling modes' mut-side WT-context). Per pool_batch's self-referential
    default, omitting wt_mean here is already exactly correct under both new
    modes -- not an oversight."""
    uniq = meta[["dms_id", "pos", "target_seq"]].drop_duplicates(
        subset=["dms_id", "pos"]).reset_index(drop=True)
    key_to_row = {(d, p): i for i, (d, p) in enumerate(zip(uniq["dms_id"], uniq["pos"]))}

    wt_seqs = uniq["target_seq"].tolist()
    positions = uniq["pos"].tolist()
    out = []
    for s in tqdm(range(0, len(uniq), batch_size), desc="embed WT (deduped)"):
        sl = slice(s, s + batch_size)
        e = backbone.embed(wt_seqs[sl], layer=layer, pooling=pooling,
                           positions=positions[sl], grad=False)
        out.append(e.float().cpu().numpy())
    X_uniq = np.concatenate(out, axis=0).astype(np.float32)

    row_to_uniq = np.array(
        [key_to_row[(d, p)] for d, p in zip(meta["dms_id"], meta["pos"])])
    return X_uniq[row_to_uniq]


def _build_or_load_wt_embeddings(args, meta: pd.DataFrame, backbone) -> np.ndarray:
    path = _wt_emb_path(args)
    if path.exists():
        with h5py.File(path, "r") as h5:
            return h5["X_wt"][:]
    n_unique = meta.drop_duplicates(subset=["dms_id", "pos"]).shape[0]
    print(f"[llr_projection] embedding {n_unique} unique WT-at-position vectors "
          f"for {len(meta)} variants")
    was_training = backbone.model.training
    backbone.model.eval()
    with torch.no_grad():
        X_wt = _embed_wt_deduped(args, meta, backbone, args.embedding_layer,
                                 args.pooling, args.score_batch_size)
    if was_training:
        backbone.model.train()
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        h5.create_dataset("X_wt", data=X_wt, compression="gzip", compression_opts=4)
    print(f"[llr_projection] cached WT embeddings {X_wt.shape} -> {path}")
    return X_wt


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return 1.0 - np.sum(an * bn, axis=1)


class LLRProjectionEvaluator:
    primary_metric = "llr_projection/spearman"

    def __init__(self, args, meta: pd.DataFrame, llr: np.ndarray,
                 X_wt: np.ndarray, X_mut: np.ndarray):
        self.args = args
        self.meta = meta.reset_index(drop=True)
        self.llr = llr
        self.X_wt = X_wt
        self.X_mut = X_mut

    @classmethod
    def from_args(cls, args, backbone=None):
        """``backbone`` MUST be the frozen (non-LoRA) backbone: LLR is a fixed
        external reference like DMS_score, not something that should drift as
        the trainable head/LoRA adapters change over a run."""
        if backbone is None:
            from gsr.backbone.esm import ESMBackbone
            device = args.device if torch.cuda.is_available() else "cpu"
            backbone = ESMBackbone(args.esm_model, device=device)
        meta = _build_or_load_meta(args)
        llr_df = _build_or_load_llr(args, meta, backbone)
        X_mut = _mutant_embeddings_from_dms_cache(args, meta, backbone)
        X_wt = _build_or_load_wt_embeddings(args, meta, backbone)
        return cls(args, meta, llr_df["llr"].to_numpy(dtype=np.float32), X_wt, X_mut)

    def reprepare(self, subsample: int) -> None:
        """No held-out split for this zero-shot metric; kept for interface
        parity since Trainer._final_full_eval calls reprepare() unconditionally."""
        return

    def _embed_rows_live(self, backbone, embed_args: dict):
        """Re-embed WT/mutant pairs through a LIVE (e.g. LoRA-finetuned)
        backbone -- no caching or dms_cache reuse here, since the whole point
        is that the backbone has moved on from whatever it was when any cache
        was built."""
        from gsr.backbone.pooling import WT_MEAN_POOLINGS, mean_pool

        bs = embed_args.get("batch_size", self.args.score_batch_size)
        layer = embed_args.get("layer", self.args.embedding_layer)
        pooling = embed_args.get("pooling", self.args.pooling)
        mut_seqs = self.meta["mutated_sequence"].tolist()
        wt_seqs = self.meta["target_seq"].tolist()
        positions = self.meta["pos"].tolist()
        was_training = backbone.model.training
        backbone.model.eval()
        wt_mean_full = None
        if pooling in WT_MEAN_POOLINGS:
            uniq_df = self.meta.drop_duplicates("dms_id")[["dms_id", "target_seq"]]
            with torch.no_grad():
                uh, ua = backbone.forward_reps(uniq_df["target_seq"].tolist(),
                                               layer=layer, grad=False)
            uniq_mean = mean_pool(uh, ua).cpu().numpy()
            gene_to_mean = dict(zip(uniq_df["dms_id"], uniq_mean))
            wt_mean_full = np.stack(
                [gene_to_mean[g] for g in self.meta["dms_id"]])
        X_mut, X_wt = [], []
        with torch.no_grad():
            for s in range(0, len(self.meta), bs):
                sl = slice(s, s + bs)
                wm = (torch.from_numpy(wt_mean_full[sl])
                      if wt_mean_full is not None else None)
                X_mut.append(backbone.embed(mut_seqs[sl], layer=layer, pooling=pooling,
                                            positions=positions[sl], grad=False,
                                            wt_mean=wm)
                             .float().cpu().numpy())
                # No wt_mean here: this is the metric's own "WT relative to
                # ITSELF" embedding (self-referential default), used as the
                # cosine-distance's WT-side reference -- unaffected by the
                # new pooling modes' mut-side WT-context concept.
                X_wt.append(backbone.embed(wt_seqs[sl], layer=layer, pooling=pooling,
                                           positions=positions[sl], grad=False)
                            .float().cpu().numpy())
        if was_training:
            backbone.model.train()
        return np.concatenate(X_wt, 0), np.concatenate(X_mut, 0)

    def effect_table(self, project_fn: Callable[[np.ndarray], np.ndarray],
                      backbone=None, embed_args: Optional[dict] = None) -> pd.DataFrame:
        if backbone is not None:
            X_wt, X_mut = self._embed_rows_live(backbone, embed_args or {})
        else:
            X_wt, X_mut = self.X_wt, self.X_mut
        z_wt = project_fn(X_wt)
        z_mut = project_fn(X_mut)
        effect = _cosine_distance(z_wt, z_mut)
        out = self.meta.copy()
        out["effect_score"] = effect
        out["llr"] = self.llr
        return out

    def evaluate(self, project_fn: Callable[[np.ndarray], np.ndarray],
                 backbone=None, embed_args=None) -> Dict[str, float]:
        table = self.effect_table(project_fn, backbone=backbone, embed_args=embed_args)
        rho = spearman(table["effect_score"].to_numpy(), table["llr"].to_numpy())
        # Signed: cosine distance (effect_score) is expected to correlate
        # NEGATIVELY with LLR (a larger mutational effect drives both a bigger
        # embedding shift from WT and a more negative log-likelihood ratio),
        # so a strongly-negative rho is the success direction, not a positive
        # one. "Correlates well" (per the metric's purpose: showing effect
        # scores sit on the same scale across genes) means large |rho|,
        # either sign -- report the magnitude as the primary/checkpoint-
        # selection metric (Trainer picks the highest value), keep the signed
        # value too for diagnostics/wandb.
        return {
            "llr_projection/spearman": abs(rho),
            "llr_projection/spearman_signed": rho,
            "llr_projection/n_variants": float(len(table)),
            "llr_projection/n_genes": float(table["uniprot_id"].nunique()),
        }
