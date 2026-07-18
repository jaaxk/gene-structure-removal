"""Zero-shot, selection-type-specific centroid evaluation.

For each selection type we split its genes into a centroid set and a query set
(gene-level, leakage-safe). From the centroid genes we build a top-quartile
('high') and bottom-quartile ('low') centroid in the *projected* embedding space
(quartiles are per-assay, so different DMS scales are handled). Each query variant
is then scored two ways and correlated (Spearman) with its true DMS score:

  * centroid difference: sim(v, high) - sim(v, low)
  * axis projection    : v . (high - low) / |high - low|

The primary metric is the mean per-type Spearman of the centroid-difference score.
The same object runs during training (project_fn = current head) and standalone
(project_fn = identity for the raw backbone, or a loaded head).
"""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np

from gsr.data.dms import gene_level_split
from gsr.eval.dms_cache import build_or_load_dms_cache
from gsr.utils.spearman import spearman


def _cosine(a, B):
    a = a / (np.linalg.norm(a) + 1e-8)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    return Bn @ a


def _neg_euclidean(a, B):
    return -np.linalg.norm(B - a[None, :], axis=1)


class CentroidDMSEvaluator:
    primary_metric = "centroid/spearman_mean"

    def __init__(self, args, embeddings: np.ndarray, meta):
        self.metric = args.eval_distance
        self.subsample = args.centroid_subsample
        self.held_out_gene_frac = args.held_out_gene_frac
        self.seed = args.seed
        self.embeddings = embeddings
        self.meta = meta.reset_index(drop=True)
        self.rng = np.random.default_rng(args.seed)
        self._prepare()

    @classmethod
    def from_args(cls, args, backbone=None):
        emb, meta = build_or_load_dms_cache(args, backbone=backbone)
        return cls(args, emb, meta)

    def save_splits(self, path) -> None:
        """Persist the per-selection-type centroid/query gene split for provenance
        (the split is deterministic from seed + held_out_gene_frac + gene set)."""
        import json
        from pathlib import Path
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "seed": self.seed,
            "held_out_gene_frac": self.held_out_gene_frac,
            "splits": self.splits,
        }
        with open(path, "w") as fh:
            json.dump(meta, fh, indent=2)

    def reprepare(self, subsample: int) -> None:
        """Rebuild centroid/query indices with a different centroid subsample
        (e.g. 0 = use all variants, for the final full-centroid eval)."""
        self.subsample = subsample
        self._prepare()

    def _prepare(self):
        """Precompute per-type centroid-high/low and query row indices."""
        m = self.meta
        # Per-assay quartile membership (scale-free across assays).
        m["_hi"] = False
        m["_lo"] = False
        for _dms, g in m.groupby("dms_id"):
            hi = g["DMS_score"] >= g["DMS_score"].quantile(0.75)
            lo = g["DMS_score"] <= g["DMS_score"].quantile(0.25)
            m.loc[g.index[hi], "_hi"] = True
            m.loc[g.index[lo], "_lo"] = True

        self.types = {}
        self.splits = {}
        for stype, g in m.groupby("selection_type"):
            genes = g["uniprot_id"].tolist()
            centroid_genes, query_genes = gene_level_split(
                genes, query_frac=self.held_out_gene_frac, seed=self.seed)
            self.splits[stype] = {
                "centroid_genes": sorted(centroid_genes),
                "query_genes": sorted(query_genes),
            }
            in_centroid = g["uniprot_id"].isin(centroid_genes)
            in_query = g["uniprot_id"].isin(query_genes)
            hi_idx = g.index[in_centroid & g["_hi"]].to_numpy()
            lo_idx = g.index[in_centroid & g["_lo"]].to_numpy()
            q_idx = g.index[in_query].to_numpy()
            if self.subsample and len(hi_idx) > self.subsample:
                hi_idx = self.rng.choice(hi_idx, self.subsample, replace=False)
            if self.subsample and len(lo_idx) > self.subsample:
                lo_idx = self.rng.choice(lo_idx, self.subsample, replace=False)
            if len(hi_idx) == 0 or len(lo_idx) == 0 or len(q_idx) < 3:
                print(f"[centroid] skipping {stype}: hi={len(hi_idx)} "
                      f"lo={len(lo_idx)} query={len(q_idx)}")
                continue
            self.types[stype] = dict(hi=hi_idx, lo=lo_idx, q=q_idx,
                                     dms=m.loc[q_idx, "DMS_score"].to_numpy())

    def _embed_rows(self, backbone, embed_args, rows) -> np.ndarray:
        """Re-embed the given meta rows through ``backbone`` (grad-free), for the
        LoRA path where DMS embeddings must reflect the finetuned backbone."""
        import torch
        seqs = self.meta["mutated_sequence"].to_numpy()
        pos = self.meta["pos"].to_numpy()
        bs = embed_args.get("batch_size", 16)
        layer = embed_args.get("layer", -1)
        pooling = embed_args.get("pooling", "concat")
        was_training = backbone.model.training
        backbone.model.eval()
        outs = []
        with torch.no_grad():
            for s in range(0, len(rows), bs):
                idx = rows[s:s + bs]
                e = backbone.embed([seqs[i] for i in idx], layer=layer,
                                   pooling=pooling,
                                   positions=[int(pos[i]) for i in idx], grad=False)
                outs.append(e.float().cpu().numpy())
        if was_training:
            backbone.model.train()
        return np.concatenate(outs, 0) if outs else np.empty((0, 0), np.float32)

    def evaluate(self, project_fn: Callable[[np.ndarray], np.ndarray],
                 backbone=None, embed_args=None) -> Dict[str, float]:
        """Compute per-type + macro centroid Spearman.

        Frozen path (backbone=None): projects the cached frozen DMS embeddings.
        LoRA path (backbone given): re-embeds the DMS variants through the current
        (finetuned) backbone so the metric reflects the adapted backbone; query rows
        are capped to ``subsample`` to bound cost.
        """
        types = self.types
        if backbone is not None:
            types = {}
            for st, t in self.types.items():
                q, dms = t["q"], t["dms"]
                if self.subsample and len(q) > self.subsample:
                    sel = self.rng.choice(len(q), self.subsample, replace=False)
                    q, dms = q[sel], dms[sel]
                types[st] = dict(hi=t["hi"], lo=t["lo"], q=q, dms=dms)
            used = np.unique(np.concatenate(
                [np.concatenate([t["hi"], t["lo"], t["q"]]) for t in types.values()]))
            fresh = self._embed_rows(backbone, embed_args or {}, used)
            full = np.zeros((len(self.meta), fresh.shape[1]), np.float32)
            full[used] = fresh
            Z = project_fn(full)
        else:
            Z = project_fn(self.embeddings)
        sim = _cosine if self.metric == "cosine" else _neg_euclidean
        out: Dict[str, float] = {}
        diffs, axes = [], []
        for stype, t in types.items():
            high_c = Z[t["hi"]].mean(0)
            low_c = Z[t["lo"]].mean(0)
            Zq = Z[t["q"]]
            score_diff = sim(high_c, Zq) - sim(low_c, Zq)
            axis = high_c - low_c
            axis = axis / (np.linalg.norm(axis) + 1e-8)
            score_axis = Zq @ axis
            rho_diff = spearman(score_diff, t["dms"])
            rho_axis = spearman(score_axis, t["dms"])
            out[f"centroid/spearman_diff/{stype}"] = rho_diff
            out[f"centroid/spearman_axis/{stype}"] = rho_axis
            # Signed: high centroid is built from the high-DMS quartile, so a
            # positive Spearman is the success direction; a negative value is a
            # real failure and must not be hidden by abs().
            if rho_diff == rho_diff:
                diffs.append(rho_diff)
            if rho_axis == rho_axis:
                axes.append(rho_axis)
        out["centroid/spearman_mean"] = float(np.mean(diffs)) if diffs else float("nan")
        out["centroid/spearman_axis_mean"] = float(np.mean(axes)) if axes else float("nan")
        return out
