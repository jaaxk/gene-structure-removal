"""Dimensionality-reduction figures (PCA / t-SNE / UMAP).

Projects embeddings to 2-D and scatter-plots them colored by gene and by DMS
selection type -- the visual check for whether gene structure has been removed
(before vs after training, backbone vs projected). Works on any embedding matrix.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

# Non-interactive backend for headless SLURM.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _reduce(emb: np.ndarray, method: str, seed: int) -> np.ndarray:
    if method == "pca":
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=seed).fit_transform(emb)
    if method == "tsne":
        from sklearn.manifold import TSNE
        return TSNE(n_components=2, random_state=seed,
                    init="pca", perplexity=min(30, max(5, len(emb) // 4))
                    ).fit_transform(emb)
    if method == "umap":
        import umap
        return umap.UMAP(n_components=2, random_state=seed).fit_transform(emb)
    raise ValueError(f"Unknown method {method!r}")


def _scatter(xy, labels, title, path):
    fig, ax = plt.subplots(figsize=(7, 6))
    uniq = list(dict.fromkeys(labels))
    cmap = plt.get_cmap("tab20", max(len(uniq), 1))
    for i, u in enumerate(uniq):
        m = np.array([l == u for l in labels])
        ax.scatter(xy[m, 0], xy[m, 1], s=6, alpha=0.6, color=cmap(i),
                   label=str(u))
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    if len(uniq) <= 20:
        ax.legend(markerscale=2, fontsize=7, loc="best", framealpha=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# Fixed, non-cycled categorical pair for the two DMS-score quartile labels
# (polarity: high vs. low), distinct from the tab20 gene palette above.
_QUARTILE_COLORS = {"high": "#D6604D", "low": "#4393C3"}


def _top_gene_mask(genes: np.ndarray, top_n: int) -> np.ndarray:
    """Boolean mask selecting only the top_n most-frequent genes' variants."""
    counts = {}
    for g in genes:
        counts[g] = counts.get(g, 0) + 1
    top = {g for g, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:top_n]}
    return np.isin(genes, list(top))


def _hi_lo_mask(meta):
    """Per-assay (dms_id) top/bottom-quartile DMS-score boolean masks, matching
    CentroidDMSEvaluator's quartile convention (drop the middle 50%)."""
    hi = np.zeros(len(meta), dtype=bool)
    lo = np.zeros(len(meta), dtype=bool)
    for _dms, g in meta.groupby("dms_id"):
        q75 = g["DMS_score"].quantile(0.75)
        q25 = g["DMS_score"].quantile(0.25)
        hi[g.index[g["DMS_score"] >= q75]] = True
        lo[g.index[g["DMS_score"] <= q25]] = True
    return hi, lo


def _subsample(idx: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if len(idx) <= max_points:
        return idx
    rng = np.random.default_rng(seed)
    return rng.choice(idx, max_points, replace=False)


def _balanced_top_gene_idx(genes: np.ndarray, top_n: int, per_gene_cap: int,
                           seed: int) -> np.ndarray:
    """Indices for the top_n most-frequent genes, subsampled PER GENE (not by a
    single global random draw) so every gene keeps up to per_gene_cap points --
    a gene with few variants isn't crowded out by a gene with thousands."""
    rng = np.random.default_rng(seed)
    counts = {}
    for g in genes:
        counts[g] = counts.get(g, 0) + 1
    top = [g for g, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:top_n]]
    out = []
    for g in top:
        gidx = np.where(genes == g)[0]
        if len(gidx) > per_gene_cap:
            gidx = rng.choice(gidx, per_gene_cap, replace=False)
        out.append(gidx)
    return np.concatenate(out) if out else np.array([], dtype=int)


def _scatter_ax(ax, xy, labels, colors=None, title=""):
    uniq = list(dict.fromkeys(labels))
    if colors is None:
        cmap = plt.get_cmap("tab20", max(len(uniq), 1))
        colors = {u: cmap(i) for i, u in enumerate(uniq)}
    for u in uniq:
        m = np.array([l == u for l in labels])
        ax.scatter(xy[m, 0], xy[m, 1], s=6, alpha=0.7, color=colors[u], label=str(u))
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    if len(uniq) <= 20:
        ax.legend(markerscale=2, fontsize=6, loc="best", framealpha=0.6)


def make_global_gene_figure(emb: np.ndarray, meta, out_path: Path, tag: str,
                            top_n: int = 20, per_gene_cap: int = 150,
                            seed: int = 0) -> Path:
    """t-SNE over ALL selection types pooled, colored by gene (top-N by count;
    each of those genes keeps up to per_gene_cap variants so every gene is
    visibly represented, not just whichever genes a global random draw favors).
    Other genes' variants are neither embedded into the reduction nor plotted,
    for a legible legend."""
    genes = meta["uniprot_id"].to_numpy()
    idx = _balanced_top_gene_idx(genes, top_n, per_gene_cap, seed)
    xy = _reduce(emb[idx], "tsne", seed)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    _scatter_ax(ax, xy, genes[idx], title=f"t-SNE by gene (top{top_n}, all selection types, {tag})")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def make_selection_type_panel(emb: np.ndarray, meta, stype: str, out_path: Path,
                              tag: str, top_n: int = 20, per_gene_cap: int = 150,
                              max_points: int = 8000, seed: int = 0) -> Path:
    """One figure, 2 subplots, for a single DMS selection type:
    (1) colored by gene (top-N by count within this type only; up to
        per_gene_cap variants per gene so every gene is visibly represented)
    (2) colored by high/low DMS-score quartile (per-assay, middle 50% dropped)
    Each subplot is its own independent t-SNE fit over just the points it plots.
    """
    stype_pos = np.where(meta["selection_type"].to_numpy() == stype)[0]
    sub_meta = meta.iloc[stype_pos].reset_index(drop=True)
    sub_emb = emb[stype_pos]

    genes = sub_meta["uniprot_id"].to_numpy()
    gidx = _balanced_top_gene_idx(genes, top_n, per_gene_cap, seed)

    hi, lo = _hi_lo_mask(sub_meta)
    qidx = _subsample(np.where(hi | lo)[0], max_points, seed)
    qlabels = np.where(hi[qidx], "high", "low")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    if len(gidx) > 3:
        xy_g = _reduce(sub_emb[gidx], "tsne", seed)
        _scatter_ax(axes[0], xy_g, genes[gidx],
                   title=f"{stype}: by gene (top{top_n}, {tag})")
    else:
        axes[0].set_title(f"{stype}: by gene -- too few genes")
    if len(qidx) > 3:
        xy_q = _reduce(sub_emb[qidx], "tsne", seed)
        _scatter_ax(axes[1], xy_q, qlabels, colors=_QUARTILE_COLORS,
                   title=f"{stype}: by DMS-score quartile ({tag})")
    else:
        axes[1].set_title(f"{stype}: by quartile -- too few points")
    fig.suptitle(f"{stype} ({tag})", fontsize=11)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def make_figures(emb: np.ndarray, meta, out_dir: Path, tag: str,
                 methods: List[str] = ("pca", "tsne", "umap"),
                 max_points: int = 5000, seed: int = 0) -> List[Path]:
    """Write dim-reduction figures colored by gene and by selection type."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(emb))
    if len(emb) > max_points:
        idx = rng.choice(idx, max_points, replace=False)
    emb = emb[idx]
    genes = meta["uniprot_id"].to_numpy()[idx]
    stypes = meta["selection_type"].to_numpy()[idx]

    saved = []
    for method in methods:
        try:
            xy = _reduce(emb, method, seed)
        except Exception as e:
            print(f"[dimred] {method} failed: {e}")
            continue
        saved.append(_scatter(xy, stypes, f"{method.upper()} by selection type ({tag})",
                              out_dir / f"{tag}_{method}_selection_type.png"))
        # Gene coloring is only legible with a limited number of genes.
        top_genes = [g for g, _ in sorted(
            {g: (genes == g).sum() for g in set(genes)}.items(),
            key=lambda kv: -kv[1])[:20]]
        gmask = np.isin(genes, top_genes)
        if gmask.sum() > 3:
            saved.append(_scatter(xy[gmask], genes[gmask],
                                  f"{method.upper()} by gene (top20, {tag})",
                                  out_dir / f"{tag}_{method}_gene.png"))
    print(f"[dimred] wrote {len(saved)} figures to {out_dir}")
    return saved
