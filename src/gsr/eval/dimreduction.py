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
