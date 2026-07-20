"""Final-eval figure for the LLR-projection metric: one PNG, 3 subplots, all
sharing x = projection effect score (cosine distance, WT vs mutant) and
y = LLR (delta = LL_mut - LL_wt), each a scatter with a fitted line and the
pooled Spearman (the actual reported metric) annotated. Only the plotted
*subset* differs per panel (for legibility); the annotated rho is always the
same full-dataset number computed by ``LLRProjectionEvaluator.evaluate``.

  1. colored by gene   -- top-20 genes by variant count (categorical, tab20)
  2. colored by DMS-score quartile -- high/low (2-color diverging pair,
     reusing dimreduction.py's fixed convention; middle 50% dropped)
  3. colored by relative position along the protein (0-100%, continuous
     sequential colormap + colorbar)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Non-interactive backend for headless SLURM (matches dimreduction.py).
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from gsr.eval.dimreduction import _QUARTILE_COLORS, _top_gene_mask


def _fit_line(ax, x: np.ndarray, y: np.ndarray) -> None:
    if len(x) < 2 or np.all(x == x[0]):
        return
    slope, intercept = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(xs, slope * xs + intercept, color="0.2", linewidth=1.5, zorder=3)


def _annotate_rho(ax, rho: float) -> None:
    ax.text(0.03, 0.96, f"Spearman ρ = {rho:.3f}", transform=ax.transAxes,
             fontsize=9, va="top", ha="left",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.75,
                       edgecolor="0.7"))


def _panel_by_gene(ax, table: pd.DataFrame, rho: float, top_n: int) -> None:
    genes = table["uniprot_id"].to_numpy()
    mask = _top_gene_mask(genes, top_n)
    sub = table.loc[mask]
    uniq = list(dict.fromkeys(sub["uniprot_id"]))
    cmap = plt.get_cmap("tab20", max(len(uniq), 1))
    for i, gene in enumerate(uniq):
        g = sub[sub["uniprot_id"] == gene]
        ax.scatter(g["effect_score"], g["llr"], s=8, alpha=0.6, color=cmap(i),
                  label=gene, linewidths=0)
    _fit_line(ax, sub["effect_score"].to_numpy(), sub["llr"].to_numpy())
    _annotate_rho(ax, rho)
    ax.set_title(f"Colored by gene (top {top_n} by variant count)", fontsize=10)
    ax.legend(markerscale=1.5, fontsize=5.5, loc="lower right", framealpha=0.7,
              ncol=2)


def _panel_by_quartile(ax, table: pd.DataFrame, rho: float) -> None:
    hi = table[table["_hi"]]
    lo = table[table["_lo"]]
    ax.scatter(hi["effect_score"], hi["llr"], s=8, alpha=0.6,
              color=_QUARTILE_COLORS["high"], label="high DMS quartile",
              linewidths=0)
    ax.scatter(lo["effect_score"], lo["llr"], s=8, alpha=0.6,
              color=_QUARTILE_COLORS["low"], label="low DMS quartile",
              linewidths=0)
    sub = pd.concat([hi, lo], ignore_index=True)
    _fit_line(ax, sub["effect_score"].to_numpy(), sub["llr"].to_numpy())
    _annotate_rho(ax, rho)
    ax.set_title("Colored by DMS-score quartile (per-assay, all selection types)",
                fontsize=10)
    ax.legend(markerscale=1.5, fontsize=7, loc="lower right", framealpha=0.7)


def _panel_by_position(ax, table: pd.DataFrame, rho: float, fig) -> None:
    sc = ax.scatter(table["effect_score"], table["llr"], s=8, alpha=0.6,
                    c=table["relative_pos"], cmap="viridis", vmin=0, vmax=1,
                    linewidths=0)
    _fit_line(ax, table["effect_score"].to_numpy(), table["llr"].to_numpy())
    _annotate_rho(ax, rho)
    ax.set_title("Colored by relative position along protein (0-100%)", fontsize=10)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("relative position", fontsize=8)


def make_llr_projection_figure(effect_table: pd.DataFrame, rho: float, out_path,
                               top_n_genes: int = 20) -> Path:
    """Write the 3-subplot LLR-vs-projection-effect-score figure and return its path.

    ``effect_table`` is ``LLRProjectionEvaluator.effect_table(...)``'s output:
    one row per (subsampled) DMS variant, columns include ``effect_score``,
    ``llr``, ``uniprot_id``, ``_hi``/``_lo``, ``relative_pos``.
    """
    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    _panel_by_gene(axes[0], effect_table, rho, top_n_genes)
    _panel_by_quartile(axes[1], effect_table, rho)
    _panel_by_position(axes[2], effect_table, rho, fig)
    for ax in axes:
        ax.set_xlabel("projection effect score (cosine distance, WT vs mutant)")
    axes[0].set_ylabel("LLR (LL_mut - LL_wt)")
    n_genes = effect_table["uniprot_id"].nunique()
    fig.suptitle(f"Projection effect score vs LLR -- {len(effect_table)} variants, "
                f"{n_genes} genes", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[llr_figure] wrote {out_path}")
    return out_path
