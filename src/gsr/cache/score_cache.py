"""Per-gene score/label cache (frozen-ESM LL/PLL).

Scores come from the FROZEN backbone, so they are deterministic and reusable
across all runs regardless of pooling, sampling, or LoRA. One file per gene holds
every single-aa variant (for marginal scorers this is free -- one sweep per gene),
so per-run quartile labels are computed over the FULL distribution and are
therefore stable/run-independent.

Cache dir is keyed by (model, scorer): different scorers never collide.
Labels are NOT stored (they depend on the run's quartile params); derive them from
the cached ``abs_delta`` with ``gsr.data.labeling.assign_labels``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from gsr import paths
from gsr.data.mutagenesis import enumerate_variants
from gsr.scoring.scorer import score_gene
from gsr.utils.hashing import seq_hash


class ScoreCache:
    def __init__(self, model: str, scorer: str):
        self.model = model
        self.scorer = scorer
        self.dir = paths.SCRATCH_ROOT / "cache" / "scores" / f"{model}_{scorer}"
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, gene_id: str) -> Path:
        # gene ids can contain characters that are awkward in filenames; hash them.
        return self.dir / f"{seq_hash(gene_id)[:16]}.parquet"

    def get_or_compute(self, gene_id: str, wt_seq: str, backbone,
                       score_batch_size: int = 16,
                       candidate_cap: int = 0, seed: int = 0) -> pd.DataFrame:
        """Return single-aa variant scores for a gene (cached or computed).

        For marginal scorers all variants are scored in one sweep (``candidate_cap``
        = 0). For the per-variant ``pll`` scorer, pass ``candidate_cap`` > 0 to score
        only a random candidate subset (scoring every variant would be infeasible);
        quartiles are then computed over that subset.
        """
        path = self._path(gene_id)
        if path.exists():
            return pd.read_parquet(path)

        variants = enumerate_variants(gene_id, wt_seq)
        if candidate_cap and len(variants) > candidate_cap:
            import numpy as np
            rng = np.random.default_rng(seed)
            keep = rng.choice(len(variants), candidate_cap, replace=False)
            variants = [variants[i] for i in keep]
        scores = score_gene(backbone, wt_seq, variants, scorer=self.scorer,
                            batch_size=score_batch_size)
        rows = [dict(gene_id=gene_id, variant_id=seq_hash(wt_seq), mutant="WT",
                     pos=0, wt_aa="", mut_aa="", is_wt=True, mutated_sequence=wt_seq,
                     wt_score=0.0, mut_score=0.0, delta=0.0, abs_delta=0.0)]
        for v, (ws, ms, delta) in zip(variants, scores):
            rows.append(dict(gene_id=gene_id, variant_id=seq_hash(v.sequence),
                             mutant=v.mutant, pos=v.pos, wt_aa=v.wt_aa,
                             mut_aa=v.mut_aa, is_wt=False,
                             mutated_sequence=v.sequence, wt_score=ws,
                             mut_score=ms, delta=delta, abs_delta=abs(delta)))
        df = pd.DataFrame(rows)
        tmp = path.with_suffix(".parquet.tmp")
        df.to_parquet(tmp, index=False)
        tmp.replace(path)  # atomic publish
        return df
