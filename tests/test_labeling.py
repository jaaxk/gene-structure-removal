import numpy as np
import pandas as pd

from gsr.data.labeling import assign_labels


def _gene_df(gene, n, seed):
    rng = np.random.default_rng(seed)
    deltas = rng.normal(size=n)
    rows = [dict(gene_id=gene, is_wt=True, abs_delta=0.0)]
    rows += [dict(gene_id=gene, is_wt=False, abs_delta=abs(d)) for d in deltas]
    return pd.DataFrame(rows)


def test_quartile_split_and_wt_same():
    df = _gene_df("g1", 100, 0)
    out = assign_labels(df, quartile_low=0.25, quartile_high=0.25, min_variants=10)
    mut = out[~out["is_wt"]]
    assert (mut["label"] == "same").sum() > 0
    assert (mut["label"] == "different").sum() > 0
    assert (mut["label"] == "middle").sum() > 0
    # WT always 'same'
    assert (out[out["is_wt"]]["label"] == "same").all()
    # 'same' variants have smaller |delta| than 'different'
    assert mut[mut.label == "same"]["abs_delta"].max() <= \
        mut[mut.label == "different"]["abs_delta"].min() + 1e-9


def test_small_genes_dropped():
    df = pd.concat([_gene_df("big", 50, 1), _gene_df("small", 3, 2)],
                   ignore_index=True)
    out = assign_labels(df, min_variants=10)
    assert set(out["gene_id"]) == {"big"}
