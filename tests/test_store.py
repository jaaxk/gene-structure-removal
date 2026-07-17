import numpy as np
import pandas as pd
import pytest

from gsr.scoring.store import VariantStore


def _make_part(gene, n, dim, seed):
    rng = np.random.default_rng(seed)
    vids = [f"{gene}_v{i}" for i in range(n)]
    df = pd.DataFrame({
        "gene_id": gene,
        "variant_id": vids,
        "mutant": [f"A{i+1}V" for i in range(n)],
        "pos": np.arange(1, n + 1),
        "wt_aa": "A", "mut_aa": "V", "seq_len": 100,
        "is_wt": False,
        "wt_score": 0.0, "mut_score": rng.normal(size=n),
        "delta": rng.normal(size=n), "abs_delta": rng.random(n),
        "label": rng.choice(["same", "different", "middle"], size=n),
    })
    emb = rng.normal(size=(n, dim)).astype(np.float32)
    return df, emb, vids, emb


def test_roundtrip_scores_and_embeddings(tmp_path):
    store = VariantStore(tmp_path / "ds")
    dim = 8
    df1, emb1, vids1, _ = _make_part("geneA", 5, dim, 0)
    df2, emb2, vids2, _ = _make_part("geneB", 7, dim, 1)
    store.write_part("shard_0000", df1, emb1)
    store.write_part("shard_0001", df2, emb2)

    scores = store.load_scores()
    assert len(scores) == 12
    assert set(scores["gene_id"]) == {"geneA", "geneB"}
    assert store.embedding_dim() == dim

    # Request in a shuffled, cross-shard order and verify exact rows come back.
    order = [vids2[3], vids1[0], vids1[4], vids2[0]]
    want = np.stack([emb2[3], emb1[0], emb1[4], emb2[0]])
    got = store.load_embeddings(order)
    assert got.shape == (4, dim)
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-6)


def test_missing_variant_raises(tmp_path):
    store = VariantStore(tmp_path / "ds")
    df, emb, vids, _ = _make_part("geneA", 3, 4, 0)
    store.write_part("shard_0000", df, emb)
    with pytest.raises(KeyError):
        store.load_embeddings(["does_not_exist"])


def test_length_mismatch_rejected(tmp_path):
    store = VariantStore(tmp_path / "ds")
    df, emb, vids, _ = _make_part("geneA", 3, 4, 0)
    with pytest.raises(AssertionError):
        store.write_part("shard_0000", df, emb[:2])
