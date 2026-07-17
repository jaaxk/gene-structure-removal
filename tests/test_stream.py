import numpy as np
import pandas as pd
import pytest

import gsr.paths as paths
from gsr.cache.embedding_cache import EmbeddingCache
from gsr.data.stream_dataset import StreamingVariantDataset
from gsr.data.training_data import select_embeddings_mode


def test_select_mode_threshold():
    # 1000 items, dim 2304: 1000*2304*4*2 = ~0.018 GB
    assert select_embeddings_mode(1000, 2304, 32, "auto") == "ram"
    # 10M items -> ~184 GB
    assert select_embeddings_mode(10_000_000, 2304, 32, "auto") == "stream"
    # explicit modes pass through
    assert select_embeddings_mode(1, 1, 32, "stream") == "stream"
    assert select_embeddings_mode(10**9, 2304, 1, "ram") == "ram"


def _make_df(vids, gene):
    return pd.DataFrame({
        "variant_id": vids,
        "gene_id": gene,
        "label_id": [0, 1] * (len(vids) // 2),
    })


def test_streaming_matches_resident(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "SCRATCH_ROOT", tmp_path)
    cache = EmbeddingCache("m", -1, "concat")
    n, d = 6, 8
    vids = [f"v{i}" for i in range(n)]
    rng = np.random.default_rng(0)
    mut = rng.normal(size=(n, d)).astype(np.float32)
    wt = rng.normal(size=(n, d)).astype(np.float32)
    assert cache.put(vids, mut, wt)

    df = _make_df(vids, "g")
    ds = StreamingVariantDataset(df, cache)
    assert ds.input_dim == d
    for i in range(n):
        m_i, w_i, lbl, gene = ds[i]
        np.testing.assert_allclose(m_i.numpy(), mut[i], atol=1e-6)
        np.testing.assert_allclose(w_i.numpy(), wt[i], atol=1e-6)


def test_streaming_requires_full_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "SCRATCH_ROOT", tmp_path)
    cache = EmbeddingCache("m", -1, "mean")
    cache.put(["a", "b"], np.zeros((2, 4), np.float32), np.zeros((2, 4), np.float32))
    df = _make_df(["a", "missing"], "g")
    with pytest.raises(ValueError):
        StreamingVariantDataset(df, cache)
