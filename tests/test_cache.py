import numpy as np

import gsr.paths as paths
from gsr.cache.embedding_cache import EmbeddingCache
from gsr.data.mutagenesis import enumerate_variants


def test_enumerate_variants_complete():
    seq = "MACD"  # 4 canonical sites
    vs = enumerate_variants("g", seq)
    assert len(vs) == 4 * 19
    keys = {(v.pos, v.mut_aa) for v in vs}
    assert len(keys) == len(vs)
    for v in vs:
        assert v.mut_aa != v.wt_aa
        assert v.sequence[v.pos - 1] == v.mut_aa


def test_embedding_cache_roundtrip_and_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "SCRATCH_ROOT", tmp_path)
    cache = EmbeddingCache("m", -1, "concat")
    ids = ["a", "b", "c"]
    mut = np.arange(3 * 4, dtype=np.float32).reshape(3, 4)
    wt = mut + 100
    assert cache.put(ids, mut, wt) is True

    # Full hit, shuffled order.
    got_mut, got_wt, missing = cache.get(["c", "a"])
    assert missing == []
    np.testing.assert_allclose(got_mut[0], mut[2])
    np.testing.assert_allclose(got_wt[1], wt[0])

    # Partial: one known, one unknown.
    _, _, missing = cache.get(["a", "zzz"])
    assert missing == [1]


def test_embedding_cache_lock_skips_save(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "SCRATCH_ROOT", tmp_path)
    cache = EmbeddingCache("m", -1, "mean")
    cache.lock_path.parent.mkdir(parents=True, exist_ok=True)
    cache.lock_path.write_text("held by other")  # simulate another writer
    saved = cache.put(["a"], np.zeros((1, 4), np.float32), np.zeros((1, 4), np.float32))
    assert saved is False  # did not save, did not crash
