import numpy as np

import gsr.paths as paths
from gsr.cache.wt_mean_cache import WtMeanCache


def test_wt_mean_cache_roundtrip_and_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "SCRATCH_ROOT", tmp_path)
    cache = WtMeanCache("m", -1)
    seqs = ["MACD", "GGGG", "PEPTIDE"]
    X = np.arange(3 * 4, dtype=np.float32).reshape(3, 4)
    assert cache.put(seqs, X) is True

    got, missing = cache.get(["GGGG", "MACD"])
    assert missing == []
    np.testing.assert_allclose(got[0], X[1])
    np.testing.assert_allclose(got[1], X[0])

    _, missing = cache.get(["MACD", "UNKNOWNSEQ"])
    assert missing == [1]


def test_wt_mean_cache_dedups_identical_sequences(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "SCRATCH_ROOT", tmp_path)
    cache = WtMeanCache("m", -1)
    # Same sequence appearing twice (e.g. two different genes/assays sharing
    # an identical WT) should collapse to one cached row.
    seqs = ["MACD", "MACD"]
    X = np.array([[1., 2.], [999., 999.]], dtype=np.float32)
    assert cache.put(seqs, X) is True
    with cache.open_readonly() as h5:
        assert h5["X"].shape[0] == 1


def test_wt_mean_cache_get_with_broadcast_repeats(tmp_path, monkeypatch):
    # Regression test: get() is normally called with a BROADCAST list (the
    # same WT sequence repeated once per variant of a gene) -- h5py's fancy
    # indexing rejects duplicate/non-increasing row selections, so this must
    # dedup before touching the HDF5 file.
    monkeypatch.setattr(paths, "SCRATCH_ROOT", tmp_path)
    cache = WtMeanCache("m", -1)
    seqs = ["AAAA", "BBBB", "CCCC"]
    X = np.array([[1., 1.], [2., 2.], [3., 3.]], dtype=np.float32)
    assert cache.put(seqs, X) is True

    broadcast = ["BBBB", "BBBB", "AAAA", "CCCC", "BBBB", "AAAA"]
    got, missing = cache.get(broadcast)
    assert missing == []
    expected = np.array([[2., 2.], [2., 2.], [1., 1.], [3., 3.], [2., 2.], [1., 1.]],
                        dtype=np.float32)
    np.testing.assert_allclose(got, expected)


def test_wt_mean_cache_lock_skips_save(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "SCRATCH_ROOT", tmp_path)
    cache = WtMeanCache("m", -1)
    cache.lock_path.parent.mkdir(parents=True, exist_ok=True)
    cache.lock_path.write_text("held by other")  # simulate another writer
    saved = cache.put(["MACD"], np.zeros((1, 4), np.float32))
    assert saved is False  # did not save, did not crash
