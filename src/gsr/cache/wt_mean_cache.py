"""Content-addressed cache of per-gene WT mean-pooled embeddings.

Mean-pooling a WT sequence is position-independent and identical for every
variant of that gene -- computing it once per unique WT sequence (instead of
once per variant row) is the whole point of this cache. Keyed by the content
hash of the WT sequence (``gsr.utils.hashing.seq_hash``, the same convention
``ScoreCache`` already uses for WT rows) rather than gene_id/dms_id strings,
so the same cache is shared safely across UniRef ``gene_id``s (training) and
ProteinGym ``dms_id``s (eval) with zero namespace collision risk, and gets a
free dedup bonus if two different genes happen to share an identical WT
sequence. Keyed by (model, layer) only -- NOT pooling, since raw mean
pooling doesn't depend on the variant's own pooling choice, so the same
cache is reused across both WT-mean-aware pooling modes.

Mirrors ``EmbeddingCache``'s HDF5 + JSON-sidecar + writelock convention, but
with a single dataset (one row per unique WT sequence) instead of a mut/wt
pair.
"""

from __future__ import annotations

import json
import os
from typing import List, Tuple

import h5py
import numpy as np
import torch

from gsr import paths
from gsr.utils.hashing import seq_hash


class WtMeanCache:
    def __init__(self, model: str, layer: int):
        self.dir = paths.SCRATCH_ROOT / "cache" / "wt_mean" / f"{model}_L{layer}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.h5_path = self.dir / "wt_mean.h5"
        self.sidecar = self.dir / "hash_to_row.json"
        self.lock_path = self.dir / "wt_mean.h5.writelock"

    # --- index ------------------------------------------------------------
    def _load_index(self) -> dict:
        if self.sidecar.exists():
            with open(self.sidecar) as fh:
                return json.load(fh)
        return {}

    def missing_seqs(self, wt_seqs: List[str]) -> List[str]:
        """Unique WT sequences not yet cached."""
        index = self._load_index()
        uniq = sorted(set(wt_seqs))
        return [s for s in uniq if seq_hash(s) not in index]

    def open_readonly(self):
        return h5py.File(self.h5_path, "r")

    # --- read ---------------------------------------------------------------
    def get(self, wt_seqs: List[str]) -> Tuple[np.ndarray, List[int]]:
        """Bulk-read, broadcasting repeats. Returns (X (n,D), missing_positions)
        -- missing_positions indexes into wt_seqs for entries not cached. If the
        cache is empty, all positions are missing and X is zero-width."""
        index = self._load_index()
        hashes = [seq_hash(s) for s in wt_seqs]
        present = [(i, index[h]) for i, h in enumerate(hashes) if h in index]
        missing = [i for i, h in enumerate(hashes) if h not in index]
        if not present or not self.h5_path.exists():
            return np.empty((0, 0), np.float32), list(range(len(wt_seqs)))
        out_idx = np.array([p[0] for p in present])
        rows = np.array([p[1] for p in present])
        order = np.argsort(rows)
        try:
            with h5py.File(self.h5_path, "r") as h5:
                D = h5["X"].shape[1]
                X = np.full((len(wt_seqs), D), np.nan, np.float32)
                X[out_idx[order]] = h5["X"][rows[order]]
        except (OSError, KeyError):
            return np.empty((0, 0), np.float32), list(range(len(wt_seqs)))
        return X, missing

    # --- write --------------------------------------------------------------
    def _acquire_lock(self) -> bool:
        try:
            fd = os.open(str(self.lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{os.getpid()}@{os.uname().nodename}".encode())
            os.close(fd)
            return True
        except FileExistsError:
            return False

    def _release_lock(self) -> None:
        try:
            os.unlink(self.lock_path)
        except FileNotFoundError:
            pass

    def put(self, wt_seqs: List[str], X: np.ndarray) -> bool:
        """Persist new unique rows (dedup by content hash within this call
        too). Returns False (without saving) if another writer holds the
        lock -- caller keeps them in memory regardless."""
        if not self._acquire_lock():
            return False
        try:
            index = self._load_index()
            X = np.asarray(X, np.float32)
            seen = set()
            new_hashes, new_rows = [], []
            for i, s in enumerate(wt_seqs):
                h = seq_hash(s)
                if h in index or h in seen:
                    continue
                seen.add(h)
                new_hashes.append(h)
                new_rows.append(X[i])
            if not new_hashes:
                return True
            D = X.shape[1]
            with h5py.File(self.h5_path, "a") as h5:
                if "X" not in h5:
                    h5.create_dataset("X", shape=(0, D), maxshape=(None, D),
                                      dtype="float32", chunks=(1024, D))
                start = h5["X"].shape[0]
                h5["X"].resize(start + len(new_hashes), axis=0)
                h5["X"][start:] = np.stack(new_rows)
            for offset, h in enumerate(new_hashes):
                index[h] = start + offset
            tmp = self.sidecar.with_suffix(".json.tmp")
            with open(tmp, "w") as fh:
                json.dump(index, fh)
            tmp.replace(self.sidecar)
            return True
        finally:
            self._release_lock()


def ensure_and_broadcast(cache: WtMeanCache, backbone, wt_seqs: List[str],
                         batch_size: int) -> torch.Tensor:
    """Fill cache misses (one forward pass per UNIQUE missing WT sequence,
    batched -- never once per variant row), then return a (len(wt_seqs), D)
    tensor aligned to wt_seqs (repeats reused from the cache, no recompute)."""
    from gsr.backbone.pooling import mean_pool

    missing = cache.missing_seqs(wt_seqs)
    if missing:
        was_training = backbone.model.training
        backbone.model.eval()
        new_means = []
        with torch.no_grad():
            for s in range(0, len(missing), batch_size):
                chunk = missing[s:s + batch_size]
                hidden, attn = backbone.forward_reps(chunk, grad=False)
                new_means.append(mean_pool(hidden, attn).float().cpu().numpy())
        if was_training:
            backbone.model.train()
        new_means = np.concatenate(new_means, axis=0)
        if not cache.put(missing, new_means):
            print("[wt_mean_cache] WARNING: cache locked by another writer; "
                  "some WT means were NOT persisted this run (kept in memory).")

    X, missing_pos = cache.get(wt_seqs)
    if missing_pos:
        # Another writer's concurrent put() raced us, or persistence failed
        # above -- fill any still-missing rows directly from what we just
        # computed rather than erroring.
        computed = dict(zip(missing, new_means)) if missing else {}
        for i in missing_pos:
            s = wt_seqs[i]
            if s in computed:
                if X.shape[1] == 0:
                    X = np.zeros((len(wt_seqs), computed[s].shape[0]), np.float32)
                X[i] = computed[s]
            else:
                raise RuntimeError(
                    f"[wt_mean_cache] missing WT mean for a sequence not in "
                    f"this call's fill set (unexpected): {s[:40]}...")
    return torch.from_numpy(X)
