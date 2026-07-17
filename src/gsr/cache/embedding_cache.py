"""Content-addressed embedding cache (frozen backbone), shared across runs.

Stores per-variant mutant + position-matched wild-type embeddings keyed by
``variant_id`` (SHA1 of the mutated sequence). The cache dir is keyed by
(model, layer, pooling) so pooling variants never collide.

Design goals from usage on the cluster:
- **Bulk reads only.** Callers read everything they need for a run in one
  ``get`` at startup and hold it resident, so the training loop never does
  per-batch H5 I/O (which starves the GPU and triggers low-util cancellation).
- **Graceful misses.** ``get`` returns which ids are missing; the caller computes
  them live and calls ``put``.
- **Safe concurrent writes.** ``put`` takes an exclusive writelock; if another
  writer holds it, ``put`` returns False (caller keeps the embeddings in memory
  but does not persist them) instead of blocking or corrupting the file.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np

from gsr import paths


class EmbeddingCache:
    def __init__(self, model: str, layer: int, pooling: str):
        self.dir = (paths.SCRATCH_ROOT / "cache" / "embeddings" /
                    f"{model}_L{layer}_{pooling}")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.h5_path = self.dir / "emb.h5"
        self.sidecar = self.dir / "hash_to_id.json"
        self.lock_path = self.dir / "emb.h5.writelock"

    # --- index ----------------------------------------------------------
    def _load_index(self) -> dict:
        if self.sidecar.exists():
            with open(self.sidecar) as fh:
                return json.load(fh)
        return {}

    # --- read -----------------------------------------------------------
    def get(self, variant_ids: List[str]) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """Bulk-read mut+wt embeddings; missing rows are NaN.

        Returns (mut (n,D), wt (n,D), missing_positions) where missing_positions
        indexes into ``variant_ids`` for entries not in the cache. If the cache is
        empty, all positions are missing and the arrays are zero-width.
        """
        index = self._load_index()
        present = [(i, index[v]) for i, v in enumerate(variant_ids) if v in index]
        missing = [i for i, v in enumerate(variant_ids) if v not in index]
        if not present or not self.h5_path.exists():
            return (np.empty((0, 0), np.float32), np.empty((0, 0), np.float32),
                    list(range(len(variant_ids))))
        out_idx = np.array([p[0] for p in present])
        rows = np.array([p[1] for p in present])
        order = np.argsort(rows)
        try:
            with h5py.File(self.h5_path, "r") as h5:
                D = h5["X_mut"].shape[1]
                mut = np.full((len(variant_ids), D), np.nan, np.float32)
                wt = np.full((len(variant_ids), D), np.nan, np.float32)
                mut[out_idx[order]] = h5["X_mut"][rows[order]]
                wt[out_idx[order]] = h5["X_wt"][rows[order]]
        except (OSError, KeyError):
            # Corrupt/partial read -> treat everything as missing (recompute).
            return (np.empty((0, 0), np.float32), np.empty((0, 0), np.float32),
                    list(range(len(variant_ids))))
        return mut, wt, missing

    # --- write ----------------------------------------------------------
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

    def put(self, variant_ids: List[str], mut: np.ndarray, wt: np.ndarray) -> bool:
        """Persist new (mut, wt) rows. Returns False (without saving) if another
        writer holds the lock -- the caller keeps them in memory regardless."""
        if not self._acquire_lock():
            return False
        try:
            index = self._load_index()
            new = [(i, v) for i, v in enumerate(variant_ids) if v not in index]
            if not new:
                return True
            mut = np.asarray(mut, np.float32)
            wt = np.asarray(wt, np.float32)
            D = mut.shape[1]
            with h5py.File(self.h5_path, "a") as h5:
                if "X_mut" not in h5:
                    for name in ("X_mut", "X_wt"):
                        h5.create_dataset(name, shape=(0, D), maxshape=(None, D),
                                          dtype="float32", chunks=(1024, D))
                start = h5["X_mut"].shape[0]
                add_mut = np.stack([mut[i] for i, _ in new])
                add_wt = np.stack([wt[i] for i, _ in new])
                for name, arr in (("X_mut", add_mut), ("X_wt", add_wt)):
                    h5[name].resize(start + len(new), axis=0)
                    h5[name][start:] = arr
            for offset, (_, vid) in enumerate(new):
                index[vid] = start + offset
            tmp = self.sidecar.with_suffix(".json.tmp")
            with open(tmp, "w") as fh:
                json.dump(index, fh)
            tmp.replace(self.sidecar)
            return True
        finally:
            self._release_lock()
