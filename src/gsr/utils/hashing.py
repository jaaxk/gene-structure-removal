"""Stable content hashing for sequences.

A variant is identified by the SHA1 of its (mutated) amino-acid sequence. This
matches the convention used by the sibling repo's embedding store and lets us
key embeddings/scores by content rather than by a fragile positional index, so
the same sequence is never scored or embedded twice.
"""

from __future__ import annotations

import hashlib


def seq_hash(sequence: str) -> str:
    """SHA1 hex digest of an amino-acid sequence (case-sensitive, as-is)."""
    return hashlib.sha1(sequence.encode("utf-8")).hexdigest()
