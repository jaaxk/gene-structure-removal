"""Load wild-type protein sequences (human UniRef90) for dataset construction.

Reads a FASTA of UniRef representative sequences (see scripts/download_uniref.py),
applies a length cap (PLL/embedding cost is O(L)), and yields one record per gene.
The FASTA header id is used as the gene id.

Non-canonical residues (e.g. ``X``, ``U``, ``B``, ``Z``) are allowed in the stored
sequence but are never chosen as mutation sites downstream (see mutagenesis.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List

from gsr.args import AA_ALPHABET


@dataclass(frozen=True)
class WTRecord:
    gene_id: str          # UniRef representative id (from FASTA header)
    sequence: str         # wild-type amino-acid sequence


def _iter_fasta(path: Path) -> Iterator[WTRecord]:
    """Minimal streaming FASTA parser (no biopython dependency needed)."""
    header = None
    chunks: List[str] = []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield WTRecord(header, "".join(chunks))
                # id = first whitespace-delimited token after '>'
                header = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line.strip())
    if header is not None:
        yield WTRecord(header, "".join(chunks))


def load_wildtypes(
    fasta_path: Path,
    max_seq_len: int,
    min_seq_len: int = 20,
    require_canonical_frac: float = 0.9,
) -> List[WTRecord]:
    """Load WT records, filtering by length and canonical-residue content.

    Args:
        fasta_path: FASTA of UniRef representative sequences.
        max_seq_len: drop sequences longer than this (length cap).
        min_seq_len: drop trivially short sequences.
        require_canonical_frac: drop sequences with too few canonical AAs
            (guards against fragments that are mostly ``X``).
    """
    fasta_path = Path(fasta_path)
    canonical = set(AA_ALPHABET)
    records: List[WTRecord] = []
    n_seen = n_len = n_canon = 0
    for rec in _iter_fasta(fasta_path):
        n_seen += 1
        L = len(rec.sequence)
        if L < min_seq_len or L > max_seq_len:
            n_len += 1
            continue
        frac = sum(c in canonical for c in rec.sequence) / max(L, 1)
        if frac < require_canonical_frac:
            n_canon += 1
            continue
        records.append(rec)
    print(
        f"[uniref] {fasta_path.name}: {n_seen} sequences read, kept {len(records)} "
        f"(dropped {n_len} by length [{min_seq_len},{max_seq_len}], "
        f"{n_canon} by canonical-frac < {require_canonical_frac})"
    )
    if not records:
        raise ValueError(
            f"No wild-type sequences survived filtering in {fasta_path}. "
            "Check max_seq_len and the FASTA contents."
        )
    return records
