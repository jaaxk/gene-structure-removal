"""Generate single-amino-acid variants for a wild-type sequence.

We only ever embed the variants we actually use in training batches, so instead
of enumerating all 19*L substitutions we *sample* a configurable number per gene.
Sampling is done with a caller-supplied RNG for reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from gsr.args import AA_ALPHABET


@dataclass(frozen=True)
class Variant:
    gene_id: str
    mutant: str        # HGVS-like short form, 1-indexed, e.g. "A123V"
    pos: int           # 1-indexed position in the sequence
    wt_aa: str
    mut_aa: str
    sequence: str      # full mutated amino-acid sequence


def _canonical_sites(sequence: str) -> List[int]:
    """0-indexed positions whose WT residue is a canonical amino acid."""
    canonical = set(AA_ALPHABET)
    return [i for i, c in enumerate(sequence) if c in canonical]


def enumerate_variants(gene_id: str, sequence: str) -> List[Variant]:
    """All single-aa substitutions at canonical sites (19 per site).

    Used for scoring: masked/wt-marginal produce the log-likelihood for every
    variant in one sweep per gene, so scoring the full set is as cheap as a
    subset and yields run-independent quartile thresholds.
    """
    variants: List[Variant] = []
    for site in _canonical_sites(sequence):
        wt_aa = sequence[site]
        for alt in AA_ALPHABET:
            if alt == wt_aa:
                continue
            variants.append(Variant(
                gene_id=gene_id, mutant=f"{wt_aa}{site + 1}{alt}", pos=site + 1,
                wt_aa=wt_aa, mut_aa=alt,
                sequence=sequence[:site] + alt + sequence[site + 1:]))
    return variants


def sample_variants(
    gene_id: str,
    sequence: str,
    n: int,
    rng: np.random.Generator,
) -> List[Variant]:
    """Sample up to ``n`` distinct single-aa variants of ``sequence``.

    Each candidate is a (position, alternative-aa) pair where the WT residue is
    canonical and the alternative differs from WT. If fewer than ``n`` candidates
    exist, all are returned.
    """
    sites = _canonical_sites(sequence)
    # Enumerate candidate (site_idx, alt_aa_idx) pairs lazily via flat index to
    # avoid materializing the full list for long proteins.
    n_alt = len(AA_ALPHABET)
    total = len(sites) * n_alt
    if total == 0:
        return []
    take = min(n, total)
    # Oversample flat indices then filter out no-op (alt == wt) collisions and
    # dedup, topping up until we have `take` distinct real substitutions.
    chosen: dict[int, tuple[int, str]] = {}
    attempts = 0
    max_attempts = 20 * take + 100
    while len(chosen) < take and attempts < max_attempts:
        flat = rng.integers(0, total, size=take * 2)
        for f in flat:
            f = int(f)
            site = sites[f // n_alt]
            alt = AA_ALPHABET[f % n_alt]
            if alt == sequence[site]:
                continue
            chosen.setdefault(f, (site, alt))
            if len(chosen) >= take:
                break
        attempts += 1

    variants: List[Variant] = []
    for site, alt in chosen.values():
        wt_aa = sequence[site]
        mut_seq = sequence[:site] + alt + sequence[site + 1:]
        variants.append(
            Variant(
                gene_id=gene_id,
                mutant=f"{wt_aa}{site + 1}{alt}",
                pos=site + 1,
                wt_aa=wt_aa,
                mut_aa=alt,
                sequence=mut_seq,
            )
        )
    return variants
