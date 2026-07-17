import numpy as np

from gsr.args import AA_ALPHABET
from gsr.data.mutagenesis import sample_variants


def test_sample_basic():
    seq = "MACDEFGHIK"
    rng = np.random.default_rng(0)
    variants = sample_variants("g1", seq, n=15, rng=rng)
    assert len(variants) == 15
    for v in variants:
        assert v.gene_id == "g1"
        assert v.wt_aa == seq[v.pos - 1]
        assert v.mut_aa != v.wt_aa
        assert v.mut_aa in AA_ALPHABET
        # mutated sequence differs from WT at exactly one position
        assert len(v.sequence) == len(seq)
        diffs = [i for i in range(len(seq)) if v.sequence[i] != seq[i]]
        assert diffs == [v.pos - 1]
        assert v.mutant == f"{v.wt_aa}{v.pos}{v.mut_aa}"


def test_distinct_variants():
    seq = "MACDEFGHIK"
    rng = np.random.default_rng(1)
    variants = sample_variants("g1", seq, n=100, rng=rng)
    # 10 positions * 19 alternatives = 190 possible; all should be distinct
    keys = {(v.pos, v.mut_aa) for v in variants}
    assert len(keys) == len(variants)


def test_cap_at_available():
    seq = "MA"  # 2 canonical positions -> 2*19 = 38 possible
    rng = np.random.default_rng(2)
    variants = sample_variants("g1", seq, n=1000, rng=rng)
    assert len(variants) == 38


def test_noncanonical_sites_skipped():
    seq = "MXXA"  # only positions 1 and 4 are canonical
    rng = np.random.default_rng(3)
    variants = sample_variants("g1", seq, n=1000, rng=rng)
    assert {v.pos for v in variants} <= {1, 4}
