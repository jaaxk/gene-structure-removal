from gsr.data.sampler import GeneBatchSampler
from gsr.losses.base import DIFFERENT, SAME


class _StubDataset:
    """Minimal stand-in exposing indices_by_gene_and_label()."""

    def __init__(self, by_gl):
        self._by_gl = by_gl

    def indices_by_gene_and_label(self):
        return self._by_gl


def _stub(n_genes=5, per_label=10):
    by_gl = {}
    idx = 0
    for g in range(n_genes):
        by_gl[g] = {SAME: [], DIFFERENT: []}
        for _ in range(per_label):
            by_gl[g][SAME].append(idx); idx += 1
            by_gl[g][DIFFERENT].append(idx); idx += 1
    return _StubDataset(by_gl)


def test_gene_diverse_one_batch_per_gene():
    ds = _stub(n_genes=5)
    s = GeneBatchSampler(ds, batch_size=8, batch_mode="gene_diverse", seed=0)
    batches = list(s)
    assert len(batches) == 5           # every gene once per epoch
    assert len(s) == 5


def test_gene_diverse_single_gene_per_batch():
    ds = _stub(n_genes=4, per_label=10)
    s = GeneBatchSampler(ds, batch_size=8, batch_mode="gene_diverse", seed=1)
    gene_of = {}
    for g, by_label in ds._by_gl.items():
        for lab in by_label.values():
            for i in lab:
                gene_of[i] = g
    for batch in s:
        assert len({gene_of[i] for i in batch}) == 1  # all from one gene


def test_reshuffle_changes_variants_across_epochs():
    ds = _stub(n_genes=3, per_label=20)
    s = GeneBatchSampler(ds, batch_size=8, batch_mode="gene_diverse", seed=0)
    s.set_epoch(0); e0 = [set(b) for b in s]
    s.set_epoch(1); e1 = [set(b) for b in s]
    # At least some batch content differs between epochs.
    assert e0 != e1
