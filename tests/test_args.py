import pytest

from gsr.args import parse_args


BASE = ["--run_name", "t", "--dataset_name", "d"]


def test_defaults_parse():
    a = parse_args(BASE)
    assert a.scorer == "masked_marginal"
    assert a.pooling == "concat"
    assert a.loss_type == "wt_anchored_bce"
    assert a.batch_mode == "gene_diverse"


def test_cross_gene_divisibility_enforced():
    with pytest.raises(AssertionError):
        parse_args(BASE + ["--batch_mode", "cross_gene",
                           "--genes_per_batch", "3", "--batch_size", "64"])


def test_quartile_sum_bound():
    with pytest.raises(AssertionError):
        parse_args(BASE + ["--quartile_low", "0.6", "--quartile_high", "0.6"])
