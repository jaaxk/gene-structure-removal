import pytest

from gsr.args import parse_args


def test_defaults_parse():
    a = parse_args(["--run_name", "t"])
    assert a.scorer == "masked_marginal"
    assert a.pooling == "mean"
    assert a.loss_type == "contrastive_ce"
    assert a.batch_mode == "gene_diverse"


def test_cross_gene_divisibility_enforced():
    with pytest.raises(AssertionError):
        parse_args(["--run_name", "t", "--batch_mode", "cross_gene",
                    "--genes_per_batch", "3", "--batch_size", "64"])


def test_quartile_sum_bound():
    with pytest.raises(AssertionError):
        parse_args(["--run_name", "t", "--quartile_low", "0.6",
                    "--quartile_high", "0.6"])
