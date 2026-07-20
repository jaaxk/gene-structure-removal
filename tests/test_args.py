import pytest

from gsr.args import parse_args


BASE = ["--run_name", "t"]


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


def test_new_pooling_choices_parse():
    a = parse_args(BASE + ["--pooling", "wt_mut_mean_concat"])
    assert a.pooling == "wt_mut_mean_concat"
    a = parse_args(BASE + ["--pooling", "wt_subtracted_mean"])
    assert a.pooling == "wt_subtracted_mean"


def test_wt_subtracted_mean_with_wt_anchored_bce_warns(capsys):
    parse_args(BASE + ["--pooling", "wt_subtracted_mean",
                       "--loss_type", "wt_anchored_bce"])
    out = capsys.readouterr().out
    assert "wt_subtracted_mean" in out and "wt_anchored_bce" in out


def test_wt_subtracted_mean_with_other_loss_does_not_warn(capsys):
    parse_args(BASE + ["--pooling", "wt_subtracted_mean",
                       "--loss_type", "contrastive_ce"])
    out = capsys.readouterr().out
    assert "wt_anchored_bce" not in out
