import torch

from gsr.backbone.pooling import output_dim, pool_batch


def _synthetic():
    # (B=2, L=4, D=3), no padding (attn all ones) -- values chosen so
    # mean/position vectors are easy to compute by hand.
    hidden = torch.tensor([
        [[1., 1., 1.], [2., 2., 2.], [3., 3., 3.], [4., 4., 4.]],
        [[5., 5., 5.], [6., 6., 6.], [7., 7., 7.], [8., 8., 8.]],
    ])
    attn = torch.ones(2, 4)
    positions = [1, 2]
    return hidden, attn, positions


def test_output_dim():
    assert output_dim(3, "mean") == 3
    assert output_dim(3, "mutated_position") == 3
    assert output_dim(3, "concat") == 6
    assert output_dim(3, "wt_mut_mean_concat") == 9
    assert output_dim(3, "wt_subtracted_mean") == 3


def test_wt_subtracted_mean_defaults_to_zero_without_wt_mean():
    hidden, attn, positions = _synthetic()
    out = pool_batch(hidden, attn, positions, "wt_subtracted_mean")
    torch.testing.assert_close(out, torch.zeros(2, 3))


def test_wt_subtracted_mean_with_explicit_wt_mean():
    hidden, attn, positions = _synthetic()
    wt_mean = torch.tensor([[10., 10., 10.], [20., 20., 20.]])
    out = pool_batch(hidden, attn, positions, "wt_subtracted_mean", wt_mean=wt_mean)
    mean = torch.tensor([[2.5, 2.5, 2.5], [6.5, 6.5, 6.5]])
    torch.testing.assert_close(out, mean - wt_mean)


def test_wt_mut_mean_concat_with_explicit_wt_mean():
    hidden, attn, positions = _synthetic()
    wt_mean = torch.tensor([[10., 10., 10.], [20., 20., 20.]])
    out = pool_batch(hidden, attn, positions, "wt_mut_mean_concat", wt_mean=wt_mean)
    mean = torch.tensor([[2.5, 2.5, 2.5], [6.5, 6.5, 6.5]])
    pos_vec = torch.tensor([[2., 2., 2.], [7., 7., 7.]])
    torch.testing.assert_close(out[:, :3], mean)
    torch.testing.assert_close(out[:, 3:6], pos_vec)
    torch.testing.assert_close(out[:, 6:], wt_mean)


def test_old_modes_unaffected_by_wt_mean_arg():
    hidden, attn, positions = _synthetic()
    wt_mean = torch.tensor([[99., 99., 99.], [-1., -1., -1.]])
    for pooling in ("mean", "mutated_position", "concat"):
        without = pool_batch(hidden, attn, positions, pooling)
        with_arg = pool_batch(hidden, attn, positions, pooling, wt_mean=wt_mean)
        torch.testing.assert_close(without, with_arg)


def test_wt_mean_mismatched_dtype_is_cast():
    # Regression test: WtMeanCache round-trips wt_mean through numpy (always
    # float32) and other call sites .cpu().numpy() it -- pool_batch must cast
    # (and, on a real GPU run, move device) rather than erroring or silently
    # upcasting the whole computation.
    hidden, attn, positions = _synthetic()
    wt_mean_f64 = torch.tensor([[10., 10., 10.], [20., 20., 20.]], dtype=torch.float64)
    out = pool_batch(hidden, attn, positions, "wt_subtracted_mean", wt_mean=wt_mean_f64)
    assert out.dtype == hidden.dtype
    mean = torch.tensor([[2.5, 2.5, 2.5], [6.5, 6.5, 6.5]])
    torch.testing.assert_close(out, mean - wt_mean_f64.to(mean.dtype))
