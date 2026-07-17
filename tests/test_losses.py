import torch

from gsr.losses.base import DIFFERENT, MIDDLE, SAME
from gsr.losses.contrastive_ce import ContrastiveCELoss
from gsr.losses.ntxent import NTXentLoss
from gsr.losses.triplet import TripletLoss
from gsr.losses.wt_anchored_bce import WTAnchoredBCELoss


def _batch():
    torch.manual_seed(0)
    z_mut = torch.randn(8, 16, requires_grad=True)
    z_wt = torch.randn(8, 16, requires_grad=True)
    y = torch.tensor([SAME, SAME, DIFFERENT, DIFFERENT,
                      SAME, DIFFERENT, MIDDLE, MIDDLE])
    return z_mut, z_wt, y


def test_contrastive_ce_grad_and_middle_excluded():
    z_mut, z_wt, y = _batch()
    loss, m = ContrastiveCELoss(distance_metric="cosine")(z_mut, z_wt, y)
    assert torch.isfinite(loss)
    loss.backward()
    assert z_mut.grad is not None
    assert m["n_pairs"] == 15  # C(6,2) over the 6 non-middle items


def test_wt_anchored_pulls_same_together():
    # 'same' pairs identical (sim=1 -> prob 1), 'different' pairs opposite.
    z_mut = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    z_wt = torch.tensor([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]])
    y = torch.tensor([SAME, SAME, DIFFERENT, DIFFERENT])
    loss, m = WTAnchoredBCELoss(distance_metric="cosine")(z_mut, z_wt, y)
    assert m["pair_acc"] == 1.0
    assert m["n_items"] == 4


def test_wt_anchored_grad_flows():
    z_mut, z_wt, y = _batch()
    loss, m = WTAnchoredBCELoss()(z_mut, z_wt, y)
    assert torch.isfinite(loss)
    loss.backward()
    assert z_mut.grad is not None and z_wt.grad is not None


def test_ntxent_and_triplet_finite():
    z_mut, z_wt, y = _batch()
    for loss_fn in (NTXentLoss(temperature=0.1), TripletLoss(margin=1.0)):
        loss, m = loss_fn(z_mut.detach().requires_grad_(True), z_wt, y)
        assert torch.isfinite(loss)


def test_all_middle_returns_zero():
    z = torch.randn(4, 8, requires_grad=True)
    y = torch.full((4,), MIDDLE)
    loss, m = WTAnchoredBCELoss()(z, z, y)
    assert float(loss) == 0.0
