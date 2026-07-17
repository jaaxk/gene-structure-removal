import torch

from gsr.losses.base import DIFFERENT, MIDDLE, SAME
from gsr.losses.contrastive_ce import ContrastiveCELoss
from gsr.losses.ntxent import NTXentLoss
from gsr.losses.triplet import TripletLoss


def _batch():
    torch.manual_seed(0)
    z = torch.randn(8, 16, requires_grad=True)
    y = torch.tensor([SAME, SAME, DIFFERENT, DIFFERENT,
                      SAME, DIFFERENT, MIDDLE, MIDDLE])
    return z, y


def test_contrastive_ce_grad_and_middle_excluded():
    z, y = _batch()
    loss_fn = ContrastiveCELoss(distance_metric="cosine")
    loss, m = loss_fn(z, y)
    assert torch.isfinite(loss)
    loss.backward()
    assert z.grad is not None
    # 6 valid items -> C(6,2)=15 pairs; the 2 middle items contribute none
    assert m["n_pairs"] == 15


def test_contrastive_ce_separates_groups():
    # Perfectly separated groups should give near-zero loss once scaled.
    z = torch.tensor([[5.0, 0.0], [5.0, 0.1], [-5.0, 0.0], [-5.0, 0.1]])
    y = torch.tensor([SAME, SAME, DIFFERENT, DIFFERENT])
    loss_fn = ContrastiveCELoss(distance_metric="cosine")
    loss, m = loss_fn(z, y)
    assert m["pair_acc"] == 1.0


def test_ntxent_and_triplet_finite():
    z, y = _batch()
    for loss_fn in (NTXentLoss(temperature=0.1), TripletLoss(margin=1.0)):
        loss, m = loss_fn(z.detach().requires_grad_(True), y)
        assert torch.isfinite(loss)


def test_all_middle_returns_zero():
    z = torch.randn(4, 8, requires_grad=True)
    y = torch.full((4,), MIDDLE)
    loss, m = ContrastiveCELoss()(z, y)
    assert float(loss) == 0.0
