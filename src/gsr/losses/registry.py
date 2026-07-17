"""Build a loss module from parsed args (``--loss_type``)."""

from __future__ import annotations

from gsr.losses.base import BaseLoss
from gsr.losses.contrastive_ce import ContrastiveCELoss
from gsr.losses.ntxent import NTXentLoss
from gsr.losses.triplet import TripletLoss


def build_loss(args) -> BaseLoss:
    if args.loss_type == "contrastive_ce":
        return ContrastiveCELoss(distance_metric=args.distance_metric,
                                 use_learnable_scale=args.use_learnable_scale)
    if args.loss_type == "ntxent":
        return NTXentLoss(temperature=args.ntxent_temperature)
    if args.loss_type == "triplet":
        return TripletLoss(margin=args.triplet_margin)
    raise ValueError(f"Unknown loss_type {args.loss_type!r}")
