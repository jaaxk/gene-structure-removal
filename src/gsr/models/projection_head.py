"""Configurable MLP projection head.

Maps a (frozen) ESM embedding into the learned space where gene structure is
removed. Depth, widths, dropout, activation, and normalization are all config-
driven so the head can be swept for hyperparameter tuning. The default
(input -> 1024 -> 512 -> 256) is a 3-layer MLP.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

_ACT = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU, "tanh": nn.Tanh}


def _norm(kind: str, dim: int) -> nn.Module:
    if kind == "layernorm":
        return nn.LayerNorm(dim)
    if kind == "batchnorm":
        return nn.BatchNorm1d(dim)
    if kind == "none":
        return nn.Identity()
    raise ValueError(f"Unknown norm {kind!r}")


class ProjectionHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        out_dim: int,
        dropout: float = 0.1,
        activation: str = "gelu",
        norm: str = "layernorm",
    ):
        super().__init__()
        if activation not in _ACT:
            raise ValueError(f"Unknown activation {activation!r}")
        dims = [input_dim] + list(hidden_dims) + [out_dim]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            is_last = i == len(dims) - 2
            if not is_last:  # no norm/activation/dropout after the output layer
                layers.append(_norm(norm, dims[i + 1]))
                layers.append(_ACT[activation]())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)
        self.input_dim = input_dim
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
