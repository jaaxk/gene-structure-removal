import torch

from gsr.models.projection_head import ProjectionHead


def test_output_shape_and_depth():
    head = ProjectionHead(input_dim=32, hidden_dims=[16, 8], out_dim=4,
                          dropout=0.1, activation="gelu", norm="layernorm")
    x = torch.randn(5, 32)
    y = head(x)
    assert y.shape == (5, 4)
    n_linear = sum(1 for m in head.net if isinstance(m, torch.nn.Linear))
    assert n_linear == 3  # input->16->8->4 is a 3-layer MLP


def test_no_activation_after_output():
    head = ProjectionHead(32, [16], 4, norm="none", dropout=0.0)
    assert isinstance(head.net[-1], torch.nn.Linear)
