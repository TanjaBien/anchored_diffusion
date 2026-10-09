import torch
import torch.nn as nn


class ToyMLP(nn.Module):
    """x0-predicting SiLU MLP for the shocked circle.

    The time enters as the raw scalar t / n_steps, repeated t_dim times: no Fourier features
    anywhere. On a thin manifold such features scale with ||x|| and made the 10-D runs collapse
    towards the origin.
    """

    def __init__(self, dim, n_steps, hidden=512, layers=5, t_dim=64):
        super().__init__()
        self.n_steps = n_steps
        self.t_dim = t_dim
        width, seq = dim + t_dim, []
        for _ in range(layers):
            seq += [nn.Linear(width, hidden), nn.SiLU()]
            width = hidden
        seq += [nn.Linear(width, dim)]
        self.net = nn.Sequential(*seq)

    def forward(self, x, t, *_args, **_kwargs):
        time = (t.float() / self.n_steps).reshape(-1, 1).expand(-1, self.t_dim)
        return self.net(torch.cat([x, time], dim=-1))
