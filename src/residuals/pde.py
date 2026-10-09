"""Residual operators of the three PDEs.

Each maps a batch of fields, shaped like the data (batch, channels, n0, n1), to its residual,
shaped (batch, n0 * n1, residual channels). The training loss, the sampler and the metrics all use
this one function.
"""
import torch

from src.residuals.stencils import FiniteDifference
from src.utils.tensors import generalized_image_to_b_xy_c

TRIM = 5   # advection and Burgers leave this many rows next to the boundary unevaluated


class AdvectionResidual:
    """u_t + beta u_x on a (t, x) grid, periodic in x. Rows and columns within TRIM of the edge are
    set to zero, keeping one-sided stencil artefacts out of the training signal."""

    gov_eqs = 'advection'

    def __init__(self, pixels_t, pixels_x, domain_length_t=1.0, domain_length_x=1.0, beta=1.0, device='cpu'):
        self.dt = domain_length_t / pixels_t
        self.dx = domain_length_x / pixels_x
        self.beta = beta
        self.d_t = FiniteDifference(0, 1, self.dt, periodic=False, device=device)
        self.d_x = FiniteDifference(1, 1, self.dx, periodic=True, device=device)

    def calculate_residual(self, fields):
        u = fields[:, 0]
        eq_0 = self.d_t(u) + self.beta * self.d_x(u)
        eq_0[:, 0:TRIM, :] = 0.0
        eq_0[:, -TRIM:, :] = 0.0
        eq_0[:, :, 0:TRIM] = 0.0
        eq_0[:, :, -TRIM:] = 0.0
        return generalized_image_to_b_xy_c(eq_0.unsqueeze(1))


class BurgersResidual:
    """u_t + u u_x - nu u_xx on a (t, x) grid. The spatial terms are spectral (periodic in x), the
    product dealiased by the 2/3 rule; u_t is a finite difference. The TRIM rows next to the initial
    and final time are set to zero."""

    gov_eqs = 'burgers'

    def __init__(self, pixels_t, pixels_x, domain_length_t=1.0, domain_length_x=1.0, nu=0.001, device='cpu'):
        self.nu = nu
        self.d_t = FiniteDifference(0, 1, domain_length_t / pixels_t, periodic=False, device=device)
        freq = torch.fft.fftfreq(pixels_x, d=domain_length_x / pixels_x, device=device)
        self.k = (2j * torch.pi * freq).reshape(1, 1, pixels_x)
        self.dealias = (freq.abs() <= freq.abs().max() * (2.0 / 3.0)).to(device=device).reshape(1, 1, pixels_x)

    def calculate_residual(self, fields):
        u = fields[:, 0]
        u_hat = torch.fft.fft(u, dim=-1)
        u_xx = torch.fft.ifft((self.k ** 2) * u_hat, dim=-1).real
        dealiased = u_hat * self.dealias
        convection = torch.fft.ifft(dealiased, dim=-1).real * torch.fft.ifft(self.k * dealiased, dim=-1).real

        eq_0 = self.d_t(u) + convection - self.nu * u_xx
        trim = min(TRIM, eq_0.shape[1] // 2)
        eq_0[:, :trim, :] = 0.0
        eq_0[:, -trim:, :] = 0.0
        return generalized_image_to_b_xy_c(eq_0.unsqueeze(1))


class DarcyResidual:
    """-div(K grad p) - f on the unit square (fields: channel 0 = p, channel 1 = K), with zero-flux
    Neumann conditions on the four edges. Three residual channels: the PDE at every node and the
    normal derivative of p on the edges (zero in the interior). f is +10 in the corner square
    [0, 0.125]^2 and -10 in the opposite one."""

    gov_eqs = 'darcy'

    def __init__(self, pixels=64, domain_length=1.0, device='cpu'):
        self.pixels = pixels
        h = domain_length / (pixels - 1)          # nodes on the boundary
        self.d_0 = FiniteDifference(0, 1, h, device=device)
        self.d_1 = FiniteDifference(1, 1, h, device=device)
        self.d_00 = FiniteDifference(0, 2, h, device=device)
        self.d_11 = FiniteDifference(1, 2, h, device=device)
        x = torch.linspace(0.0, 1.0, steps=pixels)
        grid_0, grid_1 = torch.meshgrid(x, x, indexing='ij')
        self.source = source_term(grid_0, grid_1).reshape(1, pixels, pixels).to(device)

    def calculate_residual(self, fields):
        p, permeability = fields[:, 0], fields[:, 1]
        p_0, p_1 = self.d_0(p), self.d_1(p)
        grad_p = torch.stack([p_0, p_1], dim=-3)
        p_00, p_11 = self.d_00(p), self.d_11(p)
        k_0, k_1 = self.d_0(permeability), self.d_1(permeability)
        eq_0 = (-permeability * p_00 - k_0 * p_0) + (-permeability * p_11 - k_1 * p_1) - self.source.to(p.dtype)
        eq_0 = generalized_image_to_b_xy_c(eq_0.unsqueeze(1)).squeeze(-1)

        # outward normal derivative of p on each edge; the second axis is stored reversed, so the
        # signs of its two edges are swapped
        boundary = torch.zeros_like(grad_p)
        boundary[:, 0, 0, :] = -grad_p[:, 0, 0, :]
        boundary[:, 0, -1, :] = grad_p[:, 0, -1, :]
        boundary[:, 1, :, 0] = grad_p[:, 1, :, 0]
        boundary[:, 1, :, -1] = -grad_p[:, 1, :, -1]
        return torch.cat([eq_0.unsqueeze(-1), generalized_image_to_b_xy_c(boundary)], dim=-1)


def source_term(x, y, w=0.125, r=10.0):
    result = torch.zeros_like(x)
    lower = torch.logical_and(torch.abs(x - 0.5 * w) <= 0.5 * w, torch.abs(y - 0.5 * w) <= 0.5 * w)
    upper = torch.logical_and(torch.abs(x - 1 + 0.5 * w) <= 0.5 * w, torch.abs(y - 1 + 0.5 * w) <= 0.5 * w)
    result[lower] = r
    result[upper] = -r
    return result
