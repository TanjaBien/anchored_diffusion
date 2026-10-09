"""Finite-difference derivatives of 2-D fields as convolutions, second-order accurate (findiff stencils).

The interior uses the central stencil; without periodic boundaries the edge rows and columns use
findiff's one-sided stencils, so the derivative is defined on the whole grid.
"""
import torch
import torch.nn.functional as F
from findiff import FinDiff


class FiniteDifference:
    """d^order / d axis^order of fields shaped (batch, ..., n0, n1) on a grid with the given spacing."""

    def __init__(self, axis, order, spacing, periodic=False, device='cpu', acc=2):
        stencils = FinDiff(axis, spacing, order, acc=acc).stencil((99, 99)).data
        self.periodic = periodic
        self.inner_offset = max(max(abs(i), abs(j)) for i, j in stencils[('C', 'C')])
        self.offset = max(max(abs(i), abs(j)) for key, stencil in stencils.items() if key != ('C', 'C')
                          for i, j in stencil)
        self.kernels = {}
        for key, stencil in stencils.items():
            mid = self.inner_offset if key == ('C', 'C') else self.offset
            kernel = torch.zeros((1, 1, 2 * mid + 1, 2 * mid + 1), device=device)
            for (i, j), value in stencil.items():
                kernel[0, 0, mid + i, mid + j] = value
            self.kernels[key] = kernel

    def _conv(self, x, key):
        kernel = self.kernels[key].repeat((x.size(1), 1, 1, 1))
        return F.conv2d(x, kernel, groups=x.size(1))

    def __call__(self, x):
        original_size = x.size()
        batch_size, *_, height, width = original_size
        x = x.view(batch_size, -1, height, width)
        inner = self.inner_offset

        if self.periodic:
            x = F.pad(x, (inner, inner, inner, inner), mode='circular')
            return self._conv(x, ('C', 'C')).view(original_size)

        interior = self._conv(x, ('C', 'C'))
        # the edges: the one-sided stencils, on the padded field next to each edge and corner
        x_ext = F.pad(x, (self.offset, self.offset, self.offset, self.offset), mode='constant', value=0)
        reach = 2 * self.offset + inner
        out = torch.zeros_like(x)
        out[:, :, inner:-inner, inner:-inner] = interior
        out[:, :, 0:inner, :] = self._conv(x_ext[:, :, 0:reach, :], ('L', 'C'))
        out[:, :, -inner:, :] = self._conv(x_ext[:, :, -reach:, :], ('H', 'C'))
        out[:, :, :, 0:inner] = self._conv(x_ext[:, :, :, 0:reach], ('C', 'L'))
        out[:, :, :, -inner:] = self._conv(x_ext[:, :, :, -reach:], ('C', 'H'))
        out[:, :, 0:inner, 0:inner] = self._conv(x_ext[:, :, 0:reach, 0:reach], ('L', 'L'))
        out[:, :, 0:inner, -inner:] = self._conv(x_ext[:, :, 0:reach, -reach:], ('L', 'H'))
        out[:, :, -inner:, 0:inner] = self._conv(x_ext[:, :, -reach:, 0:reach], ('H', 'L'))
        out[:, :, -inner:, -inner:] = self._conv(x_ext[:, :, -reach:, -reach:], ('H', 'H'))
        return out.view(original_size)
