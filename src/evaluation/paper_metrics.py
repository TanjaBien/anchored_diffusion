"""The metrics of the paper's tables, computed from generated samples and reference data.

    PDEs   mean|R|       mean over samples of the per-sample mean |R| over the entries the
                         operator evaluates (Burgers/advection blank a few boundary rows, which
                         are left out; Darcy: the PDE channel on the whole grid)
           MMD           unbiased MMD^2 with an RBF kernel (median bandwidth) between per-sample
                         features, standardised by the reference
           sigma ratio   median sigma(x) of the samples over the median of the data, with
                         sigma(x) = ||J_R(x) - J_R(0)||_F (RMS over outputs, Hutchinson estimate)
           std ratio     mean per-sample std of the field, samples over data
           sol. error    Darcy only: median ||p - p*(K)|| / ||p*(K)||, p* the solution of the PDE
                         for the sample's own K
    toy    mean|R|, circular W1 of the angle, sigma ratio (mean exact ||grad R|| of the samples over
           the data's) and the median distance to the data plane ||x_{3:D}||
"""
import numpy as np
import torch
from scipy.spatial.distance import cdist

from src.config import device
from src.residuals.pde import source_term
from src.residuals.toy import toy_residual
from src.utils.linearization import residual_jvp
from src.utils.tensors import generalized_b_xy_c_to_image

SIGMA_PROBES = 2
SIGMA_SEED = 0
FEATURES_1D = ('max_grad', 'tv', 'mean_u', 'std_u', 'max_abs_u')
MMD_FEATURES_1D = ('max_grad', 'tv', 'std_u', 'max_abs_u')
FEATURES_DARCY = ('std_p', 'max_grad_p', 'mean_logK', 'std_logK', 'corr_logK')
DARCY_CORRELATION_LAG = 6      # pixels, ~ the prior's length scale 0.1 on the 64-point unit square
CIRCULAR_W1_GRID = 4096


# ---------------------------------------------------------------- features and MMD
def physical_features(u, dx):
    """(n, 5) per-sample features of 1-D space-time fields u (n, T, X): max |du/dx|, total
    variation along x, mean, std and max |u|."""
    n = u.shape[0]
    grad = np.abs(np.gradient(u, dx, axis=-1))
    return np.stack([
        grad.reshape(n, -1).max(axis=1),
        np.abs(np.diff(u, axis=-1)).reshape(n, -1).sum(axis=1),
        u.reshape(n, -1).mean(axis=1),
        u.reshape(n, -1).std(axis=1),
        np.abs(u).reshape(n, -1).max(axis=1),
    ], axis=1)


def mmd_rbf(x, y):
    """Unbiased MMD^2 with a Gaussian kernel whose bandwidth is the median pooled distance."""
    z = np.vstack([x, y])
    dists = cdist(z, z, 'euclidean')
    median_dist = float(np.median(dists[dists > 0])) if (dists > 0).any() else 1.0
    gamma = 1.0 / (2.0 * median_dist ** 2 + 1e-30)
    kxx = np.exp(-gamma * cdist(x, x, 'sqeuclidean'))
    kyy = np.exp(-gamma * cdist(y, y, 'sqeuclidean'))
    kxy = np.exp(-gamma * cdist(x, y, 'sqeuclidean'))
    np.fill_diagonal(kxx, 0.0)
    np.fill_diagonal(kyy, 0.0)
    n, m = len(x), len(y)
    mmd2 = kxx.sum() / (n * (n - 1)) + kyy.sum() / (m * (m - 1)) - 2.0 * kxy.mean()
    return float(max(mmd2, 0.0))


def standardised_mmd(features, reference_features, eps):
    mean, std = reference_features.mean(0), reference_features.std(0) + eps
    return mmd_rbf((features - mean) / std, (reference_features - mean) / std)


def max_gradient_2d(fields, channel=0):
    """max |grad| per sample by central differences on the unit square."""
    u = fields[:, channel].astype(np.float64)
    h = 1.0 / (u.shape[-1] - 1)
    g0, g1 = np.gradient(u, h, axis=(1, 2))
    return np.sqrt(g0 ** 2 + g1 ** 2).reshape(len(u), -1).max(axis=1)


def darcy_features(fields, lag=DARCY_CORRELATION_LAG):
    """(n, 5): std(p), max|grad p|, mean and std of log K, and the lag-`lag` autocorrelation of
    log K averaged over both axes."""
    p = fields[:, 0].astype(np.float64)
    log_k = np.log(np.clip(fields[:, 1].astype(np.float64), 1e-6, None))
    centred = log_k - log_k.mean(axis=(1, 2), keepdims=True)
    variance = (centred ** 2).mean(axis=(1, 2)) + 1e-12
    corr = 0.5 * ((centred[:, lag:] * centred[:, :-lag]).mean(axis=(1, 2))
                  + (centred[:, :, lag:] * centred[:, :, :-lag]).mean(axis=(1, 2))) / variance
    return np.stack([p.reshape(len(p), -1).std(axis=1), max_gradient_2d(fields),
                     log_k.mean(axis=(1, 2)), np.sqrt(variance), corr], axis=1)


# ---------------------------------------------------------------- residual and sensitivity
def _pde_residual(operator, block, gov_eqs):
    values = operator.calculate_residual(block)
    if gov_eqs == 'darcy':            # (b, xy, PDE + 2 BC channels): the PDE channel as an image
        return generalized_b_xy_c_to_image(values[..., :1])[:, 0]
    return values


def evaluated_entries(operator, gov_eqs, field_shape):
    """The residual entries the operator evaluates: it zeroes the rest, which no random field hits."""
    probe = torch.randn(2, *field_shape, device=device) + (2.0 if gov_eqs == 'darcy' else 0.0)
    values = _pde_residual(operator, probe, gov_eqs).detach()
    return (values.abs() > 0).all(dim=0).reshape(-1)


def mean_abs_residual(fields, operator, gov_eqs, chunk=32):
    """Mean |R| per sample over the evaluated entries."""
    keep = evaluated_entries(operator, gov_eqs, fields.shape[1:])
    out = []
    with torch.no_grad():
        for start in range(0, len(fields), chunk):
            block = torch.as_tensor(fields[start:start + chunk], device=device)
            values = _pde_residual(operator, block, gov_eqs).abs().reshape(len(block), -1)
            out.append(values[:, keep].mean(dim=1).cpu().numpy())
    return np.concatenate(out)


def sensitivity(fields, operator, chunk=16, probes=SIGMA_PROBES, seed=SIGMA_SEED):
    """sigma(x) = ||J_R(x) - J_R(0)||_F, RMS over all outputs, from `probes` Gaussian tangents.

    Subtracting J_R(0) keeps the part of the Jacobian that depends on the state; on a linear
    residual it is zero. The tangents come from a seeded generator, so sigma is reproducible.
    """
    generator = torch.Generator().manual_seed(seed)
    out = []
    for start in range(0, len(fields), chunk):
        cpu_block = torch.as_tensor(np.ascontiguousarray(fields[start:start + chunk]))
        block = cpu_block.to(device)
        zero, total = torch.zeros_like(block), torch.zeros(len(block), device=device)
        for _ in range(probes):
            tangent = torch.randn(cpu_block.shape, generator=generator, dtype=cpu_block.dtype).to(device)
            difference = (residual_jvp(operator.calculate_residual, block, tangent)
                          - residual_jvp(operator.calculate_residual, zero, tangent))
            total = total + difference.detach().pow(2).reshape(len(block), -1).mean(dim=1)
        out.append(torch.sqrt(total / probes).cpu().numpy())
    return np.concatenate(out)


# ---------------------------------------------------------------- Darcy forward error
class PressureSolver:
    """The Darcy problem's linear system for a given K, solved for p.

    -div(K grad p) = f at every node (findiff, acc 2), zero Neumann rows on the four edges and the
    trapezoidal integral of p = 0, solved in the least-squares sense through the normal equations
    and a float64 Cholesky. On the validation data this reproduces the stored p to ~3e-5.
    """

    def __init__(self, n=64):
        import scipy.sparse as sp
        from findiff import FinDiff
        h, shape = 1.0 / (n - 1), (n, n)
        mat = lambda axis, order: sp.csr_matrix(FinDiff(axis, h, order, acc=2).matrix(shape))
        self.sp = sp
        self.d0, self.d1, self.d00, self.d11 = mat(0, 1), mat(1, 1), mat(0, 2), mat(1, 2)
        edge = np.zeros(shape, dtype=bool)
        first_0, last_0, first_1, last_1 = (edge.copy() for _ in range(4))
        first_0[0, :], last_0[-1, :], first_1[:, 0], last_1[:, -1] = True, True, True, True
        self.bc = sp.vstack([-self.d0[first_0.ravel()], self.d0[last_0.ravel()],
                             self.d1[first_1.ravel()], -self.d1[last_1.ravel()]])
        # trapezoidal weights: 1 at the corners, 2 on the edges, 4 inside, times h^2 / 4
        weights = np.full(shape, 4.0)
        weights[0, :] = weights[-1, :] = weights[:, 0] = weights[:, -1] = 2.0
        weights[0, 0] = weights[0, -1] = weights[-1, 0] = weights[-1, -1] = 1.0
        self.weights = (weights * h ** 2 / 4.0).reshape(1, -1)
        grid = np.linspace(0.0, 1.0, n)
        x, y = np.meshgrid(grid, grid, indexing='ij')
        source = source_term(torch.as_tensor(x), torch.as_tensor(y)).numpy().ravel()
        self.rhs = np.concatenate([source, np.zeros(self.bc.shape[0] + 1)])

    @staticmethod
    def _torch(matrix):
        matrix = matrix.tocoo()
        return torch.sparse_coo_tensor(np.vstack([matrix.row, matrix.col]), matrix.data, matrix.shape,
                                       device=device, dtype=torch.float64)

    def solve(self, permeability):
        sp, k = self.sp, permeability.astype(np.float64).ravel()
        a = (-sp.diags(k) @ self.d00 - sp.diags(self.d0 @ k) @ self.d0
             - sp.diags(k) @ self.d11 - sp.diags(self.d1 @ k) @ self.d1)
        m = sp.vstack([a, self.bc, sp.csr_matrix(self.weights)])
        normal = torch.sparse.mm(self._torch(m.T), self._torch(m).to_dense())
        rhs = torch.as_tensor(m.T @ self.rhs, device=device)
        factor, info = torch.linalg.cholesky_ex(normal)
        if info.item() != 0:
            return None
        return torch.cholesky_solve(rhs[:, None], factor)[:, 0].cpu().numpy().reshape(permeability.shape)

    def solution_error(self, fields):
        """||p - p*(K)|| / ||p*(K)|| per sample, p shifted to zero trapezoidal mean first (the
        constant is the gauge of the Neumann problem); NaN where the solve fails."""
        out = np.full(len(fields), np.nan)
        for i, (p, k) in enumerate(zip(fields[:, 0].astype(np.float64), fields[:, 1])):
            solution = self.solve(k)
            if solution is None:
                continue
            p = p - float(self.weights @ p.ravel())
            out[i] = np.linalg.norm(p - solution) / np.linalg.norm(solution)
        return out


# ---------------------------------------------------------------- toy
def circular_w1(generated_theta, data_theta, grid=CIRCULAR_W1_GRID):
    """W1 between two distributions of angles on the circle, in radians."""
    edges = np.linspace(0, 2 * np.pi, grid + 1)
    generated_cdf = np.searchsorted(np.sort(generated_theta), edges[1:], side='right') / len(generated_theta)
    data_cdf = np.searchsorted(np.sort(data_theta), edges[1:], side='right') / len(data_theta)
    difference = generated_cdf - data_cdf
    return float(np.mean(np.abs(difference - np.median(difference))) * 2 * np.pi)


def residual_gradient_norm(points, profile):
    """Exact ||grad R(x)|| per point."""
    x = torch.as_tensor(np.ascontiguousarray(points, dtype=np.float32)).requires_grad_(True)
    (gradient,) = torch.autograd.grad(toy_residual(profile, x).sum(), x)
    return gradient.norm(dim=-1).numpy()


def toy_paper_metrics(samples, reference, profile):
    angle = lambda x: np.mod(np.arctan2(x[:, 1], x[:, 0]), 2 * np.pi)
    with torch.no_grad():
        residual = toy_residual(profile, torch.as_tensor(samples)).abs().numpy()
    off_plane = np.linalg.norm(samples[:, 2:], axis=1) if samples.shape[1] > 2 else np.zeros(len(samples))
    return {
        'paper/residual': float(residual.mean()),
        'paper/w1': circular_w1(angle(samples), angle(reference)),
        # the mean, not the median: the shocks are narrow, so the median sits on the flat part
        # of the circle and is ~1 for every run
        'paper/sigma_ratio': float(residual_gradient_norm(samples, profile).mean()
                                   / residual_gradient_norm(reference, profile).mean()),
        'paper/off_plane': float(np.median(off_plane)),
    }


# ---------------------------------------------------------------- entry point
def paper_metrics(gov_eqs, samples, reference, operator, profile=None):
    """The table columns for one run. samples and reference are float32 arrays shaped like the
    data; operator is the dataset's residual (for Darcy, evaluated on the whole grid)."""
    samples = np.ascontiguousarray(samples, dtype=np.float32)
    reference = np.ascontiguousarray(reference, dtype=np.float32)
    if gov_eqs == 'toy':
        return toy_paper_metrics(samples, reference, profile)

    metrics = {'paper/residual': float(mean_abs_residual(samples, operator, gov_eqs).mean())}
    if gov_eqs != 'advection':         # advection's residual is linear: sigma is zero everywhere
        chunk = 32 if gov_eqs == 'darcy' else 16
        sigma, sigma_data = sensitivity(samples, operator, chunk), sensitivity(reference, operator, chunk)
        metrics['paper/sigma_ratio'] = float(np.median(sigma) / np.median(sigma_data))

    if gov_eqs == 'darcy':
        metrics['paper/mmd'] = standardised_mmd(darcy_features(samples), darcy_features(reference), 1e-12)
        for name, channel in (('p', 0), ('K', 1)):
            std = lambda x: x[:, channel].reshape(len(x), -1).std(axis=1).mean()
            metrics[f'paper/std_{name}_ratio'] = float(std(samples) / std(reference))
        metrics['paper/solution_error'] = float(np.nanmedian(PressureSolver(samples.shape[-1]).solution_error(samples)))
        return metrics

    keep = [FEATURES_1D.index(name) for name in MMD_FEATURES_1D]
    features = physical_features(samples[:, 0], 1.0 / samples.shape[-1])
    reference_features = physical_features(reference[:, 0], 1.0 / reference.shape[-1])
    metrics['paper/mmd'] = standardised_mmd(features[:, keep], reference_features[:, keep], 1e-8)
    std_index = FEATURES_1D.index('std_u')
    metrics['paper/std_u_ratio'] = float(features[:, std_index].mean()
                                         / (reference_features[:, std_index].mean() + 1e-12))
    return metrics
