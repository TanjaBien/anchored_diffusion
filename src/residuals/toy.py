"""The shocked circle.

The data lie on the unit circle in the first two coordinates of R^D (all other coordinates zero).
The residual keeps the whole unit sphere as its zero set but varies in stiffness along it,

    R(x) = (||x||^2 - 1) * g(x),

so on the sphere the sensitivity is sigma = ||grad R|| = 2 g(x): choosing the residual is choosing
the profile g. The sphere strictly contains the data circle, so the residual cannot see a sample
that drifts along the sphere out of the data plane.
"""
import torch

PROFILES = {
    # four narrow shocks, all equally strong
    'narrow': dict(heights=[30.0] * 4, widths=[0.10] * 4),
    # the same shocks, graded in strength: sigma varies by a factor of 101 along the circle
    'graded': dict(heights=[4.0, 12.0, 35.0, 100.0], widths=[0.10] * 4),
    # equal height, four different widths
    'widths': dict(heights=[30.0] * 4, widths=[0.06, 0.12, 0.25, 0.50]),
    # control: constant stiffness, the plain unit-sphere residual
    'flat': dict(heights=[], widths=[]),
}

RADIAL_MASK_SCALE = 0.8


def bump_centres(dim, device=None, dtype=torch.float32):
    """The four shock centres as unit vectors, +e1, +e2, -e1, -e2: theta = 0, pi/2, pi, 3pi/2."""
    centres = torch.zeros(4, dim, device=device, dtype=dtype)
    centres[0, 0], centres[1, 1], centres[2, 0], centres[3, 1] = 1.0, 1.0, -1.0, -1.0
    return centres


def _radial_mask(radius):
    """Fades the bumps out at the origin, where the direction x/||x|| is undefined.

    Without it grad R has a tangential 1/||x|| blow-up near the origin, which is where the x0
    prediction sits at high noise. The mask is 1 on the sphere, so the stiffness there is untouched.
    """
    a2 = RADIAL_MASK_SCALE ** 2
    return (1.0 + a2) * radius ** 2 / (radius ** 2 + a2)


def stiffness_of(profile, x):
    """g(x) for vectors in any dimension.

    Each bump is von Mises-Fisher in the direction, exp(-2 (1 - x_hat . c) / w^2). Written in the
    dot product rather than the geodesic angle, because acos has an infinite derivative at
    x_hat = +-c and a sample exactly on a bump centre would otherwise produce a NaN gradient.
    """
    spec = PROFILES[profile]
    radius = x.norm(dim=-1)
    total = torch.ones_like(radius)
    if not spec['heights']:
        return total
    direction = x / radius.clamp_min(1e-12).unsqueeze(-1)
    centres = bump_centres(x.shape[-1], device=x.device, dtype=x.dtype)
    mask = _radial_mask(radius)
    for centre, height, width in zip(centres, spec['heights'], spec['widths']):
        cosine = (direction * centre).sum(-1)
        total = total + mask * height * torch.exp(-2.0 * (1.0 - cosine) / width ** 2)
    return total


def stiffness_on_circle(profile, theta):
    """g along the data circle, for plotting against the angle."""
    return stiffness_of(profile, torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1))


def toy_residual(profile, x):
    return ((x ** 2).sum(-1) - 1.0) * stiffness_of(profile, x)


class ToyResidual:
    gov_eqs = 'toy'

    def __init__(self, profile='graded'):
        if profile not in PROFILES:
            raise ValueError(f'Unknown toy profile {profile!r}; use one of {sorted(PROFILES)}.')
        self.profile = profile

    def calculate_residual(self, x):
        """R(x), one value per sample."""
        return toy_residual(self.profile, x)
