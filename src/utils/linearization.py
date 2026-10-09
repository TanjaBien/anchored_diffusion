"""Anchored (linearised) and scale-normalised physics terms.

Both ingredients of the general construction are built from the residual operator and the
training sample alone:

    linearisation   J_F(x0) d              in place of   F(x0 + d)
    normalisation   / E_v ||J_F(x0) v||^2  with v ~ N(0, I)

The Jacobian-vector product is taken with forward-mode autodiff on the residual function
itself, so no equation-specific code is needed and the discretisation is shared with the
residual exactly.  It is exact: unlike a finite difference (F(x0 + h d) - F(x0)) / h it
carries no step size.
"""
import torch


def residual_jvp(residual_fn, x0, delta):
    """J_F(x0) . delta.  The anchor is detached, so gradients reach the model through delta."""
    _, tangent = torch.func.jvp(residual_fn, (x0.detach(),), (delta,))
    return tangent


def _per_sample_mean_square(tensor):
    return tensor.pow(2).reshape(tensor.shape[0], -1).mean(dim=1)


EXACT_OUTPUT_LIMIT = 16


def _exact_jacobian_scale(residual_fn, anchor):
    """||J||_F^2 / n_out by one reverse-mode pass per output component.

    Used when the residual has few outputs per sample. There the Hutchinson estimate below is
    unusable: with a single output the estimate is proportional to a chi-squared variable with
    one degree of freedom, whose reciprocal has infinite mean, so dividing by it is unstable.
    """
    out, vjp_fn = torch.func.vjp(residual_fn, anchor)
    batch_size = out.shape[0]
    n_out = out.numel() // batch_size
    total = torch.zeros(batch_size, device=anchor.device, dtype=anchor.dtype)
    for index in range(n_out):
        cotangent = torch.zeros(batch_size, n_out, device=out.device, dtype=out.dtype)
        cotangent[:, index] = 1.0
        gradient = vjp_fn(cotangent.reshape(out.shape))[0]
        total = total + gradient.reshape(batch_size, -1).pow(2).sum(dim=1)
    return total / n_out


def jacobian_scale(residual_fn, x0, probes=1, generator=None):
    """||J_F(x0)||_F^2, in the same per-element units as the numerator.

    With v ~ N(0, I),  E_v mean((J v)^2) = ||J||_F^2 / n_out, and the numerator is
    ||J d||^2 / n_out, so the ratio of the two is exactly ||J d||^2 / ||J||_F^2.
    The estimate averages over the residual's output elements, so it is accurate for a field
    residual; for a residual with few outputs it is computed exactly instead.
    """
    anchor = x0.detach()
    with torch.no_grad():
        probe_out = residual_fn(anchor)
    if probe_out.numel() // anchor.shape[0] <= EXACT_OUTPUT_LIMIT:
        return _exact_jacobian_scale(residual_fn, anchor)

    total = torch.zeros(anchor.shape[0], device=anchor.device, dtype=anchor.dtype)
    for _ in range(probes):
        probe = torch.randn(anchor.shape, device=anchor.device, dtype=anchor.dtype,
                            generator=generator)
        total = total + _per_sample_mean_square(residual_jvp(residual_fn, anchor, probe).detach())
    return total / probes


def physics_term(residual_fn, x0, x0_pred, plain_residual=None, linearize=True, normalize=True,
                 probes=1, eps=1e-12, generator=None):
    """Per-sample physics penalty for any of the four ablation arms.

    linearize=False, normalize=False  ->  ||F(x0_pred)||^2            (the PIDM penalty)
    linearize=True,  normalize=False  ->  ||J_F(x0) d||^2
    linearize=False, normalize=True   ->  ||F(x0_pred)||^2 / ||J_F(x0)||_F^2
    linearize=True,  normalize=True   ->  ||J_F(x0) d||^2 / ||J_F(x0)||_F^2
    """
    if linearize:
        quantity = residual_jvp(residual_fn, x0, x0_pred - x0.detach())
    elif plain_residual is not None:
        quantity = plain_residual
    else:
        quantity = residual_fn(x0_pred)

    per_sample = _per_sample_mean_square(quantity)
    if not normalize:
        return per_sample, quantity
    scale = jacobian_scale(residual_fn, x0, probes=probes, generator=generator)
    return per_sample / (scale + eps), quantity
