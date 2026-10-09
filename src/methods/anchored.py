"""The training objective: the data term plus a physics term, in the four combinations of the two
ingredients of the anchored construction.

    L = c_data * lambda_t * ||x0 - x0_hat||^2  +  c * w_t * P(x0, x0_hat)

    linearize  normalize   P
    false      false       ||R(x0_hat)||^2                          PIDM's penalty
    true       false       ||J_R(x0) d||^2                          d = x0_hat - x0
    false      true        ||R(x0_hat)||^2 / ||J_R(x0)||_F^2
    true       true        ||J_R(x0) d||^2 / ||J_R(x0)||_F^2        the anchored term

The term is applied at every step and to every sample.
"""
import torch

from src.diffusion_process.schedule import extract
from src.methods.utils import denoise
from src.utils.linearization import physics_term


def physics_time_weight(diffusion, t, config):
    """The weight w_t of the physics term at noise level t.

    'posterior' is 1/(2 beta_tilde_t), the weight PIDM's Gaussian log-likelihood of the residual
    applies. 'data' is the data term's own weight min(SNR_t, gamma), so the two terms keep the
    ratio c at every noise level. 'none' is flat in t.
    """
    mode = config['physics_time_weight']
    if mode == 'none':
        return None
    if mode == 'data':
        return extract(diffusion.diff_dict['p2_loss_weight'], t, torch.zeros(len(t), device=t.device))
    if mode != 'posterior':
        raise ValueError(f"unknown physics_time_weight {mode!r}; use 'posterior', 'data' or 'none'")
    return 0.5 / extract(diffusion.diff_dict['posterior_variance_clipped'], t, torch.zeros(len(t), device=t.device))


_PROBE_GENERATORS = {}


def _probe_generator(config, device):
    """The generator of the normalisation's random tangents, or None for the global RNG.

    A dedicated generator keeps the global stream (noise and t) identical to the DM run of the
    same seed, so a normalised run differs from it only through the physics term. It is created
    once per process and advances across steps; a resumed run restarts it from the seed.
    """
    seed = config.get('physics_probe_seed', -1)
    seed = -1 if seed is None else int(seed)
    if seed < 0:
        return None
    key = (seed, str(device))
    if key not in _PROBE_GENERATORS:
        _PROBE_GENERATORS[key] = torch.Generator(device=device).manual_seed(seed)
    return _PROBE_GENERATORS[key]


def loss(model, diffusion, residual, x_0, config):
    """Returns the loss and, for logging, the data term, the mean |quantity| the physics term
    penalises, and the physics term."""
    t, x0_hat, data_loss = denoise(model, diffusion, x_0)
    data_loss = config['c_data'] * data_loss.mean()
    if config['active_c_residual'] == 0.0:
        return data_loss, data_loss.item(), 0.0, 0.0

    normalize = bool(config['physics_normalize'])
    per_sample, quantity = physics_term(
        residual.calculate_residual,
        x_0,
        x0_hat,
        linearize=bool(config['physics_linearize']),
        normalize=normalize,
        probes=int(config['physics_probes']),
        generator=_probe_generator(config, x_0.device) if normalize else None,
    )
    weight = physics_time_weight(diffusion, t, config)
    if weight is not None:
        per_sample = per_sample * weight.reshape(per_sample.shape)
    physics_loss = config['active_c_residual'] * per_sample.mean()
    return data_loss + physics_loss, data_loss.item(), quantity.detach().abs().mean().item(), physics_loss.item()
