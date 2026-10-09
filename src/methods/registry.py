"""The training objectives, as presets of one loss. Each loss is called as

    loss(model, diffusion, residual, x_0, config) -> (loss, data term, mean |physics quantity|, physics term)

Every method except `dm` is the same objective (`src.methods.anchored`) with a different
physics term, so the methods differ only in the three switches below:

    method    physics_linearize  physics_normalize  physics_time_weight
    pidm      false              false              posterior    ||R(x0_hat)||^2, PIDM's penalty
    lin       true               false              data         ||J_R(x0) d||^2
    norm      false              true               data         ||R(x0_hat)||^2 / ||J_R(x0)||_F^2
    anchored  true               true               data         ||J_R(x0) d||^2 / ||J_R(x0)||_F^2
    dm        (no physics term: the data-only diffusion model)

A switch given explicitly in the YAML config or on the command line overrides the preset.
"""
from dataclasses import dataclass
from typing import Callable, Optional

from . import anchored, standard_diffusion


@dataclass(frozen=True)
class TrainingMethod:
    name: str
    loss_fn: Callable
    physics_linearize: Optional[bool] = None
    physics_normalize: Optional[bool] = None
    physics_time_weight: Optional[str] = None

    @property
    def uses_residual(self):
        return self.physics_linearize is not None


METHODS = {
    'dm': TrainingMethod('dm', standard_diffusion.loss),
    'pidm': TrainingMethod('pidm', anchored.loss, False, False, 'posterior'),
    'lin': TrainingMethod('lin', anchored.loss, True, False, 'data'),
    'norm': TrainingMethod('norm', anchored.loss, False, True, 'data'),
    'anchored': TrainingMethod('anchored', anchored.loss, True, True, 'data'),
}

PHYSICS_SWITCHES = ('physics_linearize', 'physics_normalize', 'physics_time_weight')


def get_training_method(name):
    try:
        return METHODS[name]
    except KeyError as exc:
        available = ', '.join(sorted(METHODS))
        raise ValueError(f'Unknown method {name!r}. Available methods: {available}.') from exc


def apply_method_preset(config):
    """Fill the physics switches the config leaves unset (None) from the method's preset."""
    method = get_training_method(config['method'])
    for key in PHYSICS_SWITCHES:
        if config.get(key) is None:
            config[key] = getattr(method, key)
    return config
