from src.residuals.pde import AdvectionResidual, BurgersResidual, DarcyResidual
from src.residuals.toy import ToyResidual


def build_residual(setup, device='cpu'):
    """The residual operator of the dataset, on the grid of its data."""
    if setup.gov_eqs == 'advection':
        return AdvectionResidual(*setup.field_shape, *setup.domain_lengths, beta=setup.beta, device=device)
    if setup.gov_eqs == 'burgers':
        return BurgersResidual(*setup.field_shape, *setup.domain_lengths, nu=setup.nu, device=device)
    if setup.gov_eqs == 'darcy':
        return DarcyResidual(setup.field_shape[0], device=device)
    if setup.gov_eqs == 'toy':
        return ToyResidual(setup.toy_profile)
    raise ValueError(f'Unknown dataset: {setup.gov_eqs}')


__all__ = ['AdvectionResidual', 'BurgersResidual', 'DarcyResidual', 'ToyResidual', 'build_residual']
