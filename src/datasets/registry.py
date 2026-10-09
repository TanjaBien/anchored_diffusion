import math
import os
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader, random_split

from src.datasets.common import DarcyDataset, HDF5TensorDataset, SingleTensorDataset, cycle


@dataclass
class ProblemSetup:
    gov_eqs: str
    channels: int
    train_dataset: Any
    valid_dataset: Any
    train_batch_size: int
    train_iterations: int
    diff_steps: int
    field_shape: tuple = ()             # grid of one field, e.g. (t, x); empty for the toy
    domain_lengths: tuple = ()          # physical extent along each grid axis
    beta: float = 1.0                   # advection speed
    nu: float = 0.001                   # Burgers viscosity
    toy_profile: str = 'graded'


def choose(value, default):
    return default if value is None else value


def hdf5_metadata(path, key='tensor'):
    import h5py

    with h5py.File(path, 'r') as f:
        attrs = dict(f.attrs)
        coordinates = {name: torch.tensor(f[name][:], dtype=torch.float64)
                       for name in ('t-coordinate', 'x-coordinate') if name in f}
    return attrs, coordinates


def hdf5_attr(attrs, name, default):
    for key, value in attrs.items():
        if str(key).lower() == name.lower():
            return float(value)
    return default


def domain_length(coordinate, pixels):
    """Grid spacing times the number of points, from the stored coordinates; 1 if there are none."""
    if coordinate is None or len(coordinate) < 2:
        return 1.0
    return abs(float(coordinate[1] - coordinate[0])) * pixels


def split_dataset(dataset, seed):
    """90 % training, 10 % validation."""
    val_size = max(1, int(0.1 * len(dataset)))
    return random_split(dataset, [len(dataset) - val_size, val_size],
                        generator=torch.Generator().manual_seed(seed))


def build_problem_setup(config):
    gov_eqs = config['gov_eqs']
    data_dir = config['data_dir']

    if gov_eqs in {'advection', 'burgers'}:
        path = (f'{data_dir}/advection/1d_advection_beta{config["advection_beta"]}.h5' if gov_eqs == 'advection'
                else f'{data_dir}/burgers/1d_burgers_nu{config["burgers_nu"]}.h5')
        attrs, coordinates = hdf5_metadata(path)
        dataset = HDF5TensorDataset(path)
        train_dataset, valid_dataset = split_dataset(dataset, data_split_seed_of(config))
        pixels_t, pixels_x = dataset.shape[-2], dataset.shape[-1]
        return ProblemSetup(
            gov_eqs=gov_eqs,
            channels=1,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            train_batch_size=choose(config['train_batch_size'], 8),
            train_iterations=choose(config['train_iterations'], 100000),
            diff_steps=config['diff_steps'],
            field_shape=(pixels_t, pixels_x),
            domain_lengths=(domain_length(coordinates.get('t-coordinate'), pixels_t),
                            domain_length(coordinates.get('x-coordinate'), pixels_x)),
            beta=hdf5_attr(attrs, 'beta', config['advection_beta']),
            nu=hdf5_attr(attrs, 'nu', config['burgers_nu']),
        )

    if gov_eqs == 'darcy':
        train_dataset = DarcyDataset(f'{data_dir}/darcy/train')
        valid_dataset = DarcyDataset(f'{data_dir}/darcy/valid')
        pixels = train_dataset.data.shape[-1]
        return ProblemSetup(
            gov_eqs=gov_eqs,
            channels=2,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            train_batch_size=choose(config['train_batch_size'], 8),
            train_iterations=choose(config['train_iterations'], 100000),
            diff_steps=config['diff_steps'],
            field_shape=(pixels, pixels),
            domain_lengths=(1.0, 1.0),
        )

    if gov_eqs == 'toy':
        # synthetic data, so the two splits are independent draws
        dim, seed = config['toy_dim'], config['seed']
        return ProblemSetup(
            gov_eqs=gov_eqs,
            channels=dim,
            train_dataset=SingleTensorDataset(sample_circle(config['toy_num_samples'], dim, 10_000 + seed)),
            valid_dataset=SingleTensorDataset(sample_circle(config['toy_valid_samples'], dim, 20_000 + seed)),
            train_batch_size=choose(config['train_batch_size'], 512),
            train_iterations=choose(config['train_iterations'], 100000),
            diff_steps=config['diff_steps'],
            toy_profile=config['toy_profile'],
        )

    raise ValueError(f'Unknown dataset {gov_eqs!r}; use advection, burgers, darcy or toy.')


def data_split_seed_of(config):
    return config['seed'] if config.get('data_split_seed') is None else config['data_split_seed']


def make_loader(dataset, batch_size, num_workers=None):
    """An endless iterator over the dataset in a fixed order."""
    if num_workers is None:
        num_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', 4))
    kwargs = {'batch_size': batch_size, 'shuffle': False, 'num_workers': num_workers, 'pin_memory': True}
    if num_workers > 0:
        kwargs['persistent_workers'] = True
    return cycle(DataLoader(dataset, **kwargs))


def sample_circle(size, dim, seed):
    """Uniform on the unit circle in the first two coordinates of R^dim, zeros elsewhere."""
    generator = torch.Generator().manual_seed(seed)
    theta = 2.0 * math.pi * torch.rand(size, generator=generator)
    data = torch.zeros(size, dim)
    data[:, 0], data[:, 1] = torch.cos(theta), torch.sin(theta)
    return data
