import logging
from pathlib import Path
from typing import Union

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchfsm.mesh import MeshGrid
from torchfsm.operator import LinearAdvection
from torchfsm.traj_recorder import AutoRecorder
from tqdm import tqdm


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AdvectionDataGenerator:
    def __init__(
        self,
        L: float = 1.0,
        N: int = 256,
        dt: float = 1 / 256,
        steps: int = 256,
        beta: float = 1.0,
        device: str = None,
        seed: int = 42,
    ):
        self.L = L
        self.N = N
        self.dt = dt
        self.steps = steps
        self.beta = beta
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.mesh = MeshGrid([(0, self.L, self.N)], device=self.device)
        self.x_grid = self.mesh.bc_mesh_grid()
        self.advection_op = -1 * LinearAdvection(self.beta)
        self.dataset_tensor = None

        self.seed = seed
        self.set_seed(self.seed)

    def _generate_ic(self, max_freq: int = 8, num_waves: int = 2) -> torch.Tensor:
        u_0 = torch.zeros_like(self.x_grid)

        for _ in range(num_waves):
            l_i = torch.randint(1, max_freq + 1, (1,), device=self.device).item()
            A_i = torch.rand(1, device=self.device).item()
            phi_i = torch.rand(1, device=self.device).item() * 2 * torch.pi
            u_0 += A_i * torch.sin(2 * torch.pi * l_i * self.x_grid / self.L + phi_i)

        if torch.rand(1).item() < 0.1:
            sign = 1 if torch.rand(1).item() > 0.5 else -1
            u_0 = sign * torch.abs(u_0)

        if torch.rand(1).item() < 0.1:
            window = torch.sin(torch.pi * self.x_grid / self.L) ** 2
            u_0 = u_0 * window

        return u_0

    def generate_dataset(self, num_samples: int, max_freq: int = 5) -> None:
        logger.info('Generating %s trajectories on %s.', num_samples, self.device)
        all_trajectories = []

        for _ in tqdm(range(num_samples), desc='Simulating PDEs'):
            u_0 = self._generate_ic(max_freq=max_freq)
            traj = self.advection_op.integrate(
                u_0=u_0,
                mesh=self.mesh,
                dt=self.dt,
                step=self.steps,
                trajectory_recorder=AutoRecorder(),
            )

            traj_tensor = torch.stack(traj) if isinstance(traj, list) else traj
            traj_tensor = traj_tensor.squeeze().unsqueeze(0).cpu()
            all_trajectories.append(traj_tensor)

        self.dataset_tensor = torch.stack(all_trajectories)
        logger.info('Generated dataset with shape %s.', self.dataset_tensor.shape)

    def save_hdf5(self, filepath: Union[str, Path], use_double: bool = False) -> None:
        if self.dataset_tensor is None:
            raise ValueError('Empty dataset.')

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        dtype = np.float64 if use_double else np.float32

        with h5py.File(filepath, 'w') as f:
            f.create_dataset('tensor', data=self.dataset_tensor.numpy().astype(dtype))

        logger.info("Dataset saved to '%s' as %s.", filepath, dtype.__name__)

    def plot_samples(self, num_plots: int = 3, save_dir: Union[str, Path] = 'plots') -> None:
        if self.dataset_tensor is None:
            raise ValueError('Empty dataset.')

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        for i in range(min(num_plots, len(self.dataset_tensor))):
            sample = self.dataset_tensor[i, 0].numpy()

            plt.figure(figsize=(10, 6))
            image = plt.imshow(
                sample,
                aspect='auto',
                origin='lower',
                cmap='RdBu_r',
                extent=[0, self.L, 0, self.dt * self.steps],
            )
            plt.colorbar(image, label='u(x, t)')
            plt.xlabel('Space (x)')
            plt.ylabel('Time (t)')
            plt.title(f'Sample {i + 1}: 1D Advection Trajectory')

            save_path = save_dir / f'sample_{i + 1:03d}.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

        logger.info("%s plots were saved to '%s'.", num_plots, save_dir)

    def set_seed(self, seed: int):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    num_samples = 10000
    beta = 1.0
    save_path = f'data/advection/1d_advection_beta{beta}.h5'

    generator = AdvectionDataGenerator(L=1.0, N=256, dt=1 / 256, steps=255, beta=beta)
    generator.generate_dataset(num_samples=num_samples)
    generator.plot_samples(num_plots=5, save_dir='data/advection/visualizations')
    generator.save_hdf5(filepath=save_path)
