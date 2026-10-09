import argparse
import logging
import os
from pathlib import Path
from typing import Union

os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchfsm.mesh import MeshGrid
from torchfsm.pde import Burgers
from torchfsm.traj_recorder import AutoRecorder, IntervalController
from tqdm import tqdm


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BurgersDataGenerator:
    def __init__(
        self,
        L: float = 1.0,
        N: int = 1024,
        save_dt: float = 1e-4,
        save_steps: int = 256,
        internal_substeps: int = 1,
        nu: float = 0.001,
        device: str = None,
        seed: int = 42,
        use_double: bool = True,
    ):
        self.L = L
        self.N = N
        self.save_dt = save_dt
        self.save_steps = save_steps
        self.internal_substeps = internal_substeps
        self.nu = nu
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.float64 if use_double else torch.float32

        self.dt = save_dt / internal_substeps
        self.total_steps = save_steps * internal_substeps
        self.mesh = MeshGrid([(0, self.L, self.N)], device=self.device, dtype=self.dtype)
        self.x_grid = self.mesh.bc_mesh_grid().reshape(self.N)
        self.burgers_op = Burgers(self.nu)
        self.dataset_tensor = None

        self.seed = seed
        self.set_seed(self.seed)

    def _generate_ic(
        self,
        max_freq: int = 5,
        num_waves: int = 4,
        amplitude_scale: float = 1.0,
        min_amplitude: float = 0.5,
        max_amplitude: float = 1.5,
    ) -> torch.Tensor:
        u_0 = torch.zeros_like(self.x_grid)

        for _ in range(num_waves):
            freq = torch.randint(1, max_freq + 1, (1,), device=self.device).item()
            amplitude = amplitude_scale * torch.randn(1, device=self.device, dtype=self.dtype).item() / max(1, freq)
            phase = torch.rand(1, device=self.device, dtype=self.dtype).item() * 2 * torch.pi
            u_0 += amplitude * torch.sin(2 * torch.pi * freq * self.x_grid / self.L + phase)

        u_0 -= u_0.mean()
        max_abs = u_0.abs().max().clamp_min(1e-8)
        target_amplitude = min_amplitude + (
            max_amplitude - min_amplitude
        ) * torch.rand(1, device=self.device, dtype=self.dtype).item()
        u_0 = target_amplitude * u_0 / max_abs
        return u_0.reshape(1, 1, self.N)

    def generate_dataset(
        self,
        num_samples: int,
        max_freq: int = 5,
        num_waves: int = 4,
        amplitude_scale: float = 1.0,
        min_amplitude: float = 0.5,
        max_amplitude: float = 1.5,
    ) -> None:
        logger.info(
            'Generating %s Burgers trajectories on %s with N=%s, save_dt=%s, save_steps=%s.',
            num_samples,
            self.device,
            self.N,
            self.save_dt,
            self.save_steps,
        )
        all_trajectories = []

        recorder = AutoRecorder(
            control_func=IntervalController(interval=self.internal_substeps),
            include_initial_state=True,
        )
        for _ in tqdm(range(num_samples), desc='Simulating Burgers PDEs'):
            u_0 = self._generate_ic(
                max_freq=max_freq,
                num_waves=num_waves,
                amplitude_scale=amplitude_scale,
                min_amplitude=min_amplitude,
                max_amplitude=max_amplitude,
            )
            traj = self.burgers_op.integrate(
                u_0=u_0,
                mesh=self.mesh,
                dt=self.dt,
                step=self.total_steps,
                trajectory_recorder=recorder,
                nan_check=True,
            )
            traj_tensor = traj.squeeze(0).permute(1, 0, 2).contiguous().cpu()
            all_trajectories.append(traj_tensor)

        self.dataset_tensor = torch.stack(all_trajectories)
        logger.info('Generated dataset with shape %s.', self.dataset_tensor.shape)

    def save_hdf5(self, filepath: Union[str, Path], use_double: bool = None) -> None:
        if self.dataset_tensor is None:
            raise ValueError('Empty dataset.')

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        write_double = self.dtype == torch.float64 if use_double is None else use_double
        dtype = np.float64 if write_double else np.float32

        t_coordinate = np.arange(self.dataset_tensor.shape[-2], dtype=np.float64) * self.save_dt
        x_coordinate = np.arange(self.N, dtype=np.float64) * self.L / self.N

        with h5py.File(filepath, 'w') as f:
            f.attrs['nu'] = self.nu
            f.attrs['L'] = self.L
            f.attrs['save_dt'] = self.save_dt
            f.attrs['internal_dt'] = self.dt
            f.attrs['internal_substeps'] = self.internal_substeps
            f.create_dataset('tensor', data=self.dataset_tensor.numpy().astype(dtype))
            f.create_dataset('t-coordinate', data=t_coordinate)
            f.create_dataset('x-coordinate', data=x_coordinate)

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
                extent=[0, self.L, 0, self.save_dt * (sample.shape[0] - 1)],
            )
            plt.colorbar(image, label='u(x, t)')
            plt.xlabel('Space (x)')
            plt.ylabel('Time (t)')
            plt.title(f'Sample {i + 1}: 1D Burgers Trajectory')

            save_path = save_dir / f'sample_{i + 1:03d}.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

        logger.info('%s plots were saved to %s.', num_plots, save_dir)

    def set_seed(self, seed: int):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def parse_args():
    # the defaults are the settings of the paper's dataset, data/burgers/1d_burgers_nu0.001.h5
    parser = argparse.ArgumentParser(description='Generate 1D Burgers data with torchfsm.')
    parser.add_argument('--num_samples', type=int, default=10000)
    parser.add_argument('--L', type=float, default=1.0)
    parser.add_argument('--N', type=int, default=512)
    parser.add_argument('--save_dt', type=float, default=0.0005)
    parser.add_argument('--save_steps', type=int, default=255)
    parser.add_argument('--internal_substeps', type=int, default=1)
    parser.add_argument('--nu', type=float, default=0.001)
    parser.add_argument('--max_freq', type=int, default=4)
    parser.add_argument('--num_waves', type=int, default=4)
    parser.add_argument('--amplitude_scale', type=float, default=1.0)
    parser.add_argument('--min_amplitude', type=float, default=0.8)
    parser.add_argument('--max_amplitude', type=float, default=1.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default=None)
    parser.add_argument('--use_double', action='store_true', default=True)
    parser.add_argument('--use_float32', dest='use_double', action='store_false')
    parser.add_argument('--save_path', default=None)
    parser.add_argument('--plot_samples', type=int, default=5)
    parser.add_argument('--plot_dir', default='data/burgers/visualizations')
    return parser.parse_args()


def main():
    args = parse_args()
    save_path = args.save_path or f'data/burgers/1d_burgers_nu{args.nu}.h5'
    generator = BurgersDataGenerator(
        L=args.L,
        N=args.N,
        save_dt=args.save_dt,
        save_steps=args.save_steps,
        internal_substeps=args.internal_substeps,
        nu=args.nu,
        device=args.device,
        seed=args.seed,
        use_double=args.use_double,
    )
    generator.generate_dataset(
        num_samples=args.num_samples,
        max_freq=args.max_freq,
        num_waves=args.num_waves,
        amplitude_scale=args.amplitude_scale,
        min_amplitude=args.min_amplitude,
        max_amplitude=args.max_amplitude,
    )
    if args.plot_samples > 0:
        generator.plot_samples(num_plots=args.plot_samples, save_dir=args.plot_dir)
    generator.save_hdf5(filepath=save_path, use_double=args.use_double)


if __name__ == '__main__':
    main()
