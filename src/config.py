"""Command-line options, YAML configs, and the run's device and seeds.

A config file sets any of the options below; the command line overrides it. A run saves its full
configuration to <run>/model/model.yaml.
"""
import argparse
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

from src.methods import METHODS, apply_method_preset
from src.residuals.toy import PROFILES

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def fix_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {'true', '1', 'yes', 'y'}:
        return True
    if value in {'false', '0', 'no', 'n'}:
        return False
    raise argparse.ArgumentTypeError(f'Expected a boolean value, got {value!r}.')


def build_parser():
    parser = argparse.ArgumentParser(description='Train a diffusion model with the anchored physics term.')
    run = parser.add_argument_group('run')
    run.add_argument('--config', default=None, help='YAML file whose values replace the defaults below.')
    run.add_argument('--name', default='run', help='Run name; the run is saved to <output_root>/<dataset>/<timestamp>_<dataset>_<name>.')
    run.add_argument('--output_root', default='./trained_models')
    run.add_argument('--resume_checkpoint', default=None, help='Checkpoint file or run directory to continue training from; the run continues in its own directory.')
    run.add_argument('--seed', type=int, default=0, help='Seed of the weights, the noise and t.')
    run.add_argument('--data_split_seed', type=int, default=None, help='Seed of the train/validation split. Defaults to --seed.')
    run.add_argument('--num_workers', type=int, default=None, help='Data loader workers. Defaults to SLURM_CPUS_PER_TASK or 4.')
    run.add_argument('--log_freq', type=int, default=20, help='Log the training losses every N iterations.')
    run.add_argument('--wandb_track', type=str_to_bool, default=False, help='Log to Weights & Biases.')
    run.add_argument('--wandb_project', default='anchored')

    data = parser.add_argument_group('data')
    data.add_argument('--gov_eqs', choices=['advection', 'burgers', 'darcy', 'toy'], default='burgers', help='The dataset.')
    data.add_argument('--data_dir', default='./data')
    data.add_argument('--advection_beta', type=float, default=1.0)
    data.add_argument('--burgers_nu', type=float, default=0.001)
    data.add_argument('--toy_profile', choices=sorted(PROFILES), default='graded', help='Stiffness profile of the shocked circle; flat is the control.')
    data.add_argument('--toy_dim', type=int, default=10, help='Dimension D of the shocked circle.')
    data.add_argument('--toy_num_samples', type=int, default=20000, help='Training points of the shocked circle.')
    data.add_argument('--toy_valid_samples', type=int, default=20000, help='Held-out points of the shocked circle.')

    method = parser.add_argument_group('objective')
    method.add_argument('--method', choices=sorted(METHODS), default='anchored', help='anchored, pidm, dm (data only), or lin and norm (one ingredient each). See src/methods/registry.py.')
    method.add_argument('--c_data', type=float, default=1.0, help='Weight of the data term.')
    method.add_argument('--c_residual', type=float, default=1.0, help='Weight c of the physics term.')
    method.add_argument('--physics_linearize', type=str_to_bool, default=None, help='Charge ||J_R(x0) (x0_hat - x0)||^2 instead of ||R(x0_hat)||^2. Unset: from --method.')
    method.add_argument('--physics_normalize', type=str_to_bool, default=None, help='Divide the physics term by ||J_R(x0)||_F^2. Unset: from --method.')
    method.add_argument('--physics_time_weight', choices=['posterior', 'data', 'none'], default=None, help="Weight of the physics term over t: 'posterior' 1/(2 beta_tilde_t) as in PIDM, 'data' the data term's min(SNR_t, gamma), 'none' flat. Unset: from --method.")
    method.add_argument('--physics_probes', type=int, default=1, help='Random tangents of the ||J_R(x0)||_F^2 estimate.')
    method.add_argument('--physics_probe_seed', type=int, default=0, help='Seed of the tangents\' own generator, which leaves the noise and t of the DM run with the same --seed unchanged; -1 draws them from the global RNG.')

    training = parser.add_argument_group('training')
    training.add_argument('--diff_steps', type=int, default=100, help='Diffusion steps T.')
    training.add_argument('--snr_gamma', type=float, default=5.0, help='The data term is weighted by min(SNR_t, gamma).')
    training.add_argument('--train_iterations', type=int, default=None, help='Default: 100000.')
    training.add_argument('--train_batch_size', type=int, default=None, help='Default: 8 on the PDEs, 512 on the toy.')
    training.add_argument('--learning_rate', type=float, default=1e-4)
    training.add_argument('--ema_decay', type=float, default=0.0, help='Decay of an exponential moving average of the weights, which is then used for every evaluation and checkpoint; 0 disables it.')
    training.add_argument('--ema_start', type=int, default=1000, help='Iteration from which the average is updated; before it, it is the current weights.')
    training.add_argument('--holdout_size', type=int, default=256, help='Training fields held out of training (from a split seeded by --holdout_seed). The paper\'s PDE runs hold out 256.')
    training.add_argument('--holdout_seed', type=int, default=12345)
    training.add_argument('--test_eval_freq', type=int, default=500, help='Log the loss on a validation batch every N iterations; 0 disables it.')
    training.add_argument('--checkpoint_freq', type=int, default=5000, help='Save a checkpoint every N iterations; the final one is always saved.')
    training.add_argument('--sample_residual_freq', type=int, default=5000, help='Every N iterations, draw --sample_count samples and log their residual; 0 disables it.')
    training.add_argument('--sample_count', type=int, default=9)

    evaluation = parser.add_argument_group('evaluation after training')
    evaluation.add_argument('--generate_after_training', type=str_to_bool, default=True, help='Draw the evaluation samples and compute the metrics when training ends.')
    evaluation.add_argument('--generation_num_samples', type=int, default=1000)
    evaluation.add_argument('--generation_eval_batch_size', type=int, default=32)
    return parser


def parser_keys(parser):
    return {action.dest for action in parser._actions if action.dest != 'help'}


def read_yaml(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Config file not found: {path}')
    with path.open() as f:
        return yaml.safe_load(f) or {}


def run_config_path(checkpoint):
    """model.yaml of the run a checkpoint (file or run directory) belongs to."""
    checkpoint = Path(checkpoint)
    run_dir = checkpoint if checkpoint.is_dir() else checkpoint.parent.parent
    return run_dir / 'model' / 'model.yaml'


def parse_args_and_config(argv=None):
    """Defaults < config file (or, when resuming without one, the run's saved config) < command line."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', default=None)
    pre.add_argument('--resume_checkpoint', default=None)
    pre_args, _ = pre.parse_known_args(argv)

    parser = build_parser()
    keys = parser_keys(parser)
    if pre_args.config is not None:
        values = read_yaml(pre_args.config)
        unknown = sorted(set(values) - keys)
        if unknown:
            raise ValueError(f'Unknown config option(s) in {pre_args.config}: {", ".join(unknown)}')
    elif pre_args.resume_checkpoint is not None:
        # a saved run config also holds bookkeeping entries and options of older versions
        values = {k: v for k, v in read_yaml(run_config_path(pre_args.resume_checkpoint)).items() if k in keys}
    else:
        values = {}
    parser.set_defaults(**values)
    args = parser.parse_args(argv)
    return args, finalize_config(vars(args).copy())


def add_datetime_prefix(name, gov_eqs):
    if re.match(rf'^\d{{8}}_\d{{6}}_{re.escape(gov_eqs)}(?:_|$)', name) is not None:
        return name
    return f'{datetime.now().strftime("%Y%m%d_%H%M%S")}_{gov_eqs}_{name}'


def finalize_config(config):
    config['name'] = add_datetime_prefix(config['name'], config['gov_eqs'])
    config['active_c_residual'] = 0.0 if config['method'] == 'dm' else config['c_residual']
    return apply_method_preset(config)


def run_directory(config):
    return Path(config['output_root']) / config['gov_eqs'] / config['name']


def latest_checkpoint(path):
    """A checkpoint file as given, or the newest checkpoint_<step>.pt of a run directory."""
    path = Path(path)
    if path.is_file():
        return path
    steps = [(int(m.group(1)), p) for p in (path / 'model').glob('checkpoint_*.pt')
             if (m := re.search(r'checkpoint_(\d+)\.pt$', p.name))]
    if not steps:
        raise FileNotFoundError(f'No checkpoint_<step>.pt under {path / "model"}')
    return max(steps)[1]


def checkpoint_step(path):
    match = re.search(r'checkpoint_(\d+)\.pt$', str(path))
    return None if match is None else int(match.group(1))
