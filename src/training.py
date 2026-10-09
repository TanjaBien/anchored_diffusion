"""The training loop: fixed budget, optional EMA of the weights, periodic diagnostics and checkpoints."""
import contextlib

import torch
from torch.utils.data import Subset
from tqdm import tqdm

from src.datasets import make_loader
from src.config import device
from src.diffusion_process import EMA, save_model
from src.evaluation.paper_metrics import toy_paper_metrics
from src.methods import get_training_method


def train_step(model, diffusion, residual, batch, config, optimizer):
    model.train()
    loss, data_loss, quantity, physics_loss = get_training_method(config['method']).loss_fn(
        model, diffusion, residual, batch, config)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.item(), data_loss, quantity, physics_loss


def hold_out(setup, config):
    """Take config['holdout_size'] fields out of the training split, chosen by a fixed seed.

    The paper's PDE runs reserved them for a stopping objective that the final protocol no longer
    uses; they are still left out so that every run trains on exactly the same fields.
    """
    size = int(config['holdout_size'])
    if size <= 0 or size >= len(setup.train_dataset):
        return
    order = torch.randperm(len(setup.train_dataset), generator=torch.Generator().manual_seed(int(config['holdout_seed'])))
    setup.train_dataset = Subset(setup.train_dataset, order[size:].tolist())


@contextlib.contextmanager
def isolated_rng(seed):
    """Run a block on its own random stream, so that diagnostics leave the training noise unchanged."""
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        torch.manual_seed(seed)
        yield
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def sample_diagnostics(model, diffusion, residual, setup, config):
    """Residual of a few fresh samples (on the toy also their distance to the data and angular W1)."""
    shape = (config['sample_count'], setup.channels, *setup.field_shape)
    samples = diffusion.p_sample_loop(model, shape)
    with torch.no_grad():
        values = residual.calculate_residual(samples)
    metrics = {'samples/residual_mean_abs': float(values.abs().mean())}
    if setup.gov_eqs == 'toy':
        reference = setup.valid_dataset.tensors[:len(samples)].numpy()
        metrics.update({f'samples/{k.split("/")[1]}': v for k, v in
                        toy_paper_metrics(samples.cpu().numpy(), reference, setup.toy_profile).items()})
    return samples, values, metrics


def setup_logging(config):
    if not config['wandb_track']:
        return lambda metrics, step: None, lambda: None
    import wandb
    wandb.init(project=config['wandb_project'], name=config['name'], config=config)
    return wandb.log, wandb.finish


def train(config, setup, model, residual, diffusion, run_dir, start_iteration=0):
    hold_out(setup, config)
    train_loader = make_loader(setup.train_dataset, setup.train_batch_size, config['num_workers'])
    valid_loader = make_loader(setup.valid_dataset, setup.train_batch_size, config['num_workers'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    ema = None
    if config['ema_decay'] > 0.0:
        ema = EMA(config['ema_decay'])
        ema.register(model)
    log, finish = setup_logging(config)

    progress = tqdm(range(start_iteration, setup.train_iterations + 1))
    try:
        for iteration in progress:
            batch = next(train_loader).to(device)
            loss, data_loss, quantity, physics_loss = train_step(model, diffusion, residual, batch, config, optimizer)
            # every evaluation and checkpoint below uses the averaged weights
            if ema is not None:
                if iteration > config['ema_start']:
                    ema.update(model)
                else:
                    ema.register(model)
                ema.ema(model)
            model.eval()

            if config['log_freq'] > 0 and iteration % config['log_freq'] == 0:
                progress.set_description(f'loss {loss:.3e}')
                log({'loss': loss, 'loss_data': data_loss, 'residual_mean_abs': quantity,
                     'loss_physics': physics_loss}, step=iteration)

            if config['test_eval_freq'] > 0 and iteration % config['test_eval_freq'] == 0:
                with torch.no_grad():
                    test = get_training_method(config['method']).loss_fn(
                        model, diffusion, residual, next(valid_loader).to(device), config)
                log({'loss_test': float(test[0]), 'loss_data_test': test[1]}, step=iteration)

            if config['checkpoint_freq'] > 0 and iteration > 0 and iteration % config['checkpoint_freq'] == 0:
                save_model(config, model, iteration, str(run_dir))

            if config['sample_residual_freq'] > 0 and (iteration % config['sample_residual_freq'] == 0
                                                       or iteration == setup.train_iterations):
                # own random stream: how often this runs does not change the training
                with isolated_rng(config['seed'] * 1_000_003 + iteration):
                    samples, values, metrics = sample_diagnostics(model, diffusion, residual, setup, config)
                log(metrics, step=iteration)
                if config['wandb_track']:
                    import wandb
                    import matplotlib.pyplot as plt
                    from src.visualization.fields import sample_figure
                    figure = sample_figure(samples, values, setup.field_shape, title=f'step {iteration}')
                    log({'samples': wandb.Image(figure)}, step=iteration)
                    plt.close(figure)

            if ema is not None:
                ema.restore(model)
    finally:
        finish()

    if ema is not None:
        ema.ema(model, backup=False)
    save_model(config, model, setup.train_iterations, str(run_dir))
    return model


def describe(model, config):
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return (f'{config["gov_eqs"]}: method {config["method"]}, c = {config["active_c_residual"]:g}, '
            f'{count / 1e6:.2f}M parameters')

