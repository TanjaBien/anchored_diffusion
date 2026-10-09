"""Training cost of one objective: time per training step and per epoch, nothing else.

Builds the run exactly as main.py does, from the same command-line flags, and calls
src.training.train_step on the real training loader -- but with no evaluation, sampling, logging or
checkpointing, so the time is the training step itself. Each epoch is one pass over the training
fields (the held-out fields left out, as in training). The first epoch is warm-up and is reported
but not averaged.

    python -m src.evaluation.benchmark --bench_epochs 5 --bench_out results/training_cost.csv \
        --bench_label PIDM <the main.py flags of the run>

Appends one row per run to the CSV: the step time (CUDA events around train_step, data loading
excluded), the epoch wall time (data loading included), both as mean and std over epochs 2..N,
and the peak GPU memory.
"""
import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

from src.config import device, fix_seeds, parse_args_and_config
from src.datasets import build_problem_setup, make_loader
from src.diffusion_process import DenoisingDiffusion
from src.models import build_model
from src.residuals import build_residual
from src.training import hold_out, train_step


def bench_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--bench_epochs', type=int, default=5)
    parser.add_argument('--bench_out', default='results/training_cost.csv')
    parser.add_argument('--bench_label', required=True)
    parser.add_argument('--bench_steps', type=int, default=None,
                        help='cap on the steps per epoch: every dataset is then timed over the same number of steps')
    own, rest = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + rest          # main.py's parser sees only its own flags
    return own


def main():
    own = bench_args()
    _args, config = parse_args_and_config()
    fix_seeds(config['seed'])
    setup = build_problem_setup(config)
    model = build_model(setup)
    residual = build_residual(setup, device)
    diffusion = DenoisingDiffusion(config['diff_steps'], device, config['snr_gamma'])
    hold_out(setup, config)
    train_loader = make_loader(setup.train_dataset, setup.train_batch_size, config['num_workers'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    steps_per_epoch = math.ceil(len(setup.train_dataset) / setup.train_batch_size)
    if own.bench_steps:
        steps_per_epoch = min(steps_per_epoch, own.bench_steps)
    gpu = torch.cuda.get_device_name(0)
    print(f'{own.bench_label} {config["gov_eqs"]}: {len(setup.train_dataset)} fields, '
          f'{steps_per_epoch} steps per epoch, {own.bench_epochs} epochs on {gpu}', flush=True)

    torch.cuda.reset_peak_memory_stats()
    step_ms, epoch_s = [], []
    for epoch in range(own.bench_epochs):
        events = []
        torch.cuda.synchronize()
        epoch_start = time.perf_counter()
        for _ in range(steps_per_epoch):
            batch = next(train_loader).to(device)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            train_step(model, diffusion, residual, batch, config, optimizer)
            end.record()
            events.append((start, end))
        torch.cuda.synchronize()
        epoch_s.append(time.perf_counter() - epoch_start)
        step_ms.append(float(np.mean([s.elapsed_time(e) for s, e in events])))
        print(f'  epoch {epoch + 1}: {epoch_s[-1]:.1f} s, {step_ms[-1]:.1f} ms per step',
              flush=True)

    kept_ms, kept_s = np.array(step_ms[1:]), np.array(epoch_s[1:])
    row = {
        'label': own.bench_label, 'equation': config['gov_eqs'],
        'c': config['c_residual'], 'linearize': config.get('physics_linearize'),
        'normalize': config.get('physics_normalize'),
        'time_weight': config.get('physics_time_weight'), 'batch': setup.train_batch_size,
        'steps_per_epoch': steps_per_epoch, 'epochs_kept': len(kept_ms),
        'step_ms_mean': kept_ms.mean(), 'step_ms_std': kept_ms.std(ddof=1),
        'epoch_s_mean': kept_s.mean(), 'epoch_s_std': kept_s.std(ddof=1),
        'peak_mem_gb': torch.cuda.max_memory_allocated() / 2**30,
        'gpu': gpu, 'epoch_s_all': ' '.join(f'{s:.2f}' for s in epoch_s),
        'step_ms_all': ' '.join(f'{s:.2f}' for s in step_ms),
    }
    out = Path(own.bench_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    new = not out.exists()
    with out.open('a', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)
    print(f'{own.bench_label} {config["gov_eqs"]}: {row["step_ms_mean"]:.1f} +- '
          f'{row["step_ms_std"]:.1f} ms per step, {row["epoch_s_mean"]:.1f} s per epoch, '
          f'peak {row["peak_mem_gb"]:.2f} GB -> {out}', flush=True)


if __name__ == '__main__':
    main()
