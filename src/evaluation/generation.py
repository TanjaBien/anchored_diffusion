"""Draw a trained run's evaluation samples and compute the paper's metrics.

    python -m src.evaluation.generation trained_models/burgers/<run>
    python -m src.evaluation.generation trained_models/burgers --all

Writes <run>/evaluation/samples.npy and <run>/evaluation/report.json. The report holds the table
columns as paper/* (src/evaluation/paper_metrics.py) and a summary of the samples' residual. The
sampler is seeded with the run's --seed, so a run evaluates to the same samples every time.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.config import build_parser, device, finalize_config, fix_seeds, latest_checkpoint, parser_keys, read_yaml
from src.datasets import build_problem_setup
from src.diffusion_process import DenoisingDiffusion, load_model
from src.evaluation.paper_metrics import paper_metrics
from src.models import build_model
from src.residuals import build_residual

EVALUATION_DIR = 'evaluation'


def load_run_config(run_dir):
    """The run's saved configuration, on top of today's defaults (older runs lack newer options)."""
    parser = build_parser()
    config = vars(parser.parse_args([]))
    keys = parser_keys(parser)
    config.update({k: v for k, v in read_yaml(Path(run_dir) / 'model' / 'model.yaml').items() if k in keys})
    return finalize_config(config)


def generate(model, diffusion, setup, num_samples, batch_size):
    batches = []
    for start in tqdm(range(0, num_samples, batch_size), desc='sampling'):
        shape = (min(batch_size, num_samples - start), setup.channels, *setup.field_shape)
        batches.append(diffusion.p_sample_loop(model, shape).cpu())
    return torch.cat(batches)


def reference_fields(dataset, count):
    return torch.stack([torch.as_tensor(dataset[i]) for i in range(min(count, len(dataset)))]).float()


def evaluate_run(run_dir, checkpoint=None, num_samples=None, batch_size=None, data_dir=None, force=False):
    run_dir = Path(run_dir)
    report_path = run_dir / EVALUATION_DIR / 'report.json'
    if report_path.exists() and not force:
        print(f'{run_dir.name}: already evaluated ({report_path}); --force to redo')
        return json.loads(report_path.read_text())

    config = load_run_config(run_dir)
    if data_dir is not None:
        config['data_dir'] = data_dir
    num_samples = num_samples or config['generation_num_samples']
    batch_size = batch_size or config['generation_eval_batch_size']
    checkpoint = latest_checkpoint(checkpoint or run_dir)

    fix_seeds(config['seed'])
    setup = build_problem_setup(config)
    model = build_model(setup)
    load_model(checkpoint, model)
    model.eval()
    residual = build_residual(setup, device)
    diffusion = DenoisingDiffusion(config['diff_steps'], device, config['snr_gamma'])

    samples = generate(model, diffusion, setup, num_samples, batch_size)
    reference = reference_fields(setup.valid_dataset, num_samples)
    with torch.no_grad():
        per_sample = torch.cat([
            residual.calculate_residual(samples[i:i + batch_size].to(device)).abs()
            .reshape(min(batch_size, len(samples) - i), -1).mean(dim=1).cpu()
            for i in range(0, len(samples), batch_size)])

    report = {
        'dataset': setup.gov_eqs,
        'method': config['method'],
        'c_residual': config['active_c_residual'],
        'checkpoint': str(checkpoint),
        'num_samples': len(samples),
        'reference_count': len(reference),
        'residual/mean_abs': float(per_sample.mean()),
        'residual/median_abs': float(per_sample.median()),
    }
    report.update(paper_metrics(setup.gov_eqs, samples.numpy(), reference.numpy(), residual,
                                profile=setup.toy_profile))

    report_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(report_path.parent / 'samples.npy', samples.numpy())
    report_path.write_text(json.dumps(report, indent=2))
    columns = '  '.join(f'{k[6:]} {v:.4g}' for k, v in report.items() if k.startswith('paper/'))
    print(f'{run_dir.name}: {columns}\nwrote {report_path}')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('path', help='A run directory, or with --all a directory of runs.')
    parser.add_argument('--all', action='store_true', help='Evaluate every run directory under path.')
    parser.add_argument('--checkpoint', default=None, help='A checkpoint file; default: the run\'s newest.')
    parser.add_argument('--num_samples', type=int, default=None, help='Default: the run\'s generation_num_samples.')
    parser.add_argument('--eval_batch_size', type=int, default=None)
    parser.add_argument('--data_dir', default=None, help='Data root, if not the run\'s.')
    parser.add_argument('--force', action='store_true', help='Re-evaluate runs that already have a report.')
    args = parser.parse_args()

    runs = (sorted(p.parent.parent for p in Path(args.path).glob('*/model/model.yaml')) if args.all
            else [Path(args.path)])
    for run_dir in runs:
        evaluate_run(run_dir, args.checkpoint, args.num_samples, args.eval_batch_size, args.data_dir, args.force)


if __name__ == '__main__':
    main()
