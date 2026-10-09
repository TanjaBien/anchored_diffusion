"""Check the datasets before training: their size and value range, the residual of the ground-truth
fields under the operator the models are trained with, for Darcy the error of the stored pressure
against a fresh solve of the PDE, and a figure of a few fields with their residual.

    python -m src.datasets.validation                       # every dataset
    python -m src.datasets.validation burgers darcy --count 200

Writes results/data_validation/report.json and results/data_validation/<dataset>.png.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.config import build_parser, device, finalize_config, read_yaml
from src.datasets import build_problem_setup
from src.evaluation.paper_metrics import PressureSolver, mean_abs_residual
from src.residuals import build_residual

DATASETS = ('advection', 'burgers', 'darcy', 'toy')


def dataset_config(dataset, data_dir):
    config = vars(build_parser().parse_args([]))
    config.update(read_yaml(f'config/{dataset}.yaml'))
    config['data_dir'] = data_dir
    return finalize_config(config)


def first(dataset, count):
    return torch.stack([torch.as_tensor(dataset[i]) for i in range(min(count, len(dataset)))]).float()


def check(dataset, count, data_dir, out_dir):
    setup = build_problem_setup(dataset_config(dataset, data_dir))
    residual = build_residual(setup, device)
    report = {'train_size': len(setup.train_dataset), 'valid_size': len(setup.valid_dataset)}
    for split, data in (('train', setup.train_dataset), ('valid', setup.valid_dataset)):
        fields = first(data, count)
        report[f'{split}/field_shape'] = list(fields.shape[1:])
        report.update({f'{split}/min': float(fields.min()), f'{split}/max': float(fields.max()),
                       f'{split}/mean': float(fields.mean()), f'{split}/std': float(fields.std())})
        if dataset == 'toy':
            values = residual.calculate_residual(fields).abs().numpy()
        else:
            values = mean_abs_residual(fields.numpy(), residual, dataset)
        report[f'{split}/residual_mean_abs'] = float(values.mean())
        report[f'{split}/residual_max_abs'] = float(values.max())

    fields = first(setup.valid_dataset, count)
    if dataset == 'darcy':
        errors = PressureSolver(fields.shape[-1]).solution_error(fields[:min(count, 100)].numpy())
        report['valid/solution_error_median'] = float(np.nanmedian(errors))

    import matplotlib.pyplot as plt
    from src.visualization.fields import sample_figure
    with torch.no_grad():
        values = residual.calculate_residual(fields[:4].to(device))
    figure = sample_figure(fields if dataset == 'toy' else fields[:4], values, setup.field_shape,
                           title=f'{dataset}: validation fields')
    figure.savefig(out_dir / f'{dataset}.png', dpi=110, bbox_inches='tight')
    plt.close(figure)

    summary = '  '.join(f'{k} {v:.3g}' for k, v in report.items()
                        if isinstance(v, float) and ('residual' in k or 'solution' in k))
    print(f'{dataset}: {report["train_size"]} train / {report["valid_size"]} valid fields  {summary}')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('datasets', nargs='*', help=f'Any of {", ".join(DATASETS)}; default: all.')
    parser.add_argument('--count', type=int, default=1000, help='Fields per split to check.')
    parser.add_argument('--data_dir', default='./data')
    parser.add_argument('--out_dir', default='results/data_validation')
    args = parser.parse_args()
    unknown = set(args.datasets) - set(DATASETS)
    if unknown:
        parser.error(f'unknown dataset(s) {sorted(unknown)}; use {", ".join(DATASETS)}')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {dataset: check(dataset, args.count, args.data_dir, out_dir) for dataset in args.datasets or DATASETS}
    (out_dir / 'report.json').write_text(json.dumps(report, indent=2))
    print(f'wrote {out_dir / "report.json"} and the figures')


if __name__ == '__main__':
    main()
