"""Collect the evaluated runs of one dataset into the paper's table: one row per method and weight.

    python -m src.evaluation.tables burgers
    python -m src.evaluation.tables toy --output_root ./trained_models --profile graded --dim 10

Reads every run under <output_root>/<dataset>/ that has a generation report, takes the method and
weight from its model/model.yaml and the columns from the report's paper/* metrics, and writes
results/table_<dataset>.csv next to printing the table. If a method and weight was run more than
once, the newest run is used.
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

COLUMNS = {
    'toy': ['residual', 'w1', 'sigma_ratio', 'off_plane'],
    'burgers': ['residual', 'mmd', 'sigma_ratio', 'std_u_ratio'],
    'advection': ['residual', 'mmd', 'std_u_ratio'],
    'darcy': ['residual', 'solution_error', 'mmd', 'sigma_ratio', 'std_p_ratio', 'std_K_ratio'],
}
METHOD_ORDER = ['dm', 'pidm', 'lin', 'norm', 'anchored']
REPORT = Path('evaluation') / 'report.json'


def collect(dataset, output_root, profile=None, dim=None):
    rows = []
    for run in sorted((Path(output_root) / dataset).glob('*')):
        report_path, config_path = run / REPORT, run / 'model' / 'model.yaml'
        if not report_path.exists() or not config_path.exists():
            continue
        config = yaml.safe_load(config_path.read_text())
        if dataset == 'toy' and ((profile and config.get('toy_profile') != profile)
                                 or (dim and int(config.get('toy_dim', 0)) != dim)):
            continue
        report = json.loads(report_path.read_text())
        row = {'method': config['method'],
               'c': 0.0 if config['method'] == 'dm' else float(config['c_residual']),
               'run': run.name, 'mtime': report_path.stat().st_mtime}
        row.update({column: report.get(f'paper/{column}') for column in COLUMNS[dataset]})
        rows.append(row)
    if not rows:
        raise SystemExit(f'no evaluated runs under {Path(output_root) / dataset}')
    frame = pd.DataFrame(rows).sort_values('mtime').drop_duplicates(['method', 'c'], keep='last')
    frame['order'] = frame['method'].map({m: i for i, m in enumerate(METHOD_ORDER)})
    return frame.sort_values(['order', 'c']).drop(columns=['order', 'mtime']).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('dataset', choices=sorted(COLUMNS))
    parser.add_argument('--output_root', default='./trained_models')
    parser.add_argument('--profile', default=None, help='toy only: keep runs of this stiffness profile')
    parser.add_argument('--dim', type=int, default=None, help='toy only: keep runs of this dimension')
    parser.add_argument('--out', default=None, help='CSV path; default results/table_<dataset>.csv')
    args = parser.parse_args()

    frame = collect(args.dataset, args.output_root, args.profile, args.dim)
    out = Path(args.out or f'results/table_{args.dataset}.csv')
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    with pd.option_context('display.float_format', '{:.3g}'.format, 'display.width', 160):
        print(frame.drop(columns=['run']).to_string(index=False))
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
