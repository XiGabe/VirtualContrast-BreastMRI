"""Recompute all seven Table 1 settings from aligned raw logits."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

from .utils import write_csv

SEEDS = (2026, 2027, 2028)
COLUMNS = ('nc', 'vce_ce', 'vce', 'ce')
STRATEGIES = {
    'NC': ('nc',), 'VCE (CE classifier)': ('vce_ce',),
    'VCE (VCE classifier)': ('vce',), 'CE': ('ce',),
    'NC + VCE (CE classifier)': ('nc', 'vce_ce'),
    'NC + VCE (VCE classifier)': ('nc', 'vce'), 'NC + CE': ('nc', 'ce'),
}


def load_predictions(directory):
    frames = [pd.read_csv(Path(directory) / f'seed_{seed}.csv').sort_values('sample_id').reset_index(drop=True)
              for seed in SEEDS]
    fields = ['sample_id', 'patient', 'dataset', 'label']
    for frame in frames:
        if not frame.sample_id.is_unique or set(frame.label) != {0, 1}:
            raise ValueError('Predictions must have unique samples and both classes')
        if not np.isfinite(frame[list(COLUMNS)].to_numpy()).all():
            raise ValueError('Nonfinite logits')
        pd.testing.assert_frame_equal(frame[fields], frames[0][fields])
    return frames


def evaluate(directory, output):
    frames = load_predictions(directory)
    sources = sorted(frames[0].dataset.unique())
    for source in sources:
        if set(frames[0].loc[frames[0].dataset == source, 'label']) != {0, 1}:
            raise ValueError(f'AUROC requires both labels in dataset {source}')
    records, repeats = [], []
    for name, columns in STRATEGIES.items():
        row = dict(strategy=name)
        for source in ['overall', *sources]:
            values = []
            for seed, frame in zip(SEEDS, frames):
                d = frame if source == 'overall' else frame[frame.dataset == source]
                z = d[list(columns)].mean(axis=1)
                values.append([roc_auc_score(d.label, z), average_precision_score(d.label, z)])
                repeats.append(dict(strategy=name, source=source, seed=seed,
                                    auroc=values[-1][0], auprc=values[-1][1]))
            values = np.array(values)
            for k, metric in enumerate(('auroc', 'auprc')):
                row[f'{source}_{metric}_mean'] = values[:, k].mean()
                row[f'{source}_{metric}_sd'] = values[:, k].std(ddof=1)
        records.append(row)
    output = Path(output)
    write_csv(output / 'table1.csv', records)
    write_csv(output / 'per_repeat.csv', repeats)
    headers = ['Strategy', 'AUROC', 'AUPRC', *[f'{source} AUROC' for source in sources]]
    lines = ['| ' + ' | '.join(headers) + ' |', '|' + '---|' * len(headers)]
    for r in records:
        keys = ['overall_auroc', 'overall_auprc', *[f'{source}_auroc' for source in sources]]
        values = [f'{r[k+"_mean"]:.3f} ± {r[k+"_sd"]:.3f}' for k in keys]
        lines.append('| ' + ' | '.join([r['strategy'], *values]) + ' |')
    (output / 'table1.md').write_text('\n'.join(lines) + '\n')
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions', type=Path, default=Path('reproducibility/predictions'))
    p.add_argument('--output', type=Path, default=Path('outputs/results'))
    args = p.parse_args()
    evaluate(args.predictions, args.output)


if __name__ == '__main__':
    main()
