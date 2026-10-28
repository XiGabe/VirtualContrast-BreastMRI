"""Patient-level confidence and random routing."""
import argparse
from pathlib import Path
import numpy as np
from scipy.special import expit
from .evaluate import load_predictions
from .utils import write_csv, write_json

def batch_metrics(y, scores):
    """Exact tie-aware ROC AUC and average precision for score rows."""
    order = np.argsort(-scores, axis=1, kind='stable')
    sorted_scores = np.take_along_axis(scores, order, axis=1)
    labels = y[order]
    tp = np.cumsum(labels, axis=1)
    fp = np.cumsum(1 - labels, axis=1)
    ends = np.concatenate((sorted_scores[:, :-1] != sorted_scores[:, 1:],
                           np.ones((len(scores), 1), dtype=bool)), axis=1)
    # Previous tie-group cumulative counts, carried forward to each endpoint.
    previous_tp = np.maximum.accumulate(np.where(ends, tp, 0), axis=1)
    previous_fp = np.maximum.accumulate(np.where(ends, fp, 0), axis=1)
    previous_tp = np.pad(previous_tp[:, :-1], ((0, 0), (1, 0)))
    previous_fp = np.pad(previous_fp[:, :-1], ((0, 0), (1, 0)))
    positives, negatives = y.sum(), (1-y).sum()
    auc = (np.where(ends, (fp-previous_fp)*(tp+previous_tp)*.5, 0).sum(1)
           / (positives * negatives))
    ap = (np.where(ends, (tp-previous_tp)*tp/(tp+fp), 0).sum(1) / positives)
    return np.stack((auc, ap), axis=-1)


def simulate(frames, permutations=2000, seed=2026):
    meta = frames[0]
    patients, owner = np.unique(meta.patient.to_numpy(), return_inverse=True)
    y = meta.label.to_numpy()
    a = np.stack([d.nc.to_numpy() for d in frames])
    fusion = np.stack([(d.nc.to_numpy() + d.vce_ce.to_numpy()) / 2 for d in frames])
    real = np.stack([d.ce.to_numpy() for d in frames])
    breast_confidence = np.maximum(expit(a), 1-expit(a)).mean(0)
    confidence = np.array([breast_confidence[owner == i].min() for i in range(len(patients))])
    ranks = np.argsort(np.argsort(-confidence, kind='stable'))
    rng = np.random.default_rng(seed)
    random_ranks = np.argsort(np.stack([rng.permutation(len(patients)) for _ in range(permutations)]), axis=1)
    counts = np.floor(np.arange(101) * len(patients) / 100 + .5).astype(int)
    rows = []
    for level, count in enumerate(counts):
        chosen = ranks[owner] < count
        selective = np.mean([batch_metrics(y, np.where(chosen, fusion[k], real[k])[None])[0] for k in range(3)], axis=0)
        random = np.zeros((permutations, 2))
        chosen_random = random_ranks[:, owner] < count
        for k in range(3):
            random += batch_metrics(y, np.where(chosen_random, fusion[k], real[k])) / 3
        row = dict(requested_percent=level, patients=int(count), actual_percent=100*count/len(patients))
        for k, metric in enumerate(('auroc', 'auprc')):
            row.update({f'confidence_{metric}': selective[k], f'random_{metric}': random[:, k].mean(),
                        f'random_{metric}_p025': np.quantile(random[:, k], .025),
                        f'random_{metric}_p975': np.quantile(random[:, k], .975)})
        rows.append(row)
    return rows


def plot(rows, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(5, 4.5), sharex=True)
    x = [r['actual_percent'] for r in rows]
    for ax, metric in zip(axes, ('auroc', 'auprc')):
        ax.fill_between(x, [r[f'random_{metric}_p025'] for r in rows],
                        [r[f'random_{metric}_p975'] for r in rows], color='gray', alpha=.2,
                        label='95% randomization interval')
        ax.plot(x, [r[f'confidence_{metric}'] for r in rows], label='Confidence-based')
        ax.plot(x, [r[f'random_{metric}'] for r in rows], '--', color='gray', label='Random')
        ax.set_ylabel(metric.upper())
        ax.spines[['top', 'right']].set_visible(False)
    axes[-1].set(xlabel='Patients assigned to NC + virtual CE (%)', xlim=(0, 100))
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    for extension in ('pdf', 'png'):
        fig.savefig(Path(output) / f'figure2.{extension}', dpi=200)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions', type=Path, default=Path('reproducibility/predictions'))
    p.add_argument('--output', type=Path, default=Path('outputs/results'))
    p.add_argument('--config', type=Path, default=Path('configs/default.json'))
    args = p.parse_args()
    import json
    config = json.loads(args.config.read_text())['routing']
    rows = simulate(load_predictions(args.predictions), config['permutations'], config['seed'])
    write_csv(args.output / 'routing.csv', rows)
    write_json(args.output / 'routing_protocol.json', dict(**config,
        interval='Pointwise 2.5th–97.5th percentiles across random patient assignments',
        aggregation='Metrics averaged after independent evaluation of three classifier repeats'))
    plot(rows, args.output)


if __name__ == '__main__':
    main()
