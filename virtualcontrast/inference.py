from __future__ import annotations

import itertools

import numpy as np
import torch


def _starts(length: int, patch: int, overlap: float) -> list[int]:
    if length <= patch:
        return [0]
    step = max(1, int(round(patch * (1 - overlap))))
    values = list(range(0, length - patch + 1, step))
    if values[-1] != length - patch:
        values.append(length - patch)
    return values


def gaussian_importance(size: tuple[int, int, int]) -> np.ndarray:
    axes = []
    for length in size:
        coordinate = np.arange(length, dtype=np.float32) - (length - 1) / 2
        axes.append(np.exp(-0.5 * (coordinate / max(length / 8, 1)) ** 2))
    weight = axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :]
    return np.maximum(weight / weight.max(), 1e-3).astype(np.float32)


@torch.inference_mode()
def sliding_window_predict(
    model: torch.nn.Module,
    pre: np.ndarray,
    valid: np.ndarray,
    device: torch.device,
    patch_size: tuple[int, int, int],
    overlap: float = 0.5,
    batch_size: int = 1,
) -> np.ndarray:
    original_shape = pre.shape
    padding = [(0, max(0, patch - length)) for length, patch in zip(pre.shape, patch_size, strict=True)]
    padded = np.pad(pre, padding)
    locations = list(itertools.product(*[
        _starts(length, patch, overlap) for length, patch in zip(padded.shape, patch_size, strict=True)
    ]))
    prediction_sum = np.zeros(padded.shape, dtype=np.float32)
    weight_sum = np.zeros(padded.shape, dtype=np.float32)
    importance = gaussian_importance(patch_size)
    model.eval()
    for offset in range(0, len(locations), batch_size):
        members = locations[offset:offset + batch_size]
        patches = np.stack([
            padded[d:d + patch_size[0], h:h + patch_size[1], w:w + patch_size[2]]
            for d, h, w in members
        ])[:, None]
        tensor = torch.from_numpy(patches).to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            outputs = model(tensor)[:, 0]
        for (d, h, w), output in zip(members, outputs.float().cpu().numpy(), strict=True):
            region = (slice(d, d + patch_size[0]), slice(h, h + patch_size[1]), slice(w, w + patch_size[2]))
            prediction_sum[region] += output * importance
            weight_sum[region] += importance
    prediction = prediction_sum / np.maximum(weight_sum, 1e-8)
    prediction = prediction[tuple(slice(0, length) for length in original_shape)]
    return prediction * valid



def generate(model, rows, cache, device, patch=(64,256,256), batch_size=8, checkpoint_hash=None):
    from .data import load_case, cache_path
    from .utils import write_json, sha256
    from pathlib import Path
    import json
    marker = Path(cache) / 'vce' / 'provenance.json'
    identity = dict(generator_sha256=checkpoint_hash, patch=list(patch), overlap=.5,
                    precision='bf16' if device.type == 'cuda' else 'fp32', batch_size=batch_size)
    if marker.exists():
        if json.loads(marker.read_text()) != identity:
            raise ValueError('VCE directory belongs to another generator/configuration')
    elif marker.parent.exists() and any(marker.parent.glob('*.npy')):
        raise ValueError('Existing VCE files lack directory provenance')
    write_json(marker, identity)
    for i, row in enumerate(rows):
        dst = cache_path(cache, row, 'vce')
        c = load_case(cache, row, paired=False)
        nc_hash = sha256(cache_path(cache, row, 'nc'))
        if dst.exists() and dst.with_suffix('.json').exists():
            previous = json.loads(dst.with_suffix('.json').read_text())
            existing = np.load(dst, mmap_mode='r')
            if (previous.get('generator_sha256') != checkpoint_hash or previous.get('nc_sha256') != nc_hash or existing.shape != c['pre'].shape
                    or existing.dtype != np.float16 or not np.isfinite(existing).all()):
                raise ValueError(f'Invalid existing VCE cache: {dst}')
            continue
        delta = sliding_window_predict(model, c['pre'], c['valid'], device, tuple(patch), .5, batch_size)
        synthetic = (np.clip(c['pre'] + delta, 0, 1) * c['valid']).astype(np.float16)
        if not np.isfinite(synthetic).all():
            raise ValueError('Nonfinite synthetic image')
        dst.parent.mkdir(parents=True, exist_ok=True)
        temporary = dst.with_suffix('.tmp')
        with temporary.open('wb') as handle:
            np.save(handle, synthetic)
        temporary.replace(dst)
        write_json(dst.with_suffix('.json'), dict(c['metadata'], generator_sha256=checkpoint_hash, nc_sha256=nc_hash))
        print(f'Generated {i+1}/{len(rows)}', flush=True)


@torch.inference_mode()
def predict(model, loader, device):
    records = []
    model.eval()
    for volumes, labels, rows in loader:
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
            logits = model(volumes).float().cpu().tolist()
        records.extend(dict(sample_id=r['sample_id'], patient=r['patient'], dataset=r['dataset'],
                            label=int(y), logit=z) for r, y, z in zip(rows, labels, logits))
    return records


def main():
    import argparse
    import json
    from pathlib import Path
    from torch.utils.data import DataLoader
    from .data import read_manifest, BreastDataset, collate
    from .generator import Generator
    from .classifier import Classifier
    from .utils import load_weights, sha256, write_csv
    p = argparse.ArgumentParser(description='Generate VCE or predict all four study logit columns')
    p.add_argument('mode', choices=['generate', 'predict'])
    p.add_argument('--manifest', type=Path, default=Path('reproducibility/cohort.csv'))
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--weights', type=Path, default=Path('weights'))
    p.add_argument('--config', type=Path, default=Path('configs/default.json'))
    p.add_argument('--output', type=Path, default=Path('outputs/predictions'))
    p.add_argument('--split', choices=['train', 'val', 'test', 'all'], default='test')
    p.add_argument('--device', default='cuda')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--chunk-size', type=int, default=160)
    args = p.parse_args()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    rows = [r for r in read_manifest(args.manifest) if args.split == 'all' or r['split'] == args.split]
    if not rows:
        raise ValueError('Empty split')
    if args.mode == 'generate':
        state = load_weights(args.weights / 'generator.pt', 'generator')
        model = Generator().to(device)
        model.load_state_dict(state['model'], strict=True)
        patch = json.loads(args.config.read_text())['generator']['patch']
        generate(model, rows, args.cache, device, patch, args.batch_size, sha256(args.weights / 'generator.pt'))
    else:
        for seed in (2026,2027,2028):
            result = {r['sample_id']: dict(sample_id=r['sample_id'], patient=r['patient'], dataset=r['dataset'], label=int(r['label'])) for r in rows}
            for name, modality, column in [('nc', 'nc', 'nc'), ('ce', 'ce', 'ce'), ('ce', 'vce', 'vce_ce'), ('vce', 'vce', 'vce')]:
                state = load_weights(args.weights / f'{name}_{seed}.pt', 'classifier')
                if state['modality'] != name or state['seed'] != seed:
                    raise ValueError('Wrong classifier identity')
                model = Classifier(pretrained=False, chunk_size=args.chunk_size).to(device)
                model.load_state_dict(state['model'], strict=True)
                loader = DataLoader(BreastDataset(rows, args.cache, modality), batch_size=args.batch_size, collate_fn=collate)
                for r in predict(model, loader, device):
                    result[r['sample_id']][column] = r['logit']
                del model, state
            write_csv(args.output / f'seed_{seed}.csv', list(result.values()))


if __name__ == '__main__':
    main()
