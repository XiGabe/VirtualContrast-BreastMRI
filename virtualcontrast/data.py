"""Fixed cohort, paired caches, and the two study training datasets."""
import csv
import hashlib
import json
import random
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from scipy.ndimage import distance_transform_cdt, gaussian_filter
from torch.utils.data import Dataset, Sampler


def read_manifest(path):
    with open(path) as handle:
        rows = list(csv.DictReader(handle))
    seen, patients, groups = set(), {}, {}
    for r in rows:
        if r['sample_id'] in seen or r['split'] not in ('train', 'val', 'test') or r['label'] not in ('0', '1'):
            raise ValueError('Duplicate sample or invalid split/label')
        seen.add(r['sample_id'])
        for field, table in [('patient', patients), ('group', groups)]:
            if table.setdefault(r[field], r['split']) != r['split']:
                raise ValueError(f'{field} crosses splits: {r[field]}')
    if not rows:
        raise ValueError('Empty manifest')
    return rows


def cache_path(cache, row, modality):
    return Path(cache) / modality / (row['sample_id'] + '.npy')


def validate_cache(rows, cache, modalities):
    """Validate the requested cached representations."""
    digest = hashlib.sha256()
    generators = set()
    for row in rows:
        meta = json.loads(cache_path(cache, row, 'nc').with_suffix('.json').read_text())
        for modality in modalities:
            path = cache_path(cache, row, modality)
            array = np.load(path, mmap_mode='r')
            if array.ndim != 3 or (modality != 'breast' and array.dtype != np.float16):
                raise ValueError(f'Invalid cached array: {path}')
            if 'shape' in meta and list(array.shape) != meta['shape']:
                raise ValueError(f'Cache geometry mismatch: {path}')
            if modality == 'vce':
                generated = json.loads(path.with_suffix('.json').read_text())
                if not generated.get('generator_sha256'):
                    raise ValueError('VCE cache lacks generator identity')
                generators.add(generated['generator_sha256'])
            digest.update(json.dumps([row['sample_id'], modality, list(array.shape), meta], sort_keys=True).encode())
    if len(generators) > 1:
        raise ValueError('VCE cache mixes generator checkpoints')
    digest.update(json.dumps(sorted(generators)).encode())
    return digest.hexdigest()


def valid_mask(shape, meta):
    valid = np.zeros(shape, dtype=bool)
    h, w = meta['pad_lower']
    nh, nw = meta['resampled_shape_lr_ap']
    valid[:, h:h+nh, w:w+nw] = True
    return valid


def load_case(cache, row, paired=True):
    pre = np.load(cache_path(cache, row, 'nc')).astype(np.float32)
    meta = json.loads(cache_path(cache, row, 'nc').with_suffix('.json').read_text())
    valid = valid_mask(pre.shape, meta)
    result = dict(pre=pre, valid=valid, metadata=meta)
    if paired:
        post = np.load(cache_path(cache, row, 'ce')).astype(np.float32)
        breast = np.load(cache_path(cache, row, 'breast')).astype(bool) & valid
        if pre.shape != post.shape or pre.shape != breast.shape or not breast.any():
            raise ValueError(f'Invalid pair: {row["sample_id"]}')
        spacing = meta['effective_spacing']
        result.update(post=post, breast=breast, spacing_dhw=np.array([spacing[2], *spacing[:2]], dtype=np.float32))
    if any(not np.isfinite(v).all() for v in result.values() if isinstance(v, np.ndarray)):
        raise ValueError('Nonfinite image')
    return result


class BreastDataset(Dataset):
    def __init__(self, rows, cache, modality, augment=False):
        self.rows, self.cache, self.modality, self.augment = rows, cache, modality, augment
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, i):
        row = self.rows[i]
        x = torch.from_numpy(np.load(cache_path(self.cache, row, self.modality)).astype(np.float32))
        if x.ndim != 3 or not torch.isfinite(x).all():
            raise ValueError(f'Invalid image: {row["sample_id"]}')
        if self.augment:
            if torch.rand(()) < .5:
                x = x.flip(1)
            foreground = x > 0
            x = (x * torch.empty(()).uniform_(.9, 1.1)
                 + torch.randn_like(x) * torch.empty(()).uniform_(0, .015) * foreground).clamp(0, 1)
        return x, float(row['label']), row


def collate(batch):
    xs, ys, rows = zip(*batch)
    return list(xs), torch.tensor(ys), list(rows)


class BalancedSampler(Sampler):
    def __init__(self, rows, seed):
        groups = Counter((r['dataset'], r['label']) for r in rows)
        self.weights = torch.tensor([1 / groups[(r['dataset'], r['label'])] for r in rows], dtype=torch.double)
        self.seed, self.epoch = seed, 0
    def __len__(self):
        return len(self.weights)
    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return iter(torch.multinomial(self.weights, len(self), True, generator=generator).tolist())

def make_guidance(
    pre: np.ndarray,
    post: np.ndarray,
    breast: np.ndarray,
    spacing_dhw: np.ndarray,
    boundary_ramp_mm: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    positive = np.maximum(post - pre, 0) * breast
    values = positive[breast]
    scale = float(np.percentile(values, 95)) if values.size else 0.0
    guidance = np.clip(positive / max(scale, 1e-6), 0, 1) if scale > 0 else np.zeros_like(positive)
    # The problematic registration rim is the in-plane breast/chest boundary. A
    # slice-wise chessboard transform is deterministic and much cheaper than a
    # full-volume Euclidean transform while retaining physical-mm scaling.
    distance = np.stack([
        distance_transform_cdt(section, metric="chessboard") for section in breast
    ]).astype(np.float32)
    distance = np.maximum(distance, 0) * float(min(spacing_dhw[1:]))
    boundary = 0.25 + 0.75 * np.clip(distance / boundary_ramp_mm, 0, 1)
    boundary *= breast
    return guidance.astype(np.float32), boundary.astype(np.float32)

def crop_with_padding(array: np.ndarray, start: tuple[int, int, int], size: tuple[int, int, int]) -> np.ndarray:
    output = np.zeros(size, dtype=array.dtype)
    source, target = [], []
    for offset, length, available in zip(start, size, array.shape, strict=True):
        source_start = max(offset, 0)
        source_stop = min(offset + length, available)
        target_start = source_start - offset
        source.append(slice(source_start, source_stop))
        target.append(slice(target_start, target_start + max(0, source_stop - source_start)))
    output[tuple(target)] = array[tuple(source)]
    return output

def _axis_start(length: int, patch: int, lower: int, upper: int, rng: random.Random) -> int:
    if length <= patch:
        return -(patch - length) // 2
    maximum = length - patch
    # Fully contain the breast extent whenever it fits inside the patch.
    contain_low = max(0, upper - patch)
    contain_high = min(lower, maximum)
    if contain_low <= contain_high:
        return rng.randint(contain_low, contain_high)
    # Otherwise sample a contiguous window through the breast extent.
    inside_low = max(0, lower)
    inside_high = min(maximum, upper - patch)
    if inside_low <= inside_high:
        return rng.randint(inside_low, inside_high)
    return min(max((lower + upper - patch) // 2, 0), maximum)

def sample_start(breast: np.ndarray, patch_size: tuple[int, int, int], rng: random.Random) -> tuple[int, int, int]:
    coordinates = np.argwhere(breast)
    lower = coordinates.min(0)
    upper = coordinates.max(0) + 1
    return tuple(
        _axis_start(length, patch, int(lo), int(hi), rng)
        for length, patch, lo, hi in zip(breast.shape, patch_size, lower, upper, strict=True)
    )
def make_soft_enhancement(pre, post, breast, spacing_dhw, low, high, sigma_mm=1.):
    """Fixed-scale enhancement ramp from smoothed signed subtraction.
    """
    if not 0 < low < high or sigma_mm <= 0:
        raise ValueError('Require 0 < low < high and positive smoothing')
    if pre.shape != post.shape or pre.shape != breast.shape:
        raise ValueError('Unaligned paired arrays')
    if not np.isfinite(pre).all() or not np.isfinite(post).all():
        raise ValueError('Nonfinite paired image')
    spacing=np.asarray(spacing_dhw,dtype=np.float32)
    if spacing.shape != (3,) or np.any(spacing <= 0):
        raise ValueError('Invalid spacing')
    b=breast.astype(np.float32)
    sigma=sigma_mm/spacing
    den=gaussian_filter(b,sigma,mode='constant')
    smooth=gaussian_filter((post.astype(np.float32)-pre.astype(np.float32))*b,sigma,mode='constant')/np.maximum(den,1e-6)
    smooth*=b
    q=np.clip((smooth-low)/(high-low),0,1)*b
    return q.astype(np.float32),smooth.astype(np.float32)

def enhancement_start(q, boundary, patch, rng: random.Random):
    """Randomly include an enhancement anchor, without putting it at center.

    None means no eligible enhancement; caller must fall back to breast sampling.
    """
    weights=q*(boundary>=.5)
    indices=np.flatnonzero(weights>0)
    if not len(indices): return None
    cumulative=np.cumsum(weights.ravel()[indices],dtype=np.float64)
    index=indices[min(np.searchsorted(cumulative,rng.random()*cumulative[-1]),len(indices)-1)]
    anchor=np.unravel_index(index,q.shape)
    start=[]
    for length,size,a in zip(q.shape,patch,anchor,strict=True):
        if length<=size: start.append(-(size-length)//2)
        else: start.append(rng.randint(max(0,int(a)-size+1),min(int(a),length-size)))
    return tuple(start)

class GeneratorDataset(Dataset):
    def __init__(self, rows, cache, patch=(64,256,256), seed=2026):
        if any(r['split'] != 'train' for r in rows):
            raise ValueError('Generator training must use train only')
        self.rows, self.cache, self.patch, self.seed = rows, cache, tuple(patch), seed
        self.epoch = 0
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, i):
        c = load_case(self.cache, self.rows[i])
        c['guidance'], c['boundary'] = make_guidance(c['pre'], c['post'], c['breast'], c['spacing_dhw'])
        low = .046331580728292465
        c['target'], _ = make_soft_enhancement(c['pre'], c['post'], c['breast'], c['spacing_dhw'], low, 2*low)
        rng = random.Random(self.seed + self.epoch * len(self.rows) + i)
        anchor_rng = random.Random(self.seed + 1000 + self.epoch * len(self.rows) + i)
        start = enhancement_start(c['target'], c['boundary'], self.patch, anchor_rng) if anchor_rng.random() < .5 else None
        ordinary = sample_start(c['breast'], self.patch, rng)
        if start is None:
            start = ordinary
        axes = tuple(a for a in (1, 2) if rng.random() < .5)
        result = {}
        for key in ('pre', 'post', 'breast', 'valid', 'guidance', 'boundary', 'target'):
            x = crop_with_padding(c[key], start, self.patch)
            if axes:
                x = np.flip(x, axes)
            result[key] = torch.from_numpy(np.ascontiguousarray(x[None])).float()
        return result
