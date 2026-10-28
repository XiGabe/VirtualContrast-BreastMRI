"""Build the study cache directly from paired bilateral NIfTI images."""
import argparse
import csv
import json
from pathlib import Path

import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np
from scipy.ndimage import affine_transform, gaussian_filter

from .data import read_manifest
from .utils import sha256, write_json


def prepare_breast(pre, post, mask, affine, reference):
    """Arrays are native RAS single-breast volumes; output axes are SI,LR,AP."""
    if pre.shape != mask.shape or (post is not None and pre.shape != post.shape) or not mask.any():
        raise ValueError('Misaligned pair or empty breast mask')
    spacing = nib.affines.voxel_sizes(affine)
    bounds = [np.flatnonzero(mask.any(axis=tuple(j for j in range(3) if j != i))) for i in range(3)]
    margin = np.ceil(10 / spacing).astype(int)
    lower = np.maximum(0, np.array([x[0] for x in bounds])-margin)
    upper = np.minimum(pre.shape, np.array([x[-1]+1 for x in bounds])+margin)
    size = upper-lower
    target_spacing = np.array([.8,.8,2.])
    canvas = np.array([280,336])
    fit = min(1., float(np.min(canvas*target_spacing[:2]/(size[:2]*spacing[:2]))))
    target_spacing[:2] /= fit
    shape = np.ceil(size*spacing/target_spacing-1e-7).astype(int)
    shape[:2] = np.minimum(shape[:2], canvas)
    pad = (canvas-shape[:2])//2
    ratio = target_spacing/spacing
    transform = np.eye(4)
    transform[:3,:3] = np.diag(ratio)
    transform[:3,3] = lower + .5*ratio-.5 - np.r_[pad,0]*ratio
    target_affine = affine@transform
    lo, hi = map(float, reference)
    if not np.isfinite(reference).all() or hi <= lo:
        raise ValueError('Invalid NC normalization')
    sl = tuple(slice(int(l),int(u)) for l,u in zip(lower,upper))
    outputs = {}
    for modality, array in [('nc',pre), ('ce',post)]:
        if array is None:
            continue
        if not np.isfinite(array).all():
            raise ValueError('Nonfinite image')
        u = np.clip((array[sl]-lo)/(hi-lo),0,3)/3
        u = gaussian_filter(u, np.maximum((ratio-1)/2,0), mode='nearest')
        x = affine_transform(u,np.diag(ratio),offset=.5*ratio-.5,output_shape=tuple(shape),
                             order=1,mode='nearest',prefilter=False)
        x = np.pad(x,((pad[0],canvas[0]-shape[0]-pad[0]),(pad[1],canvas[1]-shape[1]-pad[1]),(0,0)))
        outputs[modality] = x.transpose(2,0,1).astype(np.float16)
    target_shape = (int(canvas[0]),int(canvas[1]),int(shape[2]))
    mask_img = nib.Nifti1Image(mask.astype(np.uint8), affine)
    outputs['breast'] = (np.asarray(resample_from_to(mask_img, (target_shape,target_affine), order=0).dataobj)
                         .transpose(2,0,1) > 0).astype(np.uint8)
    meta = dict(shape=list(outputs['nc'].shape), array_axes='SI,LR,AP',
                effective_spacing=target_spacing.tolist(), pad_lower=pad.tolist(),
                resampled_shape_lr_ap=shape[:2].tolist(), canvas_affine_lr_ap_si=target_affine.tolist(),
                crop_lower=lower.tolist(), crop_upper=upper.tolist(), overflow_fit_scale=fit,
                reference=dict(pre_p005=float(lo), scale=float(hi-lo)))
    return outputs, meta


def split_pair(nc_path, ce_path, divider_path, rows):
    ref = nib.load(nc_path)
    images = [nib.load(divider_path)] + ([nib.load(ce_path)] if ce_path else [])
    if len(ref.shape) != 3 or any(i.shape != ref.shape or not np.allclose(i.affine,ref.affine,atol=1e-4) for i in images):
        raise ValueError('Expected aligned 3D NC/first CE/divider images')
    tr = nib.orientations.ornt_transform(nib.orientations.io_orientation(ref.affine), nib.orientations.axcodes2ornt(('R','A','S')))
    pre = nib.orientations.apply_orientation(ref.get_fdata(dtype=np.float32),tr)
    divider = nib.orientations.apply_orientation(images[0].get_fdata(),tr).astype(np.uint8)
    if not set(np.unique(divider)).issubset({0,1,2}):
        raise ValueError('Divider mask must contain labels 0,1,2')
    post = nib.orientations.apply_orientation(images[1].get_fdata(dtype=np.float32),tr) if ce_path else None
    affine = ref.affine @ nib.orientations.inv_ornt_aff(tr,ref.shape)
    means = []
    for label in (1,2):
        counts = (divider == label).sum(axis=(1,2),dtype=np.int64)
        if not counts.sum():
            raise ValueError('Divider is missing one breast')
        means.append(float(np.dot(np.arange(len(counts)),counts)/counts.sum()))
    # Partition between breast centroids using the cohort side convention.
    labels = {'left':1,'right':2} if means[0] < means[1] else {'left':2,'right':1}
    cut = int(np.floor(sum(means)/2))+1
    ranges = {'left':(0,cut), 'right':(cut,pre.shape[0])}
    values = pre[divider > 0]
    if values.size < 1000:
        raise ValueError('Insufficient breast support for normalization')
    reference = np.percentile(values,[.5,99.5])
    for row in rows:
        lower,upper = ranges[row['side']]
        shift = np.eye(4)
        shift[0,3] = lower
        outputs,meta = prepare_breast(pre[lower:upper],post[lower:upper] if post is not None else None,
                                    divider[lower:upper] == labels[row['side']], affine@shift, reference)
        yield row,outputs,meta


def run_divider(source, output, model_folder):
    import os
    os.environ.setdefault('nnUNet_compile','false')
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    import torch
    predictor = nnUNetPredictor(tile_step_size=.5, use_gaussian=True, use_mirroring=True,
        perform_everything_on_device=True, device=torch.device('cuda',0), verbose=False,
        verbose_preprocessing=False, allow_tqdm=False)
    predictor.initialize_from_trained_model_folder(str(model_folder), use_folds=('all',), checkpoint_name='checkpoint_final.pth')
    predictor.predict_from_files([[str(source)]],[str(output)[:-7]], save_probabilities=False,
        overwrite=False,num_processes_preprocessing=2,num_processes_segmentation_export=2)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sources', type=Path, required=True, help='CSV: patient,nc,ce,divider; paths relative to this CSV')
    p.add_argument('--manifest',type=Path,default=Path('reproducibility/cohort.csv'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--divider-model',type=Path,help='Optional BreastDivider model folder to generate missing masks')
    p.add_argument('--nc-only',action='store_true',help='Preprocess NC volumes only')
    args = p.parse_args()
    rows = read_manifest(args.manifest)
    with args.sources.open() as handle:
        sources = list(csv.DictReader(handle))
    if len({s['patient'] for s in sources}) != len(sources):
        raise ValueError('Duplicate source patient')
    source_by_id = {s['patient']:s for s in sources}
    patients = list(dict.fromkeys(r['patient'] for r in rows))
    if set(patients) - source_by_id.keys():
        raise ValueError('Missing patients in source index')
    for modality in ('nc','ce','breast'):
        (args.output/modality).mkdir(parents=True,exist_ok=True)
    protocol = dict(manifest_sha256=sha256(args.manifest), sources_sha256=sha256(args.sources), nc_only=args.nc_only)
    marker = args.output/'preprocessing.json'
    if marker.exists() and json.loads(marker.read_text()) != protocol:
        raise ValueError('Output belongs to a different preprocessing run')
    write_json(marker,protocol)
    for i,patient in enumerate(patients):
        source = source_by_id[patient]
        paths = {k:args.sources.parent/source[k] for k in ('nc','divider')}
        ce = None if args.nc_only else args.sources.parent/source['ce']
        if not paths['divider'].exists() and args.divider_model:
            paths['divider'].parent.mkdir(parents=True,exist_ok=True)
            run_divider(paths['nc'],paths['divider'],args.divider_model)
        members = [r for r in rows if r['patient'] == patient]
        for row,outputs,meta in split_pair(paths['nc'],ce,paths['divider'],members):
            for modality,array in outputs.items():
                dst = args.output/modality/(row['sample_id']+'.npy')
                if dst.exists():
                    if not np.array_equal(np.load(dst),array):
                        raise ValueError(f'Existing cache differs: {dst}')
                else:
                    np.save(dst,array)
            write_json(args.output/'nc'/(row['sample_id']+'.json'),dict(meta,sample_id=row['sample_id']))
        print(f'Prepared {i+1}/{len(patients)} patients',flush=True)
    write_json(args.output/'completed.json',dict(**protocol,patients=len(patients),breasts=len(rows)))


if __name__ == '__main__':
    main()
