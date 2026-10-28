"""Rebuild the study example panels from local caches and evaluation-only masks."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--cache', type=Path, required=True)
p.add_argument('--cases', type=Path, default=Path('reproducibility/figure1.json'))
p.add_argument('--tumor-sources', type=Path, required=True, help='CSV: sample_id,path to original evaluation tumor NIfTI')
p.add_argument('--output', type=Path, default=Path('outputs/results'))
a = p.parse_args()
with a.tumor_sources.open() as f:
    masks = {r['sample_id']:a.tumor_sources.parent/r['path'] for r in csv.DictReader(f)}
cases = json.loads(a.cases.read_text())['cases']
plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42})
fig,axes = plt.subplots(len(cases),3,figsize=(9,10.3),squeeze=False)
fig.subplots_adjust(left=.065,right=.995,bottom=.005,top=.96,wspace=.035,hspace=.045)
for i,c in enumerate(cases):
    sid,z=c['sample_id'],c['slice']
    (y0,x0),(y1,x1)=c['crop_yx']
    meta=json.loads((a.cache/'nc'/f'{sid}.json').read_text())
    tumor=None
    if c['tumor_contour']:
        shape=meta['shape'];target=(shape[1],shape[2],shape[0])
        tumor=np.asarray(resample_from_to(nib.load(masks[sid]),(target,np.array(meta['canvas_affine_lr_ap_si'])),order=0).dataobj).transpose(2,0,1)[z,y0:y1,x0:x1]
    for j,modality in enumerate(['nc','ce','vce']):
        image=np.load(a.cache/modality/f'{sid}.npy',mmap_mode='r')[z,y0:y1,x0:x1]
        ax=axes[i,j];ax.set_facecolor('black')
        ax.imshow(image,cmap='gray',vmin=0,vmax=.65,interpolation='bilinear')
        if tumor is not None and tumor.any():ax.contour(tumor,levels=[.5],colors=['#00BFFF'],linewidths=1.6)
        h,w=image.shape;dh=max(h,w/1.14);dw=dh*1.14;cx,cy=(w-1)/2,(h-1)/2
        ax.set_xlim(cx-dw/2,cx+dw/2);ax.set_ylim(cy+dh/2,cy-dh/2);ax.set_box_aspect(1/1.14)
        ax.set_xticks([]);ax.set_yticks([])
        for spine in ax.spines.values():spine.set_visible(False)
        if i==0:ax.set_title(['Non contrast MRI','Acquired contrast MRI','Virtual contrast MRI'][j],fontsize=10,fontweight='bold',color='#233445',pad=10)
fig.text(.026,.73,'Cancer',rotation=90,va='center',ha='center',fontsize=12,fontweight='bold',color='#233445')
fig.text(.026,.25,'Non-cancer',rotation=90,va='center',ha='center',fontsize=12,fontweight='bold',color='#233445')
a.output.mkdir(parents=True,exist_ok=True)
for ext in ['png','pdf']:fig.savefig(a.output/f'figure1.{ext}',dpi=300,bbox_inches='tight',pad_inches=.03)
plt.close(fig)
