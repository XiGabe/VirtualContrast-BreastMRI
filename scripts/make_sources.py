"""Index converted NC and first-CE NIfTI files."""
import argparse
import csv
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--root', type=Path, required=True, help='NIfTI root, with dataset/source-ID/source-ID_{0,1}.nii.gz')
p.add_argument('--mapping', type=Path, default=Path('reproducibility/source_mapping.csv'))
p.add_argument('--output', type=Path, default=Path('data/sources.csv'))
p.add_argument('--masks', type=Path, default=Path('data/divider'))
a = p.parse_args()
with a.mapping.open() as f:
    records = list(csv.DictReader(f))
rows = {}
for r in records:
    sid = r['source_patient_id']
    folder = a.root / r['dataset'] / sid
    row = dict(patient=r['patient'], nc=str((folder/f'{sid}_0.nii.gz').resolve()),
               ce=str((folder/f'{sid}_1.nii.gz').resolve()),
               divider=str((a.masks/f'{r["patient"]}.nii.gz').resolve()))
    if r['patient'] in rows and rows[r['patient']] != row:
        raise ValueError('Ambiguous duplicate source patient')
    rows[r['patient']] = row
for row in rows.values():
    for key in ['nc','ce']:
        if not Path(row[key]).is_file():
            raise FileNotFoundError(row[key])
a.output.parent.mkdir(parents=True, exist_ok=True)
with a.output.open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['patient','nc','ce','divider'], lineterminator='\n')
    w.writeheader()
    w.writerows(rows.values())
