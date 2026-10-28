"""Download a published release, or verify an existing local weight directory."""
import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--tag', default='v1.0.0')
p.add_argument('--output', type=Path, default=Path('weights'))
p.add_argument('--manifest', type=Path, default=Path('reproducibility/weights.json'))
p.add_argument('--verify-only', action='store_true')
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
for r in json.loads(a.manifest.read_text())['artifacts']:
    dst = a.output / r['file']
    if not dst.exists() and not a.verify_only:
        url = f'https://github.com/XiGabe/VirtualContrast-BreastMRI/releases/download/{a.tag}/{r["file"]}'
        temporary = dst.with_suffix('.download')
        urllib.request.urlretrieve(url, temporary)
        temporary.replace(dst)
    h = hashlib.sha256()
    with dst.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''):
            h.update(chunk)
    if dst.stat().st_size != r['bytes'] or h.hexdigest() != r['sha256']:
        raise ValueError(f'Weight checksum mismatch: {dst}')
    print(f'OK {dst.name}')
