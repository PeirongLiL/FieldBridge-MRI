"""Portable pair manifests and axial-window loading."""
from __future__ import annotations
import argparse
import csv
from collections import defaultdict
from pathlib import Path

def build_manifest(pairs: Path, output: Path, bidirectional: bool = True):
    groups = defaultdict(list)
    with Path(pairs).open(newline='') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            groups[(row['source_dataset'], row['pair_id'], row['modality'])].append(row)
    rows = []
    for (dataset, pair, modality), records in sorted(groups.items()):
        if len(records) != 2 or records[0]['field_strength'] == records[1]['field_strength']:
            raise ValueError(f'Ambiguous or incomplete pair: {dataset}/{pair}/{modality}')
        order = {'64mT': 0, '1.5T': 1, '3T': 2, '7T': 3}
        records.sort(key=lambda r: order[r['field_strength']])
        directions = [records, records[::-1]] if bidirectional else [records]
        for source, target in directions:
            rows.append(dict(source_dataset=dataset, pair_id=pair, modality=modality,
                             source_field=source['field_strength'], target_field=target['field_strength'],
                             source=source['relative_path'], target=target['relative_path'],
                             split_group=source['split_group']))
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fields = ['source_dataset','pair_id','modality','source_field','target_field','source','target','split_group']
    with Path(output).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    return len(rows)

def challenge_field(field: str) -> str:
    return '0.1T' if field == '64mT' else field

def load_window(path: Path, center: int, width: int = 15):
    """Return float32 (slices, x, y); edge-repeat within z=[72,292).

    Gzip NIfTI is suitable for examples. Cache decompressed arrays for training.
    """
    import nibabel as nib
    import numpy as np
    if width <= 0 or width % 2 != 1 or not 72 <= center < 292:
        raise ValueError('Use positive odd width and a center in [72,292)')
    image = nib.load(str(path))
    if len(image.shape) != 3 or image.shape[2] < 292:
        raise ValueError('Expected a full-volume image with at least 292 axial slices')
    indices = np.clip(np.arange(center-width//2, center+width//2+1), 72, 291)
    lo, hi = int(indices.min()), int(indices.max())+1
    block = np.asarray(image.dataobj[:, :, lo:hi], dtype=np.float32)
    return np.moveaxis(block[:, :, indices-lo], -1, 0).copy()

def load_pair(row, root: Path, center: int, width: int = 15):
    return dict(source=load_window(Path(root)/row['source'], center, width),
                target=load_window(Path(root)/row['target'], center, width),
                modality=row['modality'], source_field=challenge_field(row['source_field']),
                target_field=challenge_field(row['target_field']), split_group=row['split_group'])

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pairs', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--forward-only', action='store_true')
    a = ap.parse_args()
    print(f'Wrote {build_manifest(a.pairs, a.output, not a.forward_only)} directed volume pairs')

if __name__ == '__main__':
    main()
