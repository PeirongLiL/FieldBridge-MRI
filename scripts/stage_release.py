"""Stage the selected final volumes and a portable pair table for publication."""
import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    rows = []
    total_bytes = 0
    for dataset in sorted(a.source.iterdir()):
        if not dataset.is_dir():
            continue
        for base, dirs, files in os.walk(dataset, followlinks=True):
            for filename in sorted(files):
                if not filename.endswith('_record.json'):
                    continue
                record = json.loads((Path(base)/filename).read_text())
                spec = record.get('spec', {})
                final = record.get('final', {})
                if not spec or not isinstance(final, dict) or not final.get('path'):
                    continue
                src = Path(base)/Path(final['path']).name
                if not src.is_file():
                    continue
                pair = spec['pair_id']
                modality = 'T1W' if spec['modality'] == 'TGT' else spec['modality']
                field = '64mT' if spec['field_strength'] == '0.1T' else spec['field_strength']
                relative = Path('data')/dataset.name/pair/f'{pair}_{modality}_{field}.nii.gz'
                dest = a.output/relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                target = src.resolve()
                if dest.is_symlink():
                    if dest.resolve() != target:
                        raise ValueError(f'Conflicting staged file: {relative}')
                elif dest.exists():
                    raise ValueError(f'Destination exists: {relative}')
                else:
                    dest.symlink_to(target)
                rows.append(dict(fieldbridge_id=pair, source_dataset=dataset.name,
                                 pair_id=pair, modality=modality, field_strength=field,
                                 paired_field_strength='', relative_path=relative.as_posix(),
                                 split_group=f'{dataset.name}:{pair}'))
                total_bytes += src.stat().st_size
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['source_dataset'], row['pair_id'], row['modality'])].append(row)
    for group in grouped.values():
        if len(group) != 2 or group[0]['field_strength'] == group[1]['field_strength']:
            raise ValueError('Cannot write a paired manifest for incomplete/ambiguous records')
        group[0]['paired_field_strength'] = group[1]['field_strength']
        group[1]['paired_field_strength'] = group[0]['field_strength']
    rows.sort(key=lambda r:(r['source_dataset'],r['pair_id'],r['modality'],r['field_strength']))
    if not rows:
        raise ValueError('No selected final volumes found')
    with (a.output/'pairs.tsv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter='\t')
        writer.writeheader(); writer.writerows(rows)
    counts = []
    for dataset in sorted({r['source_dataset'] for r in rows}):
        subset = [r for r in rows if r['source_dataset']==dataset]
        counts.append(dict(source_dataset=dataset, case_pairs=len({r['pair_id'] for r in subset}),
                           modality_pairs=len(subset)//2, volumes=len(subset)))
    with (a.output/'dataset_summary.tsv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(counts[0]), delimiter='\t')
        writer.writeheader(); writer.writerows(counts)
    print(json.dumps(dict(volumes=len(rows), modality_pairs=len(grouped), bytes=total_bytes, datasets=counts)),flush=True)

if __name__=='__main__':
    main()
