"""Run the released processing functions with explicit input paths."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import SimpleITK as sitk
from .core import VolumeSpec, reconstruct_raw, n4_and_minmax, register_masks, write_image

def prepare(spec_path: Path, reference: Path, output: Path):
    """Apply saved transforms then the original masked N4/min-max routine."""
    payload = json.loads(spec_path.read_text())
    specs = payload if isinstance(payload, list) else [payload]
    fixed = sitk.ReadImage(str(reference), sitk.sitkFloat32)
    for item in specs:
        item = dict(item)
        for key in ('source','mask_mni','transform_to_mni','intermediate_reference','transform_to_intermediate'):
            if item.get(key):
                p = Path(item[key])
                item[key] = str(p if p.is_absolute() else spec_path.parent/p)
        spec = VolumeSpec(**item)
        for component in (spec.dataset, spec.pair_id, spec.subject_id, spec.modality, spec.field_strength):
            if not component or component in {'.','..'} or '/' in component or '\\' in component:
                raise ValueError('Identifiers must be single path components')
        raw = reconstruct_raw(spec, fixed)
        _, normalized, _ = n4_and_minmax(raw, Path(spec.mask_mni))
        mod = 'T1W' if spec.modality == 'TGT' else spec.modality
        field = '64mT' if spec.field_strength == '0.1T' else spec.field_strength
        destination = output/spec.dataset/spec.pair_id/f'{spec.subject_id}_{mod}_{field}.nii.gz'
        if destination.exists():
            raise FileExistsError(f'Use a fresh output directory: {destination}')
        write_image(normalized, destination)
        print(destination, flush=True)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--threads', type=int, default=8)
    sub = ap.add_subparsers(dest='command', required=True)
    run = sub.add_parser('apply')
    run.add_argument('--specs', type=Path, required=True)
    run.add_argument('--reference', type=Path, required=True)
    run.add_argument('--output', type=Path, required=True)
    reg = sub.add_parser('register-masks')
    reg.add_argument('--fixed-mask', type=Path, required=True)
    reg.add_argument('--moving-mask', type=Path, required=True)
    reg.add_argument('--output', type=Path, required=True)
    reg.add_argument('--seed', type=int, default=20260907)
    a = ap.parse_args()
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(a.threads)
    if a.command == 'apply':
        prepare(a.specs, a.reference, a.output)
    else:
        if a.output.exists():
            raise FileExistsError(a.output)
        fixed = sitk.ReadImage(str(a.fixed_mask), sitk.sitkUInt8)
        moving = sitk.ReadImage(str(a.moving_mask), sitk.sitkUInt8)
        tx, method, _, _ = register_masks(fixed, moving, a.seed)
        a.output.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteTransform(tx, str(a.output))
        print(f'Saved {method} transform to {a.output}')

if __name__ == '__main__':
    main()
