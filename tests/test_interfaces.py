"""Small synthetic-image tests; no study images are read."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
import nibabel as nib
import numpy as np
import SimpleITK as sitk
from fieldbridge.adapter import build_manifest, build_task3_manifest, load_window, challenge_field
from collections import Counter
from fieldbridge.preprocess import prepare

class Interfaces(unittest.TestCase):
    def test_full_release_all_modalities(self):
        pairs = Path(__file__).resolve().parents[1] / 'metadata/pairs.tsv'
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'pairs.csv'
            self.assertEqual(build_manifest(pairs, out), 518)
            with out.open(newline='') as f:
                rows = list(csv.DictReader(f))
            counts = Counter(r['modality'].replace('-', '') for r in rows)
            self.assertEqual(counts, {'T1W': 322, 'T2W': 174, 'T2FLAIR': 22})
            self.assertEqual(len({r['split_group'] for r in rows}), 161)
            self.assertEqual(build_manifest(pairs, out, False), 259)
            self.assertEqual(build_task3_manifest(pairs, out, Path(tmp)), 113960)
            groups = {}
            with out.open(newline='') as f:
                for r in csv.DictReader(f):
                    key = (r['dataset'], r['pair_id'])
                    groups.setdefault(key, []).append(int(r['slice_idx']))
                    self.assertTrue(r['source_path'].endswith('#z=' + r['slice_idx']))
                    self.assertNotEqual(r['source_field'], '64mT')
                    self.assertIn(r['modality'], ('T1W', 'T2W', 'T2FLAIR'))
            self.assertEqual(len(groups), 518)
            self.assertTrue(all(z == list(range(72, 292)) for z in groups.values()))

    def test_bidirectional_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [dict(source_dataset='test',pair_id='s1',modality='T1W',
                         field_strength=f,relative_path=f'{f}.nii.gz',split_group='test:s1')
                    for f in ['3T','64mT']]
            with (root/'pairs.tsv').open('w') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t')
                writer.writeheader(); writer.writerows(rows)
            self.assertEqual(build_manifest(root/'pairs.tsv',root/'out.csv'),2)
            with (root/'out.csv').open() as f:
                output=list(csv.DictReader(f))
            self.assertEqual(output[0]['source_field'],'64mT')
            self.assertEqual(output[0]['source'],output[1]['target'])
            self.assertEqual(output[0]['target'],output[1]['source'])
            self.assertEqual(challenge_field('64mT'),'0.1T')

    def test_window_edges_and_axis_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'slices.nii.gz'
            arr=np.broadcast_to(np.arange(364,dtype=np.float32),(4,5,364)).copy()
            nib.save(nib.Nifti1Image(arr,np.eye(4)),path)
            default = load_window(path, 79)
            self.assertEqual(default.shape, (7,4,5))
            np.testing.assert_array_equal(default[:,0,0], np.arange(76,83))
            first=load_window(path,72,15)
            self.assertEqual(first.shape,(15,4,5))
            np.testing.assert_array_equal(first[:,0,0],np.clip(np.arange(65,80),72,291))
            last=load_window(path,291,15)
            np.testing.assert_array_equal(last[:,0,0],np.clip(np.arange(284,299),72,291))

    def test_saved_transform_and_n4_export(self):
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            z,y,x=np.indices((24,24,24))
            mask=((x-12)**2+(y-12)**2+(z-12)**2<9**2).astype(np.uint8)
            arr=(100+2*x+y+0.2*z).astype(np.float32)*mask
            image=sitk.GetImageFromArray(arr)
            mask_image=sitk.GetImageFromArray(mask); mask_image.CopyInformation(image)
            sitk.WriteImage(image,str(root/'source.nii.gz'))
            sitk.WriteImage(mask_image,str(root/'mask.nii.gz'))
            sitk.WriteTransform(sitk.Transform(3,sitk.sitkIdentity),str(root/'identity.tfm'))
            spec=dict(dataset='test',pair_id='s1',subject_id='s1',modality='T1W',field_strength='3T',
                      source='source.nii.gz',mask_mni='mask.nii.gz',transform_to_mni='identity.tfm')
            (root/'spec.json').write_text(json.dumps(spec))
            prepare(root/'spec.json',root/'source.nii.gz',root/'out')
            out=sitk.ReadImage(str(root/'out/test/s1/s1_T1W_3T.nii.gz'))
            values=sitk.GetArrayFromImage(out)
            self.assertEqual(out.GetSize(),image.GetSize())
            self.assertTrue(np.isfinite(values).all())
            self.assertAlmostEqual(float(values[mask>0].min()),0)
            self.assertAlmostEqual(float(values.max()),1)
            self.assertTrue((values[mask==0]==0).all())

if __name__=='__main__':
    unittest.main()
