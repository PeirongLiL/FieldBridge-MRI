# MRI-PairedField (public repository: FieldBridge-MRI)

MRI-PairedField is the dataset name used in the manuscript. The public GitHub and Hugging Face repositories retain the release name **FieldBridge-MRI**; both names refer to the same v1.0 dataset.

Unified preparation of eight paired brain MRI sources, with a data interface for MRIxFields2026 Task 3.

**Dataset:** [lipeirong/FieldBridge-MRI on Hugging Face](https://huggingface.co/datasets/lipeirong/FieldBridge-MRI)

The collection contains 161 case-level pairs, 259 same-modality pairs and 518 full-volume NIfTI images across 64 mT, 1.5 T, 3 T and 7 T. Available contrasts are T1W, T2W and T2-FLAIR. The code release contains the numerical processing functions used in preparing the data, with portable command-line wrappers and a paired-data loader.

## Install

```bash
git clone https://github.com/PeirongLiL/FieldBridge-MRI.git
cd FieldBridge-MRI
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install huggingface_hub
```

For the source processing environment, use Python 3.10 and `pip install -r requirements-processing.txt`. CPU preprocessing uses SimpleITK; the data adapter does not require a GPU or a particular model framework.

## Download the data

```bash
hf download lipeirong/FieldBridge-MRI --type dataset --local-dir ./data
```

For one source and the pairing metadata:

```bash
hf download lipeirong/FieldBridge-MRI --type dataset --include 'data/unc/**' --include '*.tsv' --local-dir ./data
```

The repository is approximately 20.36 GB. Each file is a full float32 NIfTI image with shape `(364, 436, 364)` at 0.5 mm isotropic spacing, with masked intensities in `[0, 1]` and zero background. The 0.5 mm spacing is a resampling grid, not the native resolution of every scan.

## Build a training manifest

```bash
fieldbridge-manifest --pairs ./data/pairs.tsv --output ./paired_bidirectional.csv
```

This writes 518 directed volume-pair records from the 259 modality-pairs. Use `--forward-only` for 259 lower-to-higher field records. Each record contains source and target paths relative to the data download root, physical field labels, modality and participant grouping information. All records sharing `split_group` should remain in the same participant partition.

```python
import csv
from pathlib import Path
from fieldbridge.adapter import load_pair

with open('paired_bidirectional.csv', newline='') as f:
    row = next(csv.DictReader(f))
sample = load_pair(row, Path('data'), center=150, width=7)
print(sample['source'].shape)  # (7, 364, 436)
print(sample['source_field'], sample['target_field'], sample['modality'])
```

The default width is seven slices. The paper's training protocol uses centers 79–284 (zero-based, 206 positions), so all seven slices are real, without edge repetition. The general loader also accepts centers in `z=72:292`; at those interval boundaries it repeats the nearest available slice. Explicit `width=15` remains supported for older callers; use `width=1` for individual slices. Physical `64mT` labels map to `0.1T` only in the challenge-facing adapter. Arrays have axes `(slices, x, y)` and retain the released `[0,1]` intensity range. UNSB training maps them to `[-1,1]`. Cache decompressed arrays for large training jobs rather than repeatedly decompressing gzip files.

### Metadata fields and source labels

`metadata/pairs.tsv` has one row per volume. The fields are `fieldbridge_id` (unique volume identifier), `source_dataset` (stable machine-readable source code), `pair_id` (same-subject pairing key), `modality`, `field_strength`, `paired_field_strength`, `relative_path`, and `split_group` (participant-level grouping key). Join two rows by `source_dataset`, `pair_id` and `modality`; the two rows must have reciprocal field-strength values. Keep all rows with the same `split_group` in one partition.

The source-code mapping is: `openneuro_hfc` = ULF; `zenodo_20281403` = LUMC; `unc` = UNC; `osf` = Subcortical; `adni` = ADNI; `penn` = Penn; `figshare_26075713` = Hippocampal; `lausanne_zenodo1438358` = Vim. The processed derivatives contain 161 subject-level pairings, 259 same-modality pairs and 518 volumes.

### Full-data seven-slice training manifest

All available modalities are retained: **161 T1W, 87 T2W and 11 T2-FLAIR pairs**, totaling **259 pairs**. The 161 participant-level groups are used for splitting, not for choosing one modality per participant. Do not deduplicate training pairs by participant alone.

For the revised `full259` MRIxFields loader, export its slice-record schema:

```bash
fieldbridge-manifest --pairs ./data/pairs.tsv \
  --output ./external_full259_bidirectional.csv \
  --task3-slices --data-root ./data
```

This exports 518 directed pairs × 220 slice records = **113,960 CSV rows**, using absolute `image.nii.gz#z=72` paths. It writes metadata only, not cropped or duplicated images. The training loader assembles 206 seven-slice windows per directed pair, yielding **106,708 external windows**. These overlapping windows are not independent subjects. The schema uses `T2FLAIR` and maps `64mT` to `0.1T`; published metadata preserve the physical field strengths.

Use the full-data manifest in both unpaired pretraining and paired fine-tuning. A checkpoint pretrained on the older 161-pair subset is not a full-data pretraining run. This repository provides the data interface, not the complete UNSB trainer, checkpoints or third-party network implementation. The exporter matches the revised remote loader's field names and slice conventions; regression tests cover all 259 released pairs.

```bash
python -m unittest discover -s tests -v
```

## Preprocess source images

`fieldbridge/core.py` retains the numerical operations from the processing scripts used for this collection:

- mask-based rigid/affine registration and mask cleanup;
- one-step or two-step resampling using saved SimpleITK transforms;
- N4 with shrink factor 2, iterations `[50, 50, 30, 20]` and threshold `1e-6`;
- min-max normalization over finite positive brain voxels after N4, with zero background.

The portable wrappers use explicit paths. They do not download raw datasets, install SynthSeg weights or reproduce the original workstation directory layout. Supply source NIfTI images, masks, the reference image and the appropriate transforms for each source. `examples/volume_specs.json` illustrates a two-step 7 T → 3 T → reference path; its input paths are placeholders to replace with local files.

```bash
# Estimate a transform from two prepared masks.
fieldbridge-preprocess register-masks \
  --fixed-mask inputs/subject001_3T_mask.nii.gz \
  --moving-mask inputs/subject001_7T_mask.nii.gz \
  --output inputs/subject001_7T_to_3T.h5

# Resample using saved transforms, then run masked N4 and normalization.
fieldbridge-preprocess --threads 8 apply \
  --specs examples/volume_specs.json \
  --reference inputs/reference.nii.gz \
  --output processed
```

SimpleITK resampling transforms map fixed-grid physical points to moving-image coordinates. The descriptive filenames above refer to the image alignment direction, not an instruction to invert the transform. The `transform_to_mni` key is retained for compatibility with the original processing records and denotes the MRIxFields reference-grid transform.

Source-specific acquisition selection and mask preparation are described in [Processing notes](docs/PROCESSING.md). The reference image and any source-specific segmentation inputs must be supplied separately. Source images are linearly resampled; binary masks use nearest-neighbour interpolation. The shared reference grid is not labelled as a generic MNI152 template.

## Repository contents

| Path | Purpose |
|---|---|
| `fieldbridge/core.py` | Extracted registration, resampling and intensity processing functions |
| `fieldbridge/preprocess.py` | Portable preprocessing command-line wrapper |
| `fieldbridge/adapter.py` | Bidirectional manifests and axial-window loading |
| `scripts/stage_release.py` | Organize processed volume records into the published layout |
| `metadata/pairs.tsv` | Volume-level pairing metadata for the release |
| `metadata/dataset_summary.tsv` | Counts for each source |
| `examples/volume_specs.json` | Input specification template for preprocessing |
| `docs/PROCESSING.md` | Source-specific processing notes and implementation boundaries |

The release uses one public version number. Internal processing iterations are not separate dataset versions. No MRI model weights or image-translation model are included.

## Sources, access conditions and citation

Please cite the manuscript under the name MRI-PairedField and the original dataset paper or record for every source used. Access, redistribution and citation remain subject to the original source terms. ADNI is controlled-access and requires the ADNI application and data-use agreement; raw ADNI files are not redistributed here. For the other sources, check the linked record for its current licence or access requirements. The full source table and persistent links are maintained in the [Hugging Face dataset card](https://huggingface.co/datasets/lipeirong/FieldBridge-MRI).

Relevant methods include [N4ITK](https://doi.org/10.54294/jculxw), [SynthSeg](https://doi.org/10.1016/j.media.2023.102789) and [BIDS](https://doi.org/10.1038/sdata.2016.44). The MRIxFields adapter follows the collection's existing Task 3 workflow and is not an official challenge repository.
