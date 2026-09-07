# Processing notes

## Source preparation

| Source | Pairing and preparation |
|---|---|
| OpenNeuro HFC | Retain paired HFC 64 mT and GE 3 T acquisitions; exclude the separate HFE session. Propagate the paired 3 T brain mask. T1W and T2W share the participant's spatial mapping. |
| Zenodo 20281403 | Retain 11 matched 64 mT/3 T cases with T1W, T2W and T2-FLAIR. Propagate the 3 T mask to the aligned low-field side. |
| UNC | Retain 10 paired 3 T/7 T cases with T1W and T2W. Reuse the prepared source-to-reference transform chain. |
| OSF 4nrku | Retain 10 T1W 3 T/7 T pairs; use the prepared 3 T anchor and source masks. |
| ADNI | Retain 47 T1W 1.5 T/3 T pairs. Use SynthSeg masks and a 1.5 T → 3 T → reference path. |
| Penn | Retain 30 T1W and 23 T2W 3 T/7 T pairs. Propagate the T1W transform to the other available contrast. |
| Figshare 26075713 | Retain 20 T1W and T2W pairs. Align 7 T to the paired 3 T image and then to the reference grid. Retain one source copy where BNU and Figshare overlap. |
| Lausanne | Retain 10 T1W 3 T/7 T pairs, represented by the 3 T anchor and paired alignment. |

The released image files are already processed. Raw-data users must reconstruct the same acquisition selection and prepare compatible masks and transforms before running `apply`. An image filename alone cannot identify the correct acquisition in every raw repository.

## Shared endpoint

The reference-grid affine is:

```text
[[0.5, 0,   0,   -91.5],
 [0,   0.5, 0,  -126.0],
 [0,   0,   0.5, -72.0],
 [0,   0,   0,     1.0]]
```

The grid has shape `(364,436,364)`. The processing workspace built a reference from MRIxFields prospective 7 T T1W training images and a corresponding brain mask. These reference files are not included in this code repository. `apply --reference` accepts the prepared reference image; supply the exact reference used for the run you want to reproduce.

The N4 estimator uses finite positive intensities within the supplied mask. Its log bias field is estimated on a factor-two reduced image, linearly interpolated to the full grid, and applied before per-volume min-max normalization. Normalization uses the retained finite positive brain voxels and clips only floating-point roundoff to `[0,1]`. No percentile truncation or histogram matching is applied.

## Relationship to the original implementation

`core.py` extracts the registration/resampling helpers from the source processing implementation and the common N4/min-max functions from its intensity module. Their numerical function bodies are retained. The public `preprocess.py` wrapper exposes explicit input paths and writes only the final image. It omits workstation-specific orchestration and intermediate-report generation. `adapter.py` is a portable implementation of the volume-pair and 220-slice interface, with an explicitly documented edge-repeat rule for optional windows.

Consequently, this repository exposes the implemented processing operations, but a new raw-data run still depends on source acquisition selection, source masks and saved transforms. New registration runs may differ from the existing derivatives because optimizer sampling and input masks affect the transform. The published derivatives remain the fixed data release.
