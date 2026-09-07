"""Processing functions extracted from the scripts used for FieldBridge-MRI.

Original numerical operations are retained; repository-specific orchestration
is exposed separately through the portable CLI.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import nibabel as nib
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_closing, binary_dilation, binary_fill_holes, binary_opening, label

TARGET_SHAPE = (364, 436, 364)
TARGET_AFFINE = np.array([[0.5,0,0,-91.5],[0,0.5,0,-126],[0,0,0.5,-72],[0,0,0,1]], dtype=np.float64)

@dataclass
class VolumeSpec:
    dataset: str
    pair_id: str
    subject_id: str
    modality: str
    field_strength: str
    source: str
    mask_mni: str
    transform_to_mni: str
    intermediate_reference: str | None = None
    transform_to_intermediate: str | None = None


def same_geometry(a: sitk.Image, b: sitk.Image, tol: float = 1e-5) -> bool:
    return (
        a.GetSize() == b.GetSize()
        and np.allclose(a.GetSpacing(), b.GetSpacing(), atol=tol)
        and np.allclose(a.GetOrigin(), b.GetOrigin(), atol=tol)
        and np.allclose(a.GetDirection(), b.GetDirection(), atol=tol)
    )


def reconstruct_raw(spec: VolumeSpec, fixed: sitk.Image) -> sitk.Image:
    moving = sitk.ReadImage(spec.source, sitk.sitkFloat32)
    if spec.transform_to_intermediate:
        if not spec.intermediate_reference:
            raise ValueError("intermediate reference required")
        intermediate = sitk.ReadImage(spec.intermediate_reference, sitk.sitkFloat32)
        tx_first = sitk.ReadTransform(spec.transform_to_intermediate)
        moving = sitk.Resample(
            moving, intermediate, tx_first, sitk.sitkLinear, 0.0, sitk.sitkFloat32
        )
    tx_mni = sitk.ReadTransform(spec.transform_to_mni)
    return sitk.Resample(moving, fixed, tx_mni, sitk.sitkLinear, 0.0, sitk.sitkFloat32)


def n4_and_minmax(raw: sitk.Image, mask_path: Path) -> tuple[sitk.Image, sitk.Image, dict[str, Any]]:
    mask = sitk.ReadImage(str(mask_path), sitk.sitkUInt8)
    if not same_geometry(mask, raw):
        mask = sitk.Resample(
            mask, raw, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8
        )
    raw_arr = sitk.GetArrayFromImage(raw).astype(np.float32, copy=False)
    mask_arr = sitk.GetArrayFromImage(mask) > 0
    valid = mask_arr & np.isfinite(raw_arr) & (raw_arr > 0)
    if not np.any(valid):
        raise ValueError(f"empty positive brain mask for {mask_path}")

    # N4 operates on positive intensities. Voxels inside the retained brain
    # mask that are non-positive are excluded from the N4 estimation mask.
    n4_mask_arr = valid.astype(np.uint8)
    n4_mask = sitk.GetImageFromArray(n4_mask_arr)
    n4_mask.CopyInformation(raw)

    n4 = sitk.N4BiasFieldCorrectionImageFilter()
    n4.SetMaximumNumberOfIterations([50, 50, 30, 20])
    n4.SetConvergenceThreshold(1e-6)
    img_small = sitk.Shrink(raw, [2, 2, 2])
    mask_small = sitk.Shrink(n4_mask, [2, 2, 2])
    n4.Execute(img_small, mask_small)
    log_bias_small = n4.GetLogBiasFieldAsImage(img_small)
    log_bias = sitk.Resample(
        log_bias_small,
        raw,
        sitk.Transform(),
        sitk.sitkLinear,
        0.0,
        sitk.sitkFloat32,
    )
    corrected = raw / sitk.Exp(log_bias)
    corrected_arr = sitk.GetArrayFromImage(corrected).astype(np.float32, copy=False)
    finite = valid & np.isfinite(corrected_arr)
    values = corrected_arr[finite]
    lo, hi = float(values.min()), float(values.max())
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        raise ValueError(f"degenerate N4 brain intensity range: {(lo, hi)}")
    normalized_arr = np.zeros_like(corrected_arr, dtype=np.float32)
    normalized_arr[finite] = (corrected_arr[finite] - lo) / (hi - lo)
    # This clip only protects against floating-point roundoff after exact
    # min-max; it is not percentile clipping.
    normalized_arr[finite] = np.clip(normalized_arr[finite], 0.0, 1.0)
    corrected_masked = np.zeros_like(corrected_arr, dtype=np.float32)
    corrected_masked[finite] = corrected_arr[finite]
    corrected_img = sitk.GetImageFromArray(corrected_masked)
    corrected_img.CopyInformation(raw)
    normalized_img = sitk.GetImageFromArray(normalized_arr)
    normalized_img.CopyInformation(raw)
    return corrected_img, normalized_img, {
        "brain_voxels": int(finite.sum()),
        "raw_brain_min": float(raw_arr[valid].min()),
        "raw_brain_max": float(raw_arr[valid].max()),
        "n4_brain_min": lo,
        "n4_brain_max": hi,
        "normalized_min": float(normalized_arr[finite].min()),
        "normalized_max": float(normalized_arr[finite].max()),
        "outside_mask_nonzero": int(np.count_nonzero(normalized_arr[~finite])),
        "n4_shrink_factor": 2,
        "n4_iterations": [50, 50, 30, 20],
    }


def write_image(image: sitk.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(path), True)


def canonical(path: Path) -> nib.Nifti1Image:
    return nib.as_closest_canonical(nib.load(path))


def largest_component(mask: np.ndarray) -> np.ndarray:
    lab, n = label(mask)
    if n == 0:
        raise ValueError("empty mask")
    count = np.bincount(lab.ravel()); count[0] = 0
    return lab == count.argmax()


def clean_mask(mask: np.ndarray, dilate: int = 1) -> np.ndarray:
    mask = largest_component(mask.astype(bool))
    mask = binary_closing(mask, iterations=2)
    mask = binary_opening(mask, iterations=1)
    mask = binary_fill_holes(mask)
    if dilate:
        mask = binary_dilation(mask, iterations=dilate)
    return largest_component(mask).astype(np.uint8)


def save_nifti(data: np.ndarray, affine: np.ndarray, path: Path, dtype=np.float32) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(data.astype(dtype), affine)
    image.header.set_data_dtype(dtype)
    nib.save(image, path)


def sitk_distance(mask: sitk.Image) -> sitk.Image:
    d = sitk.SignedMaurerDistanceMap(sitk.Cast(mask, sitk.sitkUInt8), insideIsPositive=True, squaredDistance=False, useImageSpacing=True)
    return sitk.Clamp(d, lowerBound=-20.0, upperBound=20.0)


def coarse_mask(image: sitk.Image, spacing: float = 2.0) -> sitk.Image:
    old_spacing, old_size = np.asarray(image.GetSpacing()), np.asarray(image.GetSize())
    new_spacing = np.full(3, spacing)
    new_size = np.maximum(1, np.rint(old_size * old_spacing / new_spacing).astype(int))
    return sitk.Resample(image, [int(x) for x in new_size], sitk.Transform(3, sitk.sitkIdentity), sitk.sitkNearestNeighbor,
                         image.GetOrigin(), tuple(float(x) for x in new_spacing), image.GetDirection(), 0, sitk.sitkUInt8)


def resample(moving: sitk.Image, fixed: sitk.Image, tx: sitk.Transform, interp=sitk.sitkLinear, pixel=sitk.sitkFloat32) -> sitk.Image:
    return sitk.Resample(moving, fixed, tx, interp, 0.0, pixel)


def mask_dice(fixed: sitk.Image, moving: sitk.Image, tx: sitk.Transform) -> float:
    a = sitk.GetArrayFromImage(fixed > 0)
    b = sitk.GetArrayFromImage(resample(moving, fixed, tx, sitk.sitkNearestNeighbor, sitk.sitkUInt8) > 0)
    return float(2 * np.logical_and(a, b).sum() / max(a.sum() + b.sum(), 1))


def register_masks(fixed_full: sitk.Image, moving_full: sitk.Image, seed: int) -> tuple[sitk.Transform, str, float, float]:
    fixed, moving = coarse_mask(fixed_full), coarse_mask(moving_full)
    fd, md = sitk_distance(fixed), sitk_distance(moving)
    initial = sitk.CenteredTransformInitializer(fd, md, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.MOMENTS)
    candidates = [("moments", initial, mask_dice(fixed, moving, initial), 1.0)]
    reg = sitk.ImageRegistrationMethod(); reg.SetMetricAsMeanSquares(); reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.05, seed=seed); reg.SetInterpolator(sitk.sitkLinear); reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel([4, 2, 1]); reg.SetSmoothingSigmasPerLevel([2, 1, 0]); reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg.SetOptimizerAsRegularStepGradientDescent(1.0, 1e-4, 250, 1e-8); reg.SetInitialTransform(initial, inPlace=False)
    rigid = reg.Execute(fd, md); candidates.append(("rigid_mask_distance", rigid, mask_dice(fixed, moving, rigid), 1.0))
    rb = rigid.GetBackTransform() if isinstance(rigid, sitk.CompositeTransform) else rigid
    affine = sitk.AffineTransform(3); affine.SetCenter(rb.GetCenter()); affine.SetMatrix(rb.GetMatrix()); affine.SetTranslation(rb.GetTranslation())
    ar = sitk.ImageRegistrationMethod(); ar.SetMetricAsMeanSquares(); ar.SetMetricSamplingStrategy(ar.RANDOM)
    ar.SetMetricSamplingPercentage(0.05, seed=seed); ar.SetInterpolator(sitk.sitkLinear); ar.SetOptimizerScalesFromPhysicalShift()
    ar.SetShrinkFactorsPerLevel([4, 2, 1]); ar.SetSmoothingSigmasPerLevel([2, 1, 0]); ar.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    ar.SetOptimizerAsGradientDescentLineSearch(learningRate=0.2, numberOfIterations=150, convergenceMinimumValue=1e-6, convergenceWindowSize=15)
    ar.SetInitialTransform(affine, inPlace=False)
    try:
        atx = ar.Execute(fd, md)
        ab = atx.GetBackTransform() if isinstance(atx, sitk.CompositeTransform) else atx
        det = float(np.linalg.det(np.asarray(ab.GetMatrix()).reshape(3, 3)))
        if 0.5 <= det <= 2.0:
            candidates.append(("rigid_affine_mask_distance", atx, mask_dice(fixed, moving, atx), det))
    except RuntimeError:
        pass
    method, tx, _, det = max(candidates, key=lambda x: x[2])
    dice = mask_dice(fixed_full, moving_full, tx)
    return tx, method, dice, det


def xyz(image: sitk.Image) -> np.ndarray:
    return sitk.GetArrayFromImage(image).transpose(2, 1, 0).astype(np.float32)
