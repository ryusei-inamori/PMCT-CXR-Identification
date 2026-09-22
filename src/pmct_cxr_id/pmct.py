from __future__ import annotations
from pathlib import Path
from typing import Optional
import shutil
import subprocess
import tempfile
import numpy as np
import pydicom
from pydicom.errors import InvalidDicomError
import SimpleITK as sitk


def is_dicom(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        return True
    except (InvalidDicomError, Exception):
        return False


def dicom_files(folder: Path) -> list[Path]:
    return [p for p in sorted(folder.iterdir()) if p.is_file() and is_dicom(p)]


def select_series(ct_dir: Path) -> Path:
    candidates = []
    for d in sorted(p for p in ct_dir.iterdir() if p.is_dir()):
        n = len(dicom_files(d))
        if n:
            candidates.append((n, d))
    if not candidates:
        raise RuntimeError(f"No DICOM sequence found under {ct_dir}")
    candidates.sort(key=lambda t: (-t[0], str(t[1])))
    return candidates[0][1]


def read_longest_series(folder: Path) -> sitk.Image:
    reader = sitk.ImageSeriesReader()
    uids = reader.GetGDCMSeriesIDs(str(folder))
    if not uids:
        raise RuntimeError(f"No DICOM series found: {folder}")
    groups = [(len(reader.GetGDCMSeriesFileNames(str(folder), uid)), uid) for uid in uids]
    _, uid = max(groups)
    reader.SetFileNames(reader.GetGDCMSeriesFileNames(str(folder), uid))
    return reader.Execute()


def reorient(img: sitk.Image, orientation: str = "RAI") -> sitk.Image:
    f = sitk.DICOMOrientImageFilter()
    f.SetDesiredCoordinateOrientation(orientation)
    return f.Execute(img)


def resample_isotropic(img: sitk.Image, spacing_mm: float = 1.0) -> sitk.Image:
    old_spacing = img.GetSpacing()
    old_size = img.GetSize()
    new_spacing = (spacing_mm, spacing_mm, spacing_mm)
    new_size = [max(1, int(round(old_size[i] * old_spacing[i] / spacing_mm))) for i in range(3)]
    return sitk.Resample(
        img,
        new_size,
        sitk.Transform(),
        sitk.sitkLinear,
        img.GetOrigin(),
        new_spacing,
        img.GetDirection(),
        -1024.0,
        sitk.sitkFloat32,
    )


def clip_hu_int16(img: sitk.Image) -> sitk.Image:
    a = np.clip(sitk.GetArrayFromImage(img), -1024, 3071).astype(np.int16)
    out = sitk.GetImageFromArray(a)
    out.CopyInformation(img)
    return out


def run_totalsegmentator(input_nifti: Path, output_dir: Path, task: str, roi_subset=None, fast=False):
    cmd = ["TotalSegmentator", "-i", str(input_nifti), "-o", str(output_dir), "--task", task]
    if roi_subset:
        cmd += ["--roi_subset"] + list(roi_subset)
    if fast:
        cmd.append("--fast")
    subprocess.run(cmd, check=True)


def find_mask(base: Path, name: str) -> Path:
    direct = base / f"{name}.nii.gz"
    if direct.exists():
        return direct
    hits = list(base.rglob(f"{name}.nii.gz"))
    if not hits:
        raise FileNotFoundError(f"TotalSegmentator mask not found: {name}")
    return hits[0]


def mask_on_reference(path: Path, ref: sitk.Image) -> sitk.Image:
    m = sitk.Cast(sitk.ReadImage(str(path)) > 0, sitk.sitkUInt8)
    same = (
        m.GetSize() == ref.GetSize()
        and np.allclose(m.GetSpacing(), ref.GetSpacing())
        and np.allclose(m.GetOrigin(), ref.GetOrigin())
        and np.allclose(m.GetDirection(), ref.GetDirection())
    )
    if same:
        return m
    return sitk.Resample(m, ref, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)


def physical_z_per_slice(img: sitk.Image) -> np.ndarray:
    size = img.GetSize()
    x, y = size[0] // 2, size[1] // 2
    return np.asarray([img.TransformIndexToPhysicalPoint((x, y, z))[2] for z in range(size[2])])


def superior_is_small_index(img: sitk.Image) -> bool:
    z = physical_z_per_slice(img)
    return bool(z[0] <= z[-1])


def superior_edge(mask: sitk.Image) -> int:
    arr = sitk.GetArrayFromImage(mask).astype(bool)
    area = arr.reshape(arr.shape[0], -1).sum(1)
    nz = np.where(area > 0)[0]
    if not len(nz):
        raise RuntimeError("Empty anatomical mask")
    threshold = max(int(0.10 * area[nz].max()), 50)
    valid = np.where(area >= threshold)[0]
    if not len(valid):
        valid = nz
    return int(valid.min() if superior_is_small_index(mask) else valid.max())


def superior_slice_mask(mask: sitk.Image, threshold_index: int) -> sitk.Image:
    arr = sitk.GetArrayFromImage(mask).astype(np.uint8)
    out = np.zeros_like(arr)
    if superior_is_small_index(mask):
        out[: threshold_index + 1] = arr[: threshold_index + 1]
    else:
        out[threshold_index:] = arr[threshold_index:]
    img = sitk.GetImageFromArray(out)
    img.CopyInformation(mask)
    return img


def crop_and_mask(ct: sitk.Image, mask: sitk.Image, z0: int, z1: int, outside_hu=-1000) -> sitk.Image:
    if z1 < z0:
        z0, z1 = z1, z0
    ct_a = sitk.GetArrayFromImage(ct)
    m_a = sitk.GetArrayFromImage(mask).astype(bool)
    crop = np.where(m_a[z0:z1+1], ct_a[z0:z1+1], outside_hu).astype(np.int16)
    out = sitk.GetImageFromArray(crop)
    out.SetSpacing(ct.GetSpacing())
    out.SetDirection(ct.GetDirection())
    out.SetOrigin(ct.TransformIndexToPhysicalPoint((0, 0, z0)))
    return out


def prepare_patient(
    patient_dir: Path,
    output_dir: Path,
    ct_dirname: str = "CT",
    orientation: str = "RAI",
    spacing_mm: float = 1.0,
    outside_hu: int = -1000,
    fast: bool = False,
    include_extremities_above_t5: bool = False,
):
    ct_dir = patient_dir / ct_dirname
    sequence = select_series(ct_dir)
    image = read_longest_series(sequence)
    image = reorient(image, orientation)
    image = resample_isotropic(image, spacing_mm)
    image = clip_hu_int16(image)

    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pmct_cxr_") as td:
        work = Path(td)
        input_nii = work / "input_1mm.nii.gz"
        sitk.WriteImage(image, str(input_nii))
        body_dir, anatomy_dir = work / "body", work / "anatomy"
        body_dir.mkdir(); anatomy_dir.mkdir()
        run_totalsegmentator(input_nii, body_dir, task="body", fast=fast)
        roi = ["vertebrae_C7", "vertebrae_L3", "vertebrae_T5"]
        run_totalsegmentator(input_nii, anatomy_dir, task="total", roi_subset=roi, fast=fast)

        body_trunc = mask_on_reference(find_mask(body_dir, "body_trunc"), image)
        c7 = mask_on_reference(find_mask(anatomy_dir, "vertebrae_C7"), image)
        l3 = mask_on_reference(find_mask(anatomy_dir, "vertebrae_L3"), image)
        z0, z1 = superior_edge(c7), superior_edge(l3)

        body_mask = body_trunc
        if include_extremities_above_t5:
            ext = mask_on_reference(find_mask(body_dir, "body_extremities"), image)
            t5 = mask_on_reference(find_mask(anatomy_dir, "vertebrae_T5"), image)
            ext_sup = superior_slice_mask(ext, superior_edge(t5))
            body_mask = sitk.Cast(sitk.Or(body_trunc > 0, ext_sup > 0), sitk.sitkUInt8)

        prepared = crop_and_mask(image, body_mask, z0, z1, outside_hu=outside_hu)
        out = output_dir / "ct_body_C7toL3.nii.gz"
        sitk.WriteImage(prepared, str(out))
        return out
