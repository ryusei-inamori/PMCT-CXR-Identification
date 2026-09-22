from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
import SimpleITK as sitk


def hu_to_relative_mu(hu: np.ndarray, mu_water: float = 0.02, bone_scale: float = 1.0, bone_threshold_hu: float = 250.0):
    mu = mu_water * (1.0 + hu / 1000.0)
    mu = np.clip(mu, 0.0, None)
    if bone_scale != 1.0:
        mu = mu.copy()
        mu[hu >= bone_threshold_hu] *= bone_scale
    return mu.astype(np.float32)


def attenuation_projection(volume_zyx: np.ndarray, spacing_zyx, axis: int = 1, mu_water=0.02, bone_scale=1.0, bone_threshold_hu=250.0):
    step_cm = float(spacing_zyx[axis]) / 10.0
    mu = hu_to_relative_mu(volume_zyx, mu_water, bone_scale, bone_threshold_hu)
    return np.sum(mu, axis=axis, dtype=np.float32) * step_cm


def robust_uint8(image: np.ndarray, p_low=1.0, p_high=99.0):
    lo, hi = np.percentile(image, [p_low, p_high])
    if hi <= lo:
        return np.zeros_like(image, dtype=np.uint8)
    x = np.clip(image, lo, hi)
    x = (x - lo) / (hi - lo)
    return np.round(x * 255).astype(np.uint8)


def finish_image(image: np.ndarray, use_clahe=True, clahe_clip=2.0, clahe_grid=(8, 8), usm_amount=0.6, usm_radius=1):
    out = image
    if use_clahe:
        out = cv2.createCLAHE(clipLimit=float(clahe_clip), tileGridSize=tuple(clahe_grid)).apply(out)
    if usm_amount > 0:
        k = int(usm_radius) * 2 + 1
        blur = cv2.GaussianBlur(out, (k, k), 0)
        out = cv2.addWeighted(out, 1.0 + float(usm_amount), blur, -float(usm_amount), 0)
    return out


def make_raysum(nifti_path: str | Path, output_size=(320, 320), use_clahe=True, usm_amount=0.6):
    img = sitk.ReadImage(str(nifti_path))
    vol = sitk.GetArrayFromImage(img).astype(np.float32)  # z,y,x
    sx, sy, sz = img.GetSpacing()
    projection = attenuation_projection(vol, (sz, sy, sx), axis=1)
    u8 = robust_uint8(projection)
    u8 = finish_image(u8, use_clahe=use_clahe, usm_amount=usm_amount)
    if output_size:
        h, w = output_size
        u8 = cv2.resize(u8, (w, h), interpolation=cv2.INTER_AREA)
    return u8
