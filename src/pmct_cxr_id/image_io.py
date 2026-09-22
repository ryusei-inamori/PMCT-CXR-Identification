from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
import torch


def read_gray(path: str | Path, size_hw=(320, 320)) -> np.ndarray:
    path = str(path)
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return cv2.resize(img, (size_hw[1], size_hw[0]), interpolation=cv2.INTER_AREA)


def gray_to_tensor(gray: np.ndarray, device=None) -> torch.Tensor:
    x = torch.from_numpy((gray.astype(np.float32) / 255.0)[None, None, ...])
    return x.to(device) if device is not None else x
