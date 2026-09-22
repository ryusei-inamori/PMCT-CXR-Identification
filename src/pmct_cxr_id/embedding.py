from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from .image_io import read_gray, gray_to_tensor


@torch.inference_mode()
def embed_paths(model, paths, device, mode="paper1280", batch_size=32, size_hw=(320,320)) -> np.ndarray:
    out = []
    for start in tqdm(range(0, len(paths), batch_size), desc="Embedding", leave=False):
        batch = []
        for p in paths[start:start+batch_size]:
            batch.append(gray_to_tensor(read_gray(p, size_hw=size_hw))[0])
        x = torch.stack(batch).to(device, non_blocking=True)
        z = F.normalize(model.encode(x, mode=mode), dim=1)
        out.append(z.cpu().numpy().astype(np.float32))
    if not out:
        return np.empty((0, 0), dtype=np.float32)
    return np.concatenate(out, axis=0)
