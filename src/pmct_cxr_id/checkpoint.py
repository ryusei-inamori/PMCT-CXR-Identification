from __future__ import annotations
from pathlib import Path
import hashlib
import torch
from .model import CNN


def sha256sum(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def checkpoint_state(path: str | Path):
    obj = torch.load(str(path), map_location="cpu")
    if not isinstance(obj, dict):
        raise TypeError("Expected a dict checkpoint")
    state = obj.get("model", obj.get("state_dict", obj))
    return obj, state


def infer_architecture(state: dict) -> dict:
    w0 = state["classifier.fc.0.weight"]
    w3 = state["classifier.fc.3.weight"]
    wm = state["classifier.metric.W"]
    return {
        "backbone_dim": int(w0.shape[1]),
        "hidden_dim": int(w0.shape[0]),
        "projection_output_dim": int(w3.shape[0]),
        "n_classes": int(wm.shape[0]),
        "metric_dim": int(wm.shape[1]),
    }


def load_model(path: str | Path, device: str | torch.device = "cpu") -> CNN:
    ckpt, state = checkpoint_state(path)
    arch = infer_architecture(state)
    if arch["backbone_dim"] != 1536 or arch["projection_output_dim"] != 1536:
        raise ValueError(f"Unsupported recovered architecture: {arch}")
    model = CNN(
        imagenet_init=False,
        n_classes=arch["n_classes"],
        hidden_dim=arch["hidden_dim"],
        dropout=0.5,
    )
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model
