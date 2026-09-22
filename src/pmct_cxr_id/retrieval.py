from __future__ import annotations
import numpy as np
import torch


def topk_cosine_chunked(query_embeddings: np.ndarray, gallery_embeddings: np.ndarray, k: int, device: str, gallery_chunk=8192):
    """Return top-k gallery indices/scores without materializing the full similarity matrix."""
    q = torch.from_numpy(np.asarray(query_embeddings, dtype=np.float32)).to(device)
    best_scores = torch.full((q.shape[0], k), -float("inf"), device=device)
    best_indices = torch.full((q.shape[0], k), -1, dtype=torch.long, device=device)

    for start in range(0, len(gallery_embeddings), gallery_chunk):
        g_np = np.asarray(gallery_embeddings[start:start+gallery_chunk], dtype=np.float32)
        g = torch.from_numpy(g_np).to(device)
        scores = q @ g.T
        kk = min(k, g.shape[0])
        local_scores, local_idx = torch.topk(scores, k=kk, dim=1)
        local_idx += start
        if kk < k:
            pad_s = torch.full((q.shape[0], k-kk), -float("inf"), device=device)
            pad_i = torch.full((q.shape[0], k-kk), -1, dtype=torch.long, device=device)
            local_scores = torch.cat([local_scores, pad_s], 1)
            local_idx = torch.cat([local_idx, pad_i], 1)
        merged_s = torch.cat([best_scores, local_scores], 1)
        merged_i = torch.cat([best_indices, local_idx], 1)
        best_scores, choose = torch.topk(merged_s, k=k, dim=1)
        best_indices = torch.gather(merged_i, 1, choose)
    return best_indices.cpu().numpy(), best_scores.cpu().numpy()
