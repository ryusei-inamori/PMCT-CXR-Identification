from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import binomtest


def evaluate_topk(queries: pd.DataFrame, gallery: pd.DataFrame, top_indices: np.ndarray, top_scores: np.ndarray, ks=(1,5,10,20,30,40,50)):
    rows = []
    gallery_ids = gallery.identity_id.astype(str).to_numpy()
    gallery_paths = gallery.image_path.astype(str).to_numpy()
    for qi, q in queries.reset_index(drop=True).iterrows():
        idx = top_indices[qi]
        valid = idx >= 0
        idx = idx[valid]
        ranked_ids = gallery_ids[idx]
        ranked_paths = gallery_paths[idx]
        row = {
            "query_id": q.query_id,
            "identity_id": str(q.identity_id),
            "top1_identity_id": ranked_ids[0] if len(ranked_ids) else "",
            "top1_image_path": ranked_paths[0] if len(ranked_paths) else "",
            "top1_score": float(top_scores[qi][valid][0]) if valid.any() else np.nan,
        }
        hit_positions = np.where(ranked_ids == str(q.identity_id))[0]
        row["correct_rank"] = int(hit_positions[0] + 1) if len(hit_positions) else -1
        for k in ks:
            row[f"top{k}_correct"] = int(np.any(ranked_ids[:k] == str(q.identity_id)))
        rows.append(row)
    per_query = pd.DataFrame(rows)
    summary = {"n_queries": len(per_query)}
    for k in ks:
        summary[f"top{k}_rate"] = float(per_query[f"top{k}_correct"].mean()) if len(per_query) else np.nan
    return per_query, pd.DataFrame([summary])


def exact_mcnemar(a: pd.DataFrame, b: pd.DataFrame, ks=(1,5,10,20,30,40,50)) -> pd.DataFrame:
    merged = a.merge(b, on=["query_id", "identity_id"], suffixes=("_a", "_b"), validate="one_to_one")
    rows = []
    for k in ks:
        ca = merged[f"top{k}_correct_a"].astype(bool)
        cb = merged[f"top{k}_correct_b"].astype(bool)
        a_only = int((ca & ~cb).sum())
        b_only = int((~ca & cb).sum())
        n = a_only + b_only
        p = float(binomtest(a_only, n=n, p=0.5, alternative="two-sided").pvalue) if n else 1.0
        rows.append({"k": k, "a_correct_b_wrong": a_only, "a_wrong_b_correct": b_only, "discordant": n, "p_exact": p})
    return pd.DataFrame(rows)
