#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from pmct_cxr_id.checkpoint import load_model
from pmct_cxr_id.embedding import embed_paths
from pmct_cxr_id.gallery import load_queries, load_gallery, select_category
from pmct_cxr_id.retrieval import topk_cosine_chunked
from pmct_cxr_id.evaluation import evaluate_topk, exact_mcnemar

p = argparse.ArgumentParser(description="PMCT RaySum to AM-CXR retrieval evaluation")
p.add_argument("--checkpoint", type=Path, required=True)
p.add_argument("--queries", type=Path, required=True)
p.add_argument("--gallery", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--embedding", choices=["paper1280","preadacos1536","backbone1536"], default="paper1280")
p.add_argument("--categories", nargs="+", choices=["oldest","nearest","all"], default=["oldest","nearest","all"])
p.add_argument("--batch-size", type=int, default=32)
p.add_argument("--gallery-chunk", type=int, default=8192)
p.add_argument("--device", default="cuda")
p.add_argument("--topk", type=int, nargs="+", default=[1,5,10,20,30,40,50])
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
device = a.device if (a.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
model = load_model(a.checkpoint, device=device)
queries = load_queries(a.queries)
gallery_master = load_gallery(a.gallery).reset_index(drop=True)

print("Embedding queries...")
q_emb = embed_paths(model, queries.image_path.tolist(), device=device, mode=a.embedding, batch_size=a.batch_size)

print("Embedding master gallery...")
g_emb = embed_paths(model, gallery_master.image_path.tolist(), device=device, mode=a.embedding, batch_size=a.batch_size)
path_to_index = {str(p): i for i, p in enumerate(gallery_master.image_path.astype(str))}

results = {}
for category in a.categories:
    print(f"Evaluating {category}...")
    selected = select_category(gallery_master, queries, category).reset_index(drop=True)
    selected.to_csv(a.output / f"gallery_{category}.csv", index=False)
    idx_master = np.array([path_to_index[str(p)] for p in selected.image_path], dtype=np.int64)
    emb = g_emb[idx_master]
    maxk = max(a.topk)
    inds, scores = topk_cosine_chunked(q_emb, emb, k=maxk, device=device, gallery_chunk=a.gallery_chunk)
    per_query, summary = evaluate_topk(queries, selected, inds, scores, ks=tuple(a.topk))
    per_query.to_csv(a.output / f"per_query_{category}.csv", index=False)
    summary.insert(0, "category", category)
    summary.insert(1, "embedding", a.embedding)
    summary.to_csv(a.output / f"summary_{category}.csv", index=False)
    results[category] = per_query
    print(summary.to_string(index=False))

if "oldest" in results and "nearest" in results:
    mc = exact_mcnemar(results["oldest"], results["nearest"], ks=tuple(a.topk))
    mc.insert(0, "comparison", "oldest_vs_nearest")
    mc.to_csv(a.output / "mcnemar_oldest_vs_nearest.csv", index=False)
    print(mc.to_string(index=False))
