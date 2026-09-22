#!/usr/bin/env python3
import argparse, json
from pmct_cxr_id.checkpoint import checkpoint_state, infer_architecture, sha256sum

p = argparse.ArgumentParser()
p.add_argument("checkpoint")
a = p.parse_args()
ckpt, state = checkpoint_state(a.checkpoint)
out = {
    "sha256": sha256sum(a.checkpoint),
    "epoch": ckpt.get("epoch"),
    "architecture": infer_architecture(state),
    "scheduler": ckpt.get("scheduler"),
}
print(json.dumps(out, indent=2, default=str))
