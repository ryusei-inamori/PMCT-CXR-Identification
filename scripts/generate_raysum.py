#!/usr/bin/env python3
import argparse
from pathlib import Path
import cv2
from pmct_cxr_id.raysum import make_raysum

p = argparse.ArgumentParser(description="Generate attenuation-weighted RaySum images")
p.add_argument("--input-root", type=Path, required=True)
p.add_argument("--output-root", type=Path, required=True)
p.add_argument("--input-name", default="ct_body_C7toL3.nii.gz")
p.add_argument("--size", type=int, nargs=2, default=(320,320), metavar=("H","W"))
p.add_argument("--no-clahe", action="store_true")
p.add_argument("--usm-amount", type=float, default=0.6)
p.add_argument("--overwrite", action="store_true")
a = p.parse_args()
a.output_root.mkdir(parents=True, exist_ok=True)
for case in sorted(x for x in a.input_root.iterdir() if x.is_dir()):
    src = case / a.input_name
    dst = a.output_root / f"{case.name}_raysum.png"
    if not src.exists():
        print(f"[SKIP] missing {src}"); continue
    if dst.exists() and not a.overwrite:
        print(f"[SKIP] exists {dst}"); continue
    img = make_raysum(src, output_size=tuple(a.size), use_clahe=not a.no_clahe, usm_amount=a.usm_amount)
    if not cv2.imwrite(str(dst), img):
        raise RuntimeError(f"Failed to write {dst}")
    print(f"[OK] {dst}")
