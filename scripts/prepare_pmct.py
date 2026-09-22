#!/usr/bin/env python3
import argparse
from pathlib import Path
from pmct_cxr_id.pmct import prepare_patient

p = argparse.ArgumentParser(description="Prepare C7-L3 PMCT volumes for RaySum generation")
p.add_argument("--root", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--ct-dirname", default="CT")
p.add_argument("--orientation", default="RAI")
p.add_argument("--spacing-mm", type=float, default=1.0)
p.add_argument("--outside-hu", type=int, default=-1000)
p.add_argument("--fast", action="store_true", help="Use TotalSegmentator --fast")
p.add_argument("--include-extremities-above-t5", action="store_true", help="Enable the archived T5-limited extremity-mask variant")
p.add_argument("--skip-existing", action="store_true")
a = p.parse_args()

a.output.mkdir(parents=True, exist_ok=True)
ok = fail = skip = 0
for patient in sorted(x for x in a.root.iterdir() if x.is_dir()):
    out_dir = a.output / patient.name
    target = out_dir / "ct_body_C7toL3.nii.gz"
    if a.skip_existing and target.exists():
        skip += 1
        continue
    try:
        prepare_patient(patient, out_dir, a.ct_dirname, a.orientation, a.spacing_mm, a.outside_hu, a.fast, a.include_extremities_above_t5)
        ok += 1
        print(f"[OK] {patient.name}")
    except Exception as e:
        fail += 1
        print(f"[FAIL] {patient.name}: {e}")
print(f"Done: success={ok}, skipped={skip}, failed={fail}")
