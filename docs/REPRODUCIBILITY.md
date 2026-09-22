# Reproducibility and provenance

## Why this is called a post-publication reconstruction

The recovered project directory contained several generations of research scripts rather than one frozen release. The public code therefore separates source-backed facts from reconstruction choices.

## Source-backed components

### Chest X-ray pretraining

Recovered `cxr_pre.py` documents:

- EfficientNet-B3 from torchvision;
- grayscale one-channel input;
- physical pixel-size normalization to approximately 1 mm/pixel before resizing to 320 x 320;
- random perspective and ±10-degree rotation;
- AdaCos identity classification;
- adaptive SAM with SGD;
- label smoothing;
- cosine learning-rate schedule;
- all available CXR images used for training.

The recovered checkpoint has:

- `epoch = 499` (500 completed epochs);
- 30,805 AdaCos classes;
- EfficientNet-B3 pooled feature width = 1536;
- projection `1536 -> 1280 -> 1536`;
- scheduler base LR = 0.02, `T_max = 500`, `eta_min = 1e-4`;
- adaptive SAM rho = 2.0;
- SGD momentum = 0.8;
- weight decay = 5e-4.

### C7-L3 preprocessing

Recovered `ts_raysum.py` explicitly obtains TotalSegmentator masks for `vertebrae_C7`, `vertebrae_L3`, and `vertebrae_T5`, then crops from the superior edge of C7 through the superior edge of L3.

### RaySum

Recovered `raysum_all.py` converts HU values to a relative attenuation coefficient and integrates through the AP/PA axis, followed by percentile normalization, CLAHE, and optional unsharp masking.

## Reconstruction choices

### 1280-D embedding

The article reports a 1280-dimensional embedding. The recovered checkpoint contains a 1280-unit bottleneck inside an otherwise 1536-dimensional AdaCos head. Some later recovered matching scripts used the 1536-D backbone output directly.

Therefore the public default `paper1280` is defined as:

```text
EfficientNet pooled feature (1536)
  -> Linear(1536,1280)
  -> ReLU
  -> L2 normalization
```

The alternatives `preadacos1536` and `backbone1536` are exposed explicitly so that this interpretation can be audited rather than hidden.

### 1.0-mm isotropic PMCT resampling

The recovered C7-L3 segmentation script did not explicitly resample the PMCT volume. The public implementation adds an explicit 1.0-mm isotropic resampling step because this was part of the reported PMCT preprocessing workflow. This is marked as a reconstruction rather than an archived line-for-line operation.

### Gallery timing categories

The final original script used to create all timing categories was not recovered. This repository reconstructs them directly from metadata:

- `oldest`: earliest exam date for each identity;
- `nearest`: exam date with minimum absolute difference from that identity's PMCT date;
- `all`: all examinations.

All images from a selected exam date are retained. If two dates are equally close to PMCT, the earlier one is chosen and recorded.

### Body mask variant

The recovered `ts_raysum.py` optionally combines `body_trunc` with extremity voxels superior to T5. The public default is `body_trunc` only because it is the simpler trunk-only interpretation; the archived T5-limited variant is available as an explicit flag.

## Recommended wording in papers

Until numerical equivalence to archived study outputs has been confirmed, use wording such as:

> "We used the authors' post-publication reconstruction of the published PMCT-to-CXR retrieval pipeline."

Do not state that a re-run is an exact replication unless the expected study-level metrics and sample manifests have been independently verified.

## CXR inference preprocessing caveat

The recovered ChestXray14 training script performs physical scale normalization using pixel-spacing metadata before the final 320 x 320 resize. The recovered later matching scripts, however, read already-exported grayscale images and only resize them to 320 x 320; the upstream export/preprocessing script for the study CXR gallery was not recovered.

For that reason, `scripts/retrieve.py` expects **already prepared grayscale CXR images**. Do not claim that the repository reconstructs the original CXR export step unless that source is recovered and added.

## Source audit

The public implementation was reconstructed from archived source files and one model checkpoint. Their SHA-256 hashes are recorded in [`SOURCE_MANIFEST.md`](SOURCE_MANIFEST.md) so future revisions can be traced to the same recovered materials.
