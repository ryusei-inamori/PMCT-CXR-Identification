# PMCT-CXR-Identification

**Paper-compatible, post-publication reconstruction** of the pipeline described in:

> Ryusei Inamori, Tomoya Kobayashi, Takaya Kawasumi, Hidekazu Kanayama, Yeji Kim, Yongsu Yoon, Yohei Inaba, Akihito Usui, Eichi Takaya, Yoshikazu Okamoto.  
> **Development of a deep learning–based model for personal identification using postmortem CT and antemortem chest x-rays.**  
> *Forensic Science International*. 2026;388:113087.  
> https://doi.org/10.1016/j.forsciint.2026.113087

> **If this repository contributes to an academic study, please cite the FSI article above as the primary scientific reference.**

## What this repository is

This repository reorganizes archived research code and a surviving ChestX-ray14 pretraining checkpoint into a clean, reproducible public implementation. The core workflow is:

```text
PMCT DICOM
  -> 1.0-mm isotropic volume
  -> TotalSegmentator anatomy masks
  -> C7-to-L3 trunk extraction
  -> attenuation-weighted AP RaySum
  -> 320 x 320 grayscale input
                         \
                          -> EfficientNet-B3 + AdaCos encoder -> 1280-D paper-mode embedding
                         /
AM chest X-ray ----------
  -> 1.0-mm/pixel scale normalization during pretraining
  -> 320 x 320 grayscale input

Embeddings -> cosine retrieval -> Oldest / Nearest / All gallery evaluation -> Top-k
```

The published study included 1,385 deceased individuals and 78,999 non-matching CXRs. The article reported Top-1 = 83.0% and Top-5 = 90.6% for the all-examinations setting.

## Reproducibility status

This is **not claimed to be a byte-for-byte archive of every original analysis script**. Several archived components were recovered, but the final script that generated all three timing galleries was not available in the recovered set.

| Component | Status in this repository |
|---|---|
| EfficientNet-B3 model topology | Reconstructed from archived `cnn.py` and verified against checkpoint keys |
| AdaCos | Clean reimplementation compatible with the recovered state dict |
| ChestXray14 training recipe | Reconstructed from archived `cxr_pre.py` |
| 500-epoch checkpoint | Recovered; metadata and SHA-256 documented under `weights/` |
| 1280-D paper embedding | Paper-compatible interpretation of the recovered 1280-unit bottleneck; selectable 1536-D alternatives are retained |
| PMCT 1.0-mm isotropic resampling | Explicitly implemented to align the public pipeline with the reported preprocessing |
| C7-L3 localization | Reconstructed from archived TotalSegmentator code using C7/L3 masks |
| RaySum generation | Reconstructed from archived attenuation-weighted projection code |
| Oldest / Nearest / All galleries | Reconstructed from the article's category definitions using explicit metadata |
| Exact McNemar test | Implemented directly from paired Top-k outcomes |

See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) before using the code for replication claims. The recovered source hashes are listed in [`docs/SOURCE_MANIFEST.md`](docs/SOURCE_MANIFEST.md).

## Installation

Python 3.10+ is recommended.

```bash
git clone https://github.com/YOUR-USERNAME/PMCT-CXR-Identification.git
cd PMCT-CXR-Identification
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

PMCT segmentation additionally requires TotalSegmentator:

```bash
pip install -e '.[pmct]'
```

## Model weights

The recovered checkpoint is **248,403,337 bytes**, which is larger than GitHub's normal 100-MB file limit. Do not commit it directly.

Expected file:

```text
weights/epoch_000500.cpt
```

Expected SHA-256:

```text
9c1ca4c037ba5bde3f711f738dd22c688bfc8421493462397b44e794cd4d2fed
```

Recommended hosting: a GitHub Release asset, Git LFS, or a research archive. See [`weights/README.md`](weights/README.md).

Inspect a local checkpoint with:

```bash
python scripts/inspect_checkpoint.py weights/epoch_000500.cpt
```

## 1. Prepare PMCT volumes

Input layout:

```text
PMCT_ROOT/
  <identity_id>/
    CT/
      <series_1>/
        <DICOM files>
      <series_2>/
        <DICOM files>
```

Run:

```bash
python scripts/prepare_pmct.py \
  --root /path/to/PMCT_ROOT \
  --output /path/to/prepared_pmct
```

By default this implementation:

- selects the series with the largest number of DICOM images;
- reorients to RAI;
- resamples to 1.0-mm isotropic spacing;
- runs TotalSegmentator for `body_trunc`, C7, L3, and T5;
- crops from the superior edge of C7 to the superior edge of L3, matching the recovered landmark logic;
- masks voxels outside the trunk.

The archived preprocessing variant that adds `body_extremities` superior to T5 can be enabled with `--include-extremities-above-t5`.

## 2. Generate RaySum images

```bash
python scripts/generate_raysum.py \
  --input-root /path/to/prepared_pmct \
  --output-root /path/to/raysum
```

This produces 320 x 320 PNGs from attenuation-weighted AP line integrals. The archived CLAHE + unsharp-mask finishing is enabled by default and can be disabled.

## 3. Prepare metadata

### Query metadata

`queries.csv`

```csv
query_id,identity_id,image_path,pmct_date
q0001,ID0001,/data/raysum/ID0001_raysum.png,2015-08-14
q0002,ID0002,/data/raysum/ID0002_raysum.png,2014-03-02
```

### CXR gallery metadata

`gallery.csv`

```csv
identity_id,image_path,exam_date,is_distractor
ID0001,/data/cxr/ID0001/20100110_a.png,2010-01-10,0
ID0001,/data/cxr/ID0001/20140801_a.png,2014-08-01,0
DIST_000001,/data/distractors/a.png,,1
```

For study identities, `pmct_date` comes from `queries.csv`. For distractors, `exam_date` may be blank.

## 4. Run retrieval

```bash
python scripts/retrieve.py \
  --checkpoint weights/epoch_000500.cpt \
  --queries queries.csv \
  --gallery gallery.csv \
  --output results \
  --embedding paper1280 \
  --categories oldest nearest all
```

Embedding choices:

- `paper1280` — 1280-D bottleneck after the first projection + ReLU; default because the article reports a 1280-D embedding.
- `preadacos1536` — full projection-head output immediately before AdaCos.
- `backbone1536` — pooled EfficientNet-B3 feature used by some recovered post-hoc scripts.

The output includes per-query ranks, Top-k summaries, selected gallery manifests, and exact McNemar tests comparing Oldest vs Nearest when both are requested. The retrieval script expects already-prepared grayscale CXR images; the original upstream CXR export script was not recovered.

## Gallery category definitions

The public implementation makes the timing rules explicit:

- **oldest**: all frontal CXR rows on the earliest available exam date for each study identity;
- **nearest**: all CXR rows on the exam date closest to that identity's PMCT date;
- **all**: every available CXR row;
- distractor rows are included unchanged in every category.

Ties in nearest-date selection are resolved toward the earlier exam date and are recorded in the selected manifest.

## Pretraining from ChestX-ray14

The recovered checkpoint corresponds to 30,805 identity classes and 500 epochs. A cleaned training script is provided:

```bash
python scripts/pretrain_cxr.py \
  --image-root /path/to/ChestXray14/images \
  --dataset-csv /path/to/cxr14_patient_id.csv \
  --output-dir runs/cxr_pretrain
```

Required CSV columns:

```text
path,label,OriginalImage_Width,OriginalImage_Height,
OriginalImagePixelSpacing_x,OriginalImagePixelSpacing_y
```

The public defaults reflect the recovered checkpoint: 500 epochs, 1280-unit bottleneck, LR 0.02 with cosine annealing to 1e-4, adaptive SAM rho 2.0, SGD momentum 0.8, weight decay 5e-4, and label smoothing 0.1.

## Tests

```bash
pip install -e '.[dev]'
pytest -q
```

## Data and privacy

No clinical PMCT or CXR data are included. Do not commit DICOMs, identifiers, dates, or derived images that are not approved for public release.

## Research-use notice

This is research software, not an operational identity-verification system. Retrieval results should only be used as candidate prioritization and require appropriate forensic confirmation and governance.

## Citation

Please cite the original article:

```bibtex
@article{inamori2026pmctcxr,
  title   = {Development of a deep learning--based model for personal identification using postmortem CT and antemortem chest x-rays},
  author  = {Inamori, Ryusei and Kobayashi, Tomoya and Kawasumi, Takaya and Kanayama, Hidekazu and Kim, Yeji and Yoon, Yongsu and Inaba, Yohei and Usui, Akihito and Takaya, Eichi and Okamoto, Yoshikazu},
  journal = {Forensic Science International},
  volume  = {388},
  pages   = {113087},
  year    = {2026},
  doi     = {10.1016/j.forsciint.2026.113087}
}
```

## License

New code in this repository is released under the MIT License. External tools and datasets retain their own licenses and terms. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
