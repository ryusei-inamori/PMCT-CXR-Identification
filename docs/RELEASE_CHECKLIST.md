# Public release checklist

## Before making the repository public

- [ ] Confirm the final repository name, recommended: `PMCT-CXR-Identification`.
- [ ] Replace `YOUR-USERNAME` in `README.md`.
- [ ] Add your preferred contact method if desired.
- [ ] Confirm that no clinical images, DICOM headers, local absolute paths, patient IDs, or private URLs are present.
- [ ] Confirm that the recovered checkpoint may be redistributed under your institutional/data-use rules.
- [ ] If checkpoint redistribution is approved, upload `epoch_000500.cpt` as a GitHub Release asset or research-archive asset; do not commit it directly.
- [ ] Add the checkpoint download location to `weights/README.md`.
- [ ] Run `python scripts/inspect_checkpoint.py weights/epoch_000500.cpt` and verify the documented SHA-256.
- [ ] Run `pytest -q` in a clean environment.
- [ ] Run one end-to-end test on a small, approved local dataset.
- [ ] If possible, reproduce a small subset of the original study results before calling the repository an "official implementation" rather than a "post-publication reconstruction".

## Suggested GitHub repository metadata

**Description**

> Post-publication implementation of PMCT-derived RaySum to antemortem chest X-ray retrieval for forensic personal identification (Forensic Science International, 2026).

**Topics**

`forensic-imaging`, `postmortem-ct`, `chest-xray`, `personal-identification`, `medical-image-retrieval`, `deep-learning`, `adacos`, `efficientnet`, `totalsegmentator`, `disaster-victim-identification`

## Suggested first release

Tag: `v0.1.0`

Title: `Initial public reconstruction of the PMCT-CXR identification pipeline`

Attach, if redistribution is approved:

- `epoch_000500.cpt`
- optionally a small non-clinical synthetic demo bundle

Release notes should link the original article DOI and state that clinical data are not distributed.

## Create the repository locally

```bash
cd PMCT-CXR-Identification
git init
git add .
git commit -m "Initial public release"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/PMCT-CXR-Identification.git
git push -u origin main
```

## After publication

- [ ] Confirm that GitHub renders `CITATION.cff` and shows **Cite this repository**.
- [ ] Pin the repository on your GitHub profile.
- [ ] Add the GitHub URL to your ORCID/project page/institutional profile where appropriate.
- [ ] Add a short graphical overview to the README once a non-identifiable figure is available.
- [ ] Keep the FSI paper as the preferred citation in `CITATION.cff`.
