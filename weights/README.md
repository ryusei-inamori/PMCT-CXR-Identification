# Recovered model checkpoint

Expected filename:

```text
epoch_000500.cpt
```

Metadata from the recovered file:

- size: `248403337` bytes
- SHA-256: `9c1ca4c037ba5bde3f711f738dd22c688bfc8421493462397b44e794cd4d2fed`
- saved epoch: `499`
- AdaCos classes: `30,805`
- projection: `1536 -> 1280 -> 1536`

The checkpoint is not included in the Git repository package because it exceeds GitHub's standard 100-MB per-file limit.

Recommended options:

1. attach it to a GitHub Release;
2. use Git LFS; or
3. archive it in Zenodo/another research repository and place the stable link here.

After downloading, verify:

```bash
sha256sum weights/epoch_000500.cpt
```
