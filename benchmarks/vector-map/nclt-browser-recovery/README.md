# Processed NCLT records survive browser recovery

With **Include current point and mesh records (64 MiB)** enabled, Chromium loaded
the full NCLT generated-map review bundle, produced a 1% random subset and stored
the full map and processed result in one atomic IndexedDB recovery record. A new
page restored 646,309 + 6,463 points without selecting point source files or
rerunning the filter. Both native PLY exports after reopening were byte-identical
to the downloaded browser-copy ZIP members. All 21 saved HD lanes, visibility and
display names were retained. `restored.png` is the actual restored browser view.

Independent stdlib verification checks ZIP CRCs, all SHA-256 member hashes and
source-reference fingerprints. The full cloud's 32-byte record multiset is exactly
the original canonical PLY: double XYZ and float intensity/correction. The subset
contains only original records with valid duplicate multiplicities. Receipt paths
and hashes identify the large data retained outside git. No native binary or old
mapping run/job was changed; no mapping attempts or new source audits were spent.

```sh
cd web
PW_CHROMIUM_EXECUTABLE_PATH=/usr/bin/chromium npm run test:e2e -- processed-recovery.spec.ts autosave.spec.ts
CA_REVIEW_NCLT_BUNDLE=/path/to/nclt-hd-preflight-portable-review.zip \
  PW_CHROMIUM_EXECUTABLE_PATH=/usr/bin/chromium \
  npm run test:e2e -- processed-recovery.spec.ts --grep NCLT --workers=1
```

The optional real-data test writes `/tmp/nclt-browser-recovery.zip` and its actual
screenshot. Run the independent verifier from the repository root:

```sh
python3 -I -S benchmarks/vector-map/nclt-browser-recovery/verify.py \
  --bundle /path/to/nclt-browser-recovery.zip --source /path/to/local_map.ply
```

Without arguments it checks committed evidence hashes/receipt consistency only.
The synthetic browser cases additionally verify download/reopening of the saved
ZIP, restored mesh topology/signed C2M distances, quota-failure preservation and closing warnings, retry, and corrupt-record
rejection before changing the current workspace. Existing metadata-only recovery
and manual snapshot cases remain covered.

Point-record recovery is optional, off by default and bounded to 64 MiB for the
whole recovery record, including metadata and ZIP. Unloaded original density,
pose-graph inputs, Undo and the frozen generated-map source-audit reports remain
external. Unsaved review text stays in the recovery record rather than its ZIP.
Browser eviction or cleared site data can remove the copy; keep downloaded
snapshots and original attributed review bundles. This extends recovery, not map
accuracy: extent 124 / 252.586 m and 363 legacy height disagreements remain.

University of Michigan NCLT, session 2012-06-15; N. Carlevaris-Bianco, A. K. Ushani
and R. M. Eustice, *University of Michigan North Campus long-term vision and lidar
dataset*, IJRR 2016. Dataset: <http://robots.engin.umich.edu/nclt/>. This derived
database is offered under [ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/),
with contents under [DBCL 1.0](https://opendatacommons.org/licenses/dbcl/1-0/).
Keep this credit and the [original attribution](../../../web/public/samples/ATTRIBUTION.md)
when sharing these data; the generic workspace ZIP does not copy credits itself.
