# Keep a larger NCLT workspace and all original inputs in one file

A production-browser roundtrip adds the 588,267-point June motion candidate to
the previous complete April/June workspace and saves a 73,641,698-byte v3 ZIP
(70.2 MiB), containing 1,827,525 current point records. Reopening took 33.248
seconds on this execution environment; this is not a cross-browser performance
benchmark.
It retains three current point clouds, the 199-node graph, editable HD map and
reviews, complete original April MCAP and original June generated-map review ZIP.
The browser compares every exported cloud byte before/after reopening, graph and
map JSON, reviews, original asset bytes and the downloaded original map archive.
Saved source-audit controls remain disabled for the current edited workspace.

This validates preservation. The drives/candidate keep their existing independent
local frames; they are not registered into a common map. The HD map belongs to its
original June review pair and is not transferred onto the newly imported motion
candidate. Its old display preview remains a preview and its full source stays
external. This does not establish new motion accuracy or correctness of HD rules.

Manual snapshots now allow 256 MiB uncompressed content including manifest and
metadata. Browser recovery and generated-map review imports retain their separate
64 MiB limits. ZIP CRC32 uses 1 MiB read chunks; v3 file identities use the existing
8 MiB `sha256-chunks-v1` algorithm. Stored-member Blob slices avoid an extra full
member buffer. Unit tests bound these reads and validate CRC32 with Node's zlib;
a synthetic million-point, 12-extra-attribute cloud exercises manual saving and
exact reopening above 64 MiB, while browser autosave refuses that larger record.
Existing fault tests cover malformed records/graph inputs, cancellation, edits
during staging and preservation of current work and Undo. This is not a browser
RSS bound: native exports, loaded data and staging still require memory. Legacy
v1/v2 snapshots retain their original SHA-256 verification and limits.

`large-workspace-receipt.json`, `manifest.json`, `project.json` and
`files-sha256.json` preserve the evidence. The binary workspace stays outside the
repository. The stdlib verifier rechecks every ZIP CRC32 and chunk identity,
original cloud/asset hashes, graph/map/review state and all 588,267 full 32-byte
candidate records as a multiset. Viewer indexing changes point order and adds a
PLY comment; input-record comparison therefore includes duplicate multiplicities
but does not demand the original file order. Reopening preserves the saved
browser exports byte for byte. Receipts alone do not replace the binary checks.

```sh
python benchmarks/vector-map/nclt-large-workspace/verify.py
python benchmarks/vector-map/nclt-large-workspace/verify.py \
  --zip /path/to/project.cloudanalyzer.zip \
  --legacy /workspace/.cloudanalyzer-env/exports/nclt-complete-workspace.zip \
  --candidate /workspace/.cloudanalyzer-env/nclt-motion-trials/june/loops-only/pointcloud/original-keyframes_map.ply
```

The opt-in test is `web/e2e/large-workspace.spec.ts`, using
`CLOUDANALYZER_LARGE_WORKSPACE_BASE` and `CLOUDANALYZER_LARGE_WORKSPACE_CLOUD`.
It reuses existing immutable outputs: no new SLAM fusion, native rebuild or HD
generation attempt is spent on this roundtrip. The ordinary synthetic test runs
in web CI. Source derivation is documented in the
[complete workspace](../nclt-complete-workspace/README.md) and
[motion trials](../nclt-motion-trials/README.md).

Derived NCLT metadata retains ODbL-1.0 and DBCL-1.0 terms. See
[NCLT attribution](../../../web/public/samples/ATTRIBUTION.md).
