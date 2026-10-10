# Reusable filtering on the full NCLT map and its preview

The production browser applied the same explicit SOR (8 neighbors, ratio 1)
then 0.2 m voxel recipe to the immutable June map and its original display
preview. This is point-cloud preprocessing, not regeneration or an HD accuracy test.

| Source | Input records | Output records | Validation |
|---|---:|---:|---|
| Full June point map | 646,309 | 511,570 | Byte-exact match to individually applying both filters |
| June display preview | 161,578 | 142,476 | Byte-exact match to individually applying both filters; preview flag retained |

All output double XYZ and float intensity/correction record bytes occur in their
original source. Input and output native PLY digests and the actual WASM build
digest are retained in each processing record. The input digests describe native
exports; native indexing can reorder the original file's records, so they are not
substituted for the original-file digest.

One Undo restored both sources; Redo restored both results. The HD map remained
unchanged. The 51,601,723-byte workspace ZIP retained all four point clouds, both
processing records and the original attributed review archive. Reopening restored
the same result PLY bytes and processing JSON. The test explicitly set a 256 MiB
cloud Undo budget; the default 128 MiB was too small for this batch. A separate
test verifies insufficient budgets refuse application and retain previous Undo.
Cancellation, a later native step/source failure, concurrent editing and unsafe
small voxel ranges also preserve existing work in the fault tests.

The receipt's 13,028 ms spans recipe execution, hashes/export checks, workspace
save, Undo/Redo, reopen and export verification, excluding the initial individual
filter baselines. It is a single local measurement, not a throughput or size limit.
The point-record and source membership checks also run in the standalone verifier.
Filtering does not make the preview full density, approve traffic assumptions or
establish independent map accuracy. Frozen audits remain evidence for the original
map pair, not these filtered outputs.

Run the optional browser test in `web/`:

```sh
CLOUDANALYZER_RECIPE_PLY=/path/to/original/local_map.ply CLOUDANALYZER_RECIPE_REVIEW=/path/to/display-preview.zip npx playwright test filter-recipe.spec.ts --workers=1
```

Verify this packet with the unchanged external data and optional WASM build:

```sh
python verify.py /path/to/nclt-filter-recipe-workspace.zip /path/to/original/local_map.ply /path/to/display-preview.zip /path/to/ca_wasm_bg.wasm
```

The original full-map SHA-256 is
`944865739c31dd07e44c7c02f59f0f462dd5eef03d1eada6b0cce7d8cec8c7ad`;
the original review ZIP SHA-256 is
`85e3b22b7fb53a781cf72e356946f178a6efd1e9eda4e7e03ef06edbd96cdf2a`.
The unchanged WASM is
`83dd1c46396b4545ae247d570aa755c259e968cb7bee648b61d30231dc9faba9`.
The large ZIP is external; only the receipt, screenshot and verifier are committed.
No finished mapping job, native binary, original map/evidence or HD retry budget
was modified. See the [sample attribution](../../../web/public/samples/ATTRIBUTION.md).
