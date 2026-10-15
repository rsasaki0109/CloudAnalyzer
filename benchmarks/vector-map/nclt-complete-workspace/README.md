# Resume NCLT graph inputs, edited maps and original evidence from one file

A production-browser roundtrip saves and reopens a 54,815,611-byte workspace ZIP
(52.3 MiB). It contains 1,077,680 current point records from the bundled April
pose-graph demo, 161,578 June display-preview records, a 199-node edited pose graph,
the HD map, the complete original April MCAP and the exact original June review ZIP.
Reopening needs no file reselection. The measured restoration took 35.4 seconds
on this execution environment; it is not a cross-browser performance benchmark.

The browser test verifies the reopened graph snapshot and editable HD JSON against
their pre-save values. The MCAP and review archive are byte-identical to the original
files. Every ZIP member and source fingerprint is checked. Both clouds retain
double XYZ, float intensity/correction and the preview distinction. The original
review package, its attribution and four frozen audits remain downloadable; saved
audit selection is disabled for the edited workspace.

The graph/demo and review pair are from separate drives and keep their original
local coordinate frames. This validates preservation, not cross-drive alignment.
The June review is a display-preview package: its original full point map remains
external. Frozen source audits still describe that original map pair. Archiving
them does not validate the April graph, new edits, traffic rules or accuracy.

`manifest.json` and `project.json` are the exact snapshot metadata; receipts,
the restored screenshot and `files-sha256.json` accompany them. The 53 MiB binary
ZIP stays in the execution workspace. Initial demo setup generates its map using
the existing demo workflow. Saving/reopening does not re-fuse the saved point
records, rebuild the native binary or spend an HD generation attempt; reopening
rebinds original graph inputs through the existing graph loader.

```sh
python benchmarks/vector-map/nclt-complete-workspace/verify_packet.py
python benchmarks/vector-map/nclt-complete-workspace/verify_packet.py \
  --zip /workspace/.cloudanalyzer-env/exports/nclt-complete-workspace.zip \
  --input web/public/samples/nclt-2012-04-29.mcap \
  --review /workspace/.cloudanalyzer-env/exports/nclt-hd-preflight-display-preview.zip
```

The test uses `CLOUDANALYZER_COMPLETE_WORKSPACE=1`,
`CLOUDANALYZER_REVIEW_ZIP` and `CLOUDANALYZER_WORKSPACE_OUTPUT` with
`workspace-assets.spec.ts`. Synthetic tests additionally cover edited map
preservation, original archived audit separation, constraints, duplicate scan
basenames with distinct content, changed inputs and legacy snapshots.

New snapshots keep the existing 64 MiB uncompressed/10 MiB metadata/127-cloud
limits and allow at most 2,048 ZIP members for small original scan files. Generated
review imports retain their 129-member limit. Undo and unloaded original cloud
detail are still external; browser storage can be evicted.

Derived NCLT data and screenshot retain ODbL-1.0 and DBCL-1.0 terms. See
[NCLT attribution](../../../web/public/samples/ATTRIBUTION.md).
