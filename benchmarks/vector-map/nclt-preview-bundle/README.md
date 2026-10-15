# NCLT display-only generated-map review

Actual MCP `export_mapping_preview` and `inspect_mapping_bundle` calls exported
an immutable finished/adopted NCLT pair into a v2 display package. Source data,
license and attribution are retained in the manifest. Large data stays outside git.

- Original full point map: 646,309 records, 20,682,062 bytes; unchanged and outside ZIP.
- Every fourth original 32-byte record, starting at zero: 161,578 records. Double
  XYZ and float intensity/correction bytes match exactly, with no frame change.
- ZIP: 4,787,591 bytes; 7,466,203 uncompressed bytes; 15 verified members.
  The prior exact-full-map ZIP is 18,131,032 bytes (74% reduction in ZIP bytes).
- Exact delivered HD artifacts and all four original full-source saved audits.
  Root/child run and job hashes unchanged; no geometry regeneration, native
  processing, new audit query or mapping attempt spending.
- Independent `python -I -S` inspection works with only the inspector module and
  copied ZIP, without native core or source directories.
- Actual browser opens 161,578 display records and the retained HD draft, labels
  the original 646,309-point source, disables new source checks on the preview,
  focuses original failure locations and switches all four saved protocols.

The source extent stays 124 / 252.586 m, below the requested goal. Legacy checks
retain 363 height disagreements across nine lanes and 19 problem intervals;
ground-consensus checks retain their original results. This does not resolve
estimator disagreement, establish independent accuracy, verify traffic rules or
make the map ready for deployment. Sampling does not guarantee feature coverage.

Synthetic browser checks also save/reopen a project with the preview flag and
confirm that selecting a newly loaded full source enables fresh checks. Unit
checks stream a 72 MB original PLY to a bounded subset and compare all retained
attribute records across input chunk boundaries.

```sh
python benchmarks/vector-map/nclt-preview-bundle/verify_packet.py
python benchmarks/vector-map/nclt-preview-bundle/verify_packet.py \
  --bundle /path/to/nclt-hd-preflight-display-preview.zip \
  --source /path/to/original/local_map.ply
cd web
CA_REVIEW_NCLT_PREVIEW_BUNDLE=/path/to/nclt-hd-preflight-display-preview.zip \
  npm run test:e2e -- mapping-review-nclt.spec.ts
```

`manifest.json` and `receipt.json` record exact provenance; `review.png` is the
actual browser screenshot. Integrity checks detect changes relative to this
manifest; they are not authenticity signatures. Original artifacts and raw logs
are needed for mapping continuation; this ZIP is review-only.
