# NCLT portable review of the delivered point/HD pair

This records `export_mapping_run` followed by `inspect_mapping_bundle` over
the real MCP stdio server. It exports the finished
[HD preflight repair](../nclt-hd-repair-preflight/README.md), including the adopted
child's actual combined HD patch, rather than its parent's baseline map.
No lane generation, native source query, fusion or mapping attempt was performed.

- Exact point map: 646,309 points; SHA-256
  `944865739c31dd07e44c7c02f59f0f462dd5eef03d1eada6b0cce7d8cec8c7ad`.
- ZIP: 18,131,032 bytes; 22,976,636 bytes uncompressed; 15 distinct files.
- Source extent remains **124 / 252.5863381996826 m**. The 90% goal is unmet.
- Local connected route remains 220–226 m; global longest route remains 66 m.
- All four **saved** source audits remain exact. The legacy estimator's
  363 height mismatches and existing review lanes are retained, not repaired
  or reclassified by packaging. No independent accuracy is established.
- Root and child `run.json` / `job.json` bytes remain unchanged. No HD attempts
  are spent and no run revision changes.
- An independent interpreter launched with `-I -S`, given only a copy of the
  standard-library inspection module and relocated ZIP, verifies all members.
  It has no CloudAnalyzer installation or native-core import available.

`manifest.json` is the actual exported manifest. `receipt.json` records the MCP
request, ZIP hash/size, before/after state hashes, final extent/routes, all four
saved-audit summaries and the independent interpreter result. Original paths
inside these files are historical provenance, not receiver prerequisites.

The large ZIP and point map stay in the execution environment, outside git:

```text
/workspace/.cloudanalyzer-env/exports/nclt-hd-preflight-portable-review.zip
```

Verify the receipt/manifest packet:

```sh
python benchmarks/vector-map/nclt-review-bundle/verify_packet.py
```

Given the actual ZIP, also check every member's full bytes and the four saved
audit summaries, with no raw log or native processing:

```sh
python benchmarks/vector-map/nclt-review-bundle/verify_packet.py \
  --bundle /path/to/nclt-hd-preflight-portable-review.zip
```

The exporter includes exact point-map/graph/trajectory files, Lanelet2, editable
IR, projector, layout hypothesis, retained source proposal, full final audits,
final patch checks/comparison, continuation manifest and both decision histories.
The manifest identifies duplicate roles that share one member. See
[the delivery workflow](../../../docs/commands/mapping-run.md#deliver-a-portable-review-package)
for MCP/CLI use and map-opening instructions.

This is a portable **review** package, not a resumable mapping job. Archived
histories retain original paths and may refer to dependencies not included.
Raw logs, native binaries and previous jobs are excluded. Hash verification
checks bytes against the manifest, not authenticity, accuracy or deployment
readiness. Existing width/traffic assumptions and unresolved source intervals
remain in the saved layout, audits and diagnosis.

The input is the bounded NCLT session 2012-06-15 MCAP described in
[sample attribution](../../../web/public/samples/ATTRIBUTION.md). NCLT credit,
ODbL/DBCL terms and the derived-database notice are included in the ZIP manifest
and this packet. No new dataset or independent accuracy benchmark was introduced.
