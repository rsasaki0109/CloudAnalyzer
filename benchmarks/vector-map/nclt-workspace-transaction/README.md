# Prepare a complete NCLT workspace before committing

This production-browser test reuses the immutable 54,815,611-byte ZIP from
[the complete workspace roundtrip](../nclt-complete-workspace/README.md), SHA-256
`c918ad9dfd2a9972ee385d9e9cf2c2956a3a7cf2fc5d4ece348b7a65177b7db4`.
It prepares all native point records, saved HD map and the 199-node pose graph
with its original MCAP before committing their views. It does not regenerate a
point map, fuse scans again, rebuild native binaries or spend an HD repair attempt.

The two restored PLY exports are byte-identical to the archived members:
1,077,680 April map points and 161,578 June display-preview points, each with double
XYZ and float intensity/correction attributes. Graph and HD JSON match the saved
state. The original review archive can be downloaded; its frozen audit controls
remain disabled for current edits. The recorded 89,134 ms spans import, graph/map
checks and both complete PLY exports, not import time alone or a performance limit.

Separate fault tests construct ZIPs with valid hashes but a malformed later PLY
or graph, cancel twice after a first PLY is staged, and edit current work during
staging. They verify unchanged point exports, map/graph state and usable Undo.
Duplicate display names are refused, and matching extra inputs can precede the ZIP.
Legacy snapshots wait for their external graph inputs before native preparation.

Run the opt-in browser test in `web/`:

```sh
CLOUDANALYZER_TRANSACTION_ZIP=/path/to/nclt-complete-workspace.zip npx playwright test workspace-transaction.spec.ts --workers=1
```

Run `python verify.py /path/to/nclt-complete-workspace.zip` to check the packet
hashes, original archive identity, every member and restored-export receipt.
The PNG and receipt are small review evidence; the large unchanged ZIP is external.
Neither this preservation test nor the separate-drive workspace establishes
cross-drive alignment, independent map accuracy or traffic-rule correctness.
Preparation holds both workspaces temporarily; WASM heap allocations remain
reusable after release. Failures before commit retain current work; crashes,
post-commit rendering or later recomputed analyses are outside that guarantee.
