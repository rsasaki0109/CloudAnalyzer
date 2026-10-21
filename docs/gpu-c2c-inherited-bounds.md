# Inherited GPU split bounds

The GPU nearest-candidate traversal can retain an ancestor split lower bound when descending: the near child keeps that bound, and the far child keeps the maximum of it and the local split distance squared. The bound is a maximum, not a sum, because successive splits may constrain the same axis.

This remains a rounded f32 candidate search. Its hint is verified with the existing authoritative f64 tree search, including the full search when the hint cannot certify the result. The internal `gpuInheritedBounds` control defaults to false after the measured adoption gate failed. Backend selection and exact verification defaults are unchanged.

## Measurement

Four reports compare explicit local-only and inherited-bound controls through the same compiled production Worker. Each report has six warm-up pairs and twelve timed pairs in alternating order. Dataset order is PandaSet, Autoware, Autoware, PandaSet. All tails remain in the raw reports.

Each timed request includes a fresh query snapshot, GPU dispatch/readback, four streaming CPU verification workers, ordered statistics, f32 export, normal asynchronous release submission and result transfer. Parse/index, pool/device/reference/pipeline initialization and diagnostics are excluded. Batch size is 131,072 and workgroup size is 64; both coordinate-copy experiments remain off.

The predefined gate requires every report to show at least 1% median paired reduction and at least 7 of 12 faster pairs.

| Report | Local median, ms | Inherited median, ms | Paired change | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| pandaset 01 | 114.20 | 112.85 | -1.70% | 8/12 |
| autoware 01 | 171.90 | 174.35 | +2.09% | 5/12 |
| autoware 02 | 177.60 | 187.70 | +8.04% | 4/12 |
| pandaset 02 | 128.15 | 130.70 | +1.01% | 5/12 |

Inherited bounds were faster in 22 of 48 pairs. Only the first report passed its individual gate; three reports had positive median paired changes. This does not establish a stable reduction in total operation time, so the local-bound default remains. Paired changes use per-pair ratios rather than ratios of independently computed medians. Host GPU wait is not shader duration, and overlapping phase medians must not be summed.

## Correctness and provenance

Timing uses the same unchanged real PandaSet and Autoware even/odd subsets as the [preceding production study](wasm-owned-query-export.md). No generated geometry, offset, noise, registration, averaging or thinning is introduced. Source map hashes, source credits, exact input hashes and point counts are recorded in the specifications. Subsets of one measured cloud are not independent surveys or SLAM accuracy ground truth.

Every timed result checks all displayed f32 distance bits and ordered summary fields against its indexed-cloud CPU oracle. Certificate counts are stable separately for each control; in these reports they were also equal across controls and matched the preceding packet. Coordinate and displayed-distance hashes match that packet. Standalone shared-verifier studies used a different preparation layout and must not be pooled with these counts or ordering hashes.

Untimed browser cases compare full f64 verification with brute force for both shader controls, all four workgroup sizes, partial groups, overlapping readbacks, pipeline switches, leaf-only behavior and a failed optional pipeline followed by retry. The production Worker also recovers the failed pipeline through exact CPU fallback and retries on the shared device. Five selected cases passed on software GPU and two new cases passed on physical GPU. Generated edge fixtures are used only for correctness.

Rust sources are unchanged. The optional frontend builder reuses a prior WASM build only when the complete recorded Rust input set, all Rust hashes and all four WASM/glue/type artifacts match. It rebuilds JS/WGSL with current source and artifact fingerprints, and rejects changed Rust content, changed Rust source sets and changed WASM artifacts. This avoids a redundant Rust build and its disk cache. CI verifies the same route after a full WASM build.

## Reproduction

```sh
cd web
node scripts/build-query-export.mjs --output build.json
node scripts/benchmark-inherited-bounds.mjs --output report.json --build-info build.json \
  --reference reference.ply --compared compared.ply --input-manifest inputs.json \
  --threads 4 --warmups 6 --repeats 12 --gpu physical
```

When an unchanged, identified WASM build is already present, `node scripts/build-gpu-frontend.mjs --output frontend-build.json --wasm-build-info prior-build-or-manifest.json` can replace the full build. Outputs refuse overwrite. The harness verifies build/source/artifact/input hashes before and after measurement and records the unmodified inherited-bound default separately. `--gpu software` is a compatibility smoke. No adapter identity is collected.

[Packet manifest](../benchmarks/wasm-gpu-inherited-bounds/manifest.json) retains the frozen protocol, all four raw reports, source/build fingerprints and preserved PR340 sources needed to validate earlier packets.
