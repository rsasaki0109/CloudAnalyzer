# Direct owned JavaScript query export

The production WebGPU C2C pipeline can copy exact query coordinates directly from a checked WASM slice into a newly owned JavaScript `Float64Array`. This avoids the intermediate Rust `Vec<f64>` used by the ordinary iterator export. The existing two-argument `queriesF64` API is preserved.

The candidate remains an internal opt-in (`gpuQueryOwned`). Four reports did not meet the adoption gate, so the iterator and backend defaults remain unchanged.

## Scope and inputs

The same optimized production Worker and WASM compare explicit iterator and owned-copy controls. Each timed request includes a fresh query snapshot, GPU candidate dispatch/readback, four streaming exact CPU verification workers, ordered statistics, f32 export, ordinary asynchronous release submission and output transfer. Parsing/indexing, initialization and conformance diagnostics are excluded. Batch size is 131,072.

Inputs reproduce the byte-identical real PandaSet and Autoware even/odd point subsets from the preceding [query-export study](wasm-query-export.md). There is no generated geometry, offset, noise, averaging or thinning in timing. The subsets come from the same measured source cloud and are not independent surveys or SLAM accuracy ground truth. Source credits and hashes are in the input specifications.

Each dataset has two reports, in PandaSet / Autoware / Autoware / PandaSet order. Reports contain six warm-up pairs and twelve timed pairs, alternating execution order. The adoption gate was fixed before timing: all four reports must show at least 1% median paired reduction and at least 7 of 12 faster pairs. All timing tails are retained.

## Results

| Report | Iterator median, ms | Owned median, ms | Paired change | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| pandaset 01 | 113.95 | 114.55 | -0.34% | 6/12 |
| autoware 01 | 150.00 | 148.55 | +0.30% | 5/12 |
| autoware 02 | 164.75 | 161.75 | -1.35% | 7/12 |
| pandaset 02 | 115.20 | 114.75 | -0.48% | 6/12 |

Owned copy was faster in 24 of 48 pairs. Only one report met its individual gate. This does not establish a stable reduction in total operation time. Paired changes are medians of per-pair ratios, not ratios of separately computed medians. Maxima and per-request timings remain in the raw reports. Host activity, GC, clocks and power were uncontrolled; host GPU wait is not shader execution time.

## Correctness and ownership

Both software and physical GPU smokes check exact f64 coordinate bits, signed zero, subnormals, infinities and NaN payloads. An untimed four-point conformance fixture checks independent buffers, partial/empty ranges, ownership after source and job release, and transfer detachment without affecting the owner. Real-input diagnostics check all exported f64 bits and reject invalid/overflowing ranges.

Every production GPU request checks displayed f32 distance bits and all ordered summary fields against its indexed-cloud CPU oracle. Certificate counts agree across GPU controls and trials. All four reports retain the same coordinate hashes, displayed-distance hashes and certificate counts as the preceding normal-production study. The full f64 copy diagnostic is a coordinate check; it does not add a public f64 distance export.

The ordinary application indexes loaded clouds. Certificate counts and displayed-order hashes must not be pooled with earlier standalone shared-verifier experiments that used a different preparation layout.

## Reproduction

```sh
cd web
node scripts/build-query-export.mjs --output build.json
node scripts/benchmark-owned-query-export.mjs --output report.json --build-info build.json \
  --reference reference.ply --compared compared.ply --input-manifest inputs.json \
  --threads 4 --warmups 6 --repeats 12 --gpu physical
```

Outputs refuse overwrite. The harness verifies source, build, artifact and input hashes before and after measurement, records the unmodified application default separately, and caps CPU pool capacity before importing production code. `--gpu software` is a compatibility smoke. No adapter identity is collected.

[Raw packet](../benchmarks/wasm-owned-query-export/manifest.json) includes the frozen protocol, all four reports, input specifications, source fingerprint and build metadata. Preserved PR339 sources keep earlier packets verifiable without changing their raw evidence.
