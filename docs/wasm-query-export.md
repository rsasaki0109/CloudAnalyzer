# Contiguous exact-query export in the production GPU pipeline

Contiguous slice cloning did not consistently improve complete WebGPU C2C operations. Across four real-file reports, the median paired change ranged from **2.16% shorter to 2.12% longer**, with **21/48 faster pairs**. The predeclared adoption gate failed. The ordinary iterator implementation and backend selection remain the defaults; contiguous copy is an explicit internal comparison control.

## Change and compatibility

`GpuC2cJob` already owns a contiguous `Vec<[f64; 3]>`. The candidate replaces `iter().flatten().copied().collect()` with `as_flattened().to_vec()` when exporting exact f64 coordinates for streaming verifier workers. Both paths still allocate an owned Rust result and copy it across the WASM/JS boundary. This experiment removes iterator work; it does not eliminate those allocations or transfers. No shader, search arithmetic or summary order changes.

The original two-argument Rust/JS `queriesF64` API retains its iterator implementation. `queriesF64WithCopy(first, count, true)` selects the candidate; false selects the iterator. The production Worker request accepts internal `gpuQuerySlice`, which defaults false after the failed gate. No new UI setting is added. Native tests preserve signed zero, subnormals, NaN payloads, infinities, valid partial/empty ranges, ownership after source removal and independence of exported buffers. Browser diagnostics check exact f64 coordinate bits and invalid ranges.

## Actual operation boundary

The harness runs the ordinary production distance Worker, rather than the previous standalone shared verifier. Each timed RPC includes a fresh query snapshot, rounded query packing, actual GPU dispatch/readback and prefetch, four exact CPU verifier lanes with streaming queries, ordered summaries, f32 export, normal asynchronous release submission and result transfer. Loading, pool startup, initial device/reference preparation and correctness diagnostics are excluded. Reference/device/worker caches must be reused in every warm and measured request; GPU or verifier fallback is rejected. Both controls use the same compiled WASM, worker pool, bounds/SIMD verification, 131,072-point GPU batches and summaries.

The harness caps hardware concurrency to five in a measurement-only wrapper before importing the production Worker, giving four pool lanes. It does not change the application concurrency policy. Both query-copy settings are explicit during timing; a separate untimed request records and checks the application default.

Application loading builds its ordinary index before creating the GPU job. Earlier standalone shared experiments parsed clouds directly. Candidate slots, certificate counts and displayed-order hashes must therefore not be pooled across those preparations. The full f64 coordinate diagnostic uses raw parsed order; timed GPU outputs are checked against an exact CPU oracle using the same indexed application clouds. This is not a new full-f64-distance export benchmark.

## Inputs, protocol and results

The input files reproduce the previous real even/odd point subsets byte-for-byte: PandaSet has 531,908 reference and 531,908 query points; the Autoware planning sample has 878,921 reference and 878,920 query points. Every loaded count matches its declared file count, with no benchmark thinning or generated coordinates. Original maps are unchanged. Credit: PandaSet, Hesai and Scale AI (CC-BY-4.0), and TIER IV’s Autoware sample; source URLs and attribution remain in the input specifications. These subsets are not independent surveys or SLAM-accuracy ground truth.

Each report has six warmup pairs and twelve measured pairs, alternating which copy mode runs first. Both modes occupy each position six times. The predeclared report order was PandaSet, Autoware, Autoware, PandaSet. The four-report budget is exhausted; all trials and long waits are retained. Adoption required at least a 1% median paired reduction and 7/12 faster pairs in every report.

| Report | Iterator median (ms) | Slice median (ms) | Slice / iterator paired change | Faster pairs |
|---|---:|---:|---:|---:|
| pandaset-4-01 | 139.80 | 135.70 | -2.16% | 8/12 |
| autoware-4-01 | 165.55 | 165.05 | +1.79% | 3/12 |
| autoware-4-02 | 166.70 | 168.85 | +0.24% | 6/12 |
| pandaset-4-02 | 128.60 | 128.20 | +2.12% | 4/12 |

Negative paired change means shorter duration. Paired percentages are medians of within-pair ratios, so they can differ in direction from a ratio of the separate medians. The first PandaSet report favoured slice copying; the other three did not. Requests near 634 ms remain in both controls in the first report. Host `gpuWaitMillis` is not shader execution time and does not identify the cause of those tails. Other host activity, clocks, GC and power were uncontrolled; no portable speed guarantee or memory reduction is established.

All requests preserved every displayed f32 bit, every reported ordered statistic and certificate counts against the indexed-cloud CPU oracle and matched control. Per-report full f64 coordinate-copy diagnostics also passed. The four reports use the candidate with slice as its default. After the failed gate, the original iterator default was restored and separately rebuilt and checked on software and physical GPUs. Those final packaging smokes are correctness evidence, with no additional timing reports or final-default speedup claim.

## Evidence and reproduction

The [manifest](../benchmarks/wasm-query-export/manifest.json) retains four raw reports, two exact input specifications, 183 measured source fingerprints, compiler/build/artifact provenance and the final disabled-default build record. Four measured candidate sources are preserved separately from the final source; five PR338 sources retain the provenance of all earlier shared GPU packets. The integrity tests verify the measured and final source versions and recompute the failed adoption gate.

Build the standard production package and record its inputs from the repository root:

```sh
node web/scripts/build-query-export.mjs --output NEW-build.json
node web/scripts/benchmark-query-export.mjs --output NEW-report.json --build-info NEW-build.json --reference reference.ply --compared compared.ply --input-manifest inputs.json --threads 4 --warmups 6 --repeats 12 --gpu physical
node --test web/tests/shared-gpu-c2c-order.test.mjs web/tests/shared-gpu-packets.test.mjs web/tests/query-export-packet.test.mjs
```

The builder runs the standard SIMD release WASM, TypeScript and Vite builds, checks that sources stay fixed, and fingerprints both the diagnostic and bundled WASM. It refuses existing metadata output. If using already installed matching tools locally, `CA_WASM_PACK_NO_INSTALL=1` selects wasm-pack’s no-install mode; wasm-bindgen and wasm-opt must already be on PATH. `CARGO_TARGET_DIR` can isolate build caches.

For the exact timed source version, restore the manifest’s measured-source aliases in a separate checkout before building. The current default is deliberately different. Current builds can still compare both explicit controls, but their timings must be recorded as new evidence. `--gpu software` is a compatibility smoke, and is never labelled physical-GPU timing evidence. Reports are never overwritten; no adapter identity is collected.

CI builds the recorded production package, checks both copy paths on software WebGPU, and retains the existing shared-verifier isolation/fallback smokes. Native tests, normal E2E coverage and packet integrity checks remain enabled. A separate future experiment could investigate avoiding the temporary Rust export vector; this study does not establish such a benefit.
