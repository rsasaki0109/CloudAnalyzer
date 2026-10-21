# GPU and exact CPU verification diagnostics

Internal C2C requests can set `gpuDiagnostics: true`. Ordinary requests leave it disabled. Timestamp features, query sets, timestamp buffers, extra readbacks and diagnostic timer calls are absent on the ordinary path. The backend selection, local split bounds, query copying and exact f64 authority stay unchanged.

Diagnostic requests use a separately cached device, negotiating `timestamp-query` only when supported. Both device variants can retain GPU reference buffers while sharing the exact reference and CPU worker caches. Reference invalidation releases both variants. Each of at most two readback slots owns one reusable two-entry query set and two 16-byte timestamp buffers. Both maps settle before the slot is reused, including either map failing. A failed diagnostic readback follows the existing exact CPU fallback and permits retry. Without timestamp support, GPU hints and CPU diagnostics continue with an explicit `unsupported` status.

`compute.diagnostics` records at most 256 GPU batches and 256 verification batches, with omitted-batch counters. Each verification batch has at most eight lane records. No adapter identity is recorded.

| Field | Meaning |
| --- | --- |
| `gpu[].shaderMillis` | Beginning/end compute-pass timestamp difference; excludes uploads and readbacks |
| `gpu[].queryExportMillis` | Host export of rounded query coordinates |
| `gpu[].uploadCallMillis` | Host `queue.writeBuffer` call duration |
| `gpu[].readbackMillis` | Candidate and timestamp mapping, copying and unmapping, including queued work and host scheduling |
| `verification[].queryExportMillis` | Host exact-coordinate export for the lane |
| `verification[].verifyCallMillis` | WASM verification call including imported argument copies |
| `verification[].distanceExportMillis` | WASM distance export into an owned JS array |
| `verification[].roundTripMillis` | Lane dispatch through response, including scheduling and transfers; local duration for a resident lane |

GPU and JS use separate clocks, and the GPU and CPU work overlap. Phase medians cannot be added or subtracted to isolate transfer time. Summed lane call durations are aggregate worker time, not elapsed operation time. Timestamp precision is implementation dependent; zero/quantized intervals are retained. See the [WebGPU timestamp specification](https://gpuweb.github.io/gpuweb/#timestamp-query) and [Chrome timestamp documentation](https://developer.chrome.com/blog/new-in-webgpu-121#timestamp_queries).

## Real-input measurements

[The packet](../benchmarks/wasm-gpu-diagnostics/manifest.json) contains four physical-GPU reports using the same byte-identical PandaSet and Autoware point subsets as the previous traversal study. Coordinates are unchanged. These are disjoint source-index subsets of measured maps, with possible geometric duplicates; they are not independent surveys or SLAM accuracy ground truth. Source hashes, attribution and preparation are included in the two input specifications.

Each report uses four exact workers, batches of 131,072, six warm-up pairs and twelve timed pairs, alternating order. Both GPU device/reference variants and the shared CPU cache are prepared before timed trials. The measured boundary is the complete ordinary production RPC with a fresh query snapshot, verification, ordered statistics, f32 export, ordinary release submission and response transfer. Every f32 output bit, summary field and GPU certificate count matches between controls; outputs also match the indexed CPU oracle. Untimed real-query checks cover full f64 parity and failure draining.

| Report | Off RPC median, ms | On RPC median, ms | Paired overhead median | On GPU pass sum median, ms | On host verification window median, ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| PandaSet 1 | 109.55 | 112.50 | +3.45% | 66.88 | 59.15 |
| Autoware 1 | 160.15 | 168.55 | +4.82% | 91.88 | 110.10 |
| Autoware 2 | 171.65 | 169.90 | +0.23% | 91.78 | 111.90 |
| PandaSet 2 | 114.95 | 122.75 | +4.78% | 66.32 | 73.20 |

Paired percentages are medians of within-pair ratios, so they need not match the ratio of column medians. All tails remain, including a 633.5 ms instrumented PandaSet operation. CPU/GPU clocks, power, GC and other host activity were uncontrolled. The diagnostic samples show substantial GPU traversal work alongside exact CPU verification; this supports investigating both. Host readback wait also includes command execution and scheduling. Instrumentation itself adds work and remains disabled by default. This change does not claim a production speedup.

The frontend was rebuilt against all current inputs while reusing four byte-identical WASM package artifacts and the complete unchanged Rust source set. CI also performs the full WASM build. Seven preserved source versions keep earlier measurement packets verifiable without rewriting their raw reports.

```sh
node web/scripts/build-gpu-frontend.mjs --output NEW-build.json --wasm-build-info benchmarks/wasm-gpu-inherited-bounds/manifest.json
node web/scripts/benchmark-gpu-diagnostics.mjs --output NEW-report.json --build-info NEW-build.json --reference REAL-reference.ply --compared REAL-compared.ply --input-manifest REAL-inputs.json --threads 4 --warmups 6 --repeats 12 --gpu physical
```

The reuse command requires matching local generated WASM files. Otherwise use `build-query-export.mjs --output NEW-build.json` for a full build. Smaller software-GPU runs are compatibility checks, not performance evidence.
