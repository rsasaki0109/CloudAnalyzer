# GPU parent traversal comparison

The internal `gpuParentTraversal` request enables a bounded stackless hint traversal. It failed all four warmed adoption gates on the real inputs: median paired whole-operation changes were +3.69% to +11.54%, with 4/48 faster pairs. Ordinary GPU C2C retains local split bounds. Exact f64 verification and backend selection are unchanged.

Each preorder node has one immutable u32 parent; the root uses a sentinel. Current and previous node indices distinguish descent from return. The kernel visits the near child first and checks the existing split plane before visiting its sibling. It has no explicit per-invocation DFS arrays, and stops after 4,096 entered nodes or 12,288 steps. An unfinished search still provides only a hint for authoritative CPU verification. No register occupancy or spills were measured.

Parent storage is `nodeCount * 4` bytes, one sixth of the box experiment's storage. It is created only when requested, with concurrent preparation shared and partial allocations released on failure. Initial activation reuses the original node export; a later activation may re-export candidate buffers. Existing GPU points/nodes and exact CPU reference caches are shared across modes. Local and leaf-only requests create no parent buffer. Once created it remains with the reference until invalidation; ordinary and diagnostic device variants may each retain one. Leaf-only hints ignore the flag, parent traversal ignores inherited bounds, and simultaneous box/parent requests take the exact CPU fallback.

`compute.parentTraversal` identifies the active mode, `parentTraversalBytes` reports retained storage, and `parentTraversalReused` identifies a buffer ready before the request. `parentTraversalPrepareMillis` includes host export/packing/upload readiness and serialized device-scope scheduling. It is not an isolated transfer duration.

The [measurement packet](../benchmarks/wasm-gpu-parent-traversal/manifest.json) contains four physical-GPU reports in PandaSet/Autoware/Autoware/PandaSet order. Inputs exactly reproduce the preceding study: 531,908 reference/query points for PandaSet and 878,921/878,920 for the Autoware sample. They are unchanged subsets of measured clouds with possible geometric duplicates, not independent surveys or SLAM ground truth. Specifications retain original hashes and attribution to PandaSet, Hesai and Scale AI (CC-BY-4.0), and the TIER IV sample with its original license note.

Each report uses four exact CPU workers, batches of 131,072, six warm-up pairs and twelve alternating timed pairs. Diagnostics are **off**. Timing covers the ordinary production RPC: fresh query snapshot, GPU dispatch/readback, exact streaming verification, ordered statistics, f32 export, normal release submission and response transfer. Parse/index and initial pool/device/reference/pipeline preparation are excluded from warmed trials.

The predeclared gate requires every report to improve by at least 1% median paired time with at least 7/12 faster pairs, every fresh-reference parent/local RPC ratio at most 1.05, and added parent storage at most 16 MiB per GPU reference.

| Report | Local median, ms | Parents median, ms | Paired change | Faster pairs | Parent bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| PandaSet 1 | 115.10 | 125.40 | +9.44% | 0/12 | 323,100 |
| Autoware 1 | 168.00 | 178.25 | +3.69% | 2/12 | 524,284 |
| Autoware 2 | 160.70 | 172.65 | +10.64% | 1/12 | 524,284 |
| PandaSet 2 | 125.30 | 136.55 | +11.54% | 1/12 | 323,100 |

Two fresh-reference operations per report include exact-tree preparation/import, base GPU buffers, optional parents and the full RPC. Devices, pipelines and pool are already prepared; parsing/indexing is outside timing. Cold order alternates across reports. These single samples are descriptive setup evidence and a conservative regression guard, not cold speedup estimates. PandaSet 1 failed the cold guard as well. All tails remain, including the 708.3 ms parent trial in PandaSet 1.

| Report | Cold local RPC, ms | Cold parents RPC, ms | Parent readiness interval, ms |
| --- | ---: | ---: | ---: |
| PandaSet 1 | 245.10 | 269.60 | 9.50 |
| Autoware 1 | 479.40 | 467.60 | 8.00 |
| Autoware 2 | 866.60 | 881.10 | 6.00 |
| PandaSet 2 | 285.60 | 273.60 | 4.90 |

Two diagnostic-on explanatory pairs follow two diagnostic warm-up pairs and are excluded from adoption. Median compute-pass timestamp sums were 66.75/81.56 and 66.13/80.74 ms (local/parents) for PandaSet, and 91.36/107.02 and 91.65/106.59 ms for Autoware. Instrumentation can perturb times; phases overlap and use separate clocks. Additional parent/node reads are implementation work, but this study does not isolate the cause of the regression. Clocks, power, GC and other host activity were uncontrolled.

Every displayed distance bit and ordered statistic matched the indexed CPU oracle. Certificates stayed at 235,714 for PandaSet and 421,096 for Autoware in both modes; they describe verification without another leaf, not hint accuracy. Untimed browser tests also check full f64 distances against brute force for 258 decoded real queries, one measured exact hit and two overflow guards, independent uploads after source destruction, four workgroup sizes, failed allocation/upload and pipeline retries, buffered maps, cached mode switches and concurrent mutation/removal. Software GPU checks are compatibility evidence only.

Rust and all four generated WASM package artifacts are unchanged. Current frontend sources were rebuilt and fingerprinted; CI also performs a full WASM build. Four archived source versions preserve older packet verification without modifying prior raw reports. No adapter identity is collected.

```sh
node web/scripts/build-gpu-frontend.mjs --output NEW-build.json --wasm-build-info benchmarks/wasm-gpu-box-bounds/manifest.json
node web/scripts/benchmark-parent-traversal.mjs --output NEW-report.json --build-info NEW-build.json --reference REAL-reference.ply --compared REAL-compared.ply --input-manifest REAL-inputs.json --threads 4 --warmups 6 --repeats 12 --gpu physical --cold-first local
```

Reuse requires matching generated WASM files and the complete unchanged Rust source set; otherwise use the full `build-query-export.mjs` builder.
