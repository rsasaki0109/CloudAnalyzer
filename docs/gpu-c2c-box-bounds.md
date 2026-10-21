# GPU subtree box comparison

The internal `gpuBoxBounds` request selects a rounded GPU hint traversal with subtree boxes. The real-input comparison failed its adoption gate: all 48 primary pairs were slower, by 21.61–25.56% median paired whole-operation time. Ordinary GPU C2C keeps local split bounds. Backend selection and exact f64 verification remain unchanged.

The candidate builds six-component f32 boxes from the existing rounded tree-order points and preorder node ranges, then uploads one immutable `nodeCount * 24` byte buffer. Construction and upload are lazy and shared by concurrent requests for the same GPU reference. Switching traversal reuses the existing point/node buffers. A separate compute entry point loads the boxes and evaluates their combined squared lower bound after the existing split-plane precheck. The ordinary and near-leaf entry points require four bindings; only the box traversal uses the fifth binding. Leaf-only requests ignore the box flag.

No box buffer is constructed until requested. Once present, it stays cached with its reference and is released on invalidation. The normal and timestamp-enabled device variants can each retain one box buffer. `compute.boxBounds` identifies the active traversal, `boxBoundsBytes` reports retained box storage on that GPU reference, `boxBoundsReused` identifies a ready buffer used by the request, and `boxBoundsPrepareMillis` measures export/packing/upload readiness including device-scope scheduling. This host interval is not an isolated GPU transfer duration. Failed preparation releases partial allocation and permits retry; pipeline/readback failures keep the existing exact fallback.

## Real-input results

The [packet](../benchmarks/wasm-gpu-box-bounds/manifest.json) preserves four physical-GPU reports. PandaSet uses 531,908 reference and 531,908 query points; the Autoware sample uses 878,921 reference and 878,920 query points. These reproduce the previous study's input bytes exactly. Coordinates are unchanged decoded subsets, with possible geometric duplicates, rather than independent surveys or SLAM ground truth. Attribution, original source hashes and preparation details remain in the input specifications: PandaSet, Hesai and Scale AI (CC-BY-4.0), and the TIER IV Autoware planning sample with its original license note.

Each primary report has four exact CPU workers, batches of 131,072, six warm-up pairs and twelve timed pairs, alternating order. **Diagnostics are off** for adoption trials. The boundary is the ordinary production RPC with a fresh query snapshot, GPU dispatch/readback, streaming exact workers, ordered statistics, f32 export, normal release submission and response transfer. Parsing/indexing and initial device/reference/pipeline preparation are outside these warmed trials.

The predeclared gate requires all four reports to improve by at least 1% median paired time with at least 7/12 faster pairs, plus every fresh-reference box/local RPC ratio at most 1.05 and added box storage at most 16 MiB per GPU reference. It failed every warmed report.

| Report | Local median, ms | Boxes median, ms | Paired change | Faster pairs | Box bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| PandaSet 1 | 114.65 | 139.40 | +21.61% | 0/12 | 1,938,600 |
| Autoware 1 | 155.20 | 190.35 | +23.76% | 0/12 | 3,145,704 |
| Autoware 2 | 155.45 | 191.60 | +24.63% | 0/12 | 3,145,704 |
| PandaSet 2 | 109.20 | 137.50 | +25.56% | 0/12 | 1,938,600 |

Two additional fresh-reference operations per report include exact-tree preparation/import, base GPU buffers, optional boxes and the full RPC. Devices, pipelines and pool are already prepared; input parsing/indexing remains outside. Orders alternate across reports. These single samples are a conservative regression guard and descriptive setup evidence, not an estimate of cold speedup. All tails are retained.

| Report | Cold local RPC, ms | Cold boxes RPC, ms | Box readiness interval, ms |
| --- | ---: | ---: | ---: |
| PandaSet 1 | 781.70 | 320.80 | 30.70 |
| Autoware 1 | 443.90 | 921.80 | 37.10 |
| Autoware 2 | 415.40 | 547.80 | 60.00 |
| PandaSet 2 | 278.40 | 670.60 | 22.60 |

Two explanatory diagnostic-on pairs follow two diagnostic warm-up pairs in each report and are excluded from adoption timing. Their median sums of compute-pass timestamps were 66.29/95.62 ms (local/boxes) and 66.55/95.19 ms for PandaSet; 91.52/132.87 ms and 92.34/133.60 ms for Autoware. Instrumentation may perturb timing, phases overlap, and separate clocks cannot isolate transfer duration by subtraction. The box traversal adds global loads and arithmetic, but node-visit counts were not measured, so this study does not isolate the cause of its regression. Clocks, power, GC and other host activity were uncontrolled.

Every displayed f32 distance bit and ordered statistic matched the indexed CPU oracle. Certificate counts stayed stable and happened to match across variants: 235,714 for PandaSet and 421,096 for Autoware. Certificates describe verification without evaluating another leaf; they do not measure GPU candidate accuracy. Untimed tests also compare full f64 distances against brute force for 258 real queries, a measured exact hit and two overflow guards, including source destruction, independent uploads, both allocation/upload failure retries, four workgroup sizes, inherited-bound switches, buffered maps and concurrent mutation. The local path still allocates no box resources until a box request.

Rust and all four generated WASM package artifacts are unchanged. The frontend was rebuilt and fingerprinted; CI also runs a full WASM build. Four preserved prior source versions keep earlier packets verifiable without changing their reports.

```sh
node web/scripts/build-gpu-frontend.mjs --output NEW-build.json --wasm-build-info benchmarks/wasm-gpu-diagnostics/manifest.json
node web/scripts/benchmark-box-bounds.mjs --output NEW-report.json --build-info NEW-build.json --reference REAL-reference.ply --compared REAL-compared.ply --input-manifest REAL-inputs.json --threads 4 --warmups 6 --repeats 12 --gpu physical --cold-first local
```

Reuse requires exactly matching local generated WASM files; otherwise run the full `build-query-export.mjs` builder. Software-GPU smoke runs are compatibility evidence only. No adapter identity is recorded.
