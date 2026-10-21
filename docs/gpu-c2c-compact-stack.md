# GPU stack capacity comparison

The internal `gpuCompactStack` request selects 20 ordinary DFS slots instead of 32. It failed all four warmed adoption gates on unchanged real inputs: median paired whole-operation changes were +0.33% to +3.98%, with 13/48 faster pairs. The 32-slot traversal remains default. Backend selection and authoritative f64 verification are unchanged.

The shader source factory changes only two function-local array lengths and their capacity guard. A test compares the emitted 32-slot control byte-for-byte with the archived preceding shader, then verifies that the 20-slot source differs only in those three literals. Traversal order, point/node storage, pruning, visit limits and exact CPU verification stay the same. The compact flag is ignored for leaf, box and parent entry points, and those requests reuse their ordinary cached pipelines. Inherited split bounds can still be tested with either ordinary stack capacity.

The Rust tree builder splits each internal range at its median and stops at 16 points or a coincident range. With the production limit of 2,000,000 points, the longest median-split path has depth 17; one deferred sibling per ancestor plus the current node requires at most 18 pending slots. Twenty retains two spare slots. The ordinary arrays declare 256/160 bytes per invocation for 32/20 slots. These logical sizes do not establish physical allocations, register counts, spills, occupancy or speed; those were not measured.

The two capacities share existing reference, query and readback buffers. No extra GPU reference buffer or CPU reference copy is retained. Optional box/parent storage stays at zero in both timing controls. A failed compact pipeline uses the existing complete CPU fallback and can retry. `compute.compactStack` reports whether the ordinary 20-slot variant was selected.

The [measurement packet](../benchmarks/wasm-gpu-compact-stack/manifest.json) retains four physical-GPU reports in PandaSet/Autoware/Autoware/PandaSet order. PandaSet contains 531,908 reference/query points; Autoware contains 878,921/878,920. The files reproduce the preceding study byte-for-byte, with unchanged decoded coordinates and possible geometric duplicates. They are subsets of measured clouds, not independent surveys or SLAM ground truth. Specifications retain attribution to PandaSet, Hesai and Scale AI (CC-BY-4.0), and the TIER IV sample with its original license note.

Each primary report uses four exact CPU workers, batches of 131,072, six warm-up pairs and twelve timed pairs with alternating order. Diagnostics are **off**. Timing covers the ordinary production RPC: fresh query snapshot, GPU dispatch/readback, streaming exact workers, ordered statistics, f32 export, normal release submission and response transfer. Parsing/indexing and initial pool/device/reference/pipeline preparation are excluded from warmed trials.

The predeclared gate requires all four reports to improve by at least 1% median paired time with at least 7/12 faster pairs, every fresh-reference 20/32-slot RPC ratio at most 1.05, and zero optional box/parent bytes in both controls. Every warmed gate failed.

| Report | 32-slot median, ms | 20-slot median, ms | Paired change | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| PandaSet 1 | 120.00 | 122.05 | +2.68% | 3/12 |
| Autoware 1 | 159.75 | 163.65 | +1.58% | 4/12 |
| Autoware 2 | 156.20 | 161.40 | +3.98% | 2/12 |
| PandaSet 2 | 127.85 | 128.45 | +0.33% | 4/12 |

Two fresh-reference operations per report include exact-tree preparation/import, base GPU buffers and the full RPC. Devices, pipelines and pool are already prepared; parsing/indexing is outside timing. Cold order alternates across reports. These single samples are descriptive setup evidence and a conservative regression guard, not estimates of cold speedup or its cause. All tails remain, including the 629.9 ms compact trial in PandaSet 2.

| Report | Cold 32-slot RPC, ms | Cold 20-slot RPC, ms |
| --- | ---: | ---: |
| PandaSet 1 | 278.90 | 634.80 |
| Autoware 1 | 429.40 | 386.30 |
| Autoware 2 | 719.20 | 442.40 |
| PandaSet 2 | 305.50 | 739.50 |

Two explanatory diagnostic-on pairs follow two diagnostic warm-up pairs in each report and are excluded from adoption. Median compute-pass timestamp sums were 66.81/69.04 and 66.16/68.81 ms (32/20 slots) for PandaSet, and 91.52/96.40 and 91.32/97.09 ms for Autoware. Instrumentation can perturb times; phases overlap and use separate clocks. Smaller declared arrays do not isolate a causal performance benefit. Clocks, power, GC and other host activity were uncontrolled.

Every displayed f32 distance bit and ordered statistic matched the indexed CPU oracle. Certificates stayed at 235,714 for PandaSet and 421,096 for Autoware in both controls; they describe verification without evaluating another leaf, not GPU hint accuracy. Untimed browser tests compare full f64 distances with brute force for 258 decoded real queries, one measured exact hit and two overflow guards across 21 checks. They cover four workgroup sizes, inherited switches, independent uploads, ignored flags on other entry points, cached buffer reuse, buffered/concurrent mutation/removal and failed pipeline fallback/retry. Selected existing failure tests also pass. Software GPU checks are compatibility evidence only.

Rust and all four generated WASM package artifacts are unchanged. Current frontend sources and artifacts were rebuilt and fingerprinted; CI also performs a full WASM build. Four archived source versions keep previous packets verifiable without changing earlier raw reports. No adapter identity is collected.

```sh
node web/scripts/build-gpu-frontend.mjs --output NEW-build.json --wasm-build-info benchmarks/wasm-gpu-parent-traversal/manifest.json
node web/scripts/benchmark-compact-stack.mjs --output NEW-report.json --build-info NEW-build.json --reference REAL-reference.ply --compared REAL-compared.ply --input-manifest REAL-inputs.json --threads 4 --warmups 6 --repeats 12 --gpu physical --cold-first stack32
```

Reuse requires matching generated WASM files and the complete unchanged Rust source set; otherwise use the full `build-query-export.mjs` builder.
