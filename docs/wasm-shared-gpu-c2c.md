# Shared exact GPU candidate verification experiment

An optional WASM/Rayon verifier imports one exact reference tree and shares it across verification threads. The production application keeps its existing worker pool and isolation policy.

Four counterbalanced real-file reports show paired median reductions of 2.1-4.8% against streaming pool verification and 1.5-4.8% against staged pool verification in the prepared-job GPU pipeline. Fixed-hint results against the staged pool change direction between repeats. These small effects and variable host waits do not justify a default change.

**Scope:** imports, owner query snapshot creation and query staging occur once, outside timed requests. Each report reuses one immutable job. The regular application creates query jobs per C2C operation. These results do not establish a complete application-operation speedup.

## Implementation and controls

The [standalone crate and build instructions](../web/experiments/shared-gpu-c2c/README.md) use `CandidateKdTree::from_snapshot` to preserve production GPU tree-order slots and rebuild exact f64 certificates. Rayon shares a read-only tree and owned finite query coordinates. Each distance output has a unique mutable owner; certified counts use integer reduction. Invalid hints still use complete exact search. Failed replacement preserves the old snapshot, and release follows synchronous verification.

There are independent stable serial and pinned nightly atomic/SIMD builds, with rebuilt `std`, wasm-bindgen-rayon 1.3.0 and a 1 GiB shared-memory maximum. No production cache integration, cancellation recovery, device-loss recovery or automatic backend selection is added. The driver serializes operations and drains prefetched GPU work.

The controls are production streaming verification, the same pool with queries staged once, serial verification in the threaded binary, and shared parallel verification in that same binary. Production comparisons include compiler, allocator, layout and message-path differences; they do not isolate sharing alone. Stable serial is an untimed correctness/fallback control.

## Real inputs and checks

- PandaSet 019: 1,063,816 measured source points; reference/query counts 531,908 each. Float32 source values are exported as f64 without restoring lost precision. Attribution: Hesai and Scale AI, CC-BY-4.0.
- Autoware planning sample: 1,757,841 measured source points; reference 878,921 and query 878,920. Attribution: TIER IV; the original license note is preserved without asserting an additional survey license.
- Both pairs use alternating decoded source indices and unchanged XYZ. File hashes match the preceding [shared CPU experiment](wasm-shared-c2c.md). They are subsets of one measured cloud, not independent surveys or SLAM accuracy ground truth. No downloads or generated geometry were used.
- All eight reports initialize four actual shared-memory lanes, reject unisolated threaded initialization, initialize serial fallback, preserve failed replacements, reject invalid ranges and verify release.
- Untimed checks compare every f64 bit with complete production search and with all verification controls. Every timed trial compares f32 bits, ordered f64 summary fields and certified counts. Candidate, f64 and f32 hashes are identical per dataset across all reports.

## Timing protocol

A header-free static localhost server uses `/CloudAnalyzer/app/shared-gpu-c2c/`; only the experiment registers its scoped isolation service worker. The driver reuses production WGSL, candidate search, f64 verifier pool and summary arithmetic. Pool capacity is capped at four workers before initialization. Physical WebGPU is required for timing reports. No adapter identity is collected. Browser: Chromium 153.0.8010.12 on Windows.

Each report excludes five warm-ups and keeps ten trials per mode. Fixed-hint trials reuse identical captured GPU hints. GPU trials repeat query packing, actual dispatch/readback and one-batch prefetch before verification. Both include ordered summaries, f32 export and result RPC. Batch size is 131,072; workgroup size is 64. No compiler or heavy tests ran alongside timing.

Initial cyclic rotations retained most immediate predecessors: shared usually followed the slower serial control. After long GPU-result waits, a documented revision used four Williams rows. Each full block balances positions and every directed predecessor pair. Ten measured rows contain two full blocks and two extra rows, so total balance is not exact. Both phases and every long request are published, without pooling or filtering.

Table times are medians of elapsed RPC milliseconds. Paired percentage is `median(100 * (shared / baseline - 1))` within each trial; negative is faster. It is not the ratio of separate medians and can differ in direction. Raw `measurements.*.cold` entries are validation requests after setup/guards, with shared fixed verification already exercised. They are not fresh-process cold jobs.

## Counterbalanced results

Fixed hints, verification, summaries and RPC:

| Report | Same-binary serial | Streaming pool | Staged pool | Shared | Paired shared/staged |
|---|---:|---:|---:|---:|---:|
| pandaset-4-03 | 155.76 | 68.25 | 63.34 | 58.12 | -11.93% |
| autoware-4-03 | 252.63 | 107.70 | 106.75 | 97.46 | -7.55% |
| pandaset-4-04 | 181.05 | 78.58 | 73.06 | 72.72 | +4.93% |
| autoware-4-04 | 241.29 | 111.05 | 98.20 | 86.54 | -11.54% |

Same-binary serial/parallel speedup is 2.51-2.84x. The existing pool is already parallel, so this does not describe improvement over the application.

GPU dispatch/readback, verification, summaries and RPC for the retained job:

| Report | Streaming pool | Staged pool | Shared | Paired shared/streaming | Paired shared/staged |
|---|---:|---:|---:|---:|---:|
| pandaset-4-03 | 111.96 | 108.24 | 107.37 | -3.14% | -3.33% |
| autoware-4-03 | 149.56 | 153.29 | 146.34 | -4.80% | -4.82% |
| pandaset-4-04 | 114.14 | 111.03 | 111.67 | -2.15% | -1.80% |
| autoware-4-04 | 155.25 | 153.12 | 147.14 | -4.71% | -1.52% |

## Original phase and long waits

| Original report | Fixed paired shared/staged | GPU paired shared/staged | Largest shared GPU request, ms |
|---|---:|---:|---:|
| pandaset-4-01 | -9.39% | +5.35% | 638.09 |
| autoware-4-01 | -4.16% | +1.08% | 630.37 |
| pandaset-4-02 | -1.06% | -13.44% | 646.78 |
| autoware-4-02 | -4.11% | +0.14% | 652.48 |

For example, original PandaSet 01 trial 1 took 638 ms, with 490 ms waiting for a GPU result and 116 ms in verification. In the revised phase, existing streaming and staged modes also had requests over 600 ms; shared maxima were 122-193 ms. This does not establish that order alone caused or fixed the waits: absolute timings also varied substantially. All outliers remain in the packet.

`gpuWaitMillis` is host waiting for result promises, including scheduling/readback, not a GPU kernel timestamp. Prefetch overlaps intervals; do not sum phase counters into a fabricated wall time. Other processes, clocks, power and GC remain uncontrolled.

## Preparation payload

Four verification lanes import four tree snapshots with the existing pool and one with shared verification. The common owner tree and GPU buffers remain in both alternatives.

| Dataset | Pool snapshot transfer, bytes | Shared snapshot transfer, bytes | Query transfer for staged pool or shared, bytes |
|---|---:|---:|---:|
| pandaset | 67,328,160 | 16,832,040 | 12,765,792 |
| autoware | 111,022,032 | 27,755,508 | 21,094,080 |

These are input transfer bytes, not retained allocations or RSS. The staged pool distributes queries, so its total query payload equals shared verification. Committed WASM pages are high-water counters. All controls coexist; whole-application memory savings and complete cold-operation times have not been measured.

## Reproduction and validation

Use the [standalone instructions](../web/experiments/shared-gpu-c2c/README.md). Builders and reports refuse existing outputs. Input manifests, source/build fingerprints and final unchanged-file checks are enforced. `--gpu disabled --warmups 0 --repeats 1` is a compatibility smoke using full-search sentinel hints and makes no GPU speed claim. CI rebuilds production and both experimental packages, type-checks the driver, tests the order design and runs this smoke with two actual shared lanes. Native tests check ownership, invalid hints/inputs/ranges, distance bits, ordering and certified counts under one, two and four threads.

The [packet manifest](../benchmarks/wasm-shared-gpu-c2c/manifest.json) includes all eight reports, 111 normalized source fingerprints, the standard production build recipe and the protocol revision. The [executed original script](../benchmarks/wasm-shared-gpu-c2c/executed-source/benchmark-cyclic-order.mjs) is archived by hash; restore it to its original script location to reproduce that schedule. Binary package hashes are identical across both phases. Reports identify the base commit and working-tree measurement; fingerprints identify the executed files.

The dual-build/shared-memory mechanism follows the approach examined in [visloc-rs PR #311](https://github.com/rsasaki0109/visloc-rs/pull/311) and [wasm-bindgen-rayon](https://github.com/RReverser/wasm-bindgen-rayon). Evaluate complete fresh query jobs and isolated application memory before changing the production default.
