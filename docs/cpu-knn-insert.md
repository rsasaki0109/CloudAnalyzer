# k-neighbour insertion: copy displaced hits

The exact k-neighbour search now copies displaced hits toward the back of its
sorted list, then writes the new candidate at its final position. Previously it
swapped the new candidate with every preceding hit, rewriting it at each move.
The full-list cutoff, comparison sequence, insertion after equal distances,
traversal order and numerical distance arithmetic remain the same.

On three measured clouds with **k=24**, two measurement passes found warm
whole-RPC paired median reductions of **11.73–17.94% for normal estimation** and
**6.78–15.62% for SOR**. Normals were faster in **57/60 timed pairs**, SOR in
**56/60**. Both builds reused the same immutable neighbourhood preparation, so
these figures compare the insertion kernels after the preceding preparation
improvement. They are not cold-start results or a claim for every k/cloud size.

The copy implementation is now the standard insertion path for scalar and
optional paired-SIMD k-neighbour searches. Scalar kernels remain the default;
no optional SIMD or GPU mode was enabled by this change.

## Implementation and precision

`rust/crates/ca-core/src/kdtree.rs::insert_knn` retains the original admission
condition. While filling the list, `push` still appends a slot. Once full, the
tail is replaced through the shifting operation. Each displaced tuple is copied
once, and `(slot, squaredDistance)` is written after the loop. No allocation,
tree construction, point representation, capacity policy or additional retained
buffer is introduced.

The change performs no new floating-point arithmetic. Equal distances keep
their historical order; strict cutoff and predecessor comparisons keep the old
NaN/infinity behavior. Original indices are normalized at the same point after
search. Branch pruning still uses the same final distance and visits the same
near/far subtrees. Normal covariance/reduction and SOR square-root/sum order are
unchanged.

Normal estimation, SOR and ICP normal fitting use this k-neighbour routine.
The performance measurements below cover normals/SOR only; they do not establish
an ICP gain. Ordinary C2C nearest searches and GPU candidate shaders are separate
paths. Parallel normals still use part-local neighbours at partition borders,
and SOR still queries all reached parts, as documented in
[`cpu-knn-preparation.md`](cpu-knn-preparation.md).

## Measured-source protocol

[`benchmark-knn-insert.mjs`](../web/scripts/benchmark-knn-insert.mjs) compares
the archived swap-insertion production build from the preparation step with
the current copy-insertion build. Each timing worker owns an independent cloud
and pool. Both request `knnCache:true`; identical part counts and retained bytes,
stable job IDs, exact incrementing reuse counters, non-pending preparation and
the existing 128 MiB budget are required for every pair. Source JS and WASM
interfaces stay the same; WASM glue was checked byte-identical after rebuilding.

The full measured `compared.ply` inputs from
[`c2c-real-source.md`](c2c-real-source.md) are used unchanged: PandaSet scene 019
(531,908 points), Autoware planning (878,920), and Tokyo Takanawadai (941,933).
`keepEvery=1` and the manifest's full counts/hashes are required. Settings are
k=24, upward normals and SOR ratio=1. Inputs, original source attribution and
licenses are embedded in every report. These are measured-source subsets,
not generated points, independent surveys or accuracy ground truth.

A third archived worker supplies one complete untimed output per method, then
terminates before timing. All subsequent pairs compare both timed builds with
this oracle: every normal f32 bit, or SOR kept count and byte-identical PLY.
Both timing workers use one excluded first-use calculation, five excluded
alternating warm-up pairs and ten alternating timed pairs. Loading and pool
startup are excluded. First-use observations are single samples, not cold
medians. Whole RPC times include native search/PCA/filter, messages, compact
joins, output copies and SOR filtered-cloud indexing. Diagnostics, export,
validation and result removal are outside timing.

Initial source order was PandaSet → Autoware → Tokyo; the final pass reversed
it. No builds, tests or file scans ran during timing. CPU clocks, power,
garbage collection and other applications were uncontrolled. Some pairs show
large timing spikes, and some favour the old kernel; all are retained. Neither
a kernel-only timing nor a gain at the default UI k is inferred from these
k=24 whole-RPC measurements. No GPU adapter/model metadata is collected.

## Warm whole-RPC results

Times are separate medians in milliseconds. Reduction is the median of each
pair's `(swap − copy) / swap`, rather than the ratio of the separate medians.
Positive means faster.

| Source/pass | Normal medians, swap / copy | Normal paired reduction; faster | SOR medians, swap / copy | SOR paired reduction; faster |
| --- | ---: | ---: | ---: | ---: |
| PandaSet initial | 416.50 / 354.00 | +14.93%; 10/10 | 500.75 / 455.45 | +6.78%; 8/10 |
| PandaSet final | 469.10 / 388.55 | +17.94%; 10/10 | 471.60 / 400.95 | +15.62%; 10/10 |
| Autoware initial | 770.30 / 653.60 | +14.73%; 10/10 | 583.05 / 539.85 | +7.45%; 10/10 |
| Autoware final | 678.25 / 592.70 | +13.56%; 10/10 | 684.80 / 624.70 | +8.10%; 8/10 |
| Tokyo initial | 651.45 / 580.85 | +11.73%; 7/10 | 752.60 / 656.60 | +9.97%; 10/10 |
| Tokyo final | 608.00 / 530.85 | +11.99%; 10/10 | 696.85 / 609.90 | +11.46%; 10/10 |

All comparisons retain the same prepared capacities in both builds: PandaSet
35,138,912 bytes across five parts; Autoware 58,041,728 across eight; Tokyo
62,200,604 across eight. Source geometry, partitioning and cache policy are
unchanged. These capacities exclude scratch, source storage, allocator overhead,
other caches and in-flight retired groups; they are not a process-memory cap.

## Validation

- A new insertion regression compares **30,720 updates** against the historical
  swap implementation, checking original slots and all f64 bits after every
  update. It covers k=1/3/12/24/64/128, ascending/descending/tied/random inputs,
  signed zero, infinity, negative infinity and distinct NaN payloads.
- **366 native tests** passed, including prepared/fresh normal and SOR bit
  comparisons, global filter controls and ICP fit/transform checks.
- **19 Playwright tests** passed: pure join/lifetime controls, scalar/SIMD
  distance and normal bits, parameter changes, concurrent fits, stale-source
  rejection, worker loss/recovery, global SOR and normal PLY export, 1.2M-point
  SOR, prior ICP/M3C2/Map reuse tests, and four GPU-suite regressions run with
  hardware opt-in.
- SIMD WASM production build, TypeScript/Vite build, Rust formatting, native
  Clippy and wasm32 SIMD Clippy passed with warnings denied.
- Six reports validated **180 warm-up/timed pairs and 12 excluded first-use
  pairs**, all with matching normal bits or SOR PLY against both builds and the
  archived oracle. No real-input precision tolerance is used.
- Nine CLI guards passed: missing/unknown/repeated options, missing values,
  invalid repeats, manifest mismatch, identical archive/current artifacts and
  attempted output overwrite. Existing report bytes were preserved.
- Input, manifest, benchmark script, eight implementation source files and both
  builds' artifacts are hashed before and after every run.

## Reproduction and fingerprints

Run from the repository root with the original input/manifests and preserved
production build from immediately before the insertion change:

```powershell
node web/scripts/benchmark-knn-insert.mjs --baseline-dist notes/knn-insert-baseline-20261012/dist --input notes/c2c-real-pandaset/compared.ply --input-manifest notes/c2c-real-pandaset/inputs.json --output NEWREPORT.json
```

Default is ten timed pairs; output must not exist. Baseline/input files are
ignored, so a different checkout needs equivalent preserved artifacts and
original source inputs. The adopted candidate was also saved locally in
`notes/knn-insert-candidate-20261012`, including its production build and tree
source. Earlier preparation/merge reports retain their historical fingerprints;
their percentages are not added to these results.

| Artifact | SHA-256 |
| --- | --- |
| Baseline `worker-Bi2ftahU.js` | `e85af60d8505d3f217cfabfb642d88863043d7072ddd11b5d4526819d7fc882c` |
| Baseline `c2c-worker-Cm5O4znW.js` | `c2fd1e8a56fcc67a7cc6353fbee6d00d259b5409d5e66319ef26f3a20b8f43a0` |
| Baseline `ca_wasm_bg-dx7fFc55.wasm` | `e08cc5c9d9702c1fd09767f0adfd9c6d2c86f249075f99ead53feb13cd88a520` |
| Adopted `worker-C7Qpeoqi.js` | `a0f9f1e703041aebcf7f59848293e2c580846f148e858fed3151d42c4cd45bbc` |
| Adopted `c2c-worker-D87gkoqR.js` | `4d81c6615d2acec28cefdeb194f2d7844f868b9a2ee7b8206137f5399b318bdc` |
| Adopted `ca_wasm_bg-_OVJwnKV.wasm` | `f7e3c2dab146beb2cfac1516eac3a26953260102ba7788cf8bf7c14f77bd632e` |
| Both WASM glue files | `b6f3f60c80498963e273273723b204eb485f281a40635b110f6ac080e6cf1a0f` |
| Benchmark script | `f06a68edcebaa62c6c66efbd292ab68ea2136755e0909e6628764565d4a2e3ab` |

Raw reports and SHA-256 checksums follow.

| Report | SHA-256 |
| --- | --- |
| [windows-autoware-planning-copy-insertion-final.json](../benchmarks/cpu/knn-insert/real-source/windows-autoware-planning-copy-insertion-final.json) | `ae94852774be7dbd950b6a27b8068fc728eb45ec3bbe2f0eb5915afc98d6ecc1` |
| [windows-autoware-planning-copy-insertion-initial.json](../benchmarks/cpu/knn-insert/real-source/windows-autoware-planning-copy-insertion-initial.json) | `a543754de7b1c44ebd34dfabc2096ab2681111e766d98472adc105bee0f2d253` |
| [windows-pandaset019-copy-insertion-final.json](../benchmarks/cpu/knn-insert/real-source/windows-pandaset019-copy-insertion-final.json) | `9b8ba6062d30ba45d8fa1cdb6b9fc6e915e3e8985eef5665a1caf83b94bb3170` |
| [windows-pandaset019-copy-insertion-initial.json](../benchmarks/cpu/knn-insert/real-source/windows-pandaset019-copy-insertion-initial.json) | `3a1629d6b88b610fa6ba1700af1adc66d78d1f5ab945343c9ee6b05d263476c3` |
| [windows-tokyo-takanawadai-copy-insertion-final.json](../benchmarks/cpu/knn-insert/real-source/windows-tokyo-takanawadai-copy-insertion-final.json) | `6d33e4334a9023aefd19ac82c8f3598fde98de87fbc523b8351814e9aa8f8cfd` |
| [windows-tokyo-takanawadai-copy-insertion-initial.json](../benchmarks/cpu/knn-insert/real-source/windows-tokyo-takanawadai-copy-insertion-initial.json) | `c4fe612dbbd0873678b3c2c3707fa422d5c35b846c5edefdb6b329d86e2c3c8f` |
