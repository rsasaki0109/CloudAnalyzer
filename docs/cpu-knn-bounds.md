# Combined ancestral bounds for k-neighbour search

Combining the lower bounds from earlier tree splits preserves the tested
normal/SOR outputs, but did **not** produce a consistent whole-call speedup on
three measured clouds. The experiment therefore remains **off by default**.
Across two measurement passes, combined bounds were faster in **29/60 normal
pairs** and **21/60 SOR pairs**. Earlier preparation reuse, compact joins and
copy insertion remain enabled; their historical measurements are documented
separately in [preparation](cpu-knn-preparation.md), [joins](cpu-knn-join.md) and
[insertion](cpu-knn-insert.md).

## Implementation and precision

`KdTree::nearest_k_into_with_options` selects scalar or optional paired-SIMD
leaf arithmetic and a separate compile-time bounds variant. The original
single split-plane test still runs first. When visiting a far child, the
experimental variant retains the maximum squared split distance for each axis
along the current ancestry. It sums those three nonnegative terms in the same
`(x + y) + z` order as leaf distance arithmetic. A subtree is skipped only when
the list already contains k hits and the combined lower bound is strictly
greater than its last distance. Equal bounds and failed comparisons cannot
cause this additional skip. Near-child traversal inherits the earlier bounds.

The three lower-bound values live on the query call stack. No per-node boxes,
retained tree arrays, point conversions, neighbour-list allocation or additional
prepared capacity are introduced. Admitted hits still use copy insertion;
distance calculations, hit comparisons, original-index normalization, normal
covariance/PCA and SOR square-root/reduction order remain the same.

`SorPart.localWithBounds`, `withinWithBounds` and `normalsWithBounds` expose the
variant for the worker pool. Internal filter/normal requests can opt in with
`knnBounds:true` while using preparation. `DEFAULT_KNN_BOUNDS` is false.
Existing native entry points, unprepared pool jobs, small-cloud serial paths
and preparation-budget fallback use the original split-plane test. Geometry
leases, invalidation, worker generations and retirement are unchanged.
Parallel normal estimation retains its existing part-local border behaviour;
this experiment does not change the neighbourhood definition.

## Measured-source protocol

[benchmark-knn-bounds.mjs](../web/scripts/benchmark-knn-bounds.mjs) compares two
independent workers using the **same candidate production build**, with
`knnBounds:false` (plain) and `knnBounds:true` (combined). Plain retains the
historical single split-plane pruning. Both request `knnCache:true`; every
pair requires identical part counts and retained bytes, stable preparation job
IDs, exact incrementing reuse counters and non-pending preparations within the
existing 128 MiB idle budget.

The full `compared.ply` inputs documented in [real sources](c2c-real-source.md)
are used without thinning: PandaSet scene 019 (531,908 points), Autoware planning
(878,920) and Tokyo Takanawadai (941,933). Settings are k=24, upward normals and
SOR ratio=1. Original attribution, licenses, input hashes and counts are embedded
in each report. These are measured-source subsets; they are not independent
surveys or accuracy ground truth.

A third, archived production worker from before this experiment supplies one
complete untimed output per method and terminates before timing. Every pair
checks all normal f32 bits, or the SOR kept count and every exported PLY byte,
against both current paths and this oracle. Each method has one excluded
first-use pair, five excluded alternating warm-up pairs and ten alternating
timed pairs. Loading, pool startup, cache diagnostics, output verification,
export and result removal are outside timing. Whole RPC includes native
search/PCA/filter, messages, compact joins, output copies and SOR output indexing.
Single first-use observations are not cold medians.

Initial source order was PandaSet → Autoware → Tokyo; the final pass reversed
it. No builds, tests or file scans ran during timing. CPU clocks, power, garbage
collection and other applications were uncontrolled. All spikes and losses
are retained. No kernel-only gain or gain at the default UI k is inferred.
No GPU adapter/model metadata is collected.

## Warm whole-RPC results

Times are separate medians in milliseconds. Reduction is the median of each
pair's `(plain − combined) / plain`, rather than the ratio of separate medians.
Positive means faster. Separate medians can improve while paired reduction
is negative because they do not necessarily come from the same trials.

| Source/pass | Normal medians, plain / combined | Normal paired reduction; faster | SOR medians, plain / combined | SOR paired reduction; faster |
| --- | ---: | ---: | ---: | ---: |
| PandaSet initial | 623.35 / 585.25 | −1.38%; 4/10 | 512.15 / 517.75 | −3.67%; 3/10 |
| PandaSet final | 353.45 / 357.30 | +0.70%; 5/10 | 373.60 / 372.65 | −2.89%; 4/10 |
| Autoware initial | 504.15 / 517.45 | −3.49%; 2/10 | 572.65 / 559.55 | −1.97%; 5/10 |
| Autoware final | 463.30 / 462.30 | +1.21%; 5/10 | 519.20 / 520.70 | +0.52%; 5/10 |
| Tokyo initial | 501.75 / 493.90 | +2.65%; 7/10 | 553.60 / 570.20 | −1.73%; 3/10 |
| Tokyo final | 507.70 / 507.60 | +0.48%; 6/10 | 517.90 / 532.85 | −2.87%; 1/10 |

Prepared capacity is identical in both paths: PandaSet 35,138,912 bytes across
five parts, Autoware 58,041,728 across eight and Tokyo 62,200,604 across eight.
These figures exclude scratch, source storage, allocator overhead, other caches
and in-flight retired groups; they are not a total process-memory cap.

The mixed results do not establish that the extra arithmetic costs more than
the saved searches in isolation. They establish that this candidate does not
justify enabling the extra test in the measured whole-RPC workload.

## Validation

- The new native tree regression compares original hit indices and every f64
  bit with bounds off/on, across tiny/large coordinates, projected coordinates,
  coincident and planar points, nonfinite values, multiple k values, and scalar
  and optional-SIMD dispatch. Prepared normal/SOR regressions also compare the
  bounds variant with existing fresh results.
- **367 native tests** passed. Rust formatting, native Clippy and wasm32 SIMD
  Clippy passed with warnings denied; the SIMD WASM and TypeScript/Vite
  production builds passed.
- **20 distinct Playwright tests** passed on the candidate: the new raw-WASM
  bounds test, preparation lifetime/recovery controls, compact joins, prior
  ICP/M3C2/Map reuse, normal/SOR UI and export checks, 1.2M-point SOR and four
  hardware-opt-in GPU regressions. The new bounds test checks **120**
  count/coordinate/k combinations, both normal orientations, every SOR local
  field and query-distance bits, with unchanged retained capacity.
- After returning the worker default to false, the production build and all
  **13 related bounds/preparation/join/compute-scratch tests** passed again.
- Six reports validated **180 warm-up/timed pairs and 12 excluded first-use
  pairs**, with matching normal bits or SOR PLY against both timing paths and
  the archived oracle. No real-input precision tolerance is used.
- Nine CLI guards passed, covering missing/unknown/repeated options, missing
  values, invalid repeats, manifest mismatch, identical archived/current
  artifacts and attempted output overwrite. Existing report bytes were preserved.
- Input, manifest, benchmark script, nine implementation source files and both
  production builds are hashed before and after every run.

## Reproduction and fingerprints

Run from the repository root with the original input and its manifest, the
preserved pre-experiment production build, and a current SIMD production build:

```powershell
node web/scripts/benchmark-knn-bounds.mjs --baseline-dist notes/knn-bounds-baseline-20261012/dist --input notes/c2c-real-pandaset/compared.ply --input-manifest notes/c2c-real-pandaset/inputs.json --output NEWREPORT.json
```

The default is ten timed pairs; output must not already exist. The script
sets both bounds options explicitly, so the false worker default does not
disable the comparison. Baseline/input files are ignored; another checkout
needs equivalent preserved artifacts and original inputs.

The measured build and all nine hashed implementation sources plus the script
were saved locally under `notes/knn-bounds-candidate-20261012` before changing
the default. The reports' `optimized` build label identifies this candidate,
not an adopted speedup. Its worker source hash is
`0b119553ab6f7a7ceddb51cb17f147f97efd1a60d4a0aaa905ede9d66104130f`.
The final default-off worker source hash is
`a95e9dfa752842b66b7ca2e5b819ed695bb6d1b3e835a4a4781f0f592c707665`;
the other eight measured implementation files remain identical. Historical
reports keep their original hashes. Final default-off performance was not
measured as a separate cross-build comparison with the archived oracle.

| Artifact | SHA-256 |
| --- | --- |
| Archived `worker-C7Qpeoqi.js` | `a0f9f1e703041aebcf7f59848293e2c580846f148e858fed3151d42c4cd45bbc` |
| Archived `c2c-worker-D87gkoqR.js` | `4d81c6615d2acec28cefdeb194f2d7844f868b9a2ee7b8206137f5399b318bdc` |
| Archived `ca_wasm_bg-_OVJwnKV.wasm` | `f7e3c2dab146beb2cfac1516eac3a26953260102ba7788cf8bf7c14f77bd632e` |
| Measured `worker-EVHJaOlA.js` | `0c3e4294096fb96fb841b910b79a24647d6ff78392c35dd24079d3844717bf43` |
| Final default-off `worker-DVUZlhH3.js` | `4ffe1691efecd15db17f9fe8b8435d8aab81c17653b56b058a837d2a763ffe91` |
| Measured/final `c2c-worker-DZRdZqhd.js` | `6bdc184592148f9fec7209b14eff935502465b0104646762dd1a20f3a0590f3c` |
| Measured/final `ca_wasm_bg-Cqb-Gons.wasm` | `67033ddc33d7540cbb50ef2aabd57d39672b65ca94f90aea8490a176c3d5b1b1` |
| Benchmark script | `8b991f3886e08ed405b1b8d32e6a9a4e86b996afe067d40c064cab9ae2cfc70f` |

Raw reports and checksums follow.

| Report | SHA-256 |
| --- | --- |
| [windows-autoware-planning-combined-bounds-final.json](../benchmarks/cpu/knn-bounds/real-source/windows-autoware-planning-combined-bounds-final.json) | `29ea863446aa8dd43088c545afdcb87b78711f18afa2d47495bbbe6f7338f233` |
| [windows-autoware-planning-combined-bounds-initial.json](../benchmarks/cpu/knn-bounds/real-source/windows-autoware-planning-combined-bounds-initial.json) | `03d7a26d55110706a581d46ac88c4698c53bbf937c6ff7c160354bdc1efd471a` |
| [windows-pandaset019-combined-bounds-final.json](../benchmarks/cpu/knn-bounds/real-source/windows-pandaset019-combined-bounds-final.json) | `60f9a63f9dce22dde26d865eb7d6a378f3a9035bfa4e43e51179a8ec36672e0f` |
| [windows-pandaset019-combined-bounds-initial.json](../benchmarks/cpu/knn-bounds/real-source/windows-pandaset019-combined-bounds-initial.json) | `08a09188fd08aa9efc5b93b223ca58b48b1633cf0ee763e5a45d778ed7767d39` |
| [windows-tokyo-takanawadai-combined-bounds-final.json](../benchmarks/cpu/knn-bounds/real-source/windows-tokyo-takanawadai-combined-bounds-final.json) | `cde10db5d0b0ec59d1d3abef07c8047c809210c5327cfd27b9d0e5d2408860a7` |
| [windows-tokyo-takanawadai-combined-bounds-initial.json](../benchmarks/cpu/knn-bounds/real-source/windows-tokyo-takanawadai-combined-bounds-initial.json) | `afd8a731c0c1ff29360c827ea52ab5bea2bc1785c3ccda0d47d9b6ffb7f95078` |
