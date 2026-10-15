# C2C with an owned reference tree

`cloudanalyzer_core.PreparedReference` and Rust's
`ca_core::distance::PreparedC2cReference` retain an immutable reference k-d tree
for repeated nearest-distance queries. Full Python calls shorten by 54.68–61.28%
in six measured-source comparisons, with 60/60 faster timed pairs. Every tested
full f64 output, all six statistics and canonical scalar PLY bytes match an
independent preceding release wheel. This API is opt-in; ordinary CLI,
browser/WASM/GPU dispatch and stateless calls keep their existing selection.

This report records the standalone API stage. The subsequent
[scoped scan-match integration](cpu-scan-match-reference.md) automatically
reuses one working-map tree and measures construction plus both searches
separately from these constructor-excluded API results.

```python
import cloudanalyzer_core as cc

reference_points = cc.read("reference.ply")["positions"]
reference = cc.PreparedReference(reference_points)
first = reference.distances(cc.read("scan.ply")["positions"])
next_batch = reference.distances(other_points)
native_capacity_bytes = reference.retained_bytes
```

Rust callers use `PreparedC2cReference::new(&reference_points)`, then
`distances(&queries)` or, with the `parallel` feature, `distances_par(&queries)`.
The existing `PreparedC2c` retains a fixed query/reference pair and its query
order; the new type retains only the reference, accepting arbitrary later queries.

## Ownership and exact behavior

Construction copies the reference through the existing `KdTree::new`. Python
inputs are `(N, 3)` float64 arrays; strided arrays are accepted. Empty references
raise the existing `ValueError("target is empty")` binding error. Subsequent
mutation/deletion of caller reference arrays does not affect the prepared object.
Construct another object when reference coordinates change.

Python queries copy their source coordinates and return independent `(N,)`
float64 NumPy results in source order. Native construction/search release the
GIL. Query arrays, Morton order, guesses, result buffers and scratch stay local
to each call. Changed batches/lengths and concurrent readers share only the
immutable tree. No query result or order is retained between calls. Releasing
the last object reference releases its owned vectors; there is no new global cache.

Fresh and retained parallel APIs share the same preceding kernel: original
Morton preparation/key-only unstable sorting, 4,096-point chunks, per-chunk
previous-hit guesses, scalar leaf policy and original pair-result/scatter path.
Reference construction, f64 arithmetic, square roots, ties and nonfinite/missing
behavior are unchanged. Mesh policies and GPU candidate/verification paths are
unchanged. This step adds no lower-precision geometry or unsafe access.

## Measurements

[Measured-source preparation](c2c-real-source.md) supplies unchanged full f64
coordinates and original source-index subsets. No extra thinning, generated
points, artificial displacement, noise or registration is used here. Tokyo
keeps its preceding original-point selection. These are two subsets of one
measured cloud, not independent surveys or accuracy ground truth. Small generated
fixtures are used only for correctness tests.

| Source | Reference points | Query points |
| --- | ---: | ---: |
| PandaSet 019 | 531,908 | 531,908 |
| Autoware planning | 878,921 | 878,920 |
| Tokyo Takanawadai | 941,933 | 941,933 |

Windows x86_64, 12 logical CPUs, Python 3.12.10 and NumPy 2.3.4 use frozen
release abi3 wheels built with Rust 1.98.1, optimization level 3, LTO and one
codegen unit. The new wheel is installed only in an isolated workspace venv.
The preceding frozen wheel first produces independent complete f64 distances,
six sequential-f64 statistics and canonical binary scalar PLY oracles, then
exits. Previously recorded 27 source/configuration fingerprints are compared;
changes are limited to the new reference API/binding and exports/stub. Four
additional current fingerprints cover k-d-tree source, native/Python regressions
and the benchmark controller. Original source/PLY/manifest, installed Python
package, archive and extension fingerprints are checked before and after runs.

`RAYON_NUM_THREADS=12` is set before loading the extension. Fresh and retained
calls alternate in the same current process, one computing at a time. Each
source/pass starts a new process and prepared object. An excluded first-use pair
and five warm-up pairs precede ten timed pairs. Passes run PandaSet → Autoware →
Tokyo, then reverse the source order. No builds, tests, scans or edits run during
the performance comparisons.

Repeated-call clocks include NumPy source copies, Morton work, parallel search,
result allocation/scatter, query-local cleanup and NumPy result creation. Fresh
calls additionally copy the reference and build/drop its tree. One-time retained
construction and final object destruction are outside repeated-call clocks.
Loading, first/warm-up calls, statistics, serialization and complete-output
hashing/checks are excluded. All measured queries use the original full source
order; an excluded full reverse-order check exercises changed query order.

| Source/pass | Fresh median, ms | Retained median, ms | Median paired shortening | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| PandaSet initial | 216.64 | 89.95 | +56.70% | 10/10 |
| PandaSet final | 184.53 | 84.63 | +55.93% | 10/10 |
| Autoware initial | 357.59 | 149.34 | +59.59% | 10/10 |
| Autoware final | 311.84 | 129.87 | +54.68% | 10/10 |
| Tokyo initial | 358.79 | 133.19 | +61.10% | 10/10 |
| Tokyo final | 344.66 | 133.15 | +61.28% | 10/10 |

Shortening is `median(100 * (1 - retained_ms / fresh_ms))`, not the ratio of
marginal medians. All observations remain. Absolute times change between passes;
clocks, scheduling, power, other applications, allocator/page retention and cache
effects are uncontrolled. These are whole Python API comparisons against the
current fresh API, not isolated query-kernel or tree-construction speedups.
They do not time a full CLI, ICP, coloring/serialization pipeline or GPU operation.
The preceding wheel supplies correctness oracles, not performance timings.
Separate acceleration percentages are not additive.

## Initial cost and retained storage

| Source/pass | Constructor, ms | Fresh first call, ms | Constructor + retained first call, ms | Estimated calls to amortize setup |
| --- | ---: | ---: | ---: | ---: |
| PandaSet initial | 121.26 | 207.08 | 211.89 | 1 |
| PandaSet final | 108.87 | 203.50 | 189.61 | 2 |
| Autoware initial | 190.26 | 307.20 | 332.24 | 1 |
| Autoware final | 166.86 | 258.67 | 281.63 | 1 |
| Tokyo initial | 200.97 | 346.15 | 352.93 | 1 |
| Tokyo final | 147.23 | 350.88 | 257.74 | 1 |

Each constructor is one observation including NumPy reference copies, tree
construction and temporary cleanup, with loaded inputs and an initialized Rayon
pool. These are warmed-process first observations, not cold process measurements.
Four of six constructor-plus-first-query totals are slower than the fresh first
call. Repeated reuse is the intended benefit. Estimates use
`ceil(setup_ms / median(fresh_ms - retained_ms))`; they are not promised break-even
points and can differ from observed first totals. Final object destruction is
outside the estimate and repeated-query clocks.

| Source | Native tree vector payload capacity, bytes | Caller reference array, bytes |
| --- | ---: | ---: |
| PandaSet | 18,084,896 | 12,765,792 |
| Autoware | 29,883,356 | 21,094,104 |
| Tokyo | 32,025,740 | 22,606,392 |

`retained_bytes` includes owned tree coordinates, original indices and node
vector capacities. Construction copies reference coordinates, so this entire
native payload is additional while caller arrays remain alive. Caller reference
arrays can be dropped after preparation. The count excludes vector/object
headers, allocator overhead, pages, source/results and temporary buffers. It is
not process RSS or a memory cap. Reported capacity stays constant across calls.

## Validation and reproduction

All 192 first/warm-up/timed outputs match the preceding wheel's complete f64
bytes and canonical scalar PLY bytes. Initial fresh/retained calls reproduce
all six sequential-f64 statistics; identical complete distances establish the
same statistics for subsequent calls. PLY is reconstructed outside clocks using
the native XYZ-double/scalar-float format; this adds no Python production writer.
Six full reversed-query checks preserve results in reversed source order.
Independent audit verifies old/current wheels and source archives, old distance
and PLY files, original input provenance, all summaries and retained-capacity
formulae. The old oracle also agrees with the earlier
[scan-match diagnostic](cpu-scan-match-distance-reuse.md) before-distance hashes.

All 422 native workspace tests, 46 release-wheel API tests and 46 related Python
workflow/CLI tests pass. Native/workspace, ca-py and wasm32 Clippy with warnings
denied, formatting and syntax checks pass. Three native regressions cover owned
reference lifetime, empty/changed batches, four coordinate scales, boundary/Morton
sizes, exact serial/fresh/retained comparisons, 1/2/4-thread pools, nonfinite and
overflow queries, immutable capacity and concurrent readers. Seventeen Python
regressions cover boundary sizes, f64/strided ownership, independent results,
duplicates/signed zeros/extremes, nonfinite parity, shape/dtype errors and threads.

The [benchmark script](../rust/crates/ca-py/scripts/benchmark_prepared_reference.py)
first runs `--mode oracle` in the preceding frozen-wheel environment, then
`--mode compare --oracle <new-oracle.json>` in the new frozen-wheel environment.
Both require `--source-root`, `--reference`, `--compared`, `--input-manifest`,
`--build-index`, `--build-root`, new `--work-dir` and `--output` paths; defaults
are `--threads 12 --repeats 10`. Build indexes contain archived `sourceHashes`,
extension metadata and compiler identity. The script validates archive/package
and input fingerprints and rejects existing output/work paths. Fourteen negative
controls cover modes, options, counts, wrong archives/inputs/oracles and existing
paths. Reports preserve full raw observations:

- PandaSet: [oracle](../benchmarks/cpu/c2c/real-source/windows-pandaset019-python-reference-oracle.json), [initial](../benchmarks/cpu/c2c/real-source/windows-pandaset019-python-prepared-reference-initial.json), [final](../benchmarks/cpu/c2c/real-source/windows-pandaset019-python-prepared-reference-final.json)
- Autoware: [oracle](../benchmarks/cpu/c2c/real-source/windows-autoware-planning-python-reference-oracle.json), [initial](../benchmarks/cpu/c2c/real-source/windows-autoware-planning-python-prepared-reference-initial.json), [final](../benchmarks/cpu/c2c/real-source/windows-autoware-planning-python-prepared-reference-final.json)
- Tokyo: [oracle](../benchmarks/cpu/c2c/real-source/windows-tokyo-takanawadai-python-reference-oracle.json), [initial](../benchmarks/cpu/c2c/real-source/windows-tokyo-takanawadai-python-prepared-reference-initial.json), [final](../benchmarks/cpu/c2c/real-source/windows-tokyo-takanawadai-python-prepared-reference-final.json)
