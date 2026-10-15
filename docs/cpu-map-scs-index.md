# Sparse integer neighborhoods for MapEval SCS

`ca map-evaluate` now indexes supported Wasserstein-error cells for SCS instead
of enumerating every possible neighboring grid coordinate. Complete resident
evaluation improves by 5.43–18.12% across six real-source comparisons, with
25/30 faster timed pairs, against the immediately preceding
[parallel-NN implementation](cpu-map-nn-parallel.md). Both workflows already use
parallel exact point-distance queries. Index construction/destruction, both
point trees/queries, threshold statistics and default AWD/SCS are included.
Input loading, CLI startup/serialization and PLY writing are excluded.

The subsequent [resident covariance reduction](cpu-map-covariance.md) adds
another measured change. Reproduce the SCS delta below with the frozen
pre-covariance sources; current production also includes covariance reduction
and [scalar voxel grouping](cpu-map-grouping.md).

For the default radius 5, the old implementation creates 1,330 coordinates for
each supported error cell and probes the error dictionary for each. The new
implementation creates one SciPy `cKDTree` of existing supported cells and
queries its inclusive Chebyshev (`p=inf`) radius. The search returns the same
cube neighborhood, excludes the center, and avoids probing absent cells.
It changes lookup work, rather than the Gaussian/Wasserstein calculations or
the SCS mean/std/coefficient-of-variation formulas.

## Exact ordering, guards and lifetime

Tree points follow sorted integer keys. Sorted returned point indices therefore
retain the old lexicographic neighbor sequence. Centers still follow the error
dictionary's original insertion order, including maps assembled from chunks.
Both orders matter for reproducing floating-point reductions. Zero-error,
NaN/unavailable and support-count behavior remain unchanged.

The index is used for at least 64 supported cells, a plain Python integer radius
of at least 2, and plain three-integer tuple keys. Per-axis integer minima are
subtracted before converting keys to f64. Local spans and the radius must not
exceed `2**52`, so all local integer coordinates and relevant differences are
exactly representable. Large absolute keys, including values beyond f64's
integer precision, can still qualify when their local span is small. Small
sets, radius 1, unsupported key/radius types and wider spans keep the preceding
grid lookup. Invalid float-radius behavior is preserved. Actual index failures
propagate; they do not silently substitute a different result.

Each structural evaluation builds its own index from cells that passed the
existing reference-match/minimum-point checks. It queries one center at a time;
it does not materialize all neighbor pairs. The index is released when the
neighbor iterator is exhausted, before the final SCS reduction. No global or
cross-evaluation cache is added. This helper serves resident and chunked/
streaming AWD/SCS. Public options, classifications, protocols, metrics and
artifact formats stay unchanged. The performance measurements below cover
resident evaluation; streaming behavior is checked for correctness.

## Real-source workload and memory

[Real-source preparation](c2c-real-source.md) supplies original-coordinate
reference/compared subsets with source attribution/licenses. No additional
thinning, generated points, displacement or noise is added. Tokyo retains its
preceding original-point selection. These are disjoint-index subsets of one
measured source with possible geometric duplicates, rather than independent
surveys or map-accuracy ground truth. Generated fixtures serve correctness tests.

Requests use the CLI's default protocol: 3 m voxels, minimum 100 points in both
maps, neighbor radius 5 and accuracy thresholds `(0.2,0.1,0.08,0.05,0.01)` m.
Alignment and downsampling are disabled. The existing voxel grouping,
Gaussian/Wasserstein calculation and reductions all run inside clocks.

| Source | Estimated / reference points | Supported cells | Coordinate array, bytes | Native index array, bytes | Native tree nodes |
| --- | --- | ---: | ---: | ---: | ---: |
| PandaSet 019 | 531,908 / 531,908 | 1,202 | 28,848 | 9,616 | 215 |
| Autoware planning | 878,920 / 878,921 | 3,948 | 94,752 | 31,584 | 711 |
| Tokyo Takanawadai | 941,933 / 941,933 | 1,966 | 47,184 | 15,728 | 361 |

The two array sizes total 38,464–126,336 bytes in these cases. They exclude
native node storage, Python key/temporary-coordinate containers, per-query
neighbor lists, search scratch, allocator overhead and the existing voxel/
point/result data. They are not peak RSS or a memory cap. A single neighbor
list lives at a time; large radii/dense supported sets can still produce many
neighbors. Eligibility thresholds are conservative implementation choices;
crossover behavior has not been benchmarked across machines/configurations.

## Controlled comparison

The control uses an archived preceding map module with parallel NN and grid
SCS lookup. The new workflow uses production code. Both use the same frozen
SciPy tree binary and input arrays. No tracing hooks are active during timed
calls. Timings begin with resident arrays and end when the complete
`MapEvaluateResult` returns. Artifact writes are disabled for timing; a traced
full-distance/two-PLY correctness pair runs outside performance summaries.

Windows, 12 logical CPUs, Python 3.12.10, NumPy 2.3.4, SciPy 1.17.1 and
Open3D 0.19.0 use the preceding benchmark settings: `OMP_NUM_THREADS=12` and
`OPENBLAS_NUM_THREADS=1` before imports. Large point queries use all CPU workers
in both workflows. SCS center queries run sequentially. Production does not
change environment variables. Seven Python source/configuration fingerprints
and the compiled tree fingerprint are recorded with archived copies.

Each source/pass starts a new process. Full parity, an excluded first timing
pair and two warm-up pairs precede five alternating timed pairs. Sources run
PandaSet → Autoware → Tokyo, then reverse that order. No builds, tests, scans
or edits run during comparisons.

| Source/pass | Grid-SCS median, ms | Indexed-SCS median, ms | Median paired shortening | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| PandaSet initial | 4,407.24 | 4,022.21 | +6.15% | 4/5 |
| PandaSet final | 3,572.14 | 3,399.59 | +5.43% | 4/5 |
| Autoware initial | 8,621.26 | 6,198.78 | +18.12% | 5/5 |
| Autoware final | 8,013.37 | 6,754.21 | +15.71% | 3/5 |
| Tokyo initial | 7,028.33 | 6,291.32 | +10.96% | 5/5 |
| Tokyo final | 6,618.15 | 6,207.87 | +7.24% | 4/5 |

Shortening is `median(100 * (1 - indexed_ms / grid_ms))`, not the ratio of
marginal medians. All observations, including five reversals, remain. Passes
vary; clocks, power, scheduling, other applications, allocator/page retention
and warmed caches are uncontrolled. These results measure complete resident
evaluation with default structural metrics, rather than full CLI/cold-file or
streaming performance. This delta is relative to parallel NN; separate
acceleration percentages are not additive. No GPU/browser/WASM speed is claimed.

## Validation and reproduction

All 108 parity/first/warm-up/timed results match, including every NN/structural
metric, support count and classification/protocol value. Twenty-four saved full
f64 distance arrays and all 24 complete colored raw/inlier PLY outputs match
byte-for-byte and agree with the preceding independently recorded parallel-NN
results. An independent audit recomputes 12 serial point queries and six
structural evaluations with the archived code, compares all 14,232 real-cell
neighbor sequences in order, and checks every PLY coordinate/RGB byte. Source/
library fingerprints, index payloads/node counts, report hashes and all timing
summaries are verified. Twelve negative controls reject wrong counts/data/
baseline, invalid voxel sizes/duplicate options and existing destinations
without replacing data.

Sixty-seven related Python tests pass, covering ordinary/huge absolute integer
keys, exact-span fallback, 63/64/65-cell boundaries, cube corners/self exclusion,
small/radius-1/type fallbacks, center/neighbor ordering, zero/tiny/mixed/NaN error
values, actual query failures, one-center lookup, resident/streaming metrics and
PLY parity, preceding NN parallelism, core/experimental imports and the CLI.
Syntax, whitespace and links pass. The preceding Windows upstream-harness
limitation (`resource` import) is unchanged; this change does not run that
harness. Rust binaries and browser/GPU selection stay unchanged.

The [benchmark script](../cloudanalyzer/scripts/benchmark_map_scs_index.py)
requires `--source-root`, original `--reference`/`--compared`/`--input-manifest`,
archived `--baseline`/`--baseline-sha256`, and new `--work-dir`/`--output` paths.
Defaults are `--repeats 5 --structure-voxel-size 3`. All six recorded comparisons
use those defaults. It retains all observations, complete parity/first results,
input/distance/PLY fingerprints and scoped index counts/array bytes. Reproduce
the separate preceding NN-only change with its frozen pre-index sources;
current production also includes the subsequent covariance reduction and
scalar voxel grouping.

- PandaSet: [initial](../benchmarks/cpu/map-evaluate/real-source/windows-pandaset019-scs-index-initial.json), [final](../benchmarks/cpu/map-evaluate/real-source/windows-pandaset019-scs-index-final.json)
- Autoware: [initial](../benchmarks/cpu/map-evaluate/real-source/windows-autoware-planning-scs-index-initial.json), [final](../benchmarks/cpu/map-evaluate/real-source/windows-autoware-planning-scs-index-final.json)
- Tokyo: [initial](../benchmarks/cpu/map-evaluate/real-source/windows-tokyo-takanawadai-scs-index-initial.json), [final](../benchmarks/cpu/map-evaluate/real-source/windows-tokyo-takanawadai-scs-index-final.json)
