# Parallel exact distance queries in resident MapEval

`ca map-evaluate` now uses SciPy's parallel exact nearest-neighbor query for
each direction with at least 65,536 query points after alignment/downsampling.
Smaller and empty inputs keep their preceding single-thread/empty behavior.
On the measured 12-logical-CPU host, complete resident evaluation improves by
16.57–26.35% across six real-source comparisons, with 29/30 faster timed pairs.
Both fresh tree builds, both queries, threshold reductions and default AWD/SCS
are included. Input loading, CLI startup/serialization and PLY writing are
outside those performance intervals.

The subsequent [sparse SCS lookup](cpu-map-scs-index.md) adds another resident
optimization and changes streaming neighborhood lookup while preserving outputs.
The six NN comparisons here describe their frozen pre-index source state;
reproduce the standalone NN change with that map module. Current production
contains both changes, so a new current-versus-serial run includes both deltas.

The existing SciPy `cKDTree` and its f64 Euclidean `k=1` query remain. Only the
`workers` argument changes: large queries request `-1` (all available logical
CPUs); small queries request `1`. Approximation, metric, tree construction,
reference coordinates, query order, result precision and all metric formulas
stay unchanged. Each direction runs sequentially at the strategy level and
uses parallel query work internally. No outer pool or persistent tree cache is
introduced. No supporting Rust/GPU extension is required for these queries.

The optimization applies automatically to the resident core strategy and its
experimental re-export, including aligned/voxelized requests that retain enough
query points. Each direction makes its own size decision. Empty targets still
return infinity without creating a tree; empty sources return an empty f64
array. Real construction/query failures propagate. Streaming AWD/SCS does not
use these resident nearest-neighbor queries and is unchanged. Public options,
result classification, `nn_backend=scipy_ckdtree` and artifact formats stay the
same. This is separate from the AUC/F1 evaluation used by
[`ca batch`](cpu-batch-reference.md) and
[`loop-closure-report`](cpu-loop-reference.md).

## Real-source workload and scope

[Real-source preparation](c2c-real-source.md) supplies original-coordinate
reference/compared subsets with source attribution and licenses. No additional
thinning, generated points, artificial displacement or noise is added. Tokyo
retains its preceding original-point selection. The two inputs are disjoint-
index subsets of one measured source, with possible geometric duplicates;
they are not independent surveys or map-accuracy ground truth. Generated
fixtures serve only correctness tests.

Requests use full resident XYZ, no alignment, no downsampling, and the CLI's
default MapEval protocol: 3 m voxels, minimum 100 points in both maps, neighbor
radius 5 and accuracy thresholds `(0.2, 0.1, 0.08, 0.05, 0.01)` m. This includes
the existing serial voxel grouping, Gaussian/Wasserstein fits and SCS work.
The optimization changes nearest-neighbor queries within that complete workload.

| Source | Estimated points | Reference points | Supported AWD voxels |
| --- | ---: | ---: | ---: |
| PandaSet 019 | 531,908 | 531,908 | 1,202 |
| Autoware planning | 878,920 | 878,921 | 3,948 |
| Tokyo Takanawadai | 941,933 | 941,933 | 1,966 |

The control loads an archived preceding map-evaluation module. Both workflows
use the same frozen SciPy tree binary and existing inputs, with no query hooks
active during timed calls. Trees are constructed/destroyed per direction in
both workflows. Timings start with resident arrays and end when the complete
`MapEvaluateResult` is returned. Artifact writing is disabled in timed calls;
a full-array/two-PLY correctness pair precedes them outside performance summaries.

Windows, Python 3.12.10, NumPy 2.3.4, SciPy 1.17.1 and Open3D 0.19.0 use 12
logical CPUs. Benchmark processes set `OMP_NUM_THREADS=12` and
`OPENBLAS_NUM_THREADS=1` before imports; SciPy queries request all available
CPUs through `workers=-1`. Production does not change those environment
variables. Six Python source/configuration fingerprints and the compiled tree
fingerprint are recorded with archived copies.

Each source/pass starts a new process. Full f64/PLY parity, an excluded first
timing pair and two warm-up pairs precede five alternating timed pairs. Sources
run PandaSet → Autoware → Tokyo, then reverse that order. No builds, tests,
scans or edits run during comparisons.

| Source/pass | Serial-query median, ms | Parallel-query median, ms | Median paired shortening | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| PandaSet initial | 6,344.10 | 4,672.73 | +26.35% | 5/5 |
| PandaSet final | 8,041.56 | 5,458.12 | +25.09% | 4/5 |
| Autoware initial | 9,855.22 | 7,744.26 | +17.45% | 5/5 |
| Autoware final | 9,672.05 | 8,251.43 | +16.57% | 5/5 |
| Tokyo initial | 9,048.00 | 7,497.69 | +17.13% | 5/5 |
| Tokyo final | 9,142.40 | 7,039.92 | +20.72% | 5/5 |

Shortening is `median(100 * (1 - parallel_ms / serial_ms))`, not the ratio of
marginal medians. All observations, including the PandaSet reversal, remain.
Passes vary; CPU clocks, power, scheduling, other applications, allocator/page
retention and warmed caches are uncontrolled. These are complete resident
evaluation timings with default structural metrics, rather than full CLI or
cold-file timings. They do not establish GPU/browser/WASM speed, and percentages
from different workloads are not interchangeable or additive.

Large queries now use more CPU concurrency. SciPy worker stacks/search scratch
may add temporary memory; net peak RSS is not measured. Trees/results retain
their scoped lifetimes, and there is no new long-lived reference reservation.
Concurrent callers may compete for CPUs. The small-query threshold avoids
starting a full worker group for ordinary small fixtures/requests; its exact
crossover has not been benchmarked across machines.

## Validation and reproduction

All 108 parity/first/warm-up/timed results match, including every NN threshold,
mean/Chamfer/F-score, AWD/SCS/support count and classification/protocol value.
Twenty-four saved full f64 arrays and all 24 complete raw/inlier colored PLY
outputs match byte-for-byte. An independent audit performs 12 fresh serial
SciPy queries, recomputes all threshold metrics from saved distances, recalculates
six structural evaluations with the archived preceding code, and verifies every
PLY coordinate/RGB byte, source/library fingerprint, report hash and timing
summary. Twelve negative controls cover counts/duplicates, nonfinite/negative
voxel sizes, wrong data/baseline fingerprints and existing destinations without
replacing previous outputs.

Forty-two relevant Python tests pass. They cover actual 65,535/65,536/65,537
boundaries, small/empty inputs, forced parallel queries, ties/duplicates,
strided/reversed arrays, extreme finite scales, caller ownership, independent
directions, nonfinite errors, complete metrics/PLY parity with transforms and
voxelization, core/experimental imports, AWD/SCS and the CLI invocation.
Syntax, whitespace and document-link checks pass. The existing upstream parity
harness tests could not be collected on Windows because their unchanged module
imports Unix-only `resource`; that harness was not run or modified. The measured
core strategy, CLI and direct serial/structural/output comparisons above provide
the validation scope for this change. Rust binaries and GPU selection are unchanged.

The [benchmark script](../cloudanalyzer/scripts/benchmark_map_nn_parallel.py)
requires `--source-root`, original `--reference`/`--compared`/`--input-manifest`,
archived `--baseline`/`--baseline-sha256`, and new `--work-dir`/`--output` paths.
Defaults are `--repeats 5 --structure-voxel-size 3`. Size `0` can measure an
explicit NN-only workload in a new report; all six recorded comparisons use
size `3`. Reports retain observations, complete parity/first results, input/
distance/PLY fingerprints, worker policy and timing scope.

- PandaSet: [initial](../benchmarks/cpu/map-evaluate/real-source/windows-pandaset019-nn-parallel-initial.json), [final](../benchmarks/cpu/map-evaluate/real-source/windows-pandaset019-nn-parallel-final.json)
- Autoware: [initial](../benchmarks/cpu/map-evaluate/real-source/windows-autoware-planning-nn-parallel-initial.json), [final](../benchmarks/cpu/map-evaluate/real-source/windows-autoware-planning-nn-parallel-final.json)
- Tokyo: [initial](../benchmarks/cpu/map-evaluate/real-source/windows-tokyo-takanawadai-nn-parallel-initial.json), [final](../benchmarks/cpu/map-evaluate/real-source/windows-tokyo-takanawadai-nn-parallel-final.json)
