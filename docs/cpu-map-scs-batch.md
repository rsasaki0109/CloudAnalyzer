# Bounded equal-length SCS statistics

MapEval now batches eligible SCS neighborhood statistics by equal row length.
The SCS component, including fresh spatial-index construction and queries,
is 10.38–31.92% shorter across six real-source comparisons, with 28/30 faster
timed pairs, against the preceding [Wasserstein-batch implementation](cpu-map-wasserstein-batch.md).
All ordered neighborhood values, per-center terms and complete outputs match.

Complete resident evaluation has a smaller, mixed result: paired medians range
from −2.30% to +5.99%, with 16/30 faster pairs. PandaSet final regresses in all
five pairs; several other medians are near zero. These measurements do not
establish a consistent whole-evaluation gain. The SCS component improvement
is adopted; its percentage must not describe whole evaluation acceleration.

The later [compact midpoint point trees](cpu-map-nn-midpoint.md) add a separate
measured change. Reproduce this SCS-statistics delta with its frozen sources;
current production includes both changes.

## Arithmetic and eligibility

The existing integer spatial index and one-center-at-a-time neighbor queries
remain. Each center retains its insertion order, and its neighbors retain their
lexicographic order. Empty neighborhoods contribute no SCS term.

The new reducer buffers nonempty rows, groups rows of equal length, and computes
their means and population standard deviations along the contiguous row axis.
Each row contains exactly its original ordered values. Results return to their
original center slots before final aggregation. Means at or below f64 epsilon
retain the existing zero-error `allclose` rule. Positive-mean rows retain
`std / mean`; final `nanmean` and support counts are unchanged. No worker pool
or different variance formula is introduced.

The fast path requires at least 64 distances, plain Python float values,
NumPy's default error settings (`divide/over/invalid` warn, underflow ignore),
and finite nonnegative values bounded by `sqrt(f64_max / V) / 4`, where `V` is
the number of supported error cells. Each neighborhood is a subset of those
values; this bound leaves headroom for squared deviations and their sum.
Safety is checked once before regrouping any statistics. Nonstandard values,
non-finite/negative/extreme inputs or explicitly changed error settings use
ordered scalar calls, retaining dtype coercion, warnings and exceptions.
Production does not modify error settings or thread environment variables.

Windows contain at most 1,024 buffered centers and 65,536 buffered neighbor
values. Windows below 16 rows and same-length cohorts below four rows retain
scalar reductions. A single neighborhood larger than the value budget keeps
the preceding scalar path. The window bounds constrain buffered work, rather
than the size of the one-center query already present in the implementation.

A look-ahead neighborhood may coexist with a window while it is flushed.
Input guards temporarily materialize `8 * V` bytes of distance values and
release them before queries. Each f64 matrix for buffered values has at most
`8 * 65,536 = 524,288` bytes of payload; selection, centered-deviation and
squared-deviation temporaries can coexist. Python row/reference lists, ordered
term outputs, index storage and numerical-library/allocator state add memory.
These are array payloads, not measured peak RSS or a memory cap. No persistent
cache or all-center neighbor table is added to production.

The shared structural helper covers resident and streaming evaluation.
Correctness tests cover both; these performance measurements cover resident
evaluation and its isolated SCS component, without a streaming-speed claim.

## Real-source comparison

[Source preparation and attribution](c2c-real-source.md) supply fixed original
XYZ coordinates without new thinning, displacement, noise or generated points.
Tokyo retains its preceding original-point selection. These are disjoint-index
subsets of one measured source with possible geometric duplicates, rather than
independent surveys or accuracy ground truth. Generated fixtures are used only
for correctness tests.

| Source | Estimated / reference points | Supported error cells | Nonempty SCS centers |
| --- | ---: | ---: | ---: |
| PandaSet 019 | 531,908 / 531,908 | 1,202 | 1,200 |
| Autoware planning | 878,920 / 878,921 | 3,948 | 3,943 |
| Tokyo Takanawadai | 941,933 / 941,933 | 1,966 | 1,966 |

Requests use 3 m structural voxels, minimum 100 points in both maps per
supported cell, neighbor radius 5 and thresholds `(0.2,0.1,0.08,0.05,0.01)` m.
Alignment and downsampling are disabled. Both versions already include all
preceding MapEval optimizations; only SCS statistics change.

Windows, Python 3.12.10, NumPy 2.3.4, SciPy 1.17.1 and Open3D 0.19.0 use 12
logical CPUs, `OMP_NUM_THREADS=12` and `OPENBLAS_NUM_THREADS=1` before imports.
Fifteen source/config/test/controller fingerprints, the preceding baseline
and compiled SciPy tree are recorded with archived copies.

Each source/pass starts a fresh process. Full Gaussian/distance/PLY parity,
an excluded first timing pair and two warm-up pairs precede five alternating
timed pairs. Sources run PandaSet → Autoware → Tokyo, then reverse that order.
Tracing hooks are disabled during timing. No edits, builds, tests or scans run
during comparisons, and all observations remain.

Complete clocks include both fresh NN trees/queries, threshold statistics,
voxel grouping, means/covariances, Wasserstein and SCS index construction/
destruction. Loading, imports, hashing, parity tracing, PLY writing and CLI
startup/serialization are excluded.

| Source/pass | Scalar SCS policy median, ms | Batched SCS policy median, ms | Median paired shortening | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| PandaSet initial | 1,717.79 | 1,693.74 | +5.99% | 3/5 |
| PandaSet final | 1,826.21 | 1,854.88 | −2.30% | 0/5 |
| Autoware initial | 6,308.02 | 6,300.66 | +0.12% | 3/5 |
| Autoware final | 2,277.99 | 2,154.78 | +0.99% | 3/5 |
| Tokyo initial | 2,898.49 | 3,325.43 | +1.92% | 4/5 |
| Tokyo final | 1,851.58 | 1,930.51 | +1.30% | 3/5 |

After the complete trials, the same process measures SCS with the full
per-voxel distances reconstructed from saved Gaussian maps. Component clocks
include fresh SCS index construction/queries, guards, neighborhood statistics,
ordered term-array creation and final `nanmean`. Gaussian/distance setup,
file loading and checking/hashing are excluded. The component has its own
excluded first pair, two warm-ups and five alternating pairs.

| Source/pass | Scalar SCS median, ms | Batched SCS median, ms | Median paired shortening | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| PandaSet initial | 190.24 | 180.02 | +10.38% | 4/5 |
| PandaSet final | 308.17 | 252.37 | +18.11% | 5/5 |
| Autoware initial | 1,462.74 | 1,006.49 | +31.19% | 4/5 |
| Autoware final | 375.94 | 252.81 | +31.92% | 5/5 |
| Tokyo initial | 263.98 | 227.84 | +15.81% | 5/5 |
| Tokyo final | 312.17 | 273.80 | +11.51% | 5/5 |

Shortening is `median(100 * (1 - batched_ms / scalar_ms))`, rather than the
ratio of marginal medians. For example, Tokyo's marginal whole medians are
larger with batching while its paired median shortens. Timing variation is
material; scheduling, power, other applications, allocator state and warmed
pages/caches are uncontrolled. No causal diagnosis of the variation is
established. Component and complete percentages are not additive. No full
CLI/cold-file, streaming, browser/WASM or GPU gain is claimed.

## Validation and reproduction

All 108 complete parity/first/warm-up/timed reports match. All 24 full f64 NN
distance arrays, 24 complete PLY outputs and 24 Gaussian maps match bytes,
including 192,880 saved voxel records. Twelve full Gaussian-distance arrays
contain 28,464 matching records. Twelve full SCS-term arrays contain 28,436
matching records, including center keys, neighbor counts, insertion order and
f64 terms. Complete reports/distances/PLYs and Gaussian-distance hashes also
match independently recorded preceding Wasserstein-batch results.

An independent audit recomputes twelve serial point queries, six structural
evaluations, Gaussian summaries and six complete Gaussian-distance sequences.
It checks 14,232 ordered neighborhoods against integer cube enumeration,
including isolated centers, then independently derives each scalar SCS term.
It verifies every PLY coordinate/RGB value, all source/library hashes, complete
report hashes, SCS index metadata, pair ordering and timing summaries. Twelve
negative controls reject wrong sources/baselines, invalid counts/voxel sizes,
duplicate arguments and existing destinations without replacing data.

359 related Python tests pass. The 92 new cases cover equal-length reduction
bits around pairwise-sum boundaries, tiny/large magnitudes, small/mixed windows,
63/64/65-cell and 1,023/1,024/1,025-center cases, radius 1/2/5, value-budget
boundaries and oversized neighborhoods, center order, empty cells, signed zero,
zero/tiny mean branches, non-finite/negative/extreme warning parity, explicit
error-policy and nonstandard-dtype fallback, query failures, unchanged inputs,
complete resident/streaming reports and PLY bytes. The existing Windows Unix
`resource` upstream-harness import limitation is unchanged; that separate
harness is excluded.

The [controller](../cloudanalyzer/scripts/benchmark_map_scs_batch.py) requires
`--source-root`, `--reference`, `--compared`, `--input-manifest`, `--baseline`,
`--baseline-sha256` and new `--work-dir`/`--output` destinations. Defaults are
`--repeats 5 --structure-voxel-size 3`; all six records use them. Gaussian files
use `<4q12d` (128 bytes); Gaussian-distance files use `<3qd` (32 bytes); SCS
term files use `<3qQd` (40 bytes: integer XYZ center, unsigned neighbor count,
f64 term). Full timings/reports, input/source fingerprints and paths remain
in the records. Reproduce preceding deltas with their frozen sources.

- PandaSet: [initial](../benchmarks/cpu/map-evaluate/real-source/windows-pandaset019-scs-batch-initial.json), [final](../benchmarks/cpu/map-evaluate/real-source/windows-pandaset019-scs-batch-final.json)
- Autoware: [initial](../benchmarks/cpu/map-evaluate/real-source/windows-autoware-planning-scs-batch-initial.json), [final](../benchmarks/cpu/map-evaluate/real-source/windows-autoware-planning-scs-batch-final.json)
- Tokyo: [initial](../benchmarks/cpu/map-evaluate/real-source/windows-tokyo-takanawadai-scs-batch-initial.json), [final](../benchmarks/cpu/map-evaluate/real-source/windows-tokyo-takanawadai-scs-batch-final.json)
