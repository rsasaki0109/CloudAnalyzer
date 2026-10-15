# Bounded Gaussian Wasserstein batches for MapEval

Resident MapEval now computes eligible Gaussian Wasserstein distances in
bounded NumPy batches. Complete resident evaluation improves by 9.05–27.16%
across six real-source comparisons, with 28/30 faster timed pairs, against the
preceding [coordinate-sum implementation](cpu-map-mean-sums.md). Both fresh NN
trees/queries, threshold statistics, Gaussian construction, Wasserstein and
SCS index construction/destruction are included. Loading, imports, parity
tracing, PLY writing and CLI startup/serialization are excluded.

The distance component alone improves by 87.71–91.41%, with 30/30 faster pairs.
That component percentage does not describe whole-evaluation acceleration.
All per-voxel distance records and complete evaluation outputs match the
preceding scalar code by their bytes.

The later [bounded SCS statistics](cpu-map-scs-batch.md) add a separate
component improvement. Reproduce this Wasserstein delta with its frozen
sources; current production includes both changes.

## Formula and bounded work

Supported estimated/reference Gaussian pairs are collected in estimated-map
insertion order, at most 256 pairs at a time. Missing reference cells or cells
below the support threshold do not enter a batch. The resulting distance
dictionary retains the original key and value order used by AWD/SCS.

The batched formula keeps the scalar implementation's sequence:

1. Apply the same mean/covariance identity tolerances and return zero for
   identical distributions.
2. Symmetrize and regularize both covariances with the `1e-6` eigenvalue floor,
   reconstructing each with diagonal-matrix multiplication.
3. Decompose the reconstructed first covariance again, then form its symmetric
   matrix square root and the symmetrized Bures middle matrix.
4. Decompose the middle matrix, sum its square-root eigenvalues, combine mean
   displacement and covariance traces, and apply the existing roundoff clamp.

The implementation keeps diagonal matrix multiplications and the second
decomposition of the reconstructed covariance. It does not replace them with
different algebra that could change rounding. Mean-displacement norms retain
the scalar `row @ row` dot operation. Matrix decompositions and products are
stacked; no new worker pool or different distance formula is introduced. The
public single-Gaussian function is unchanged.

Batches below 128 pairs retain ordered scalar calls. The batched path requires
ordinary f64 arrays of shapes `(3,)` and `(3,3)`, finite values bounded in
magnitude by `1e100`, and NumPy's default error settings (`divide/over/invalid`
warn, underflow ignore). The bound leaves ample headroom for covariance
products and mean norms. Nonstandard arrays, non-finite/extreme values or
explicitly changed error settings retain scalar coercion, warnings and
exceptions. Numerical-backend errors propagate; no failed result is fabricated.
Production does not change error settings or thread environment variables.

Scratch scales with the bounded batch, rather than all supported pairs.
The two mean and two covariance input stacks have `192 * B` bytes of payload:
49,152 bytes at `B=256`. Each additional `(B,3,3)` f64 array has `72 * B` bytes
(18,432 at that bound). Eigenvectors, diagonals, products, active selections,
other temporaries, Python objects and numerical-library workspace add storage.
These figures are array payloads, not measured peak RSS or a memory cap. No
cross-evaluation cache is added.

The helper is shared by resident and streaming structural evaluation. Tests
cover both, but the performance results below measure resident evaluation and
its isolated distance component. They establish no streaming speed result.

## Real-source measurement

[Source preparation and attribution](c2c-real-source.md) provide fixed original
XYZ coordinates. No new thinning, displacement, noise or generated points is
added. Tokyo retains its preceding original-point selection. These are
disjoint-index subsets of one measured source with possible geometric
duplicates, rather than independent surveys or accuracy ground truth.
Generated fixtures are used only for correctness tests.

| Source | Estimated / reference points | Supported Gaussian pairs |
| --- | ---: | ---: |
| PandaSet 019 | 531,908 / 531,908 | 1,202 |
| Autoware planning | 878,920 / 878,921 | 3,948 |
| Tokyo Takanawadai | 941,933 / 941,933 | 1,966 |

Requests use the CLI default protocol: 3 m structural voxels, at least 100
points in each map per supported cell, neighbor radius 5 and thresholds
`(0.2,0.1,0.08,0.05,0.01)` m. Alignment and downsampling are disabled. Both
versions include all preceding MapEval optimizations; only Wasserstein
aggregation changes.

Windows, Python 3.12.10, NumPy 2.3.4, SciPy 1.17.1 and Open3D 0.19.0 use 12
logical CPUs, `OMP_NUM_THREADS=12` and `OPENBLAS_NUM_THREADS=1` before imports.
Fourteen source/config/test/controller fingerprints, the preceding baseline
and the compiled SciPy tree are recorded with archived copies.

Each source/pass starts a fresh process. Full Gaussian/distance/PLY parity,
an excluded first timing pair and two warm-up pairs precede five alternating
timed pairs. Sources run PandaSet → Autoware → Tokyo, then reverse that order.
Tracing hooks are disabled during timing. No edits, tests, builds or scans run
during the comparisons. All observations remain.

| Source/pass | Scalar median, ms | Batched median, ms | Median paired shortening | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| PandaSet initial | 2,645.18 | 2,042.25 | +22.89% | 5/5 |
| PandaSet final | 1,560.74 | 1,309.11 | +15.75% | 5/5 |
| Autoware initial | 4,171.03 | 2,910.17 | +24.38% | 5/5 |
| Autoware final | 3,755.71 | 2,742.56 | +27.16% | 5/5 |
| Tokyo initial | 2,507.45 | 2,094.68 | +15.55% | 5/5 |
| Tokyo final | 2,917.24 | 2,822.51 | +9.05% | 3/5 |

After the complete trials, the same process measures the distance component
with the saved full Gaussian maps. These clocks include support filtering,
bounded batching/guards, all distance calculations and distance dictionary
construction. They exclude moment construction, SCS, file loading and output
checking/hashing. The component has its own excluded first pair, two warm-ups
and five alternating timed pairs.

| Source/pass | Scalar distance median, ms | Batched distance median, ms | Median paired shortening | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| PandaSet initial | 362.42 | 33.68 | +90.71% | 5/5 |
| PandaSet final | 290.11 | 25.71 | +90.72% | 5/5 |
| Autoware initial | 1,283.02 | 137.93 | +88.94% | 5/5 |
| Autoware final | 707.37 | 85.52 | +87.71% | 5/5 |
| Tokyo initial | 387.86 | 46.15 | +89.67% | 5/5 |
| Tokyo final | 677.75 | 57.35 | +91.41% | 5/5 |

Shortening is `median(100 * (1 - batched_ms / scalar_ms))`, rather than the
ratio of marginal medians. All six whole-evaluation medians improve; two
individual Tokyo final pairs regress. Scheduling, power, other applications,
allocator state and warmed pages/caches are uncontrolled, and passes vary.
No causal diagnosis of that timing variation is established. Component and
complete percentages are not additive. No full CLI/cold-file, streaming,
browser/WASM or GPU speed is claimed.

## Validation and reproduction

All 108 complete parity/first/warm-up/timed reports match. All 24 full f64 NN
distance arrays, 24 complete PLY outputs and 24 Gaussian maps match bytes,
including 192,880 saved voxel records. All twelve complete per-voxel Gaussian
distance arrays match their keys, order and f64 values: 28,464 saved records
are checked. Reports/distances/PLYs also match the independently recorded
preceding coordinate-sum results.

An independent audit recomputes twelve serial point queries, six structural
evaluations, Gaussian summaries and six complete Gaussian-distance sequences
with the archived scalar code. It checks every PLY coordinate/RGB value,
source/library fingerprints, full report hashes, SCS index metadata, timing
summaries and pair ordering. Twelve negative controls reject wrong sources/
baselines, invalid counts/voxel sizes, duplicate arguments and existing
destinations without replacing data.

267 related Python tests pass. The 45 new cases cover f64 parity across
contiguous/strided/reversed arrays and tiny/large magnitudes, identity and
near-identity shortcuts, rank-deficient/indefinite/asymmetric covariances,
127/128 and 255/256/257 boundaries, multiple batches, support/key ordering,
unchanged inputs, non-finite/extreme warning parity, explicit error-policy
fallback, nonstandard array coercion, backend errors, complete resident and
streaming reports, and complete PLY bytes. The existing Windows Unix
`resource` upstream-harness import limitation is unchanged; that separate
harness is excluded.

The [controller](../cloudanalyzer/scripts/benchmark_map_wasserstein_batch.py)
requires `--source-root`, `--reference`, `--compared`, `--input-manifest`,
`--baseline`, `--baseline-sha256` and new `--work-dir`/`--output` destinations.
Defaults are `--repeats 5 --structure-voxel-size 3`; all six records use them.
Gaussian files use `<4q12d` (128 bytes); Gaussian-distance files use `<3qd`
(32 bytes: integer XYZ key and f64 distance). Full reports, inputs,
fingerprints and timings remain in the JSON records. Reproduce preceding
deltas with their respective frozen sources.

- PandaSet: [initial](../benchmarks/cpu/map-evaluate/real-source/windows-pandaset019-wasserstein-batch-initial.json), [final](../benchmarks/cpu/map-evaluate/real-source/windows-pandaset019-wasserstein-batch-final.json)
- Autoware: [initial](../benchmarks/cpu/map-evaluate/real-source/windows-autoware-planning-wasserstein-batch-initial.json), [final](../benchmarks/cpu/map-evaluate/real-source/windows-autoware-planning-wasserstein-batch-final.json)
- Tokyo: [initial](../benchmarks/cpu/map-evaluate/real-source/windows-tokyo-takanawadai-wasserstein-batch-initial.json), [final](../benchmarks/cpu/map-evaluate/real-source/windows-tokyo-takanawadai-wasserstein-batch-final.json)
