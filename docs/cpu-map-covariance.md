# Symmetric grouped covariance for resident MapEval

`ca map-evaluate` now accumulates six independent covariance components with
NumPy `bincount`, then copies their symmetric entries. Complete resident
evaluation improves by 5.13–17.64% across six real-source comparisons, with
27/30 faster timed pairs, against the immediately preceding
[parallel NN](cpu-map-nn-parallel.md) and [indexed SCS](cpu-map-scs-index.md)
implementation. Both point trees/queries, threshold statistics, voxel grouping,
Gaussian/Wasserstein calculations and SCS index construction/destruction are
included. Input loading, CLI startup/serialization and PLY writing are excluded.

The subsequent [scalar voxel grouping](cpu-map-grouping.md) adds another
measured change. Reproduce the covariance delta below with the frozen
pre-grouping sources; current production also includes scalar grouping.

## Arithmetic and scratch

The preceding code materialized every point's `3 x 3` outer product and scattered
all nine components into voxel sums with `np.add.at`. The new code computes one
scalar product per point at a time and accumulates it by voxel with `bincount`.
Each voxel receives products in the original point order, starting from zero.
Finite multiplication produces identical upper/lower entries, so six component
reductions suffice. This changes aggregation work; it adds no worker pool.

Voxel keys/counts, mean accumulation, centering, sample-covariance division,
singleton covariance, dictionary ordering, covariance regularization,
Wasserstein/SCS formulas and support checks are preserved. No global or
cross-evaluation cache is added. The chunked/streaming accumulator is unchanged.
Public options, protocols, classifications, metric names and artifacts stay the
same. Performance measurements here cover resident evaluation.

Non-finite or extreme centered coordinates use the preceding outer-product
scatter. A conservative magnitude limit of `sqrt(f64_max / N) / 2` leaves
headroom for products and accumulation even if all points share one voxel.
This avoids replacing the old overflow/invalid behavior with `bincount` behavior.

The point-product temporary decreases from `72 * N` to `8 * N` bytes.
Component-result arrays additionally require `O(V)` storage (up to two `8 * V`
arrays during reassignment), where `V` is the number of voxels. The existing
centered points, grouping arrays, means and `72 * V` covariance output still
exist. These are array payloads, not measured peak RSS or a memory cap. The
non-finite/extreme fallback retains the preceding allocation.

| Source | Estimated / reference points | Estimated / reference voxels | Old product array per map, bytes | New product array per map, bytes |
| --- | --- | --- | --- | --- |
| PandaSet 019 | 531,908 / 531,908 | 9,213 / 9,186 | 38,297,376 / 38,297,376 | 4,255,264 / 4,255,264 |
| Autoware planning | 878,920 / 878,921 | 12,024 / 12,038 | 63,282,240 / 63,282,312 | 7,031,360 / 7,031,368 |
| Tokyo Takanawadai | 941,933 / 941,933 | 2,879 / 2,880 | 67,819,176 / 67,819,176 | 7,535,464 / 7,535,464 |

## Controlled real-source comparison

[Real-source preparation](c2c-real-source.md) supplies original-coordinate
subsets with source attribution/licenses. No additional thinning, displacement,
noise or generated points is added. Tokyo retains its preceding original-point
selection. These are disjoint-index subsets of one measured source with possible
geometric duplicates, rather than independent surveys or accuracy ground truth.
Generated fixtures serve correctness tests.

Requests use the CLI default protocol: 3 m voxels, minimum 100 points in both
maps, neighbor radius 5 and thresholds `(0.2,0.1,0.08,0.05,0.01)` m. Alignment
and downsampling are disabled. The control is the frozen preceding map module;
both workflows already use parallel exact NN and sparse integer SCS lookup.

Windows, 12 logical CPUs, Python 3.12.10, NumPy 2.3.4, SciPy 1.17.1 and
Open3D 0.19.0 use `OMP_NUM_THREADS=12` and `OPENBLAS_NUM_THREADS=1` before
imports. Production does not change environment variables. Eight source/config/
test/controller fingerprints and the compiled tree fingerprint are recorded
with archived copies.

Each source/pass starts a fresh process. Full Gaussian/distance/PLY parity,
an excluded first timing pair and two warm-up pairs precede five alternating
timed pairs. Tracing hooks are disabled during timing. Sources run PandaSet →
Autoware → Tokyo, then reverse that order. No builds, tests, scans or edits run
during comparisons. All observations remain.

| Source/pass | Scatter median, ms | Grouped median, ms | Median paired shortening | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| PandaSet initial | 2,907.59 | 2,668.13 | +8.49% | 4/5 |
| PandaSet final | 4,270.55 | 3,504.98 | +17.64% | 5/5 |
| Autoware initial | 8,275.54 | 7,424.88 | +5.13% | 3/5 |
| Autoware final | 5,210.05 | 4,764.02 | +8.34% | 5/5 |
| Tokyo initial | 5,406.82 | 4,913.09 | +6.57% | 5/5 |
| Tokyo final | 4,774.33 | 4,488.27 | +5.95% | 5/5 |

Shortening is `median(100 * (1 - grouped_ms / scatter_ms))`, rather than the
ratio of marginal medians. All six paired medians improve, but three individual
pairs regress. Passes vary materially; scheduling, power, other applications,
allocator/page retention and warmed caches are uncontrolled. This is a scoped
resident-evaluation result, not a full CLI/cold-file or streaming measurement.
Separate acceleration percentages are not additive. No GPU/browser/WASM speed
is claimed.

## Validation and reproduction

All 108 parity/first/warm-up/timed results match, including every metric, support
count and classification/protocol field. All 24 full f64 distance arrays,
24 complete raw/inlier PLY outputs and 24 complete voxel-Gaussian maps match
byte-for-byte. Gaussian records include integer keys/counts, three mean values
and all nine covariance entries in insertion order: 192,880 saved voxel records
are checked, including cells below AWD support. Reports, distances and PLYs also
match the preceding independently recorded indexed-SCS results.

An independent audit recomputes 12 serial point queries, six structural
evaluations and 12 Gaussian maps with the archived scatter code. It checks every
PLY coordinate/RGB value, complete Gaussian bytes, source/library fingerprints,
report hashes, timing summaries and identical SCS index metadata. Twelve
negative controls reject wrong data/baselines/counts, invalid voxel sizes,
duplicate options and existing destinations without replacing data.

Eighty-seven related Python tests pass. Twenty new cases cover point order,
strided/reversed inputs, tiny/large magnitudes, cancellation, signed zero,
subnormals, six-component use, empty groups, non-finite/extreme fallback,
overflow exceptions and resident means/covariances at multiple voxel sizes.
Preceding NN/SCS, resident/streaming metrics, PLY, core/experimental import and
CLI tests also pass. The existing Windows upstream-harness `resource` import
limitation is unchanged; that separate harness is not included. Rust binaries
and browser/GPU selection are unchanged.

The [benchmark script](../cloudanalyzer/scripts/benchmark_map_covariance.py)
requires `--source-root`, original `--reference`/`--compared`/`--input-manifest`,
archived `--baseline`/`--baseline-sha256` and new `--work-dir`/`--output` paths.
Defaults are `--repeats 5 --structure-voxel-size 3`; all six comparisons use
them. Gaussian files use `<4q12d` records (128 bytes): XYZ integer key, integer
count, f64 mean and row-major f64 covariance. Fingerprints and timings remain
in the JSON records. Reproduce preceding NN or SCS deltas with their respective
frozen sources; current production also includes scalar grouping.

- PandaSet: [initial](../benchmarks/cpu/map-evaluate/real-source/windows-pandaset019-covariance-initial.json), [final](../benchmarks/cpu/map-evaluate/real-source/windows-pandaset019-covariance-final.json)
- Autoware: [initial](../benchmarks/cpu/map-evaluate/real-source/windows-autoware-planning-covariance-initial.json), [final](../benchmarks/cpu/map-evaluate/real-source/windows-autoware-planning-covariance-final.json)
- Tokyo: [initial](../benchmarks/cpu/map-evaluate/real-source/windows-tokyo-takanawadai-covariance-initial.json), [final](../benchmarks/cpu/map-evaluate/real-source/windows-tokyo-takanawadai-covariance-final.json)
