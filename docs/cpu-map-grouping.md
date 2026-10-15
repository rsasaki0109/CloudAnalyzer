# Bounded scalar voxel grouping for resident MapEval

`ca map-evaluate` now groups integer voxel coordinates using an exact scalar
code when their local coordinate ranges fit signed int64. Complete resident
evaluation improves by 45.73–54.44% across six real-source comparisons, with
30/30 faster timed pairs, against the immediately preceding
[covariance implementation](cpu-map-covariance.md). Both workflows already
include parallel exact NN queries, indexed SCS and symmetric covariance sums.
Both point trees/queries, thresholds, voxel grouping, means/covariances,
Wasserstein calculations and SCS index construction/destruction are included.
Input loading, CLI startup/serialization and PLY writing are excluded.

The subsequent [stable streaming groups](cpu-map-stream-groups.md) accelerate
the separate chunked AWD/SCS lane. Reproduce the resident deltas here with their
frozen sources; current production also includes the chunk-builder change.

## Integer encoding and ordering

The preceding grouping calls `np.unique(indices, axis=0)` on three-int64 rows.
For eligible inputs, the new helper subtracts integer per-axis minima and
constructs a mixed-radix scalar code:

```text
wx = max_x - min_x + 1   (similarly wy and wz)
code = ((x - min_x) * wy + (y - min_y)) * wz + (z - min_z)
```

Numeric `np.unique(code)` returns unique codes, inverse memberships and counts.
Integer division/remainder reconstructs the XYZ keys. Codes increase in XYZ
lexicographic order and cannot collide, so unique keys, dictionary insertion
order, counts and every point's membership match the preceding grouping.
All arithmetic uses integers; large absolute keys do not pass through f64.
The original floor-to-int64 voxel-coordinate calculation remains unchanged.

The helper serves resident Gaussian construction only. Mean accumulation,
centering, covariance aggregation/division, singleton handling, Wasserstein/SCS
formulas and support checks are unchanged. Point-order reductions retain all
Gaussian bits. The chunked/streaming accumulator and public options, protocols,
classifications, metrics and artifact formats are unchanged. This is a serial
grouping improvement; it adds no worker pool or cross-evaluation cache.

## Bounds, scratch and fallback

At least 64 points are required. Widths and their product are calculated using
Python integers before fixed-width encoding. The bounding-box capacity
`wx * wy * wz` must be at most `2**63 - 1`. Every local coordinate, radix,
intermediate code and reconstructed key then fits signed int64. Wider ranges,
including spans across the full signed-int64 domain, and small/empty inputs
retain the preceding axis grouping. The capacity is a safety bound, not a grid
allocation: sparse coordinates never allocate the entire bounding box.

The original integer coordinate array still occupies `24 * N` bytes. The new
scalar-code buffer occupies `8 * N` bytes, with one `8 * N` axis-offset temporary
during encoding. Numeric uniqueness has its own sort/index/inverse/count
storage, and key reconstruction uses `O(V)` arrays for `V` unique voxels.
Those buffers are scoped to grouping; no code/index is retained across calls.
These are array payloads, not measured peak RSS or a memory cap. No claim is
made about a universally optimal 64-point crossover or other machines.

| Source | Estimated / reference points | Estimated / reference voxels | Bounding-box capacities, estimated / reference | Scalar-code bytes, estimated / reference |
| --- | --- | --- | --- | --- |
| PandaSet 019 | 531,908 / 531,908 | 9,213 / 9,186 | 255,420 / 259,380 | 4,255,264 / 4,255,264 |
| Autoware planning | 878,920 / 878,921 | 12,024 / 12,038 | 244,920 / 287,938 | 7,031,360 / 7,031,368 |
| Tokyo Takanawadai | 941,933 / 941,933 | 2,879 / 2,880 | 28,224 / 28,224 | 7,535,464 / 7,535,464 |

## Controlled real-source comparison

[Real-source preparation](c2c-real-source.md) supplies original-coordinate
subsets with attribution/licenses. No additional thinning, generated points,
displacement or noise is added. Tokyo retains its preceding original-point
selection. These are disjoint-index subsets of one measured source with
possible geometric duplicates, rather than independent surveys or accuracy
ground truth. Generated fixtures serve correctness tests.

Requests use the CLI default protocol: 3 m voxels, minimum 100 points in both
maps, neighbor radius 5 and thresholds `(0.2,0.1,0.08,0.05,0.01)` m. Alignment
and downsampling are disabled. The control is the archived preceding module,
including all previous map-evaluation optimizations. The new workflow uses the
production scalar-grouping helper.

Windows, 12 logical CPUs, Python 3.12.10, NumPy 2.3.4, SciPy 1.17.1 and
Open3D 0.19.0 use `OMP_NUM_THREADS=12` and `OPENBLAS_NUM_THREADS=1` before
imports. Production does not change environment variables. Nine source/config/
test/controller fingerprints and the compiled tree fingerprint are recorded
with archived copies.

Each source/pass starts a fresh process. Full Gaussian/distance/PLY parity and
grouping-bound capture, an excluded first timing pair and two warm-up pairs precede five
alternating timed pairs. Tracing hooks are disabled during timing. Sources run
PandaSet → Autoware → Tokyo, then reverse that order. No builds, tests, scans
or edits run during comparisons. All observations remain.

| Source/pass | Axis-grouping median, ms | Scalar-grouping median, ms | Median paired shortening | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| PandaSet initial | 2,520.36 | 1,259.39 | +50.06% | 5/5 |
| PandaSet final | 3,683.77 | 1,835.34 | +47.73% | 5/5 |
| Autoware initial | 5,787.52 | 3,320.17 | +45.73% | 5/5 |
| Autoware final | 5,781.75 | 2,994.57 | +47.94% | 5/5 |
| Tokyo initial | 5,057.84 | 2,307.12 | +54.39% | 5/5 |
| Tokyo final | 5,356.31 | 2,474.30 | +54.44% | 5/5 |

Shortening is `median(100 * (1 - packed_ms / axis_ms))`, rather than the ratio
of marginal medians. All 30 timed pairs improve. Passes still vary; scheduling,
power, other applications, allocator/page retention and warmed caches are
uncontrolled. This is a scoped resident-evaluation result, not a full CLI/
cold-file or streaming measurement. Separate acceleration percentages are not
additive. No GPU/browser/WASM speed is claimed.

## Validation and reproduction

All 108 parity/first/warm-up/timed results match, including every metric, support
count and classification/protocol field. All 24 full f64 distance arrays,
24 complete raw/inlier PLY outputs and 24 complete voxel-Gaussian maps match
byte-for-byte. Gaussian records contain integer keys/counts, all three mean
values and all nine covariance entries in insertion order: 192,880 saved voxel
records are checked, including cells below AWD support. They also match the
preceding independently recorded covariance results.

An independent audit compares all integer keys, counts and 9,411,046 point
memberships across 12 real-source groupings with the preceding axis algorithm,
then reconstructs every original integer key from its membership. It recomputes
12 serial point queries, six structural evaluations and 12 Gaussian maps with
the archived module. Every PLY coordinate/RGB value, full Gaussian byte stream,
source/library fingerprint, report hash, timing summary, grouping bound and SCS
index metadata is checked. Twelve negative controls reject wrong data/baselines/
counts, invalid voxel sizes, duplicate options and existing destinations without
replacing data.

The related Python suite passes 112 tests. Twenty-five new cases cover
contiguous/strided/reversed keys, large positive/negative origins, int64 minima/
maxima, constant axes, empty/singleton/63/64/65-point dispatch, the largest safe
capacity, the first unsafe capacity, full signed spans, bounding-box product
overflow and all Gaussian bits at multiple voxel sizes. Preceding covariance,
NN/SCS, resident/streaming, PLY, core/experimental import and CLI tests pass.
The existing Windows upstream-harness `resource` import limitation is unchanged;
that separate harness is not included. Rust binaries and GPU selection are
unchanged.

The [benchmark script](../cloudanalyzer/scripts/benchmark_map_grouping.py)
requires `--source-root`, original `--reference`/`--compared`/`--input-manifest`,
archived `--baseline`/`--baseline-sha256` and new `--work-dir`/`--output` paths.
Defaults are `--repeats 5 --structure-voxel-size 3`; all six comparisons use
them. Records retain original hashes, all timings, full parity and Gaussian/
distance/PLY fingerprints, plus integer grouping minima/widths/capacities.
Gaussian files use `<4q12d` records (128 bytes), as in the preceding covariance
benchmark. Reproduce preceding deltas with their respective frozen sources;
current production also includes scalar grouping.

- PandaSet: [initial](../benchmarks/cpu/map-evaluate/real-source/windows-pandaset019-grouping-initial.json), [final](../benchmarks/cpu/map-evaluate/real-source/windows-pandaset019-grouping-final.json)
- Autoware: [initial](../benchmarks/cpu/map-evaluate/real-source/windows-autoware-planning-grouping-initial.json), [final](../benchmarks/cpu/map-evaluate/real-source/windows-autoware-planning-grouping-final.json)
- Tokyo: [initial](../benchmarks/cpu/map-evaluate/real-source/windows-tokyo-takanawadai-grouping-initial.json), [final](../benchmarks/cpu/map-evaluate/real-source/windows-tokyo-takanawadai-grouping-final.json)
