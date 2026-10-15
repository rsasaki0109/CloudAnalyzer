# Exact centroid preprocessing for resident MapEval

Resident evaluation with `MapEvaluateRequest(downsample_voxel_size=0.2)` now
avoids repeated classification and batches singleton centroid means. Complete
evaluation improves by 33.67–81.01% across six real-source comparisons, with
18/18 faster timed pairs, against the archived preceding implementation.
Requested downsampling, both fresh NN trees/queries, thresholds, Gaussian/
Wasserstein work and SCS index construction/destruction are included. Input
loading, CLI startup/serialization and PLY writing are excluded.

The subsequent [conditional small-group means](cpu-map-downsample-small-groups.md)
add a measured multi-point path. Reproduce the singleton-stage deltas here with
their frozen sources; current production also includes that conditional path.

This is an optional preprocessing path in the core API and shared centroid
helper. The current `ca map-evaluate` command does not expose a downsampling
option, and the request default remains zero. These measurements do not claim
a speedup for disabled downsampling or streaming evaluation.

## Group boundaries and mean arithmetic

The preceding helper lexicographically sorts voxel keys, then calls
`np.unique(..., axis=0, return_index=True)` on those already sorted keys.
It computes one mean ndarray per voxel, retains all those arrays in a list and
finally stacks them. The new helper finds group starts from adjacent sorted-key
differences and writes directly into one preallocated centroid output.

The original lexicographic sort and point order inside each voxel are preserved.
Multi-point groups use the same `slice.mean(axis=0)` arithmetic in the same
group order. For finite inputs, singleton groups use NumPy means on batches
shaped `(B, 1, 3)`, reducing the one-point axis. This retains NumPy's sum/divide
behavior for signed zero and subnormals; the input point is not substituted
directly. Batches contain at most 65,536 singleton groups.

Non-finite coordinates retain ordered per-group mean calls. Multi-point
overflow behavior is preserved. Empty/disabled input behavior, existing
NaN/infinite voxel-size behavior, float64 conversion, centroid/key ordering and
public signatures remain unchanged. No worker pool, alternative averaging rule
or cross-call cache is added. Downstream metrics and artifacts use exactly the
preceding centroid coordinates.

## Array payloads and scope

The output occupies `24 * V` bytes for `V` voxels. Endpoint/index/mask arrays
require additional `O(V)` storage, including singleton slots and remaining
group slots. Existing point/key/sort arrays still require `O(N)` storage for
`N` input points. Adjacent-key comparison uses boolean arrays proportional to
`N`; the repeated uniqueness sort/classification is removed.

A maximum singleton block contains 1,572,864 bytes of gathered XYZ and the same
size mean output, totaling 3,145,728 bytes of those two arrays. NumPy working
storage, index arrays, allocation overhead and other evaluation data are
additional. The preceding list of `V` separate centroid ndarrays disappears.
These are array payloads, not measured peak RSS or a total memory cap.

## Controlled real-source workload

[Real-source preparation](c2c-real-source.md) supplies original-coordinate
subsets with attribution/licenses. No generated benchmark points, displacement,
noise or extra pre-thinning is added. Tokyo retains its preceding original-point
selection. These are disjoint-index subsets of one measured source with possible
geometric duplicates, rather than independent surveys or accuracy ground truth.
Generated fixtures serve correctness tests.

Both workflows start with the same full original subsets and request 0.2 m
centroid downsampling inside the clocks. Downsampling is the existing operation
being measured; its output coordinates/counts are identical between control and
current code. This differs from earlier [resident grouping measurements](cpu-map-grouping.md),
which disabled downsampling. It also differs from the separate
[streaming lane](cpu-map-stream-groups.md); percentages are not additive.

| Source | Input estimated / reference points | Output estimated / reference centroids | Singleton voxels, estimated / reference | Supported AWD cells |
| --- | --- | --- | --- | ---: |
| PandaSet 019 | 531,908 / 531,908 | 257,703 / 257,981 | 155,285 / 155,516 | 743 |
| Autoware planning | 878,920 / 878,921 | 876,199 / 876,270 | 873,480 / 873,621 | 3,933 |
| Tokyo Takanawadai | 941,933 / 941,933 | 441,512 / 441,684 | 150,758 / 151,095 | 1,638 |

The protocol remains 3 m structural voxels, minimum 100 points in both maps,
neighbor radius 5 and thresholds `(0.2,0.1,0.08,0.05,0.01)` m. Alignment is
disabled. Both workflows include all preceding parallel NN, indexed SCS,
covariance and resident-grouping optimizations. The earlier chunk-builder
change is present in both modules but is not used by this resident workload.

Windows, 12 logical CPUs, Python 3.12.10, NumPy 2.3.4, SciPy 1.17.1 and
Open3D 0.19.0 use `OMP_NUM_THREADS=12` and `OPENBLAS_NUM_THREADS=1` before
imports. Production does not change environment variables. Eleven source/
configuration/test/controller fingerprints and the compiled tree fingerprint
are recorded with archived copies.

Each source/pass starts a fresh process. Full centroid/Gaussian/distance/PLY
parity, an excluded first timing pair and one warm-up pair precede three
alternating timed pairs. Tracing hooks are disabled during timing. Sources run
PandaSet → Autoware → Tokyo, then reverse that order. No builds, tests, scans or
edits run during comparisons. All observations remain.

| Source/pass | Legacy median, ms | Batched/preallocated median, ms | Median paired shortening | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| PandaSet initial | 4,967.23 | 2,849.78 | +49.86% | 3/3 |
| PandaSet final | 6,785.59 | 3,511.36 | +48.57% | 3/3 |
| Autoware initial | 19,771.00 | 3,866.08 | +79.72% | 3/3 |
| Autoware final | 19,455.29 | 3,695.94 | +81.01% | 3/3 |
| Tokyo initial | 11,582.15 | 7,264.78 | +38.65% | 3/3 |
| Tokyo final | 10,863.38 | 7,205.94 | +33.67% | 3/3 |

Shortening is `median(100 * (1 - batched_ms / legacy_ms))`, rather than the
ratio of marginal medians. All 18 timed pairs improve. Differences between
passes remain material; scheduling, power, other applications, allocator/page
retention and warmed caches are uncontrolled. Singleton prevalence and
multi-point-group costs affect the improvement. This is a scoped resident API
result with the stated preprocessing/protocol, not a full CLI/cold-file result.
No GPU/browser/WASM speed is claimed.

## Validation and reproduction

All 72 parity/first/warm-up/timed reports match, including every NN/structural
metric, support count and classification/protocol field. All 24 complete
centroid arrays, 24 full f64 NN distance arrays, 24 complete Gaussian maps and
24 complete raw/inlier PLYs match byte-for-byte. The centroid checks cover
12,605,396 saved XYZ records. Gaussian checks cover 192,880 saved voxel records
including cells below AWD support. Initial/final passes retain the same
fingerprints and full reports.

An independent audit rebuilds 12 centroid arrays with the archived mean/list
implementation, checks singleton counts, recomputes 12 serial NN queries and
six structural evaluations, verifies complete Gaussian exports and checks every
PLY coordinate/RGB value. Source/library/input hashes, report hashes, timing
summaries and structural grouping/index metadata are checked. Sixteen negative
controls reject wrong inputs/baselines/counts, invalid voxel sizes, duplicate
options and existing destinations without replacing data.

The related suite passes 157 Python tests. Twenty-four new cases cover
contiguous/strided/reversed inputs, multiple voxel sizes, centroid order and
input snapshots, 65,535/65,536/65,537 singleton batches, signed zero, subnormals,
finite f64 extremes, cancellation, non-finite fallback, overflow exceptions,
empty/single-point/disabled inputs, existing NaN/infinite size behavior and full
resident-report parity. Preceding grouping/covariance, streaming, NN/SCS, PLY,
core/experimental import and default resident CLI tests also pass. The existing
Windows upstream-harness `resource` import limitation remains; that separate
harness is not included. Rust binaries and GPU selection are unchanged.

The [benchmark script](../cloudanalyzer/scripts/benchmark_map_downsample_centroids.py)
requires `--source-root`, original `--reference`/`--compared`/`--input-manifest`,
archived `--baseline`/`--baseline-sha256` and new `--work-dir`/`--output` paths.
Defaults are `--repeats 3 --downsample-voxel-size 0.2 --structure-voxel-size 3`;
all six comparisons use them. Its downsampling option is a benchmark argument,
not a new `ca map-evaluate` CLI option. Centroid snapshots contain row-major
`<f8` XYZ, Gaussian files use `<4q12d` records (128 bytes), and records retain
full reports, all times, fingerprints and singleton counts. Reproduce preceding
deltas with their respective frozen sources; current production also includes
this optional centroid-preprocessing change.

- PandaSet: [initial](../benchmarks/cpu/map-evaluate/real-source/windows-pandaset019-downsample-centroids-initial.json), [final](../benchmarks/cpu/map-evaluate/real-source/windows-pandaset019-downsample-centroids-final.json)
- Autoware: [initial](../benchmarks/cpu/map-evaluate/real-source/windows-autoware-planning-downsample-centroids-initial.json), [final](../benchmarks/cpu/map-evaluate/real-source/windows-autoware-planning-downsample-centroids-final.json)
- Tokyo: [initial](../benchmarks/cpu/map-evaluate/real-source/windows-tokyo-takanawadai-downsample-centroids-initial.json), [final](../benchmarks/cpu/map-evaluate/real-source/windows-tokyo-takanawadai-downsample-centroids-final.json)
