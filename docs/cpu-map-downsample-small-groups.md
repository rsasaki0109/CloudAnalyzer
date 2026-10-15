# Bounded small-group centroid means

The optional resident MapEval centroid helper now batches sufficiently many
2–8-point voxel means by equal group size. With requested 0.2 m downsampling,
complete API evaluation improves by 55.24–70.38% across four eligible PandaSet/
Tokyo comparisons, with 12/12 faster timed pairs, against the preceding
[singleton/preallocated implementation](cpu-map-downsample-centroids.md).
Autoware is ineligible and shows mixed timings; no acceleration is claimed there.
All six comparisons and all observations are retained below.

Clocks include requested downsampling, both fresh NN trees/queries, threshold
statistics, Gaussian/Wasserstein work and SCS index construction/destruction.
Input loading, CLI startup/serialization and PLY writing are excluded. This is
the core request's optional `downsample_voxel_size` path. Its default remains
zero, and `ca map-evaluate` still does not expose a downsampling option. These
results do not measure disabled downsampling or the streaming lane.

The later [resident mean-sum reduction](cpu-map-mean-sums.md) is a separate
aggregation change. Reproduce this small-group delta with its frozen sources;
current production includes both changes.

## Arithmetic and eligibility

Singleton batching, adjacent sorted-key boundaries and preallocated output
remain. The preceding helper computes every multi-point group's mean separately.
The new helper gathers equal-length small groups into arrays shaped `(B, K, 3)`
and uses `np.mean(..., axis=1)` for `K=2..8`. Original point order inside each
group and the output's lexicographic voxel order are preserved. Mean results are
written into their original output slots, rather than concatenating groups by
size. No worker pool or different averaging formula is introduced.

At least 4,096 eligible small groups are required. Inputs must be finite and
bounded by `f64_max / 8 / 2` in magnitude, leaving headroom for every batched
addition. NumPy's underflow policy must be `ignore`. Sparse small groups,
non-finite/extreme coordinates, requested underflow reporting and groups larger
than eight keep the preceding ordered multi-point mean calls. Existing
singleton handling remains. Error settings are read, not modified.

The 4,096-group cutoff is conservative, informed by these workloads, and is not
claimed to be an optimal crossover across machines or group distributions.
All public signatures, disabled/empty behavior, classifications, protocols,
sampling, downstream metrics and artifacts remain unchanged. No persistent
point/index cache is added.

## Bounded gather work

Each small-group batch gathers at most 65,536 points: `B=floor(65536/K)` groups.
On this 64-bit host, row-index payload is at most 524,288 bytes, gathered XYZ
at most 1,572,864 bytes, and the mean result at most 786,432 bytes (for `K=2`).
The sum of those payload bounds is 2,883,584 bytes. Their lifetimes may differ;
this is not measured peak RSS or a total memory cap. Additional group-size/
eligibility/slot arrays use `O(V)` storage, and the existing input/key/sort/output
arrays remain. Large groups still use per-group means. No gather is sized to
all eligible points in the map.

## Controlled real-source comparison

[Real-source preparation](c2c-real-source.md) supplies original-coordinate
subsets with attribution/licenses. No generated benchmark points, displacement,
noise or extra pre-thinning is added. Tokyo retains its preceding original-point
selection. These are disjoint-index subsets of one measured source with possible
geometric duplicates, rather than independent surveys or accuracy ground truth.
Generated fixtures serve correctness tests.

Both workflows start with identical full subsets and request the existing
0.2 m centroid operation inside clocks. All centroid coordinates/counts are
identical. The control already includes every preceding MapEval optimization,
including singleton batching/preallocation; this comparison isolates the
additional multi-point mean path. Recorded eligibility describes current policy
for each input, including the control, whose implementation still uses separate
multi-point mean calls.

| Source | Input estimated / reference points | Output estimated / reference centroids | 2–8-point groups, estimated / reference | Eligible |
| --- | --- | --- | --- | --- |
| PandaSet 019 | 531,908 / 531,908 | 257,703 / 257,981 | 97,245 / 97,296 | yes / yes |
| Autoware planning | 878,920 / 878,921 | 876,199 / 876,270 | 2,719 / 2,649 | no / no |
| Tokyo Takanawadai | 941,933 / 941,933 | 441,512 / 441,684 | 290,749 / 290,581 | yes / yes |

The protocol is 3 m structural voxels, minimum 100 points in both maps, neighbor
radius 5 and thresholds `(0.2,0.1,0.08,0.05,0.01)` m. Alignment is disabled.
Supported AWD cells remain 743, 3,933 and 1,638 respectively. Windows, 12 logical
CPUs, Python 3.12.10, NumPy 2.3.4, SciPy 1.17.1 and Open3D 0.19.0 use
`OMP_NUM_THREADS=12` and `OPENBLAS_NUM_THREADS=1` before imports. NumPy policy
is recorded as divide/overflow/invalid=`warn`, underflow=`ignore`. Production
does not change environment variables or error policy. Twelve source/config/
test/controller fingerprints and the compiled tree fingerprint are recorded
with archived copies.

Each source/pass starts a fresh process. Full centroid/Gaussian/distance/PLY
parity, an excluded first timing pair and one warm-up pair precede three
alternating timed pairs. Tracing hooks are disabled during timing. Sources run
PandaSet → Autoware → Tokyo, then reverse that order. No builds, tests, scans or
edits run during comparisons. All observations remain.

| Source/pass | Singleton-only median, ms | Small-group-policy median, ms | Median paired shortening | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| PandaSet initial | 4,304.90 | 2,256.33 | +55.24% | 3/3 |
| PandaSet final | 7,289.88 | 2,258.20 | +70.38% | 3/3 |
| Autoware initial | 6,150.35 | 6,740.48 | −16.50% | 1/3 |
| Autoware final | 6,765.26 | 6,723.07 | +2.62% | 3/3 |
| Tokyo initial | 9,979.21 | 3,082.53 | +66.76% | 3/3 |
| Tokyo final | 11,206.84 | 4,167.21 | +68.16% | 3/3 |

Shortening is `median(100 * (1 - small_ms / singleton_ms))`, rather than the
ratio of marginal medians. Eligible inputs improve in all 12 timed pairs.
The full set improves in 16/18 pairs; both slower pairs are Autoware. Its
preceding mean path is retained, but observed passes differ by −16.50% and
+2.62%, so there is no stable acceleration result for that workload. These
measurements do not establish the cause of those timing differences.

Scheduling, power, other applications, allocator/page retention and warmed
caches are uncontrolled, and pass variation is material. This is a scoped
resident API result for the stated preprocessing/protocol, not a full CLI/
cold-file result. Separate acceleration percentages are not additive. No GPU/
browser/WASM speed is claimed.

## Validation and reproduction

All 72 parity/first/warm-up/timed reports match, including every NN/structural
metric, support count and classification/protocol field. All 24 complete
centroid arrays, 24 full f64 NN arrays, 24 complete Gaussian maps and 24 complete
raw/inlier PLYs match byte-for-byte. Checks cover 12,605,396 saved centroid XYZ
records and 192,880 saved Gaussian records, including cells below AWD support.
All outputs also match the preceding independently recorded singleton-stage
results for each source/pass.

An independent audit rebuilds 12 centroid arrays with the archived helper,
checks singleton/small-group counts and eligibility, recomputes 12 serial NN
queries and six structural evaluations, verifies complete Gaussian exports,
and checks every PLY coordinate/RGB value. Original/source/library fingerprints,
report hashes, timing summaries and grouping/index metadata are checked.
Sixteen negative controls reject wrong inputs/baselines/counts, invalid voxel
sizes, duplicate options and existing destinations without replacing data.

The related suite passes 193 Python tests. Thirty-six new cases cover all group
sizes 2–8, contiguous/strided/reversed inputs and unchanged input bits, the
4,095/4,096/4,097-group cutoff, batches around 65,536 gathered points, mixed
lengths/output order, cancellation/signed zero, subnormals and large finite
values, underflow-reporting fallback/error-state preservation, extreme overflow,
large-group fallback and complete resident-report parity. Preceding singleton,
grouping/covariance, streaming, NN/SCS, PLY, core/experimental import and default
resident CLI tests also pass. The existing Windows upstream-harness `resource`
import limitation remains; that separate harness is not included. Rust binaries
and GPU selection are unchanged.

The [benchmark script](../cloudanalyzer/scripts/benchmark_map_downsample_small_groups.py)
requires `--source-root`, original `--reference`/`--compared`/`--input-manifest`,
archived `--baseline`/`--baseline-sha256` and new `--work-dir`/`--output` paths.
Defaults are `--repeats 3 --downsample-voxel-size 0.2 --structure-voxel-size 3`;
all six comparisons use them. The script requires ignored underflow for this
measured policy and records all NumPy error settings. Its downsampling option
is a benchmark argument, not a new `ca map-evaluate` option. Centroid snapshots
contain row-major `<f8` XYZ; Gaussian files use `<4q12d` records (128 bytes).
Records retain all times, full reports, fingerprints, singleton/small-group
counts and eligibility. Reproduce preceding deltas with their frozen sources;
current production also includes this conditional small-group mean path.

- PandaSet: [initial](../benchmarks/cpu/map-evaluate/real-source/windows-pandaset019-downsample-small-groups-initial.json), [final](../benchmarks/cpu/map-evaluate/real-source/windows-pandaset019-downsample-small-groups-final.json)
- Autoware: [initial](../benchmarks/cpu/map-evaluate/real-source/windows-autoware-planning-downsample-small-groups-initial.json), [final](../benchmarks/cpu/map-evaluate/real-source/windows-autoware-planning-downsample-small-groups-final.json)
- Tokyo: [initial](../benchmarks/cpu/map-evaluate/real-source/windows-tokyo-takanawadai-downsample-small-groups-initial.json), [final](../benchmarks/cpu/map-evaluate/real-source/windows-tokyo-takanawadai-downsample-small-groups-final.json)
