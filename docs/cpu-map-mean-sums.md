# Point-order coordinate sums for resident MapEval

Resident voxel means now use three NumPy `bincount` reductions for safe inputs
of at least 64 points. Each voxel receives each coordinate in its original point
order, starting from zero. The affected sum reducer is 65.02–74.36% shorter
across twelve real-source input/pass cases, with 60/60 faster timed pairs.
This is a CPU aggregation change and introduces no worker pool.

Complete resident evaluation has a much smaller, mixed result: six paired
medians range from −2.73% to +4.10%, with 18/30 faster pairs. PandaSet improves
in both passes; Autoware and Tokyo each improve in one pass and regress in the
other. These measurements do not establish a consistent whole-evaluation gain
for those two sources. The component improvement is adopted; its percentage
must not be presented as the full evaluation improvement.

The later [bounded Wasserstein batches](cpu-map-wasserstein-batch.md) add a
separate measured improvement. Reproduce the mean-sum delta with its frozen
sources; current production includes both changes.

## Arithmetic and scope

The preceding implementation allocates `(V, 3)` zero sums and uses
`np.add.at(sums, inverse, points)`. The new helper allocates the same output,
then reduces one coordinate at a time with weights `points[:, axis]`. It keeps
the existing integer memberships and coordinate addition sequence. It does
not substitute a pairwise sum, reorder points or change the mean division.

Inputs below 64 points retain scatter. Larger inputs use scatter unless their
maximum and minimum lie within `±(f64_max / N / 2)`, where `N` is the number of
points. This rejects non-finite/extreme values and leaves headroom for every
addition even if all points share one voxel. The fallback retains scatter's
overflow/invalid warnings and exceptions. NumPy error settings are unchanged.
Subnormal, cancellation and signed-zero results are checked by their f64 bits.

The private helper receives the builder's f64 XYZ arrays and nonnegative voxel
labels. Gaussian dictionary order, counts, all covariance arithmetic,
Wasserstein/SCS calculations, nearest-neighbor work, protocols and artifacts
are preserved. No persistent reference cache is added. This applies to resident
Gaussian construction, including the normal CLI protocol with downsampling
disabled. The streaming `PointAccumulator` and optional centroid helper retain
their separate implementations.

This change does not reduce every scratch allocation. NumPy can copy strided
coordinate weights into a contiguous `8 * N`-byte array, and each reduction
returns an `8 * V`-byte result. The existing `24 * V`-byte sum output, grouping
arrays, centered points and covariance output still exist. These are array
payloads, not a measured peak RSS or a memory cap.

## Real-source comparison

[Source preparation and attribution](c2c-real-source.md) supply fixed original
coordinates, without new thinning, displacement, noise or generated points.
Tokyo retains its preceding original-point selection. The pairs are
disjoint-index subsets of one measured source with possible geometric
duplicates; they are not independent surveys or accuracy ground truth.
Generated fixtures are used only for correctness tests.

| Source | Estimated / reference points | Estimated / reference voxels |
| --- | ---: | ---: |
| PandaSet 019 | 531,908 / 531,908 | 9,213 / 9,186 |
| Autoware planning | 878,920 / 878,921 | 12,024 / 12,038 |
| Tokyo Takanawadai | 941,933 / 941,933 | 2,879 / 2,880 |

Requests use 3 m structural voxels, minimum 100 points in both maps, neighbor
radius 5 and thresholds `(0.2,0.1,0.08,0.05,0.01)` m. Alignment and downsampling
are disabled. Both implementations include all preceding MapEval optimizations.
The only new production change is coordinate-sum aggregation.

Windows, Python 3.12.10, NumPy 2.3.4, SciPy 1.17.1 and Open3D 0.19.0 use 12
logical CPUs, `OMP_NUM_THREADS=12` and `OPENBLAS_NUM_THREADS=1` before imports.
Production does not set those environment variables. Thirteen source/config/
test/controller fingerprints, the preceding baseline and compiled SciPy tree
are recorded with archived copies.

Each source/pass starts a fresh process. Full parity capture, an excluded first
timing pair and two warm-up pairs precede five alternating timed pairs. Sources
run PandaSet → Autoware → Tokyo, then reverse that order. No edits, builds,
tests or scans run during timing. All observations remain.

Complete clocks include both fresh NN trees/queries, threshold statistics,
voxel grouping, sums/means/covariances, Wasserstein and SCS index construction/
destruction. Loading, imports, hashing, parity tracing, PLY writing and CLI
startup/serialization are excluded.

| Source/pass | Scatter median, ms | Coordinate median, ms | Median paired shortening | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| PandaSet initial | 1,903.97 | 1,881.88 | +3.81% | 4/5 |
| PandaSet final | 1,805.86 | 1,697.86 | +4.10% | 4/5 |
| Autoware initial | 3,616.63 | 3,579.93 | +1.51% | 4/5 |
| Autoware final | 3,142.34 | 3,357.46 | −2.73% | 1/5 |
| Tokyo initial | 2,919.17 | 3,118.22 | −2.36% | 1/5 |
| Tokyo final | 2,896.79 | 2,793.39 | +3.67% | 4/5 |

After those complete trials, the same process measures just coordinate sums
on each original array. Labels are prepared outside clocks. Allocation,
finite/magnitude guards, reductions and assignment are included; grouping,
mean division, later evaluation and output checking/hashing are excluded.
The component has its own excluded first pair, two warm-ups and five
alternating pairs. Thus the following times have a different scope.

| Source/pass/input | Scatter median, ms | Coordinate median, ms | Median paired shortening | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| PandaSet initial estimated | 48.21 | 12.04 | +73.17% | 5/5 |
| PandaSet initial reference | 55.30 | 13.69 | +74.36% | 5/5 |
| PandaSet final estimated | 36.48 | 13.33 | +70.93% | 5/5 |
| PandaSet final reference | 44.00 | 11.28 | +73.89% | 5/5 |
| Autoware initial estimated | 65.10 | 21.96 | +67.72% | 5/5 |
| Autoware initial reference | 65.01 | 22.18 | +66.67% | 5/5 |
| Autoware final estimated | 66.27 | 18.71 | +70.61% | 5/5 |
| Autoware final reference | 58.54 | 20.92 | +66.68% | 5/5 |
| Tokyo initial estimated | 91.42 | 29.76 | +66.38% | 5/5 |
| Tokyo initial reference | 73.21 | 26.23 | +65.02% | 5/5 |
| Tokyo final estimated | 70.91 | 21.36 | +71.83% | 5/5 |
| Tokyo final reference | 68.29 | 20.86 | +68.28% | 5/5 |

Shortening is `median(100 * (1 - coordinate_ms / scatter_ms))`, rather than
the ratio of marginal medians. Power/scheduling, other applications, allocator
state and warmed pages/caches are uncontrolled; no causal diagnosis of whole
timing variation is established. Component and complete percentages are not
additive. No full CLI, cold-file, streaming, browser/WASM or GPU gain is claimed.

## Validation and reproduction

All 108 complete parity/first/warm-up/timed reports match. All 24 full f64
distance arrays, 24 complete PLY outputs and 24 Gaussian maps match bytes,
including 192,880 saved voxel records below and above AWD support. Gaussian
records contain integer keys/counts, all mean coordinates and all nine
covariance entries in dictionary order. Twelve component sum hashes match the
preceding scatter arithmetic. Reports/distances/PLYs also match independently
recorded [scalar-grouping results](cpu-map-grouping.md) with the same disabled
downsampling protocol.

The independent audit recomputes twelve serial NN queries, six structural
evaluations, twelve Gaussian maps and twelve coordinate reductions. It checks
every PLY coordinate/RGB value, source/library hashes, complete report hashes,
SCS index metadata, pair ordering and timing summaries. Twelve negative
controls reject wrong sources/baselines, invalid counts/voxel sizes, duplicate
arguments and existing destinations without replacing data.

222 related Python tests pass. The 29 new cases cover f64 point-order equality,
strided/reversed inputs, tiny/subnormal/large magnitudes, cancellation, signed
zero, empty/missing groups, the 63/64/65-point boundary, three-coordinate
reduction use, exceptional-input fallback with warning/error-policy parity,
overflow/invalid exceptions, complete Gaussians and a full evaluation report.
The separate Unix `resource` upstream-harness import limitation on Windows
is unchanged; that harness is excluded.

The [controller](../cloudanalyzer/scripts/benchmark_map_mean_sums.py) requires
`--source-root`, `--reference`, `--compared`, `--input-manifest`, `--baseline`,
`--baseline-sha256` and new `--work-dir`/`--output` destinations. Defaults are
`--repeats 5 --structure-voxel-size 3`; all six records use them. Inputs,
fingerprints, full timings and normalized reports remain in the records.
Reproduce preceding deltas using their respective frozen sources.

- PandaSet: [initial](../benchmarks/cpu/map-evaluate/real-source/windows-pandaset019-mean-sums-initial.json), [final](../benchmarks/cpu/map-evaluate/real-source/windows-pandaset019-mean-sums-final.json)
- Autoware: [initial](../benchmarks/cpu/map-evaluate/real-source/windows-autoware-planning-mean-sums-initial.json), [final](../benchmarks/cpu/map-evaluate/real-source/windows-autoware-planning-mean-sums-final.json)
- Tokyo: [initial](../benchmarks/cpu/map-evaluate/real-source/windows-tokyo-takanawadai-mean-sums-initial.json), [final](../benchmarks/cpu/map-evaluate/real-source/windows-tokyo-takanawadai-mean-sums-final.json)
