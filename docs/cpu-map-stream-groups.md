# Stable chunk groups for streaming AWD/SCS

`evaluate_map_streaming` now groups each chunk once, then updates voxel moments
from contiguous point slices. Complete streaming AWD/SCS reduction improves by
69.87–82.36% across six real-source comparisons, with 18/18 faster timed pairs,
against the archived preceding streaming implementation. Both maps' chunk
iteration, classification, per-voxel updates/finalization, Wasserstein distances
and SCS index construction/destruction are included. Original-file loading,
coordinate-snapshot preparation/mapping and CLI writes are excluded.

The subsequent [optional resident centroid preprocessing](cpu-map-downsample-centroids.md)
is measured separately. Reproduce the streaming results here with their frozen
sources; current production also includes that resident-helper change.

The later [streaming accumulator reuse](cpu-map-stream-accumulators.md) removes
52.62–68.59% of accumulator constructions on these sources. Whole streaming
timings are mixed (−11.27% to +9.93%, 20/30 faster pairs); they do not establish
a consistent whole-evaluation speedup. Its control already includes the later
Wasserstein/SCS changes, so use the frozen versions for each comparison.

The subsequent [bounded moment batches](cpu-map-stream-moments.md) reduce
equal-size small groups together. Complete streaming work is 1.66–28.90%
shorter with 28/30 faster pairs and identical Gaussian/bounds bits and reports;
the preceding allocation, Wasserstein and SCS changes are already in its control.

## Stable selection and moment arithmetic

The preceding implementation calls `np.unique(indices, axis=0)` in every chunk,
then selects each voxel with `pts[inverse == group_index]`. That scans the full
chunk once per occupied voxel, even when most of its points belong elsewhere.
The new implementation reuses [bounded scalar classification](cpu-map-grouping.md),
stably sorts the inverse memberships once, gathers the chunk once and walks
contiguous groups using the counts.

Unique keys retain lexicographic order within each chunk. Stable sorting retains
the original point order inside every voxel. Chunks arrive in their original
order, and accumulator keys retain their original first-insertion order across
chunks. The same point batches therefore reach `PointAccumulator.update` in the
same sequence. Its finite filtering, mean/covariance/bounds calculation, moment
merge formula and finalization are unchanged. No worker pool or alternate
floating-point reduction is introduced.

Wasserstein/SCS formulas, support checks, classifications, protocol fields and
the separate streaming metric lane are unchanged. Streaming still reports AWD,
SCS and support counts; it does not produce resident NN metrics or colored PLYs.
Chunk boundaries affect floating-point moment merging, so comparisons use the
same boundaries rather than claiming bit identity with resident evaluation or
different chunk sizes.

Small inputs and unsafe coordinate capacities retain the preceding axis
classification through the shared helper, then use stable group extraction.
Non-finite filtering and zero-count voxel behavior are preserved. Empty chunks
are skipped, invalid shapes/sizes and source failures propagate, and the
iterator is consumed once without requesting the next chunk early.

## Scratch and lifetime

For a chunk with `C` points, the permutation array occupies `8 * C` bytes and
the gathered XYZ array occupies `24 * C` bytes. At 100,000 points, these are
800,000 and 2,400,000 bytes. Sort workspace, integer classification/unique/
inverse/count arrays and per-voxel filtering/centering/moment arrays are
additional. The preceding full-chunk-per-voxel boolean scans disappear.

The permutation and gathered-point buffers are explicitly released before the
next chunk is requested.
The original current chunk remains borrowed, and per-voxel moments still scale
with all occupied voxels across the stream. Finalization also creates Gaussian
outputs. These array payloads are not a total memory cap or peak-RSS measurement.
No grouping index or point buffer is cached across evaluations.

## Controlled real-source workload

[Real-source preparation](c2c-real-source.md) supplies original-coordinate
subsets with attribution/licenses. No new thinning, generated points,
displacement or noise is added. Tokyo retains its preceding original-point
selection. These are disjoint-index subsets of one measured source with possible
geometric duplicates, rather than independent surveys or accuracy ground truth.
Generated fixtures serve correctness tests.

The controller decodes each prepared PLY outside clocks and writes an exact XYZ
f64 snapshot. Both workflows consume the same read-only memory maps in original
point order, in chunks of 100,000 points including each final short chunk. The
snapshots match all decoded coordinates byte-for-byte. Original-file loading,
writing/mapping snapshots, hashes and parity tracing are outside timing; chunk
iteration and all streaming reduction work are inside. Pages are warmed by
parity/first/warm-up calls. This measures streaming computation, not cold storage,
CSV/LAS/PCD/PLY ingestion or the existing CLI resource-report harness.

The protocol is 3 m voxels, minimum 100 points in both maps and neighbor radius
5. Alignment/downsampling are absent. Both workflows use the preceding indexed
SCS lookup. The control already contains the earlier resident optimizations,
but its chunk builder still uses axis classification and boolean selection.

| Source | Estimated / reference points | Final estimated / reference voxels | Voxel update calls, estimated / reference |
| --- | --- | --- | --- |
| PandaSet 019 | 531,908 / 531,908 | 9,213 / 9,186 | 24,871 / 24,889 |
| Autoware planning | 878,920 / 878,921 | 12,024 / 12,038 | 25,383 / 25,402 |
| Tokyo Takanawadai | 941,933 / 941,933 | 2,879 / 2,880 | 9,174 / 9,158 |

Windows, 12 logical CPUs, Python 3.12.10, NumPy 2.3.4, SciPy 1.17.1 and
Open3D 0.19.0 use `OMP_NUM_THREADS=12` and `OPENBLAS_NUM_THREADS=1` before
imports. Production does not change environment variables. Ten source/config/
test/controller fingerprints and the compiled SCS tree fingerprint are recorded
with archived copies.

Each source/pass starts a fresh process. A full Gaussian/ordered-batch parity
pair, an excluded first timing pair and one warm-up pair precede three
alternating timed pairs. No tracing hooks are active during timing. Sources run
PandaSet → Autoware → Tokyo, then reverse that order. No builds, tests, scans
or edits run during comparisons. All observations remain.

| Source/pass | Boolean-selection median, ms | Stable-group median, ms | Median paired shortening | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| PandaSet initial | 18,308.22 | 4,104.25 | +76.74% | 3/3 |
| PandaSet final | 17,763.18 | 3,554.57 | +82.36% | 3/3 |
| Autoware initial | 22,857.56 | 5,957.08 | +69.87% | 3/3 |
| Autoware final | 15,548.11 | 3,796.32 | +75.54% | 3/3 |
| Tokyo initial | 7,580.70 | 2,125.34 | +71.95% | 3/3 |
| Tokyo final | 8,332.28 | 2,185.99 | +73.76% | 3/3 |

Shortening is `median(100 * (1 - stable_ms / mask_ms))`, rather than the ratio
of marginal medians. All 18 timed pairs improve, but times vary materially;
scheduling, power, other applications, warmed caches and allocator/page
retention are uncontrolled. This result applies to the fixed chunk/protocol
workload. It is separate from [resident grouping measurements](cpu-map-grouping.md);
acceleration percentages are not additive. No GPU/browser/WASM speed is claimed.

## Validation and reproduction

All 72 parity/first/warm-up/timed results match, including AWD/SCS, support
counts and every classification/protocol field. All 24 complete Gaussian maps
match byte-for-byte, covering 192,880 saved voxel records with integer keys/
counts, all three means and all nine covariance entries, including cells below
AWD support. All 24 ordered accumulator-input streams have matching lengths,
call/point counts and complete f64-byte hashes. These checks preserve the
preceding streaming implementation's results rather than comparing them with
the different resident moment algorithm.

An independent audit checks the original PLY/snapshot coordinate bits,
reconstructs 12 maps and their ordered update streams with the archived builder,
replays 9,411,046 point visits and recomputes six structural evaluations. It also
checks all Gaussian bytes, input/library/source fingerprints, report hashes,
timing summaries and protocol fields. Seventeen negative controls reject wrong
data/baselines/counts, invalid chunk/voxel sizes, duplicate options and existing
destinations without replacing data.

The related suite passes 133 Python tests. Twenty-one new cases cover all
Gaussian bits and update sequences across multiple chunk sizes, contiguous/
strided/reversed inputs, empty chunks, wide integer ranges, non-finite filtering,
scratch release before the next pull, one-pass iterators, source failures,
invalid sizes/shapes, full streaming-report parity and unavailable empty results.
Preceding resident grouping/covariance, NN/SCS, PLY, core/experimental import and
resident CLI tests also pass. The existing Windows `resource` import limitation
in the separate CLI/upstream resource-report harness remains; that harness is
not included. Rust binaries and GPU selection are unchanged.

The [benchmark script](../cloudanalyzer/scripts/benchmark_map_stream_groups.py)
requires `--source-root`, original `--reference`/`--compared`/`--input-manifest`,
archived `--baseline`/`--baseline-sha256` and new `--work-dir`/`--output` paths.
Defaults are `--repeats 3 --chunk-size 100000 --structure-voxel-size 3`; all six
comparisons use them. Gaussian files use `<4q12d` records (128 bytes); ordered
update hashes concatenate each batch's `<Q` point count and row-major `<f8` XYZ
bytes. JSON records retain all times, full reports, coordinate/Gaussian hashes
and ordered-update metadata. Reproduce earlier resident deltas with their
respective frozen sources; current production also accelerates chunk grouping.

- PandaSet: [initial](../benchmarks/cpu/map-evaluate/real-source/windows-pandaset019-stream-groups-initial.json), [final](../benchmarks/cpu/map-evaluate/real-source/windows-pandaset019-stream-groups-final.json)
- Autoware: [initial](../benchmarks/cpu/map-evaluate/real-source/windows-autoware-planning-stream-groups-initial.json), [final](../benchmarks/cpu/map-evaluate/real-source/windows-autoware-planning-stream-groups-final.json)
- Tokyo: [initial](../benchmarks/cpu/map-evaluate/real-source/windows-tokyo-takanawadai-stream-groups-initial.json), [final](../benchmarks/cpu/map-evaluate/real-source/windows-tokyo-takanawadai-stream-groups-final.json)
