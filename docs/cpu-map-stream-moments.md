# Bounded moment batches for streaming MapEval

Streaming AWD/SCS now reduces enough equal-size point groups together, then
merges their moments in the original order. Complete streaming reduction is
1.66–28.90% shorter across six real-source comparisons, with 28/30 faster
timed pairs, against the preceding [accumulator-reuse implementation](cpu-map-stream-accumulators.md).
Both passes improve for every source. All Gaussian bits, bounds, ordered
point-group inputs and complete reports match.

The subsequent [expanded streaming moment groups](cpu-map-stream-expanded.md)
accept 1–128-point groups with eight groups per size while retaining a
32,768-point gather limit. Complete streaming work is a further 7.98–18.80%
shorter with 27/30 faster pairs. Its control already contains this small-group
change; reproduce each comparison with its recorded frozen sources.

## Batch policy and arithmetic

The existing stable grouping still supplies each cell's points in their
original order. A chunk is eligible when it has at least 1,024 points and
128 groups, all coordinates are finite and within ±`1e150`, and NumPy's
error policy is the default (`divide/over/invalid=warn`, `under=ignore`).
Other chunks retain scalar point updates.

Group windows contain at most 1,024 consecutive occupied cells. Within a
window, groups with 1–32 points are eligible when at least 32 groups have the
same point count and at least 128 groups qualify in total. Insufficient
windows, rare sizes and wider groups retain scalar updates. A count histogram
selects eligible sizes without allocating by coordinate extent.

For each eligible size, the builder gathers a `(groups, points, 3)` array,
computes the same mean over each point axis, subtracts that mean, and computes
each centered transpose-product for covariance moments. It also reduces
minimum/maximum coordinates. The magnitude guard leaves headroom for these
small-group sums and products. Singleton means retain NumPy's arithmetic,
including signed zeros and subnormals.

The builder consumes every group in the preceding lexicographic order within
each chunk. Chunk order and first-insertion dictionary order remain unchanged.
`PointAccumulator` shares private moment/bounds merge helpers between scalar
updates and the batched builder. Delta, outer-product correction, count ratios,
in-place mean/moment updates and final covariance division retain the original
scalar arithmetic and order. Its public `update` still filters finite points
and returns itself. No worker pool or persistent cache is introduced.

Batching bypasses public `update` for selected groups because their statistics
have already been reduced; validation traces the common group-input handler
for the current builder and public `update` for the archived scalar builder.
The logical groups and original point bytes are identical.

## Scratch and lifetime

A full 1,024-group moment window stores three means, nine covariance moments
and six bounds per group: 147,456 f64 payload bytes plus a 1,024-byte selection
mask. A same-size cohort gathers at most 32,768 XYZ points: 786,432 bytes each
for gathered and centered coordinates, with up to 262,144 bytes of gather
indices. Mean/matrix/reduction temporaries are additional.

Group starts add `8 * occupied_groups_in_chunk` bytes. Existing classification,
stable permutation, whole-chunk gathering, input chunks and all accumulated
cell states are also present. These bounds are array payloads, not a total
memory cap or measured peak RSS.

Initial accumulator fields copy their moment/bounds rows so they own their
data and cannot retain an entire batch window through a view. Consumed moment
windows are released before preparing the next window. Moment windows, group
starts, permutation and gathered-point buffers are released before requesting
the next input chunk; no next chunk is prefetched. Non-finite cells, invalid
inputs, source failures and backend
errors preserve the scalar path or propagate as appropriate.

## Controlled real-source workload

[Real-source preparation](c2c-real-source.md) supplies attributed/licensed
original-coordinate subsets. No new thinning, noise or generated points are
added. These are subsets of measured sources, not independent surveys or
accuracy ground truth. Generated fixtures serve correctness tests.

Both policies already include all preceding MapEval optimizations. The control
loads both the archived core builder and archived `PointAccumulator` so the
comparison includes the current merge-helper refactor. The protocol uses 3 m
voxels, minimum 100 points and neighbor radius 5. Exact read-only XYZ f64
snapshots supply 100,000-point chunks plus each final short chunk.

| Source | Estimated / reference points | Logical groups, both maps | Batched groups | Batched points |
| --- | ---: | ---: | ---: | ---: |
| PandaSet 019 | 531,908 / 531,908 | 49,760 | 29,505 | 73,769 |
| Autoware planning | 878,920 / 878,921 | 50,785 | 14,344 | 42,495 |
| Tokyo Takanawadai | 941,933 / 941,933 | 18,332 | 1,868 | 5,451 |

Clocks include chunk iteration, classification/grouping, allocation, moment
reduction/merging/finalization, Wasserstein and SCS setup/query/statistic work.
Original-file loading, coordinate snapshot preparation/mapping, hashing,
Gaussian/bounds/group/allocation tracing and CLI writes are excluded. Pages
are warmed; cold ingestion is not measured. This streaming lane produces
AWD/SCS and support counts without resident NN/PLY work.

Each source/pass starts a fresh process. A parity pair, excluded first pair
and two warm-up pairs precede five alternating timed pairs. Sources run
PandaSet → Autoware → Tokyo, then reverse that order. No builds, tests, scans
or edits run during timing, and all observations remain.

Windows, 12 logical CPUs, Python 3.12.10, NumPy 2.3.4, SciPy 1.17.1 and
Open3D 0.19.0 use `OMP_NUM_THREADS=12` and `OPENBLAS_NUM_THREADS=1` before
imports. Production does not set these variables.

| Source/pass | Scalar median, ms | Batch median, ms | Median paired shortening | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| PandaSet initial | 2,776.08 | 2,200.93 | +28.15% | 5/5 |
| PandaSet final | 2,660.81 | 1,891.89 | +28.90% | 5/5 |
| Autoware initial | 3,167.65 | 2,850.49 | +10.86% | 4/5 |
| Autoware final | 3,073.14 | 2,743.80 | +9.87% | 5/5 |
| Tokyo initial | 1,564.44 | 1,558.20 | +1.66% | 4/5 |
| Tokyo final | 1,587.27 | 1,548.42 | +2.47% | 5/5 |

Shortening is `median(100 * (1 - batch_ms / scalar_ms))`, not the ratio of
marginal medians. The smaller batched population in Tokyo accompanies a
smaller measured benefit. Scheduling, power, other applications, warmed caches
and allocator retention are uncontrolled; the results apply to this fixed
chunk/protocol workload. Earlier acceleration percentages are not additive.
These measurements establish CPU streaming-computation gains, without a GPU,
resident-evaluation, cold-I/O or peak-RSS claim.

## Validation and reproduction

All 108 parity/first/warm-up/timed reports match. All 24 complete Gaussian
maps match byte-for-byte, covering 192,880 records with integer keys/counts,
three means and nine covariance entries. All 24 complete minimum/maximum maps
also match, including cells below AWD support. Both builders receive identical
ordered group/point counts and complete f64-byte hashes for every map.

An independent audit checks original PLY/snapshot bits and rebuilds both
policies' 24 maps and ordered group streams. Across both passes' 12 map inputs,
each policy processes 237,754 logical updates and 9,411,046 point visits. Independent axis
classification predicts 91,434 batched groups, 243,430 batched point visits and
210 selected windows. It recomputes 12 structural evaluations, verifies all
Gaussian/bounds bytes, report hashes, timing summaries and 20 source/config/
test/controller fingerprints plus the compiled-tree fingerprint. Six report/
Gaussian/group-hash chains also match the preceding allocation experiment.
Nineteen negative controls reject invalid counts/sizes, duplicate options,
wrong inputs or either baseline, and existing destinations.

The related MapEval/I/O suite passes 482 tests. Fifty-three new cases check all
moment/bounds bits, initial and repeated merges, signed zeros/subnormals,
selection/count/width boundaries, unsafe values and NumPy policies, public
accumulator behavior, window/iterator lifetimes, backend failures and full
streaming parity on contiguous/strided/reversed inputs. The separate
`resource`-dependent benchmark/parity modules cannot collect on Windows and
are excluded. Rust binaries and GPU selection are unchanged.

The [controller](../cloudanalyzer/scripts/benchmark_map_stream_moments.py)
requires `--source-root`, original `--reference`/`--compared`/`--input-manifest`,
archived `--baseline`/`--baseline-sha256`, archived
`--io-baseline`/`--io-baseline-sha256`, and new `--work-dir`/`--output` paths.
Defaults are `--repeats 5 --chunk-size 100000 --structure-voxel-size 3`.
Reproduce with the recorded frozen source versions rather than later revisions.
Gaussian files use `<4q12d` records (128 bytes); bounds files use `<3q6d`
records (72 bytes). Group hashes concatenate each `<Q` batch length and
row-major `<f8` XYZ bytes. JSON retains full reports, dispatch/allocation counts,
all observations and fingerprints.

- PandaSet: [initial](../benchmarks/cpu/map-evaluate/real-source/windows-pandaset019-stream-moments-initial.json), [final](../benchmarks/cpu/map-evaluate/real-source/windows-pandaset019-stream-moments-final.json)
- Autoware: [initial](../benchmarks/cpu/map-evaluate/real-source/windows-autoware-planning-stream-moments-initial.json), [final](../benchmarks/cpu/map-evaluate/real-source/windows-autoware-planning-stream-moments-final.json)
- Tokyo: [initial](../benchmarks/cpu/map-evaluate/real-source/windows-tokyo-takanawadai-stream-moments-initial.json), [final](../benchmarks/cpu/map-evaluate/real-source/windows-tokyo-takanawadai-stream-moments-final.json)
