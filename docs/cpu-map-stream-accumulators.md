# Reusing streaming voxel accumulators

The streaming MapEval builder now constructs `PointAccumulator` only when a
cell first appears. This removes 52.62–68.59% of accumulator constructions per
two-map evaluation on the three measured sources. It preserves every update
batch, Gaussian bit and complete AWD/SCS report.

Whole streaming reduction is mixed: median paired shortening ranges from
−11.27% to +9.93%, with 20/30 faster pairs. PandaSet improves in both passes;
Autoware and Tokyo each have an improving and a regressing pass. The change
removes unnecessary allocations, but these timings do not establish a
consistent whole-evaluation speedup across sources.

The subsequent [bounded streaming moment batches](cpu-map-stream-moments.md)
reduce enough equal-size groups together, preserving all Gaussian/bounds bits
and reports. Complete streaming work is 1.66–28.90% shorter, with 28/30 faster
pairs. Those comparisons already include this allocation change; reproduce
each experiment with its recorded frozen sources.

## Allocation and arithmetic

The preceding builder uses `accumulators.setdefault(key, PointAccumulator())`.
Python constructs the default argument even when the key already exists,
then discards the new object. The new builder looks up the cell first and
constructs an accumulator only for a missing key. Even a zero-count cell
retains its original accumulator across chunks.

Stable grouping, within-cell point order, chunk order, dictionary insertion
order and all `PointAccumulator.update`/`finalize` arithmetic are unchanged.
Finite filtering, mean/covariance/bounds calculation and moment merging remain
unchanged. Updates still occur once per occupied cell per chunk. No worker
pool, persistent cache or different numerical reduction is introduced.

| Source | Estimated / reference points | Preceding constructions, both maps | Current constructions, both maps | Reduction |
| --- | ---: | ---: | ---: | ---: |
| PandaSet 019 | 531,908 / 531,908 | 49,760 | 18,399 | 63.02% |
| Autoware planning | 878,920 / 878,921 | 50,785 | 24,062 | 52.62% |
| Tokyo Takanawadai | 941,933 / 941,933 | 18,332 | 5,759 | 68.59% |

Current construction counts equal final occupied-cell counts, including cells
below AWD support. These are cumulative allocations, not peak memory or RSS.
The existing per-cell state and chunk scratch requirements remain.

## Real-source comparison

[Real-source preparation](c2c-real-source.md) supplies original-coordinate
subsets with attribution and licenses. No new thinning, noise or generated
points are added. These subsets are not independent surveys or accuracy ground
truth. Generated fixtures are used only for correctness tests.

Both policies already include the preceding grouping, Wasserstein and SCS
optimizations. Only accumulator allocation differs. The protocol uses 3 m
voxels, minimum 100 points and neighbor radius 5. Fixed XYZ f64 read-only
memory maps supply 100,000-point chunks and their final short chunks.

Clocks include chunk iteration, grouping, allocation, moment updates and
finalization, Wasserstein and SCS index/query/statistic work. Original-file
loading, coordinate snapshot preparation/mapping, hashing, parity tracing and
CLI writes are excluded. Pages are warmed; cold ingestion is not measured.
Streaming produces AWD/SCS and support counts, without resident NN/PLY work.

Each source/pass starts a fresh process. A parity pair, excluded first pair
and two warm-up pairs precede five alternating timed pairs. Sources run
PandaSet → Autoware → Tokyo, then in reverse. No builds, tests, scans or edits
run during timing, and all observations are retained.

Windows, 12 logical CPUs, Python 3.12.10, NumPy 2.3.4, SciPy 1.17.1 and
Open3D 0.19.0 use `OMP_NUM_THREADS=12` and `OPENBLAS_NUM_THREADS=1` before
imports. Production does not set these environment variables.

| Source/pass | Preceding median, ms | Reuse median, ms | Median paired shortening | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| PandaSet initial | 4,250.71 | 3,879.25 | +9.56% | 4/5 |
| PandaSet final | 3,618.34 | 3,344.20 | +9.93% | 5/5 |
| Autoware initial | 4,481.68 | 4,147.33 | +7.77% | 5/5 |
| Autoware final | 4,932.27 | 5,918.12 | −11.27% | 1/5 |
| Tokyo initial | 2,490.59 | 2,346.60 | +5.78% | 3/5 |
| Tokyo final | 2,655.15 | 2,505.56 | −1.80% | 2/5 |

Shortening is `median(100 * (1 - reuse_ms / eager_ms))`, not the ratio of
marginal medians. Times vary materially; scheduling, power, other applications,
warmed caches and allocator retention are uncontrolled. The observed
regressions remain in the results; their cause is not established. No universal
speedup, GPU, resident-evaluation or peak-RSS improvement is claimed. Earlier
[streaming grouping gains](cpu-map-stream-groups.md) use a different frozen
baseline and are not additive with this comparison.

## Validation and reproduction

All 108 parity/first/warm-up/timed reports match. All 24 saved Gaussian maps
match byte-for-byte, covering 192,880 voxel records with integer keys/counts,
three means and nine covariance entries. All 24 ordered update streams have
matching batch/point counts and complete f64-byte hashes.

An independent audit validates original PLY/snapshot bytes, rebuilds both
policies' 24 Gaussian maps, replays 12 preceding ordered update streams with
237,754 updates and 9,411,046 point visits, and checks current allocation
counts. It recomputes six structural evaluations, timing summaries, report
hashes and all 18 source/config/test/controller fingerprints plus the compiled
tree fingerprint. Seventeen negative controls reject invalid counts/sizes,
duplicate options, wrong inputs/baselines and existing destinations.

The related MapEval suite passes 413 tests. Four new cases verify one allocation
per cell across repeated chunks, original accumulator identity and point order,
empty chunks and reuse of a zero-count non-finite cell. Existing output-bit,
iterator/error, streaming-report and resident/PLY regression cases also pass.
The two separate `resource`-dependent benchmark/parity modules cannot collect
on Windows and are excluded. Rust binaries and GPU selection are unchanged.

The [controller](../cloudanalyzer/scripts/benchmark_map_stream_accumulators.py)
requires `--source-root`, original `--reference`/`--compared`/`--input-manifest`,
archived `--baseline`/`--baseline-sha256` and new `--work-dir`/`--output` paths.
Defaults are `--repeats 5 --chunk-size 100000 --structure-voxel-size 3`.
Reproduction must use the recorded frozen source versions, rather than later
production revisions. Gaussian files contain `<4q12d` records (128 bytes);
ordered update hashes concatenate each batch's `<Q` count and row-major `<f8`
XYZ bytes. JSON retains full reports, allocation counts, all times and hashes.

- PandaSet: [initial](../benchmarks/cpu/map-evaluate/real-source/windows-pandaset019-stream-accumulators-initial.json), [final](../benchmarks/cpu/map-evaluate/real-source/windows-pandaset019-stream-accumulators-final.json)
- Autoware: [initial](../benchmarks/cpu/map-evaluate/real-source/windows-autoware-planning-stream-accumulators-initial.json), [final](../benchmarks/cpu/map-evaluate/real-source/windows-autoware-planning-stream-accumulators-final.json)
- Tokyo: [initial](../benchmarks/cpu/map-evaluate/real-source/windows-tokyo-takanawadai-stream-accumulators-initial.json), [final](../benchmarks/cpu/map-evaluate/real-source/windows-tokyo-takanawadai-stream-accumulators-final.json)
