# Expanded streaming moment groups with bounded gathers

Streaming AWD/SCS now batches more equal-size groups while retaining the
32,768-point gather limit. Complete streaming computation is 7.98–18.80%
shorter across six real-source comparisons, with 27/30 faster timed pairs,
against the preceding [small-group moment batches](cpu-map-stream-moments.md).
All complete Gaussian/bounds bytes, ordered group inputs and reports match.
Both passes have positive paired medians for every source.

The subsequent [reuse of chunk verification](cpu-map-stream-finite.md) borrows
confirmed finite scalar-group coordinates instead of repeating filtering and
copying. Complete streaming work is a further 6.88–16.46% shorter with 25/30
faster pairs and identical full outputs. Its control already includes this
expansion; reproduce each comparison with its recorded frozen sources.

## Policy and memory

The preceding policy accepts 1–32-point groups with at least 32 groups of the
same size. The current policy accepts 1–128-point groups with at least eight
groups of the same size. Both require at least 128 selected groups within a
window of at most 1,024 consecutive groups. Insufficient windows, rare sizes
and groups above the size limit keep scalar updates.

Existing chunk guards remain: at least 1,024 points and 128 occupied groups,
finite coordinates within ±`1e150`, and default NumPy error settings. The
magnitude limit leaves headroom for the expanded group means and centered
transpose-products. Finite filtering and the fallback path remain available.

For a size `K`, the cohort is split into blocks with at most
`floor(32,768 / K)` groups. The minimum same-size count applies to the full
cohort within the window; a final block may contain fewer than eight groups.
No padding or approximate reduction is used. Each group's original point
order, mean, covariance moments and minimum/maximum arithmetic remain intact.
Moment merging still follows the original cell/chunk order through the same
`PointAccumulator` helpers, with initial fields owning their rows.

Gathered coordinates, means and centered coordinates are released before
preparing the next block. A gather contains at most 32,768 XYZ points:
786,432 bytes each for gathered and centered f64 coordinates, plus at most
262,144 bytes of indices. Moment-window payload remains 147,456 bytes plus
the 1,024-byte selection mask. Histograms, offsets, matrix/reduction
temporaries, group starts, whole-chunk grouping scratch and all accumulated
cell states are additional. These are payload bounds, not measured peak RSS
or a total memory cap. No worker pool or persistent cache is introduced.

## Real-source comparison

[Real-source preparation](c2c-real-source.md) supplies attributed/licensed
original-coordinate subsets. No new thinning, displacement or generated
points are added. These subsets are not independent surveys or accuracy
ground truth. Generated data is used only for correctness tests.

Both policies contain all preceding optimizations, including small-group
moment batching and accumulator reuse. The archived control includes both
core and I/O sources. Exact read-only XYZ f64 snapshots supply 100,000-point
chunks and each final short chunk. The protocol is 3 m voxels, minimum 100
points and neighbor radius 5, without alignment or downsampling.

| Source | Logical groups, both maps | Prior / expanded batched groups | Prior / expanded batched points |
| --- | ---: | ---: | ---: |
| PandaSet 019 | 49,760 | 29,505 / 40,437 | 73,769 / 225,831 |
| Autoware planning | 50,785 | 14,344 / 35,005 | 42,495 / 521,210 |
| Tokyo Takanawadai | 18,332 | 1,868 / 7,411 | 5,451 / 112,720 |

Clocks include chunk iteration, classification/grouping, allocation,
moment reduction/merging/finalization, Wasserstein and SCS setup/query/statistic
work. Original-file loading, snapshot preparation/mapping, hashing,
Gaussian/bounds/group/allocation tracing and CLI writes are excluded. Pages
are warmed. Streaming produces AWD/SCS and support counts without resident
NN/PLY work.

Each source/pass starts a fresh process. A parity pair, excluded first pair
and two warm-up pairs precede five alternating timed pairs. Sources run
PandaSet → Autoware → Tokyo, then reverse that order. No builds, tests, scans
or edits run during timing; all observations are retained.

Windows, 12 logical CPUs, Python 3.12.10, NumPy 2.3.4, SciPy 1.17.1 and
Open3D 0.19.0 use `OMP_NUM_THREADS=12` and `OPENBLAS_NUM_THREADS=1` before
imports. Production does not set these environment variables.

| Source/pass | Prior median, ms | Expanded median, ms | Median paired shortening | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| PandaSet initial | 2,981.76 | 2,502.07 | +15.77% | 5/5 |
| PandaSet final | 2,235.64 | 1,913.81 | +15.67% | 4/5 |
| Autoware initial | 3,717.35 | 2,716.98 | +14.66% | 4/5 |
| Autoware final | 3,297.30 | 2,533.98 | +18.80% | 5/5 |
| Tokyo initial | 1,844.59 | 2,197.30 | +11.55% | 4/5 |
| Tokyo final | 1,751.02 | 1,653.45 | +7.98% | 5/5 |

Shortening is `median(100 * (1 - expanded_ms / small_ms))`, not the ratio of
marginal medians. Tokyo initial has a slower expanded marginal median despite
four faster corresponding pairs and a positive paired median. Times vary
materially; scheduling, power, other applications, warmed caches and allocator
retention are uncontrolled. The cause of that variation is not established.
Results apply to this fixed workload and are not additive with earlier gains.
No universal speedup, GPU, resident-evaluation, cold-I/O or peak-RSS benefit is
claimed.

## Validation and reproduction

All 108 parity/first/warm-up/timed reports match. All 24 complete Gaussian
maps match byte-for-byte, covering 192,880 records with keys/counts, three
means and nine covariance entries. All 24 minimum/maximum maps also match,
including cells below AWD support. Both builders receive identical ordered
group counts and complete original-point byte hashes.

An independent audit rebuilds both policies' 24 maps and ordered input streams.
Across both passes' 12 map inputs, each policy processes 237,754 logical groups
and 9,411,046 point visits. Independent axis classification predicts 165,706
expanded batched groups, 1,719,522 batched point visits and 236 selected windows.
It recomputes 12 structural evaluations and checks report/timing summaries,
21 source/config/test/controller fingerprints and the compiled tree. Six
report/Gaussian/group-hash chains match the preceding moment experiment.

The audit checks 7,494 actual gather/mean blocks across both policies, totaling
1,962,952 point visits; their largest block has 3,051 points, below the 32,768
limit. Weak references confirm expanded gather/mean scratch is released before
the next block. Nineteen negative controls reject invalid counts/sizes,
duplicate options, wrong inputs or baselines and existing destinations.

The related MapEval/I/O suite passes 518 tests. Thirty-six new cases check
33–128-point moment/merge bits across scales, signed zeros/subnormals, block
boundaries/tails, full-capacity gathers and scratch release, plus eight-group
cohorts on contiguous/strided/reversed inputs. Existing tests retain scalar
and numerical-guard/report coverage. The separate `resource`-dependent
benchmark/parity modules cannot collect on Windows and are excluded. Rust
binaries and GPU selection are unchanged.

The [controller](../cloudanalyzer/scripts/benchmark_map_stream_expanded.py)
requires `--source-root`, original `--reference`/`--compared`/`--input-manifest`,
archived `--baseline`/`--baseline-sha256` and
`--io-baseline`/`--io-baseline-sha256`, plus new `--work-dir`/`--output` paths.
Defaults are `--repeats 5 --chunk-size 100000 --structure-voxel-size 3`.
Reproduce with the recorded frozen source versions. Gaussian files use
`<4q12d` records (128 bytes), bounds files `<3q6d` (72 bytes), and group hashes
concatenate each `<Q` batch length with row-major `<f8` XYZ bytes. JSON retains
full reports, dispatch/allocation counts, all observations and fingerprints.

- PandaSet: [initial](../benchmarks/cpu/map-evaluate/real-source/windows-pandaset019-stream-expanded-initial.json), [final](../benchmarks/cpu/map-evaluate/real-source/windows-pandaset019-stream-expanded-final.json)
- Autoware: [initial](../benchmarks/cpu/map-evaluate/real-source/windows-autoware-planning-stream-expanded-initial.json), [final](../benchmarks/cpu/map-evaluate/real-source/windows-autoware-planning-stream-expanded-final.json)
- Tokyo: [initial](../benchmarks/cpu/map-evaluate/real-source/windows-tokyo-takanawadai-stream-expanded-initial.json), [final](../benchmarks/cpu/map-evaluate/real-source/windows-tokyo-takanawadai-stream-expanded-final.json)
