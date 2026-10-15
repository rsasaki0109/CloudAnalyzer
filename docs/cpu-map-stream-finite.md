# Reusing chunk verification for scalar voxel updates

Streaming AWD/SCS now reuses its existing whole-chunk finite check for scalar
groups that cannot use moment batches. Complete streaming computation is
6.88–16.46% shorter across six real-source comparisons, with 25/30 faster
timed pairs, against the preceding [expanded moment implementation](cpu-map-stream-expanded.md).
All complete Gaussian/bounds bytes, ordered group inputs and reports match.
Both passes have positive paired medians for every source.

A later [moment-merge experiment](cpu-map-stream-merge-candidates.md) checks
three candidates against this baseline. Their timings include regressions;
none is adopted, and this implementation remains the runtime path.

## Verification and arithmetic

The preceding scalar path calls `PointAccumulator.update` for every remaining
group. It repeats a finite-value mask and copies the group's XYZ coordinates
even though the existing moment-batch guard has already checked the whole chunk.

The current builder reuses that verification when the chunk has at least
1,024 points and 128 occupied groups, all coordinates are finite and within
±`1e150`, and NumPy uses default error settings. Its gathered/sorted f64 XYZ
group slices are contiguous. For non-batched groups, a private accumulator
helper borrows those slices directly instead of repeating filtering/copying.

Mean, centering, transpose-product covariance moments, minimum/maximum,
moment merging and finalization keep their preceding arithmetic and order.
The helper does not write to or retain the caller's coordinates. Public
`PointAccumulator.update` retains its shape/dtype validation, finite filtering,
coordinate copy and return-self behavior before calling the same reduction
helper. Unsafe, small or non-default-error-policy chunks keep that public path.
Empty groups and failures retain their preceding behavior.

Moment-batch selection, point/gather budgets, cell/point/chunk order, dictionary
insertion order, support checks, Wasserstein/SCS and protocol fields remain
unchanged. No worker pool, GPU dispatch or persistent cache is introduced.

## Copy payload and real-source comparison

[Real-source preparation](c2c-real-source.md) supplies attributed/licensed
original-coordinate subsets. No new thinning, noise or generated points are
added. These subsets are not independent surveys or accuracy ground truth.
Generated fixtures are used only for correctness tests.

Both policies already include all preceding optimizations, including expanded
moment batches. The control uses archived core and I/O sources, so the current
public-update delegation is included in the comparison. Exact read-only XYZ
f64 snapshots supply 100,000-point chunks plus each final short chunk. The
protocol uses 3 m voxels, minimum 100 points and neighbor radius 5, without
alignment or downsampling.

| Source | Verified scalar groups, both maps | Borrowed point visits | Avoided XYZ copy payload per evaluation |
| --- | ---: | ---: | ---: |
| PandaSet 019 | 9,323 | 837,985 | 20,111,640 bytes |
| Autoware planning | 15,780 | 1,236,631 | 29,679,144 bytes |
| Tokyo Takanawadai | 10,921 | 1,771,146 | 42,507,504 bytes |

These are cumulative coordinate-copy payloads (`24 * borrowed_points`), not
live memory or peak RSS reductions. Finite masks and filter calls also disappear
for these groups. Existing whole-chunk gathering, centered arrays, moment
windows and all accumulated cell states remain. The measured inputs need no
remaining per-group filters; guarded fallback inputs still use them.

Clocks include chunk iteration, classification/grouping, allocation,
verification/filtering, moment reduction/merging/finalization, Wasserstein
and SCS setup/query/statistic work. Original-file loading, snapshot preparation/
mapping, hashing, Gaussian/bounds/group/filter/allocation tracing and CLI writes
are excluded. Pages are warmed; cold ingestion is not measured. This streaming
lane produces AWD/SCS and support counts without resident NN/PLY work.

Each source/pass starts a fresh process. A parity pair, excluded first pair
and two warm-up pairs precede five alternating timed pairs. Sources run
PandaSet → Autoware → Tokyo, then reverse that order. No builds, tests, scans
or edits run during timing, and all observations are retained.

Windows, 12 logical CPUs, Python 3.12.10, NumPy 2.3.4, SciPy 1.17.1 and
Open3D 0.19.0 use `OMP_NUM_THREADS=12` and `OPENBLAS_NUM_THREADS=1` before
imports. Production does not set these variables.

| Source/pass | Filtering median, ms | Verified median, ms | Median paired shortening | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| PandaSet initial | 2,410.79 | 2,232.96 | +6.88% | 4/5 |
| PandaSet final | 1,804.85 | 1,612.17 | +9.56% | 5/5 |
| Autoware initial | 3,770.22 | 3,097.87 | +13.32% | 5/5 |
| Autoware final | 3,216.74 | 2,767.87 | +13.86% | 4/5 |
| Tokyo initial | 1,944.05 | 1,896.75 | +8.80% | 3/5 |
| Tokyo final | 1,831.19 | 1,581.14 | +16.46% | 4/5 |

Shortening is `median(100 * (1 - finite_ms / filtered_ms))`, not the ratio of
marginal medians. Times vary materially. Autoware final includes a verified
trial taking 8,194.09 ms against its corresponding 3,336.05 ms control; that
regression remains in the result. Scheduling, power, other applications,
warmed caches and allocator retention are uncontrolled, and the cause of the
variation is not established. Results apply to this fixed workload and are
not additive with earlier gains. No universal speedup, resident-evaluation,
cold-I/O, GPU or peak-RSS benefit is claimed.

## Validation and reproduction

All 108 parity/first/warm-up/timed reports match. All 24 complete Gaussian
maps match byte-for-byte, covering 192,880 records with keys/counts, three
means and nine covariance entries. All 24 minimum/maximum maps also match,
including cells below AWD support. Complete ordered group/point hashes and
moment-batch selection counts match the preceding implementation.

An independent audit rebuilds both policies' 24 maps and ordered streams.
Across both passes' 12 map inputs, each policy processes 237,754 logical groups
and 9,411,046 point visits. It checks 72,048 verified scalar calls containing
7,691,524 point visits: each call receives the exact borrowed contiguous f64
slice, all coordinates are finite, and weak references confirm slices are
not retained after processing. Actual filter calls/point counts are traced
against the archived control. The avoided coordinate payload across these
12 inputs is 184,596,576 bytes, without a peak-memory claim.

The audit also independently predicts unchanged moment-batch selection,
checks gather budgets and lifetimes, recomputes 12 structural evaluations,
verifies report/timing summaries, 22 source/config/test/controller fingerprints
and the compiled tree, and checks six report/Gaussian/group-hash chains against
the preceding expansion experiment. Nineteen negative controls reject invalid
counts/sizes, duplicate options, wrong inputs/baselines and existing destinations.

The related MapEval/I/O suite passes 564 tests. Forty-six new cases cover
borrowed read-only offset slices across sizes/scales, every accumulator field,
input lifetime/non-mutation, contiguous/strided/reversed/float32 input handling,
rare scalar groups, filtering fallbacks and backend/iterator failures. Existing
moment, numerical-guard and report regressions also pass. Separate
`resource`-dependent benchmark/parity modules cannot collect on Windows and
are excluded. Rust binaries and GPU selection are unchanged.

The [controller](../cloudanalyzer/scripts/benchmark_map_stream_finite.py)
requires `--source-root`, original `--reference`/`--compared`/`--input-manifest`,
archived `--baseline`/`--baseline-sha256` and
`--io-baseline`/`--io-baseline-sha256`, plus new `--work-dir`/`--output` paths.
Defaults are `--repeats 5 --chunk-size 100000 --structure-voxel-size 3`.
Reproduce with recorded frozen sources. Gaussian files use `<4q12d` records
(128 bytes), bounds files `<3q6d` (72 bytes), and group hashes concatenate each
`<Q` length with row-major `<f8` XYZ bytes. JSON retains all observations,
full reports, dispatch/filter counts and fingerprints.

- PandaSet: [initial](../benchmarks/cpu/map-evaluate/real-source/windows-pandaset019-stream-finite-initial.json), [final](../benchmarks/cpu/map-evaluate/real-source/windows-pandaset019-stream-finite-final.json)
- Autoware: [initial](../benchmarks/cpu/map-evaluate/real-source/windows-autoware-planning-stream-finite-initial.json), [final](../benchmarks/cpu/map-evaluate/real-source/windows-autoware-planning-stream-finite-final.json)
- Tokyo: [initial](../benchmarks/cpu/map-evaluate/real-source/windows-tokyo-takanawadai-stream-finite-initial.json), [final](../benchmarks/cpu/map-evaluate/real-source/windows-tokyo-takanawadai-stream-finite-final.json)
