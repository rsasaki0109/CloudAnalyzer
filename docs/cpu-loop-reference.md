# Scoped reference reuse in before/after map reports

`ca loop-closure-report before.ply after.ply reference.ply` now loads one
reference snapshot for the two map evaluations. A supporting Rust core prepares
one immutable reference tree for both forward searches; reverse searches keep
their separate candidate trees. The complete map-only report function improves
by 10.77–23.88% across six real-source comparisons, with 57/60 faster timed pairs.
File loading, reference construction/destruction, four directional searches,
statistics, deltas and quality gates are included. ICP solving, optional
trajectory/posegraph sections, CLI startup and JSON/report writes are excluded.

The immediately preceding implementation called ordinary `evaluate` twice,
loading/building the reference separately for each map. Reference loads and
reference tree builds change from two to one. Source clouds are still loaded
independently, and each reference-to-source search builds its own source tree.
All distances, per-direction statistics, threshold F1/AUC, Chamfer/Hausdorff,
before/after/delta values and quality gates match the preceding workflow. Public
options and result formats stay unchanged. Ordinary `evaluate` and the earlier
[batch integration](cpu-batch-reference.md) keep their existing policies.

## Lifetime and compatibility

A private map-pair helper owns the loaded reference and optional tree. It
releases the before cloud before loading the after cloud. Returning the two
evaluation dictionaries releases both clouds/reference/tree before the report
adds optional posegraph or trajectory sections. Distances do not escape the
individual evaluation calls. No global or cross-report cache is introduced.
Constructor, search and missing-file errors propagate and unwind this scope.
The before file is loaded first, preserving its missing-file error precedence.

Each report uses one reference snapshot. Reference file edits during the report
are observed on the next invocation; source files should remain stable during
their evaluation. This snapshot behavior also applies to older/disabled cores.
`prepare_nn_reference` uses stateless dispatch when the optional extension is
missing, disabled or lacks `PreparedReference`. These paths still use one loaded
reference. `CA_DISABLE_RUST_CORE=1` selects Open3D. Their performance is not
measured by the six supporting-core comparisons below.

The native [owned reference API](cpu-c2c-python-prepared-reference.md) adds
18,084,896 bytes of tree vector capacity for PandaSet, 29,883,356 for Autoware
and 32,025,740 for Tokyo. Loaded reference coordinates occupy 12,765,792 /
21,094,104 / 22,606,392 bytes. The reference was also loaded during each previous
evaluation; retaining its tree adds concurrent payload during reverse searches.
Source clouds, reverse trees, copies, distance/query buffers, allocator overhead
and optional sections are separate. Capacity counts are not process RSS or a
memory cap; net peak RSS is not measured. Tree payload is released before
optional trajectory/posegraph work.

## Actual input/output workload and timing

[Real-source preparation](c2c-real-source.md) supplies original-coordinate
reference/compared subsets with source attribution and licenses. The before map
is the existing compared cloud. The after map is the actual full-source ICP
result recorded by the preceding [diagnostic benchmark](cpu-scan-match-distance-reuse.md).
Recorded input/reference coordinates, snapshot and transform fingerprints are
verified; applying the recorded matrix reproduces every aligned coordinate bit.
The benchmark saves a double-coordinate PLY and verifies every reloaded bit.
Preparation and solving are outside clocks; timed calls read those real files.
No artificial offset, noise, new thinning or generated coordinates are added.
Tokyo retains its preceding original-point selection.

These inputs measure the cost of a before/after reporting function using an
actual registration result. They do not measure manual loop-closure optimization,
loop-closure accuracy or independent-survey map accuracy. The reference and
before cloud are disjoint-index subsets of one measured source; geometric
duplicates remain possible. Generated fixtures serve only correctness tests.

| Source | Reference points | Before / after points |
| --- | ---: | --- |
| PandaSet 019 | 531,908 | 531,908 / 531,908 |
| Autoware planning | 878,921 | 878,920 / 878,920 |
| Tokyo Takanawadai | 941,933 | 941,933 / 941,933 |

Both workflows use the same frozen supporting release extension/kernel. The
control loads archived preceding report/evaluation modules. Production modules
are used for the new workflow. No loader/search hooks are active during timed
calls. Each invocation loads files, prepares/drops its scoped reference, runs
both distance directions for each map, calculates all default F1 thresholds and
statistics, forms deltas, and applies `min_auc_gain=0.001` /
`max_after_chamfer=0.2`. Optional trajectory/posegraph inputs are not supplied.

Windows, 12 logical CPUs, Python 3.12.10, NumPy 2.3.4 and Open3D 0.19.0 use
Rayon/OpenMP thread counts of 12 before imports. Ten current Python source/
configuration fingerprints and 31 unchanged native build fingerprints accompany
the executable. Each source/pass starts a new process. A traced full-array/
evaluation parity pair, an excluded first timing pair and five warm-up pairs
precede ten alternating timed pairs. Sources run PandaSet → Autoware → Tokyo,
then in reverse order. No builds, tests, scans or edits run during comparisons.

| Source/pass | Preceding median, ms | Shared-reference median, ms | Median paired shortening | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| PandaSet initial | 1,329.80 | 1,044.62 | +23.88% | 10/10 |
| PandaSet final | 978.48 | 801.47 | +19.04% | 10/10 |
| Autoware initial | 2,388.71 | 2,075.92 | +10.77% | 9/10 |
| Autoware final | 1,733.52 | 1,514.95 | +17.03% | 10/10 |
| Tokyo initial | 2,057.42 | 1,643.42 | +23.10% | 9/10 |
| Tokyo final | 1,964.70 | 1,745.67 | +12.92% | 9/10 |

Shortening is `median(100 * (1 - shared_ms / preceding_ms))`, not the ratio of
marginal medians. All observations, including reversals, remain. Passes vary;
OS file caches are warmed, and scheduling, clocks, power, other applications
and allocator/page retention are uncontrolled. These results measure the
map-only report function, rather than full CLI startup/cold-disk access or
optional trajectory/posegraph costs. Distinct API/diagnostic/batch percentages
are not interchangeable or additive. This step changes no browser/WASM/GPU
dispatch and makes no GPU-speed claim.

## Validation and reproduction

All 204 parity/first/warm-up/timed reports match. Forty-eight saved full f64
arrays (four directions per workflow per comparison) match byte-for-byte;
all 24 complete traced evaluations also match, including per-direction
mean/median/max/std and every threshold score. Full-density before/after
forward arrays match the preceding independent diagnostic records, and before
forward arrays match the independent preceding-wheel C2C oracle. An independent
audit recomputes all evaluation/statistic/delta/gate values from saved arrays,
reapplies all six aligned-cloud transforms, and verifies load counts, capacities,
source/executable fingerprints, report hashes and all timing summaries.

Eighty-three distinct relevant Python tests pass across selected runs with the
supporting extension. They cover native/older/Open3D dispatch, complete
directional f64/evaluation parity, default/custom/single/empty thresholds, one
reference load, reference edits during/between invocations, source/tree/array
lifetime, construction/query/load failure cleanup, optional posegraph/trajectory
sections, CLI quality gates, config-driven QA and the preceding batch integration.
The preceding extension passes 43 report/evaluation/CLI tests with four
feature-specific skips. Native sources and binaries are unchanged. Syntax,
whitespace, document links and independent evidence checks pass.

The [benchmark script](../cloudanalyzer/scripts/benchmark_loop_reference.py)
requires `--source-root`, original `--reference`/`--compared`/`--input-manifest`,
archived `--baseline`/`--baseline-sha256` and
`--baseline-evaluate`/`--baseline-evaluate-sha256`, a matching native
`--build-index`, the preceding actual `--registration-record`, and new
`--work-dir`/`--output` paths. Defaults are `--threads 12 --repeats 10`.
Fourteen negative controls cover counts/duplicates, wrong baseline/build/data/
registration/snapshot fingerprints and existing destinations without replacing
data. All observations, scopes, complete parity/first reports and distance/
aligned-coordinate fingerprints are retained.

- PandaSet: [initial](../benchmarks/cpu/loop-closure/real-source/windows-pandaset019-reference-reuse-initial.json), [final](../benchmarks/cpu/loop-closure/real-source/windows-pandaset019-reference-reuse-final.json)
- Autoware: [initial](../benchmarks/cpu/loop-closure/real-source/windows-autoware-planning-reference-reuse-initial.json), [final](../benchmarks/cpu/loop-closure/real-source/windows-autoware-planning-reference-reuse-final.json)
- Tokyo: [initial](../benchmarks/cpu/loop-closure/real-source/windows-tokyo-takanawadai-reference-reuse-initial.json), [final](../benchmarks/cpu/loop-closure/real-source/windows-tokyo-takanawadai-reference-reuse-final.json)
