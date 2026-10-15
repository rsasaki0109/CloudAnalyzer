# Scoped map reuse across SLAM diagnostic frames

`ca slam-debug --run-scan-match-debug` now shares a lazy map snapshot across
multiple selected frames. It also shares one immutable native reference tree
when those frames use the uncropped map. The measured three-attempt diagnostic
interval is 17.47–44.66% shorter across six real-source comparisons, with
29/30 faster timed pairs. Real file loading is included; ICP solving and
artifact writing are excluded.

A later [target-normal change](cpu-registration-target-normals.md) adds an
owned pre-solver normal snapshot for uncropped frames. The records here pin
the preceding implementation and use fixed normal assignment at the solver
boundary; they do not measure the later normal-estimation savings. Reproduce
these older records with their archived sources.

## Behavior and isolation

Reuse applies when at least two selected frames have manifest scan paths.
Map loading starts only after a frame's scan loads successfully. Map
downsampling follows scan preprocessing, preserving validation/error order.
The first successful map preprocessing supplies one snapshot for that run;
the raw map is then released. The map is read again in the next analysis.
Edits to the map after the snapshot is established become visible on the
next run. Scan files continue to load separately for every selected frame.

Every frame receives its own cropped/copied registration target. Solver
changes to normals, colors or covariances cannot affect later targets or
the cached coordinate snapshot. Positive crops still create and prepare a
new target/reference per frame, since the crop depends on the transformed
scan. Without cropping, a supporting Rust core's owned reference serves all
before/after queries. Missing/disabled/older cores retain stateless distance
dispatch while still sharing map loading/preprocessing.

Failures keep their per-frame error strings. Failed loads/preprocessing are
retried on the next eligible frame. After a frame failure, the prepared
reference is discarded and can be rebuilt; an established map snapshot stays
available. A `finally` block releases the map and reference before returning
the run report, including when an uncaught exception aborts the run.

Disabled execution, empty/no-path selections and a single eligible frame do
not construct a shared context. Standalone `ca scan-match-debug` keeps a fresh
map/reference per invocation. Public arguments, ranking, commands, diagnoses,
registration calls, metric definitions and output formats are unchanged.

## Controlled real-source comparison

[Real-source preparation](c2c-real-source.md) records original coordinates,
source hashes, attribution and licenses. There is no additional thinning,
added displacement, noise or generated geometry. Tokyo retains its preceding
original-point selection. These source subsets are not independent surveys
or accuracy ground truth.

Each workload makes **three diagnostic attempts on the same full source scan
and map**. It is a repeated-work benchmark, not three independent drive
frames. A generated CSV fixture ranks/dispatches those attempts; its scores
are not recorded SLAM telemetry. Both paths receive the same saved actual
full-source ICP result and normals. Reapplying its matrix verifies every
aligned coordinate/normal byte before comparison. Production keeps its usual
ICP/GICP solver; only the benchmark replaces registration with that result.

The control already uses one reference tree per attempt for its two distance
queries. The current path shares map loading and the uncropped reference
across attempts. The workload uses no cropping or downsampling and a 0.5 m
diagnostic threshold.

| Source | Scan loads, control/current | Map loads, control/current | Reference builds, control/current | Queries, both |
| --- | ---: | ---: | ---: | ---: |
| PandaSet 019 | 3 / 3 | 3 / 1 | 3 / 1 | 6 |
| Autoware planning | 3 / 3 | 3 / 1 | 3 / 1 | 6 |
| Tokyo Takanawadai | 3 / 3 | 3 / 1 | 3 / 1 | 6 |

Clocks include real PLY/CSV loading, preprocessing/cloud/normal copies,
ranking and diagnostic/report assembly, native preparation, six NN queries,
summaries, thresholds and scoped cleanup. ICP solving, snapshot preparation,
tracing, hashing/checks, coloring/PLY output and CLI writes are excluded.
Pages are warmed; cold ingestion is not measured. Results do not establish
speed gains for the entire ICP/SLAM pipeline or for cropped/downsampled runs.

Each source/pass starts a fresh process. Excluded traced parity and first-use
pairs plus two alternating warm-up pairs precede five alternating timed pairs.
Sources run PandaSet → Autoware → Tokyo, then reverse. No builds, tests, edits
or scans run during performance comparisons. All observations remain.

| Source/pass | Control median, ms | Shared-map median, ms | Median paired shortening | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| PandaSet initial | 4,364.58 | 3,573.18 | +17.47% | 5/5 |
| PandaSet final | 2,169.83 | 1,213.01 | +44.66% | 5/5 |
| Autoware initial | 2,980.93 | 2,174.13 | +26.70% | 5/5 |
| Autoware final | 4,463.19 | 2,883.47 | +24.36% | 5/5 |
| Tokyo initial | 3,378.20 | 2,439.24 | +23.53% | 5/5 |
| Tokyo final | 3,316.77 | 2,517.87 | +26.28% | 4/5 |

Shortening is `median(100 * (1 - shared_ms / control_ms))`, not the ratio of
marginal medians. Tokyo final pair 4 regressed: control 3,026.3593 ms versus
shared 3,589.7777 ms. It remains in the summary; its cause was not established.
Absolute timings and observed gains vary between passes. Scheduling, power,
other applications and allocator/page retention are uncontrolled. Separate
optimization percentages and operation scopes are not additive.

## Lifetime and memory

The owned reference now stays alive across the selected uncropped frames,
including their artifact writes, and is released at the end of the run.
Standalone scan-match diagnostics retain their preceding release before
artifact writing. Each working registration map remains private. Cropped
references retain their preceding per-frame lifetime.

| Source | Reference points | One native tree vector capacity, bytes |
| --- | ---: | ---: |
| PandaSet 019 | 531,908 | 18,084,896 |
| Autoware planning | 878,921 | 29,883,356 |
| Tokyo Takanawadai | 941,933 | 32,025,740 |

These are native coordinate/index/node capacities, not RSS. The preprocessed
map, frame clouds/normals, distance arrays, registration scratch and allocator
overhead are separate. Reuse reduces repeated allocations/constructions and
extends resource lifetime; net peak RSS has not been measured. There is no
persistent cache across analyses or new worker pool/GPU dispatch.

## Validation and reproduction

All 108 whole-run reports checked during the six comparisons match, including
their 324 attempt reports. Seventy-two full f64 distance arrays agree with the
preceding independent before/aligned distance hashes. A separate audit checks
those arrays against 12 direct native queries, reconstructs all statistic and
threshold blocks and rebuilds six complete production run reports. It verifies
one map load, one preparation and six queries per rebuilt run, with no retained
reference after return.

Thirty new tests cover full report parity against independent fresh calls,
ICP/GICP, Rust enabled/disabled, artifacts on/off, positive/nonpositive crops,
map downsampling, lazy/ineligible selections, retries, snapshot freshness,
target mutation isolation and cleanup on success/failure. Seven artifact
cases compare nine complete PLY files each between both paths (63 file
comparisons). Actual ICP/GICP tests verify identical source/target point,
normal, color and covariance inputs for every frame. Generated fixtures are
used only for correctness tests.

173 related Python/CLI tests pass with the supporting extension. With the
preceding extension, 81 pass and three feature-specific tests skip. The audit
pins 14 Python/source fingerprints, 31 unchanged native fingerprints and the
extension binary, and checks 16 invalid CLI/input/destination cases without
overwriting outputs. Native kernels and the wheel are unchanged.

Windows Python 3.12.10, NumPy 2.3.4 and Open3D 0.19.0 use the same frozen
release extension. Rayon/OpenMP thread counts are 12 and OpenBLAS is 1 before
imports. The [benchmark script](../cloudanalyzer/scripts/benchmark_slam_map_reuse.py)
requires original reference/compared clouds and manifest, captured scan-match
and SLAM baselines with both hashes, a preceding registration record, source
root and new work/output paths. Defaults are three attempts, two warm-up pairs
and five timed pairs. The baseline and registration sources are hashed; each
run refuses existing work/output destinations.

- PandaSet: [initial](../benchmarks/cpu/slam-debug/real-source/windows-pandaset019-map-reuse-initial.json), [final](../benchmarks/cpu/slam-debug/real-source/windows-pandaset019-map-reuse-final.json)
- Autoware: [initial](../benchmarks/cpu/slam-debug/real-source/windows-autoware-planning-map-reuse-initial.json), [final](../benchmarks/cpu/slam-debug/real-source/windows-autoware-planning-map-reuse-final.json)
- Tokyo: [initial](../benchmarks/cpu/slam-debug/real-source/windows-tokyo-takanawadai-map-reuse-initial.json), [final](../benchmarks/cpu/slam-debug/real-source/windows-tokyo-takanawadai-map-reuse-final.json)
