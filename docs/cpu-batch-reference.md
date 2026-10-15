# Scoped reference reuse in batch evaluation

`ca batch candidates/ --evaluate reference.ply` now loads one reference snapshot
per batch. With a supporting Rust core and more than one candidate, it prepares
one immutable reference tree for all source-to-reference searches. Reverse
searches continue to construct a separate tree for each candidate. The complete
batch function improves by 23.83–33.78% across six real-source comparisons,
with 59/60 faster timed pairs. Reference preparation and destruction, file
discovery/loading, both distance directions, statistics and quality gates are
included. CLI startup, JSON/HTML serialization and report writes are excluded.

The immediately preceding implementation loaded the reference once for
validation, then again inside each candidate's evaluation. For three candidates
this change reduces reference loads from four to one and reference tree builds
from three to one. Candidate loading and reverse searches remain independent.
F1, AUC, Chamfer, Hausdorff, point counts, quality gates, inspection commands,
compression metadata and public options retain their output formats and values
for fixed inputs. Ordinary `evaluate` still uses stateless bidirectional calls.

## Lifetime and compatibility

The loaded reference and optional native tree belong to one invocation. They
are released before the final quality/size recommendation step; a later batch
loads and prepares a new reference. Edits to the reference file during a batch
are observed on the next invocation. This snapshot behavior also applies to
Open3D and older cores. Keep candidate files stable while evaluating them.

`prepare_nn_reference` checks the optional extension capability. Older,
missing or disabled cores retain stateless distance dispatch while benefiting
from the single reference load. `CA_DISABLE_RUST_CORE=1` uses Open3D. Single-file
batches avoid an extra prepared tree. Empty directories still return before
loading the reference. Bad references fail before candidate processing; missing
or invalid candidates are skipped as before. Unexpected construction/query
failures propagate. A caught candidate failure releases its cloud; warning
records store error text so retained logs cannot keep its traceback/cloud alive.

The native [owned reference API](cpu-c2c-python-prepared-reference.md) retains
coordinate/index/node vector capacities of 18,084,896 bytes for PandaSet,
29,883,356 for Autoware and 32,025,740 for Tokyo. This payload is held throughout
the batch, alongside the loaded reference (12,765,792 / 21,094,104 / 22,606,392
coordinate bytes). Reference loading already occurred during each preceding
evaluation; the retained native tree adds concurrent payload during reverse
searches. Candidate clouds, reverse trees, copies, results, search scratch and
allocator overhead are separate. These counts are not peak RSS or a memory cap.
No global or cross-batch cache is introduced.

## Real-source workload and timing

[Real-source preparation](c2c-real-source.md) supplies original-coordinate,
disjoint-index reference/compared subsets and source attribution/licenses.
Each benchmark creates three candidates by selecting the existing compared
cloud's indices `0,stride,2*stride,...` with strides 1, 2 and 4. Selection uses
original point coordinates without averaging, displacement, noise or generated
points. Every saved/loaded f64 coordinate bit is checked against that selection.
The workload represents three point-reduction candidates; it does not represent
independent scans, compression codecs or accuracy ground truth. Tokyo retains
the preceding original-point selection. Candidate preparation/writing is outside
the measured interval. Generated fixtures serve only correctness tests.

| Source | Reference points | Candidate points, strides 1 / 2 / 4 |
| --- | ---: | --- |
| PandaSet 019 | 531,908 | 531,908 / 265,954 / 132,977 |
| Autoware planning | 878,921 | 878,920 / 439,460 / 219,730 |
| Tokyo Takanawadai | 941,933 | 941,933 / 470,967 / 235,484 |

Both workflows use the same frozen supporting release extension. The control
loads archived preceding `batch.py` and `evaluate.py`; production modules are
used for the new workflow. No loader/search hooks are active during timed calls.
Each call scans/loads the three real candidate files, runs six directional
queries, computes all evaluation statistics/default thresholds and quality gates
(`min_auc=0.95`, `max_chamfer=0.02`), and frees its scoped tree/clouds. No compressed
or baseline artifact directories are supplied. Compression/Pareto behavior is
covered by ordinary tests, rather than by these performance observations.

Windows, 12 logical CPUs, Python 3.12.10, NumPy 2.3.4 and Open3D 0.19.0 use
Rayon/OpenMP thread counts of 12 before imports. Ten current Python source/
configuration fingerprints and 31 native build fingerprints are recorded with
the executable. Each source/pass starts a new process. A traced full-array
correctness pair, an excluded first timing pair and five warm-up pairs precede
ten alternating timed pairs. Sources run PandaSet → Autoware → Tokyo, then in
reverse order. No builds, tests, scans or edits run during timings.

| Source/pass | Preceding median, ms | Shared-reference median, ms | Median paired shortening | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| PandaSet initial | 1,524.73 | 1,113.87 | +23.83% | 9/10 |
| PandaSet final | 1,412.96 | 958.49 | +32.13% | 10/10 |
| Autoware initial | 2,491.61 | 1,686.70 | +32.06% | 10/10 |
| Autoware final | 2,253.46 | 1,531.02 | +31.84% | 10/10 |
| Tokyo initial | 2,745.87 | 1,873.99 | +33.78% | 10/10 |
| Tokyo final | 2,445.21 | 1,628.75 | +33.49% | 10/10 |

Shortening is `median(100 * (1 - shared_ms / preceding_ms))`, not the ratio
of marginal medians. All observations, including the PandaSet reversal, remain.
OS file/page caches are warmed; scheduling, clocks, power, other applications
and allocator retention are uncontrolled. These results measure the batch
function with three candidates, not full CLI startup or cold-disk performance.
They differ from constructor-excluded repeated-query API timings and from
[scan-match diagnostics](cpu-scan-match-reference.md). Percentages from distinct
workloads are not interchangeable or additive. Browser/WASM/GPU dispatch is
unchanged by this step.

## Validation and reproduction

All 204 parity/first/warm-up/timed batch reports match, covering 612 candidate
results. Seventy-two saved full f64 arrays (six directions per workflow per
comparison) match byte-for-byte. Full-density source-to-reference arrays also
match the preceding independent-wheel oracle for every source/pass. An
independent audit recomputes F1/AUC/Chamfer/Hausdorff from saved arrays, verifies
all 18 candidate selections, reference load counts, native capacities, source/
executable fingerprints, report hashes and timing summaries.

Seventy-seven relevant Python tests pass with the supporting extension. Tests
cover complete per-candidate evaluation/statistics, full directional f64 parity,
custom/default/single/empty thresholds, native/older/Open3D dispatch, one
reference load, single/empty batches, invalid candidates, a reference file edit
between/during batches, cleanup on success and recoverable/unexpected errors,
compression recommendations and CLI/report/integrated checks. The preceding
extension passes 46 related tests with four feature-specific skips. Rust
sources/binaries are unchanged by this integration. Syntax/whitespace checks
and the independent audit pass.

The [benchmark script](../cloudanalyzer/scripts/benchmark_batch_reference.py)
requires `--source-root`, original `--reference`/`--compared`/`--input-manifest`,
the archived `--baseline`/`--baseline-sha256` and
`--baseline-evaluate`/`--baseline-evaluate-sha256`, a matching native
`--build-index`, and new `--work-dir`/`--output` paths. Defaults are
`--threads 12 --repeats 10`. Twelve negative controls verify counts/duplicates,
wrong baseline/build/source fingerprints and existing destinations fail without
replacing data. Reports preserve all observations, coordinates/distance hashes,
complete parity/first results and timing scope.

- PandaSet: [initial](../benchmarks/cpu/batch/real-source/windows-pandaset019-reference-reuse-initial.json), [final](../benchmarks/cpu/batch/real-source/windows-pandaset019-reference-reuse-final.json)
- Autoware: [initial](../benchmarks/cpu/batch/real-source/windows-autoware-planning-reference-reuse-initial.json), [final](../benchmarks/cpu/batch/real-source/windows-autoware-planning-reference-reuse-final.json)
- Tokyo: [initial](../benchmarks/cpu/batch/real-source/windows-tokyo-takanawadai-reference-reuse-initial.json), [final](../benchmarks/cpu/batch/real-source/windows-tokyo-takanawadai-reference-reuse-final.json)
