# Skipping unused normals in point-to-point diagnostics

A subsequent [no-op preprocessing copy experiment](cpu-scan-preprocess-borrow.md)
did not show stable diagnostic speed gains and was rejected. Production
retains this stage's scan-match implementation.

The later [binary XYZ PLY loader](cpu-ply-xyz-loading.md) accelerates eligible
file ingestion while preserving this stage's scan-match code. Its measurements
use separate frozen I/O sources and include loading.

ICP diagnostics without PLY artifacts now skip estimation of missing source
and target normals. Six real-source comparisons shorten the measured
diagnostic interval by **72.16–81.47%**, with **18/18 faster timed pairs**.
This interval includes loading and distance diagnostics but excludes the
iterative ICP solve. It is not an entire ICP/SLAM runtime measurement.

## Conditions and behavior

The change applies to `run_scan_match_debug` and selected SLAM frame
diagnostics when the method is point-to-point ICP and `artifact_dir is None`.
It covers standalone calls, scoped multi-frame calls, crops and downsampling.
The private registration option defaults to the preceding behavior.

GICP always prepares normals. Artifact-producing ICP keeps its normal
preparation and PLY normal fields. Ordinary `register` and `register_detailed`
calls retain their normal preparation, input mutation and transformed normal
outputs. Provided normals, colors and covariances are preserved; the fast
path only avoids estimating missing normals. No public CLI flag, thread
override or persistent cache is added. A missing-normal estimation failure
can no longer prevent the eligible diagnostic from reaching its ICP solve.

The installed Open3D version is 0.19.0. Its
[point-to-point estimator](https://raw.githubusercontent.com/isl-org/Open3D/v0.19.0/cpp/open3d/pipelines/registration/TransformationEstimation.cpp)
uses corresponding point coordinates for transformation and RMSE. Its
[ICP implementation](https://raw.githubusercontent.com/isl-org/Open3D/v0.19.0/cpp/open3d/pipelines/registration/Registration.cpp)
builds correspondences from point coordinates. Neither calculation consumes
normals. The no-artifact diagnostic returns distances, transforms and scalar
metrics, so the previously generated normals have no output consumer there.

The preceding [target-normal cache](cpu-registration-target-normals.md)
continues to serve GICP and artifact-producing, uncropped multi-frame calls.
Eligible ICP diagnostics leave its normal array unallocated. Crop behavior,
map/reference reuse, solver settings, validation order, per-frame failure
handling and cleanup remain in place.

## Measurements

The attributed/licensed [real-source inputs](c2c-real-source.md) retain their
measured coordinates, with no generated geometry, perturbation or additional
thinning for these timings. Each workload dispatches the same full scan/map
three times through SLAM diagnostic assembly. The dispatch CSV is a fixture;
these are not three independent driving frames or recorded SLAM telemetry.
Tokyo retains its preceding original-point selection.

The control already shares map loading, its uncropped native NN reference and
target normals. It performs three source-normal estimates and one target
estimate per run; the candidate performs none. Both paths execute actual
loading, cloud copies, preprocessing, native NN preparation/search, distance
statistics, ranking/report assembly and cleanup. The Open3D solver boundary
supplies the same saved actual full-source ICP result **after preparation**.
Construction of the transformed result is included. Native iterative solving,
coloring/PLY output, parity tracing/hashing/writes and CLI writes are excluded.

Each comparison starts a fresh process, runs an excluded parity pair and
first-use pair, then one warm-up pair and three timed pairs. Timed order is
skip → normals, normals → skip, skip → normals. Sources run PandaSet →
Autoware → Tokyo and then reverse. No edits, builds, tests or scans run during
timing. Every observation is retained; input pages are warmed.

| Source/pass | With normals median, ms | Skipped median, ms | Median paired shortening | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| PandaSet initial | 4,746.92 | 879.45 | +81.47% | 3/3 |
| PandaSet final | 5,042.07 | 969.67 | +80.80% | 3/3 |
| Autoware initial | 6,933.16 | 1,451.62 | +79.01% | 3/3 |
| Autoware final | 7,904.19 | 2,200.17 | +72.16% | 3/3 |
| Tokyo initial | 8,259.40 | 1,950.70 | +75.68% | 3/3 |
| Tokyo final | 7,431.71 | 1,732.99 | +76.51% | 3/3 |

Shortening is `median(100 * (1 - skipped_ms / with_normals_ms))`, not a ratio
of marginal medians. Autoware final pair 2, at 8,075.8840 ms versus
2,953.0195 ms, is retained along with every other pair. Scheduling, clocks,
other applications, allocator state and page retention are uncontrolled;
three pairs per case are limited evidence. The different optimization
intervals and percentages cannot be added. No cold-ingestion, peak RSS,
full iterative registration, cropped-workload or GPU speed claim is made.

## Correctness

The parity runs capture 36 complete native solver input states. Coordinates,
identity init, distance limit, point-to-point estimator and disabled scaling
match; missing normals are intentionally absent in the candidate. All 24
baseline normal estimates match the preceding recorded normal arrays.
Seventy-two captured full f64 distance arrays match byte-for-byte across
policies and the preceding distance oracle. Seventy-two complete run reports
(216 frame reports) are checked during the benchmark, including timed calls.

An independent audit recomputes 12 full native distance arrays, checks all
72 stored arrays and 72 statistics/threshold blocks, rechecks 24 stored run
reports and rebuilds six production reports. It observes 36 queries, six
map/reference preparations and 18 scan loads, verifies released references,
and rejects any missing-normal estimation on the candidate path.

The separate [actual ICP checks](../benchmarks/cpu/scan-match/real-source/windows-icp-unused-normals-solver-checks-v2.json)
execute 12 native solves on two unchanged original-index source subsets per
dataset, each containing 4,096 points, against the full real target. Six
pairs have identical relevant solver fields. Their fitness, RMSE, final
matrices and transformed point hashes match exactly. These checks use one
OpenMP/OpenBLAS thread to fix reduction order; production does not force that
setting. Parallel native ICP can vary slightly between independent solves,
as recorded in the preceding normal-reuse study. Exact parallel live-solver
output, full-source live solves and performance of these subsets are not claimed.

Thirty-six new tests cover consumer conditions, case-insensitive ICP, private
map contexts, crops/downsampling, complete fixed-solver reports, actual ICP
inputs/results, provided attributes, public defaults, GICP, validation and
failure retries. Related Python/CLI checks total 233 passes. The old Rust
extension also passes 141 related checks with three existing feature skips.
Existing complete PLY parity cases pass with normal preparation preserved.

## Reproduction and records

The [controller](../cloudanalyzer/scripts/benchmark_scan_icp_normals.py)
requires original reference/compared clouds and their manifest, preceding scan
and registration baselines with hashes, a saved actual registration record,
source root and new work/output destinations. Defaults are three attempts,
one warm-up pair and three timed pairs. Sixteen negative CLI checks confirm
that invalid options, altered inputs/baselines and existing destinations
preserve sentinels and leave new outputs absent.

Windows Python 3.12.10, NumPy 2.3.4 and Open3D 0.19.0 run with 12 OpenMP/Rayon
threads and one OpenBLAS thread for timing. Records pin 16 Python/source
fingerprints, unchanged native extension inputs and the 151,434,752-byte
Open3D CPU library SHA-256
`ab1b2f4a1126a11886a2a9f3eeab5ceaeac0bab1b53ee319eb363df7ef629d88`.
The 31 native Rust source fingerprints remain unchanged.

The initial controller stopped during parity tracing because two recorded
fields shared a name. No timed trial or successful performance report was
produced by that attempt. Its frozen files remain; all six completed reports
below use the corrected controller and distinct `v2` work/output paths.

- PandaSet: [initial](../benchmarks/cpu/scan-match/real-source/windows-pandaset019-unused-normals-v2-initial.json), [final](../benchmarks/cpu/scan-match/real-source/windows-pandaset019-unused-normals-v2-final.json)
- Autoware: [initial](../benchmarks/cpu/scan-match/real-source/windows-autoware-planning-unused-normals-v2-initial.json), [final](../benchmarks/cpu/scan-match/real-source/windows-autoware-planning-unused-normals-v2-final.json)
- Tokyo: [initial](../benchmarks/cpu/scan-match/real-source/windows-tokyo-takanawadai-unused-normals-v2-initial.json), [final](../benchmarks/cpu/scan-match/real-source/windows-tokyo-takanawadai-unused-normals-v2-final.json)
