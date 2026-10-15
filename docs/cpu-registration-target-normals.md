# Reusing target normals during repeated registration preparation

This records the frozen normal-reuse stage. The later
[no-artifact ICP optimization](cpu-scan-icp-normals.md) skips unused normal
estimation there; GICP and artifact-producing calls retain this preparation.
The measurements below keep their original source snapshots and scope.

Uncropped multi-frame SLAM diagnostics now estimate target-map normals once
per run and supply private copies to subsequent registration attempts. For
three repeated real-source attempts, the measured registration preparation
and transformed-result interval is 21.05–36.37% shorter across six comparisons,
with 18/18 faster timed pairs. Source normals are still prepared on every
attempt. ICP solver iterations are excluded from these timings.

## Preparation and ownership

The cache extends the preceding [scoped map reuse](cpu-slam-map-reuse.md).
It is private to one analysis and applies to the fixed, uncropped processed
map coordinate order. The first target-normal computation uses the existing
Open3D hybrid search with radius 0.5 m and at most 30 neighbors. Its resulting
f64 normals are copied before the native solver runs and kept read-only.
Later targets receive a writable temporary copy from which Open3D constructs
its owned normal vector; solver mutations cannot alter the saved array.

Open3D 0.19's [normal estimator](https://raw.githubusercontent.com/isl-org/Open3D/v0.19.0/cpp/open3d/geometry/EstimateNormals.cpp)
derives normals from point neighborhoods or provided covariances. Its
[GICP initialization](https://raw.githubusercontent.com/isl-org/Open3D/v0.19.0/cpp/open3d/pipelines/registration/GeneralizedICP.cpp)
uses supplied covariances or constructs them from normals. This change caches
only the pre-solver normal vectors; covariance handling and the solver remain
unchanged.

Validation and source-first normal preparation keep their original order.
Already-provided source/target normals are preserved, and supplied covariances
remain intact. ICP retains its previous normal preparation and transformed
normal outputs as well as GICP. Standalone registration/scan-match calls keep
fresh preparation. No public CLI option is added.

Positive crops keep separate normal estimation for each frame, because their
geometry can differ. Non-finite normal results are not cached. Frame failures
discard the normal snapshot together with the scoped reference, allowing the
next frame to retry. Run cleanup releases the snapshot on normal return or an
uncaught exception. Map edits remain visible on the next analysis, following
the existing per-run map snapshot behavior.

## Real-source comparison and scope

[Real-source preparation](c2c-real-source.md) supplies attributed/licensed
original-coordinate inputs. No generated geometry, perturbation, added noise
or further thinning is used for timing. Each case repeats the same full scan
and map three times; these are not independent drive frames or accuracy
ground truth. Tokyo retains its preceding original-point selection.

The control is the preceding registration implementation. Both paths create
private source/target clouds and compute source normals on each attempt.
Target preparation changes from three computations to one computation plus
owned snapshot/copy work. The Open3D ICP boundary supplies the same saved
actual full-source solver result **after normal preparation**, so normals are
computed normally in both paths. This isolates normal preparation, not the
native iterative solve.

Clocks include cloud copies, source/target normal estimation, cache finite
checks and all additional copies, fixed-result handling, transformed point/
normal output construction and cleanup. File loading, ranking, NN diagnostics,
hashing/capture writes, coloring/PLY output and actual solver iterations are
excluded. These timings do not measure the entire ICP/GICP or SLAM workflow.

Each source/pass starts a fresh process. Excluded parity and first-use pairs,
then one alternating warm-up pair, precede three alternating timed pairs.
Sources run PandaSet → Autoware → Tokyo, then reverse. No edits, builds,
tests or scans run during timing; every observation remains.

| Source/pass | Recomputed median, ms | Reused median, ms | Median paired shortening | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| PandaSet initial | 9,659.45 | 6,989.53 | +34.29% | 3/3 |
| PandaSet final | 7,985.62 | 5,437.07 | +36.37% | 3/3 |
| Autoware initial | 11,026.43 | 7,151.84 | +35.14% | 3/3 |
| Autoware final | 11,258.12 | 8,888.46 | +21.05% | 3/3 |
| Tokyo initial | 12,469.06 | 8,617.67 | +32.04% | 3/3 |
| Tokyo final | 9,367.03 | 6,477.55 | +34.16% | 3/3 |

Shortening is `median(100 * (1 - reused_ms / recomputed_ms))`, not a ratio of
marginal medians. Scheduling, clocks, other applications and allocator/page
retention are uncontrolled. Individual gains vary: Tokyo final pair 2 took
9,164.2573 ms versus 8,559.2511 ms, while PandaSet final pair 3 took
13,795.8871 ms versus 8,174.1422 ms. Both remain in the summaries, without an
assigned cause. Three pairs per case are limited evidence, not a universal
performance guarantee. Different optimization scopes/percentages are not additive.

## Correctness and live solvers

Each excluded parity pair checks all source/target normal bytes and all
transformed point/normal bytes against preceding recorded ICP arrays.
Across the six cases this covers 36 complete prepared states and 36 complete
transformed states. Every call also checks fitness, RMSE and matrix fields
after timing: 216 prepared registration results in total.

A separate audit independently estimates normals, checks 24 saved full arrays
containing 18,822,092 normal vectors, replays the recorded transforms, verifies
source-first preparation counts and confirms snapshot release. It also
reconstructs all 12 actual prepared solver-input pairs described below.

The [live ICP/GICP checks](../benchmarks/cpu/registration/real-source/windows-target-normal-reuse-solver-checks.json)
use two original-index query subsets, each at most 4,096 measured points,
against each full real-source target. They are correctness checks, not the
timed workload or independent drive frames. All source/target point, normal,
color and covariance fields, initial matrices and solver options match
bit-for-bit at the actual native boundary across 12 pairs. The native solver
executes normally in those checks. Independent executions produce matrix-entry
differences up to `3.437649054305325e-10` despite identical prepared fields;
the raw results are retained. The preparation/fixed-result comparison uses
bit equality; bit-identical live parallel solver output is not claimed.

Twenty-four new tests cover owned read-only storage, private working copies,
degenerate geometry, supplied normals/covariances, actual ICP/GICP input
parity, pre-solver snapshot timing, crops/downsampling, full reports/PLY bytes,
failure retries and release. Five full-pipeline cases compare nine PLY files
each (45 complete file comparisons). Related tests total 197 passing Python/
CLI checks. With the preceding Rust extension, 105 pass and three existing
feature-specific checks skip; normal reuse itself remains functional.

## Memory and reproduction

| Source | Target points | Additional owned normal snapshot, bytes |
| --- | ---: | ---: |
| PandaSet 019 | 531,908 | 12,765,792 |
| Autoware planning | 878,921 | 21,094,104 |
| Tokyo Takanawadai | 941,933 | 22,606,392 |

These snapshots are `24 * target_points` bytes and last for the analysis.
Each reuse also makes a temporary writable copy of that size before Open3D
owns its working normal vector. Clouds, reference trees, covariance/search
scratch and allocator overhead are separate. This avoids repeated work and
extends retained normal lifetime; peak RSS was not measured. Preprovided
normals and cropped targets do not allocate this cached normal array.

The [benchmark script](../cloudanalyzer/scripts/benchmark_registration_target_normals.py)
requires original reference/compared clouds and manifest, the preceding
registration baseline/hash, a saved actual registration record, source root
and new work/output destinations. Defaults are three attempts, one warm-up
pair and three timed pairs. It refuses existing destinations. Fifteen invalid
option/input/destination checks preserve sentinels and leave outputs absent.

Windows Python 3.12.10, NumPy 2.3.4 and Open3D 0.19.0 use 12 OpenMP threads
and one OpenBLAS thread. The records pin 14 Python/source fingerprints and the
151,434,752-byte Open3D CPU library SHA-256
`ab1b2f4a1126a11886a2a9f3eeab5ceaeac0bab1b53ee319eb363df7ef629d88`.
The 31 preceding native Rust fingerprints and wheel remain unchanged. This
step adds no GPU dispatch or persistent cache across analyses.

- PandaSet: [initial](../benchmarks/cpu/registration/real-source/windows-pandaset019-target-normals-initial.json), [final](../benchmarks/cpu/registration/real-source/windows-pandaset019-target-normals-final.json)
- Autoware: [initial](../benchmarks/cpu/registration/real-source/windows-autoware-planning-target-normals-initial.json), [final](../benchmarks/cpu/registration/real-source/windows-autoware-planning-target-normals-final.json)
- Tokyo: [initial](../benchmarks/cpu/registration/real-source/windows-tokyo-takanawadai-target-normals-initial.json), [final](../benchmarks/cpu/registration/real-source/windows-tokyo-takanawadai-target-normals-final.json)
