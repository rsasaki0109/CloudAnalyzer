# No-op scan preprocessing copy experiment

**Not adopted.** Borrowing a loaded point cloud when downsampling is disabled
removes redundant copies, but the complete diagnostic comparison does not
show a stable speed improvement. Only 17/30 timed pairs are faster; the six
median paired changes range from **11.58% slower to 15.03% faster**. Production
retains the preceding copying implementation and the
[unused-normal optimization](cpu-scan-icp-normals.md).

## Candidate and isolation

The candidate changes only the private `_downsample_if_needed` helper:

```python
def _downsample_if_needed(point_cloud, voxel_size):
    """Borrow the input for a no-op; callers copy before any mutation."""
    if voxel_size is None or voxel_size <= 0:
        return point_cloud
    return point_cloud.voxel_down_sample(float(voxel_size))
```

The control returns `o3d.geometry.PointCloud(point_cloud)` in the no-op branch.
Both paths retain independent clouds for initial transformation, cropped or
uncropped registration targets, and transformed registration results. Positive
downsampling is unchanged. The candidate also lets a private map context own
the loaded map directly rather than copying it during no-op preprocessing.

Tests confirm that solver changes to points, normals, colors and covariances
do not reach loaded clouds or the shared map snapshot. This isolation also
holds after exceptions and for crops, positive downsampling, GICP and PLY
output. Correctness was sufficient; the mixed timing result determined rejection.

## Real-source comparison

The [attributed real-source inputs](c2c-real-source.md) retain measured
coordinates, with no generated geometry, perturbation or further thinning.
Each run dispatches the same full scan/map three times through SLAM diagnostic
assembly. These are repeated attempts, not independent drive frames or
recorded SLAM telemetry. Tokyo retains its preceding original-point selection.

Both policies already skip unused ICP normals and reuse the loaded map and
its native NN reference. Timings include actual loading, preprocessing,
remaining cloud copies, native NN preparation/search, distance statistics,
ranking/report assembly, transformed result construction and cleanup. A saved
actual full-source ICP result is supplied at the native solver boundary after
preparation. Actual ICP iterations, artifacts/PLY, parity tracing/hashing/writes
and CLI writes are excluded.

Each source/pass starts a fresh process. An excluded parity pair and first-use
pair, then one warm-up pair, precede five alternating timed pairs. Timed order
starts with borrowing. Sources run PandaSet → Autoware → Tokyo, then reverse.
No edits, builds, tests or scans run during timing. All observations remain.

| Source/pass | Copying median, ms | Borrowing median, ms | Median paired shortening | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| PandaSet initial | 1,811.90 | 1,365.56 | +15.03% | 4/5 |
| PandaSet final | 1,235.07 | 1,237.73 | −0.07% | 2/5 |
| Autoware initial | 2,255.60 | 2,724.63 | −5.89% | 2/5 |
| Autoware final | 3,243.39 | 2,980.89 | +8.09% | 4/5 |
| Tokyo initial | 2,502.31 | 2,429.15 | +7.20% | 4/5 |
| Tokyo final | 2,158.39 | 2,348.01 | −11.58% | 1/5 |

Shortening is `median(100 * (1 - borrowing_ms / copying_ms))`, not the ratio
of marginal medians. Scheduling, clocks, other applications and allocator/page
state are uncontrolled. Autoware final pair 3, at 3,709.7023 ms versus
5,163.2141 ms, is retained without assigning a cause. Five pairs per case are
limited evidence. No general diagnostic speed, full ICP/SLAM runtime, cold
ingestion, peak RSS or GPU performance gain is claimed.

The parity trace observes four no-op preprocessing calls per run: three scan
calls and one map call. All four control results own independent coordinates;
all four candidate results share coordinates with their respective inputs.
For point-only inputs this omits the following coordinate copy payload per
three attempts, without measuring overall allocated bytes or RSS:

| Source | Scan points | Map points | Omitted coordinate copy payload, bytes |
| --- | ---: | ---: | ---: |
| PandaSet 019 | 531,908 | 531,908 | 51,063,168 |
| Autoware planning | 878,920 | 878,921 | 84,376,344 |
| Tokyo Takanawadai | 941,933 | 941,933 | 90,425,568 |

These are `24 * (3 * scan_points + map_points)` copied bytes avoided, not
retained-memory or peak-memory savings. Fewer copies alone did not establish
a useful improvement to the measured diagnostic interval.

## Verification and reproduction

Thirty-six complete prepared solver states match across policies, including
point/normal fields, distance, initial matrix and estimator options. All 72
captured full f64 distance arrays match byte-for-byte across policies and the
preceding oracle. The benchmark checks 96 complete run reports, containing
288 frame reports, including every timed call.

An independent audit recomputes 12 full native distance arrays, checks the
72 captured arrays and statistics/threshold blocks, verifies 24 stored run
reports, and rebuilds six reports through restored production code. It also
checks preprocessing ownership, input/source/library hashes and 16 negative
CLI controls. Existing output sentinels remain unchanged on rejected inputs.

Twenty-three new regression cases cover shared loader objects, all mutable
cloud fields, private map snapshots, failure retries, crops/downsampling,
actual ICP/GICP solver inputs, complete reports and 36 complete PLY file
comparisons. Related tests pass both for the candidate and after restoration:
167 Python tests plus 89 CLI tests. The candidate also passes 164 related
tests with the preceding Rust extension; three existing feature checks skip.

The [benchmark controller](../cloudanalyzer/scripts/benchmark_scan_preprocess_borrow.py)
is frozen with the candidate measurements. To reproduce, use an isolated
source root with the small candidate helper change above, the preceding scan
and registration baseline files with their recorded hashes, original clouds
and manifest, saved actual registration record, and new work/output paths.
Defaults are three attempts, one warm-up pair and five timed pairs. Running
against restored production without the candidate change fails its borrowing
ownership check rather than producing a successful comparison.

Records pin 17 Python/source fingerprints, the unchanged 31 native Rust
fingerprints, installed extension and Open3D CPU library. Timings use Windows
Python 3.12.10, NumPy 2.3.4, Open3D 0.19.0, 12 OpenMP/Rayon threads and one
OpenBLAS thread. The frozen candidate and baseline files remain distinct;
production scan-match source was restored byte-for-byte to the baseline.

- PandaSet: [initial](../benchmarks/cpu/scan-match/real-source/windows-pandaset019-preprocess-borrow-initial.json), [final](../benchmarks/cpu/scan-match/real-source/windows-pandaset019-preprocess-borrow-final.json)
- Autoware: [initial](../benchmarks/cpu/scan-match/real-source/windows-autoware-planning-preprocess-borrow-initial.json), [final](../benchmarks/cpu/scan-match/real-source/windows-autoware-planning-preprocess-borrow-final.json)
- Tokyo: [initial](../benchmarks/cpu/scan-match/real-source/windows-tokyo-takanawadai-preprocess-borrow-initial.json), [final](../benchmarks/cpu/scan-match/real-source/windows-tokyo-takanawadai-preprocess-borrow-final.json)
