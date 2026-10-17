# Flatwall recording inputs and source provenance

This input audit prepares the next numbered real recording, Flatwall02, for a
cross-recording prediction-policy comparison. It also rechecks all of Flatwall01
with the same independent decoder. No state estimation or reference-archive read
occurs in this stage. Preservation checks may hash existing opaque archives;
reference-pose contents are not decoded by this input audit.

| Recording | Original scans | Finite points | IMU samples | IMU-covered scans, header origin | Header minus integer timebase (ns) |
| --- | ---: | ---: | ---: | ---: | ---: |
| flatwall_01 | 186 | 4,464,000 | 3755 | 185 | -233 to 209 |
| flatwall_02 | 137 | 3,288,000 | 2775 | 135 | -218 to 222 |

Both recordings retain every original scan, including scans outside IMU coverage.
The packet reports coverage for both header-plus-offset and integer-timebase-plus-
offset timestamps. Endpoints are inclusive; every intersected IMU interval must be
at most 50 ms. Neither calculation uses reference poses or extrapolation.

The independent ROS decoder compares every CustomMsg point with PointCloud2:
coordinates, reflectivity, tag, line and integer offset time, in original order.
The isolated native reader also matches all finite XYZ values, header-plus-offset
point timestamps, and every original IMU header, vector and frame. Per-scan
digests and complete raw IMU arrays are retained. Original point payloads and the
large bags are not copied into this packet.

## What the primary sources establish

The [author's Flatwall configuration](https://staff.aist.go.jp/k.koide/projects/glim_params/config_flatwall.tar.gz)
sets acceleration scaling to 9.80665, IMU offset to zero and `T_lidar_imu` to
identity. Its comments define that transform from IMU coordinates into LiDAR
coordinates and use translation followed by an xyzw quaternion. This is the
opposite direction to CloudAnalyzer's `imu_from_lidar`; identity is unchanged by
inversion. The fetched archive is byte-identical to the previously preserved
download, so these configured settings were available before the prior trials.

The [manufacturer SDK](https://github.com/Livox-SDK/Livox-SDK/blob/9306596a2bf15c1343bc023b497465ed0a32909d/sdk_core/include/livox_def.h)
defines gyro values in rad/s and acceleration in g. The
[pinned ROS driver](https://github.com/Livox-SDK/livox_ros_driver/blob/3d240d5666129e1a3052e78ee8487a04b08fdda3/livox_ros_driver/livox_ros_driver/lddc.cpp)
copies those IMU values directly. It computes custom offsets in ns and writes the
ROS header through double seconds. The observed header/timebase differences are
consistent with rounding in that path. The bags name a replay process as caller;
the actual producer/converter version remains unidentified.

The [dataset author](https://staff.aist.go.jp/k.koide/projects/glimsupp/flatwall.html)
describes LiDAR ground truth from a rigid camera observing AprilTags. The
[paper, VI-A](https://arxiv.org/html/2407.10344v1#S6.SS1) describes tag/IMU camera
trajectory optimization, PTP synchronization and calibrated LiDAR-camera
alignment. Numeric camera extrinsics in the configuration do not establish how
the published reference text was converted. The archive-specific pose convention
and numerical reference-to-LiDAR clock mapping remain unresolved. A zero
configured IMU offset is not evidence of a zero reference offset.

## Reproduce the public checks

```sh
python benchmarks/slam/flatwall-recording-inputs/verify.py
python -m pytest cloudanalyzer/tests/test_flatwall_recording_inputs.py cloudanalyzer/tests/test_flatwall_recording_inputs_packet.py -q
```

The verifier independently rebuilds bracket/gap accounting from exact published
IMU timestamps, scan origins and offsets; it checks complete scan identities,
norm statistics, source footprints and receipt hashes. It cannot redo point
parity without the original bags, or certify physical calibration. A raw rerun
also needs rosbags, the isolated native build and a fresh protocol/hash freeze;
use a new output directory with `scripts/audit_flatwall_recording_inputs.py`.

Input acquisition uses the original size and MD5 from
[Zenodo v2](https://zenodo.org/records/7641866). The successful transfer's SHA256,
source retrieval receipts, a failed duplicate transfer and its partial-file hash
are recorded. Dataset-derived arrays and metadata carry CC BY 4.0 attribution in
`attribution.json`.

All eight reference prefixes were historically exposed and full 01 was already
evaluated. Full 02 has not been parsed here; a later comparison remains
exploratory. Retain the previous header-origin results. Freeze the next complete
eight-case experiment before new estimates, and freeze every estimate before
reading full 02 reference poses. This audit changes no estimator default or
initialization gate and makes no accuracy or speed claim.
