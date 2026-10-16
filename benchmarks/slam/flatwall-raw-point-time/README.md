# Original per-point time values

`BagReader.scans(..., raw_time_field="t")` and `messages(...,
raw_time_field="t")` now retain an explicitly named scalar source field.
`Scan.raw_time()` returns copied float64 values aligned with finite XYZ;
`raw_time_field` and `raw_time_datatype` are read-only metadata. All standard
PointField integer values fit exactly in float64, FLOAT32 values widen, and
FLOAT64 values retain their source precision. Units and clock origin are not
inferred. Constant/nonfinite values and physical point order are retained.

The default does not retain this buffer. Existing header, XYZ, intensity and
normalized `Scan.time()` behavior remain unchanged. If the requested field
shares the existing normalized-time layout, the decoder moves the existing
temporary f64 buffer into the result. A different field needs a separate
buffer. This extends retained memory lifetime; it is not a speed measurement.

The frozen audit compares all finite points with independent standard ROS
PointCloud2 decoding and the previous native build:

| Recording | Scans | Finite points | Raw point time |
| --- | ---: | ---: | --- |
| Flat-wall | 186 | 4,464,000 | Every UINT32 `t` value matches exactly |
| NCLT April control | 236 | 1,400,497 | No `t` field; an explicit request fails |
| NCLT June control | 238 | 1,311,375 | No `t` field; an explicit request fails |

All 660 scans / 7,175,872 finite points retain identical hashes of every
existing output field, both against independent decoding and the old native
build. Enabling raw time on flat-wall also preserves these legacy fields.
All 4,464,000 source time values retain their exact widened float64 bits.
The eleven decreasing steps within source point order and zero duplicates
are preserved, without sorting or repairing measurements.

The NCLT control recordings contain only XYZ and intensity fields. Their
normalized/raw time remains unavailable. They have empty MCAP schema text,
so the independent decoder explicitly supplies the standard ROS2_HUMBLE
PointCloud2 definition. The flat-wall ROS 1 recording uses ROS1_NOETIC.
The recordings themselves are not changed.

Serialization regression checks cover all eight PointField numeric datatypes,
ROS 1, both CDR byte orders, SQLite, point-data endianness, organized clouds
with row padding, finite-point filtering, copied/read-only access and constant,
nonfinite or empty results. An explicit request rejects missing/duplicate field
names, unsupported/nonscalar fields, invalid extents, overlapping rows and
incomplete point records instead of filling timing values with zero.

Reproduce with NumPy, `rosbags`, and the native Python extension. The baseline
native build is from parent commit `dc6a49a7a080d5abcb74459c7f0ab9d74427775f`;
keep this same controller for both runs and use separate output directories:

```bash
python scripts/audit_flatwall_raw_point_time.py \
  --bag /path/to/flatwall_01.bag --out /path/to/new/baseline --baseline \
  --control nclt-april web/public/samples/nclt-2012-04-29.mcap /velodyne_points \
  --control nclt-june web/public/samples/nclt-2012-06-15.mcap /velodyne_points
# Switch to the native extension built from this revision.
python scripts/audit_flatwall_raw_point_time.py \
  --bag /path/to/flatwall_01.bag --out /path/to/new/raw-audit \
  --comparison /path/to/new/baseline \
  --control nclt-april web/public/samples/nclt-2012-04-29.mcap /velodyne_points \
  --control nclt-june web/public/samples/nclt-2012-06-15.mcap /velodyne_points
```

The controller refuses existing outputs and freezes recording, native,
controller, reader and standard-type-definition hashes. It verifies the
baseline protocol link, independent fields, raw values and legacy parity.
Before/after protocols and receipts are exact original byte copies.

This is an input/precision check. No reference trajectory was parsed, no
deskew or trajectory estimated, and no runtime or accuracy score measured.
Declared units/origin, checked timestamp conversion and IMU bracketing are
still required before inertial motion compensation. See the
[sensor timing API](../../../docs/slam-sensor-timing.md) for the contract.

Dataset: Kenji Koide (AIST), [LiDAR Degeneration Test Dataset, version
2](https://zenodo.org/records/7641866), DOI
[10.5281/zenodo.7641866](https://doi.org/10.5281/zenodo.7641866),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
NCLT-derived inputs retain the University of Michigan
[attribution and ODbL/DBCL terms](../nclt-imu-frontend/README.md).
The raw flat-wall recording and reference poses are not redistributed here.
Software remains under the repository's software license.
