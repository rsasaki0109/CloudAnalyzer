# Declared integer point timestamps

`Scan.point_times_ns(unit="ns", origin="header")` converts an explicitly
retained source field into a new int64 array. Both conventions are required:
units are `s`, `ms`, `us` or `ns`; origin is the integer sensor header or zero
for an already absolute field. Names and datatypes never select a convention.

The converter uses the exact represented binary value and an integer unit
scale, rounds to nearest nanoseconds with ties to even, then adds the integer
origin. It does not multiply an epoch value in float64 or add a header in
float64. Rounding precedes origin addition, including for odd headers.
Nonfinite values and final int64 overflow are errors. A final timestamp may
fit even when its offset alone exceeds int64. Conversion preserves physical
point order and does not change or cache raw values, normalized fractions,
XYZ, intensity or headers. See the [API contract](../../../docs/slam-sensor-timing.md).

The frozen audit declares the flat-wall `UINT32 t` values to be nanosecond
offsets from the PointCloud2 integer Header. It compares every converted point
with independently decoded UINT32 values plus Python integer header arithmetic.
The baseline native build is from parent commit
`0f4c7c1fe60b4a2d29b78e1479c98f1cc11df295`; it uses checked manual integer
conversion from the existing raw-time API. The new build calls the new API
directly, with no fallback.

| Recording | Scans | Finite points | Timing check |
| --- | ---: | ---: | --- |
| Flat-wall | 186 | 4,464,000 | Every int64 point timestamp matches |
| NCLT April control | 236 | 1,400,497 | No retained source time; conversion rejects it |
| NCLT June control | 238 | 1,311,375 | No retained source time; conversion rejects it |

All 660 scans / 7,175,872 finite points retain identical existing output-field
digests against the old build and independent standard ROS decoding. Flat-wall
raw values also remain identical before and after conversion. Its ordered
point-timestamp digest is
`2433bc10419c9cb0be6a1c1fb67ae486ee672b291863c892dcdfb9583a895199`,
including the eleven decreasing source-time steps.

A separate control adds the epoch header in float64 before rounding. That
ordering changes 4,446,578 point timestamps, by -246 through +247 ns. Integer
conversion avoids those additional errors. This does not restore precision
already lost by a publisher or establish the physical clock convention.

The declared timestamps fully fit IMU coverage for 185 scans. The first scan
contains 632 points before the first IMU sample; no points follow the last
sample. The audit reports this coverage without extrapolation, clock shifts or
rewriting either source timebase. No reference trajectory was parsed, no
deskew or trajectory estimated, and no accuracy or runtime score measured.

The independent decoder explicitly supplies standard ROS1_NOETIC and
ROS2_HUMBLE types, including for the NCLT MCAP controls with empty schema text.
The Python numerical tests separately compare all unit/origin combinations
with `Fraction.from_float` arithmetic, covering floating epoch values, halfway
neighbors, subnormals, negative headers, range cancellation, copied arrays,
empty results and errors across ROS 1, both CDR byte orders and SQLite.

Reproduce with NumPy, `rosbags`, and separate old/new native Python builds.
Keep both audit scripts unchanged between runs and use new output directories:

```bash
python scripts/audit_flatwall_point_times_ns.py \
  --bag /path/to/flatwall_01.bag --out /path/to/new/baseline --baseline \
  --control nclt-april web/public/samples/nclt-2012-04-29.mcap /velodyne_points \
  --control nclt-june web/public/samples/nclt-2012-06-15.mcap /velodyne_points
# Switch to the native extension built from this revision.
python scripts/audit_flatwall_point_times_ns.py \
  --bag /path/to/flatwall_01.bag --out /path/to/new/integer-times \
  --comparison /path/to/new/baseline \
  --control nclt-april web/public/samples/nclt-2012-04-29.mcap /velodyne_points \
  --control nclt-june web/public/samples/nclt-2012-06-15.mcap /velodyne_points
```

The controller imports the frozen field-comparison helpers from
`audit_flatwall_raw_point_time.py`. It refuses existing outputs and freezes
both controllers, recordings, native build, independent reader and standard
type-definition hashes. Public protocols and receipts are exact byte copies.

Dataset: Kenji Koide (AIST), [LiDAR Degeneration Test Dataset, version
2](https://zenodo.org/records/7641866), DOI
[10.5281/zenodo.7641866](https://doi.org/10.5281/zenodo.7641866),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
NCLT-derived inputs retain the University of Michigan
[attribution and ODbL/DBCL terms](../nclt-imu-frontend/README.md).
Original flat-wall recordings and reference poses are not redistributed here.
Software remains under the repository's software license.
