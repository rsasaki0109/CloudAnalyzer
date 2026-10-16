# Flat-wall IMU recording order

ROS 1 and indexed MCAP chunks previously sorted messages by descending
recording time and popped them from the end. Equal-time groups therefore came
out in reverse physical order. A sensor batch can share one recording time
while its sensor-header timestamps increase normally. This prevented the
validated `ImuSeries` snapshot from accepting the real flat-wall recording.

The chunk queue now uses a stable ascending recording-time sort followed by
reversing the queue. Popping it preserves physical order within each equal-time
group. The reader does not sort sensor headers, repair duplicates, resample
measurements, or change message bytes. Strict header-order validation remains.
The SQLite row-order stream and unindexed MCAP path are unchanged.

The audit covers every IMU message of `flatwall_01.bag`, with an independent
ROS 1 reader on its single IMU connection. The frozen receipts retain the full
payload-order and payload-multiset hashes, exact sensor-header interval
statistics, `ImuSeries` acceptance, source/native provenance, and decoded scan
digests. `before/receipt.json` and `after/receipt.json` record separate runs; their protocols
were written before each audit. The earlier successful studies remain intact.
Git attributes preserve the original receipt/protocol bytes and their hashes.

| Observation | Before | After |
| --- | ---: | ---: |
| IMU messages | 3,755 | 3,755 |
| Positions differing from independent reader | 3,700 | 0 |
| Non-increasing sensor-header steps | 3,553 | 0 |
| Adjacent gaps over 50 ms | 201 | 0 |
| Maximum adjacent header interval | 200.036288 ms | 5.128860 ms |
| Validated `ImuSeries` | Rejected | All 3,755 samples accepted |

All raw IMU payload bytes are preserved. The decoded fields of all 660 scans
(7,175,872 finite points) have identical digests before and after the fix.

The decoded-scan control includes all 186 flat-wall scans and both existing
NCLT sample recordings. It covers exact header times, frame IDs, ordered XYZ
values, intensity and normalized point times. This is input validation, with
no trajectory estimation, ground-truth pose parsing, runtime measurement or
accuracy claim. Raw Livox acceleration still requires the explicit published
scale of 9.80665 before inertial factors; the ordering fix does not convert
vendor units. Physical per-point timing and frame conventions need separate
checks before IMU deskew or scoring.

After building the native extension from this revision, with NumPy and
`rosbags` available:

```bash
python scripts/audit_flatwall_imu_order.py \
  --bag /path/to/flatwall_01.bag \
  --out /path/to/new/audit \
  --expected-order matched \
  --scan-control web/public/samples/nclt-2012-04-29.mcap \
  --scan-control web/public/samples/nclt-2012-06-15.mcap
```

The controller requires the original recording's SHA-256, compares all native
IMU payload bytes in order with the independent reader, and refuses existing
output directories. `--comparison` accepts a prior audit directory and also
requires identical sensor-payload and decoded-scan digests. To reproduce the
old behavior, use the prior native extension and a distinct output directory
with `--expected-order mismatched`.

Serialization regression cases cover ROS 1 none/LZ4/BZ2, indexed MCAP with
little- and big-endian CDR, topic filtering, distinct recording times and
repeated reads. A SQLite tie-order control retains its existing row-order
semantics. Reversed or duplicate sensor headers still fail validation.
Those small fixtures are numerical I/O checks, not trajectory benchmarks.

Source: Kenji Koide (AIST), [LiDAR Degeneration Test Dataset (Flatwall
experiment), version 2](https://zenodo.org/records/7641866),
DOI [10.5281/zenodo.7641866](https://doi.org/10.5281/zenodo.7641866).
The source recording and derived data retain
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) attribution.
The raw bag and reference trajectory are not redistributed here. Software
remains under the repository's software license.

NCLT scan controls use the [previous frontend packet's recordings and
attribution](../nclt-imu-frontend/README.md), from the University of Michigan
North Campus Long-Term dataset. NCLT-derived contents retain their
[ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/) /
[DBCL 1.0](https://opendatacommons.org/licenses/dbcl/1-0/) terms.
