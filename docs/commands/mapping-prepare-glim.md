# Prepare GLIM global LiDAR exports

`ca mapping-prepare-glim` prepares the documented GLIM export pair for the
existing external-SLAM mapping workflow. It does not install or run GLIM.

GLIM documents these dump trajectories in TUM order `t x y z qx qy qz qw`:

| File | Frame | Estimate |
| --- | --- | --- |
| `traj_lidar.txt` | LiDAR | Global mapping, with loop closure |
| `traj_imu.txt` | IMU | Global mapping, with loop closure |
| `odom_lidar.txt` | LiDAR | Odometry, without loop closure |
| `odom_imu.txt` | IMU | Odometry, without loop closure |

This adapter selects only `traj_lidar.txt`, never substitutes another file, and
requires an explicitly supplied PLY export. Save the dump and export points from
the same optimized GLIM state. If you edit the GLIM map, regenerate the matching
trajectory too. The adapter cannot establish that two files came from the same
state merely from their filenames.

Upstream format references reviewed on 2026-10-26:
[GLIM mapping results and PLY export](https://github.com/koide3/glim/blob/master/docs/quickstart.md#mapping-result),
[global trajectory serialization](https://github.com/koide3/glim/blob/master/src/glim/mapping/global_mapping.cpp),
and [offline-viewer PLY export](https://github.com/koide3/glim/blob/master/src/glim/viewer/offline_viewer.cpp).

## Declare the producer and frame

Save a producer JSON using the external-import contract. Replace the example
values with the version, recording, terms, clock and configuration actually used:

```json
{
  "backend": "GLIM",
  "version": "<GLIM version or commit>",
  "source": "<recording and input coverage>",
  "license": "<source-data terms>",
  "coordinate_frame": "local_map",
  "length_unit": "metres",
  "up_axis": "z",
  "pose_convention": "world_from_sensor",
  "time_unit": "seconds",
  "time_basis": "<recording clock>",
  "configuration": {"description": "<actual GLIM settings and calibration>"}
}
```

Map and trajectory must already share a Z-up metre frame. No axis conversion,
alignment, clock correction or IMU-to-LiDAR transform is inferred. Frame/source
statements remain declarations, not independent accuracy or coverage checks.

## Prepare, then import

```sh
ca mapping-prepare-glim /data/glim-dump --map /data/glim-map.ply \
  --provenance producer.json --out prepared/glim
ca mapping-import /data/glim-map.ply \
  --trajectory prepared/glim/trajectory.tum \
  --provenance prepared/glim/provenance.json \
  --layout layout.json --out runs/glim
```

Use the explicit lane hypothesis described in [mapping-run](mapping-run.md).
The resulting run supports the existing inspect/draft/finish workflow and its
quality holds. It does not become deployment-ready because an external SLAM
backend produced its points.

The preparation directory must not exist. It receives:

- `trajectory.tum`: the exact original `traj_lidar.txt` bytes, with the extension
  accepted by the existing importer; timestamps and poses are not rewritten.
- `provenance.json`: the supplied producer declaration plus reserved
  `configuration.cloudanalyzer_glim_export` metadata and input hashes.
- `glim-export.json`: paths, original identities, and the shared input preflight.

The PLY is referenced in place, preserving its attributes without a large copy.
Limits match external import: 512 MiB PLY, 16 MiB trajectory, at least two finite
poses with strictly increasing timestamps and unit quaternions (tolerance 1e-5).
Preparation checks input identities again after preflight and reserves 10 GiB
free space plus its output estimate. Import checks the prepared map/trajectory
hashes and sizes again before creating a mapping job. Keep the map and prepared
files accessible and unchanged; start a new preparation after intentional edits.

Synthetic fixtures based on the documented format test byte preservation,
frame/provenance requirements, rejected alternate sensor/odometry files, changed
inputs, CLI use, and import → inspect → draft → finish. These tests do not claim
execution of GLIM or an accuracy benchmark against GLIM.

To execute GLIM on a compatible ROS2 recording first, use
[`mapping-run-glim`](mapping-run-glim.md), supplying an installed Docker image
and the complete sensor configuration. Its run report records execution evidence;
this export adapter still requires a PLY from the same optimized state.

## Link a recorded execution

When the trajectory came from `mapping-run-glim`, include its report:

```bash
ca mapping-prepare-glim runs/glim-session/dump --map exported-map.ply \
  --provenance producer.json --run-report runs/glim-session/glim-run.json \
  --out prepared/glim-linked
```

The optional report must be a regular JSON file of at most 4 MiB, with supported
schema, successful process status, integer zero exit code, unchanged input and
configuration checks, and a full immutable Docker image ID. Both its trajectory
artifact and validated trajectory snapshot must match the selected export's
SHA-256 and byte count. Older reports without trajectory validation are rejected
when this option is supplied. Omitting the option preserves the external-export
workflow and does not assert any link to a recorded execution.

Preparation copies the exact report bytes to `glim-run.json` and stores a small
execution link in the export metadata and producer configuration. Import checks
the copied report hash and records it as `glim_execution_report` among the job's
immutable dependencies. Keep this copied report accessible with the prepared
trajectory. Editing/removing it invalidates later dependency checks; changes to
the original report after copying do not alter the snapshot.

This binds a local report to exact trajectory bytes. It is not a signed runtime
attestation, and it does not prove the PLY came from the same optimized state,
validate source sensor calibration, or establish map accuracy. The original
report retains input/configuration identities; those original inputs are not
re-read by this export adapter.

After finishing a mapping run, `mapping-run-export` or `mapping-run-preview`
includes the original TUM, import declaration/preflight and optional copied GLIM
execution report in the portable review ZIP. The receiving inspector verifies
their bytes without the original workspace. Historical paths inside records
are preserved; use the archive manifest roles to locate included evidence.
