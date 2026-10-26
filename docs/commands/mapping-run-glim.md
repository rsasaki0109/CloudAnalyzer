# mapping-run-glim

Run GLIM's ROS2 bag executable in an already installed local Docker image and
retain the run evidence. This command requires Linux, Docker socket access and a
complete GLIM configuration matching the image and the sensor recording.

```bash
docker pull koide3/glim_ros2:jazzy
ca mapping-run-glim recording.mcap \
  --config sensor-config --image koide3/glim_ros2:jazzy \
  --out runs/glim-session --timeout 3600 \
  --points-topic /velodyne_points --imu-topic /ms25/raw --max-imu-gap 0.05
```

## Input preflight before launch

Supplying both `--points-topic` and `--imu-topic` runs the existing MCAP sensor
preflight before even querying Docker. `--max-imu-gap` sets its gap policy
(default 0.05 seconds). Specify the topics actually selected by your GLIM
configuration. With integrated preflight, the runner reads `config.json` and
its selected `global.config_ros` file (default `config_ros.json`), then checks
`glim_ros.points_topic` and `glim_ros.imu_topic` against the diagnostic topics.
Omitted topic keys use the bag reader's `/points` and `/imu` defaults. The
settings are never rewritten. This validates topic selection only, not sensor
calibration, timestamp offsets, scaling or module compatibility.

Commented JSON (`//` and `/* ... */`) is supported, including literal comment
markers inside strings. Each configuration file is limited to 1 MiB. Duplicate
keys, malformed JSON and external/parent-traversing `config_ros` paths are
rejected. The result and checked file hashes appear in `config_topics`.
A mismatch or unreadable/unsupported configuration returns `config_blocked`
with the reason in the run receipt, before any Docker call.

Blocked input returns exit 1 and `input_blocked`, preserving its complete
`input_preflight` in `glim-run.json`. No container, dump, process log or
configuration copy is created for that case. If the recording/configuration
changes during preflight, the run instead stops as `inputs_changed`. A passing
report is retained in the eventual run record and bound to the inventoried
recording hash. The copied configuration and current input inventory are checked again before
container launch, and process completion checks unchanged inputs afterward.

Supply both topic options or neither; one alone is an argument error. With
neither, `input_preflight_status` is explicitly `not_requested` and existing
execution behavior is retained. Integrated preflight currently supports MCAP
files only, as does the standalone checker. Directory inputs must use the
existing execution path without these options. Basic checks do not certify
calibration or estimator accuracy.

Missing Docker/image lookup errors return exit 1 with `runtime_unavailable`
and a retained run record, including any completed preflight. A new output
directory is needed for a retry; the failed record is not overwritten.

A ROS2 bag directory containing `metadata.yaml` is also accepted. The output
must be new and outside the input directories. Input symlinks are rejected.
The existing disk reserve check requires 10 GiB free beyond configuration-copy
space before launch; Docker image installation needs additional space and this
is not a runtime disk quota.

Supply a complete sensor-specific configuration, including topics, point time
fields, timestamp units, LiDAR/IMU calibration and CPU-compatible modules. For
headless use disable viewer extensions in that configuration. This command does
not infer calibration or rewrite GLIM settings. References to files outside the
mounted configuration directory are not supported. Do not assume repository
sample MCAPs contain the sensor topics and calibration required by GLIM.

The runner resolves the image to its immutable local image ID before execution.
It mounts the bag and a saved configuration copy read-only, mounts only the dump
directory writable, uses the caller's UID/GID, disables container networking,
and invokes `ros2 run glim_ros glim_rosbag` with `config_path`, `dump_path` and
`auto_quit:=true`. Image download is explicit, never part of a run.

Outputs:

- `glim-run.json`: image ID, exact command, input SHA-256 inventories, elapsed
  execution/cleanup time, exit code, input-change checks and trajectory hash.
- `glim.log`: combined process stdout/stderr, including failures.
- `config/`: configuration snapshot.
- `dump/`: GLIM-produced artifacts, including global `traj_lidar.txt` when available.

Timeout and keyboard interruption force removal of this run's uniquely named
container. Failed runs retain their logs and manifest. A zero exit code without
a nonempty regular global LiDAR trajectory is `output_missing` (symlinks are
not accepted). A zero exit with invalid trajectory contents is `output_invalid`.
The shared importer check enforces a 16 MiB limit, at least two finite TUM rows
with eight columns, strictly increasing timestamps, unit quaternions within
1e-5 absolute tolerance, and finite path length. The producer's `.txt` extension
is accepted without renaming. Contents are checked from a bounded byte snapshot
and the file hash is compared again after validation.

`trajectory_check` records row count, first/last timestamps, path length and the
validated byte count/SHA-256; `trajectory_error` explains a failure. Existing
process failure/timeout status is retained if its partial output is also invalid.
Changed inputs invalidate the run. `process_succeeded` now means zero exit plus
valid trajectory representation. It does not establish full bag coverage,
common-frame truth, sensor calibration or accuracy. Time ordering follows the
importer's float64 TUM representation; sub-float64 timestamp differences may be
rejected as duplicate. Runtime dependency versions beyond the image ID are not
separately inventoried.

Export the optimized map as PLY from the same saved GLIM state, then use
[`mapping-prepare-glim`](mapping-prepare-glim.md) and `mapping-import`. The runner
does not automatically export a PLY or invent frame/time provenance. Keep the
run manifest alongside the explicit producer declaration for traceability.

Implementation follows the upstream
[GLIM ROS2 bag executable](https://github.com/koide3/glim_ros2/blob/master/src/glim_rosbag.cpp)
and [official Docker instructions](https://github.com/koide3/glim/blob/master/docs/docker.md).

Validation in this environment covers orchestration contracts with a substituted
container process (success, failure, missing output, timeout, input mutation,
symlinks and output isolation). Pulling the official Jazzy image failed during
layer extraction with `no space left on device`; Docker reclaimed the failed
pull and reported no retained images/containers. Real GLIM execution and accuracy
are therefore not yet validated here.

Before allocating a GLIM run, use the separate read-only
[`mapping-check-glim`](mapping-check-glim.md) diagnostic on an MCAP input. It
checks declared IMU availability and header timing. The repository demo bags
now decode with standard ROS tools, but mark inertial
measurements unavailable and lack full IMU timestamp coverage; they are not validated GLIM benchmark recordings.

To recheck the saved run later without starting Docker, use
[`mapping-inspect-glim`](mapping-inspect-glim.md). It reports current file
integrity separately from the original execution outcome.

## Preparation failures and interrupted runs

Docker image lookup has a separate 30-second timeout. A timeout or lookup failure
is `runtime_unavailable`; interruption is `interrupted`. Both retain
`failure_stage: image_lookup`. Configuration-copy, copy-verification or dump
directory preparation errors return `preparation_failed` (or `interrupted`)
with `failure_stage: configuration_copy`, without launching a container. Partial
configuration copies remain with the failed receipt for inspection.

Interruptions during container cleanup or output verification are also recorded
with their respective failure stage. Interrupted cleanup explicitly leaves
removal unconfirmed; no successful cleanup is claimed. `mapping-inspect-glim`
shows the saved failure stage/error alongside current file integrity.

Run receipts use temporary-file replacement so a failed update does not truncate
the previous JSON. If the filesystem cannot save the final receipt, the command
raises an error and the last saved state can remain nonterminal; hard process
kills and machine failure likewise cannot guarantee a final failure record.
This is not crash recovery or a power-loss durability guarantee.


Input inventories accept regular files and directories only; symlinks, FIFOs,
sockets and device nodes are rejected before copying configuration or contacting
Docker. The 10,000-entry limit includes directories and stops traversal as soon
as exceeded. Directory access errors propagate instead of silently excluding
unreadable inputs. These checks do not provide protection against concurrent
filesystem replacement; keep input directories unchanged during execution.
