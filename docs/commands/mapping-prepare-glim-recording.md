# mapping-prepare-glim-recording

Combine an explicit PointCloud2 channel and a separately recorded raw Imu channel
into a new ROS2 MCAP. Install `cloudanalyzer[ros]` (rosbags, mcap, zstandard).

```bash
ca mapping-prepare-glim-recording lidar.mcap \
  --imu original-imu.mcap --points-topic /velodyne_points --imu-topic /ms25/raw \
  --out runs/prepared-recording
```

Both sources must be indexed ROS2 MCAPs with complete `ros2msg` definitions and
CDR payloads. A selected topic must correspond to exactly one channel. The same
source may supply both channels. Only the two selected channels are copied;
other channels, file metadata and attachments are excluded.

The output directory must be new. `recording.mcap` preserves selected CDR
payloads, header/log/publish timestamps, sequences, schemas and channel metadata.
Messages are interleaved in log-time order. The command does not transform axes,
resample, replace missing measurements, align clocks, relabel frames or deskew.
It hashes both sources before and after writing and records output identity,
selected topics and message counts in `preparation.json`.

The existing `mapping-check-glim` diagnostic runs on the combined recording.
Exit 0 / `basic_checks_passed` means its basic sensor checks passed, not that
calibration, per-point timing or GLIM settings were verified. The `--max-imu-gap` option (seconds, default 0.05) is forwarded to the
diagnostic and saved in the receipt. Intervals over this limit anywhere in the
selected IMU stream block the output. Exit 1 /
`blocked` retains the recording and diagnostic so problems can be inspected.
Preparation errors retain a failure receipt and may leave an incomplete output;
never use such a run. Initial validation failures create no output. Input files
and existing output directories are not overwritten. The disk precheck reserves
10 GiB plus the combined compressed input sizes; this estimate is not a quota.

For NCLT, original IMU vectors must retain their native axes and appropriate
frame label (`nclt_ms25_native` for the existing pilot). Bundled orientation-demo
IMU messages remain marked unavailable and are rejected by the diagnostic.
See [frame conventions](../slam-nclt-body-frames.md) and
[raw IMU pilot requirements](../slam-nclt-imu-pilot.md). Merely combining two
recordings does not establish that they came from the same session or share a
clock. Verify their acquisition provenance before interpreting the output.

Tests use actual demo point payloads with explicitly synthetic IMU fixtures to
check preservation, coverage failures, unavailable gyro, topic errors and CLI
exit status. No original raw NCLT IMU recording is present in this workspace;
real GLIM estimation and accuracy remain unverified.

If preparation raises an error or is interrupted with Ctrl+C after the initial
receipt was saved, `preparation.json` records `preparation_failed` or `interrupted`
and a `failure_stage`: `recording_write`, `input_verification`,
`output_verification` or `input_preflight`. The exception still propagates;
partial output is retained for inspection and is not approved for execution.

Receipts use atomic temporary-file replacement. A failed final replacement keeps
the previous valid JSON, which may still say `preparing`. Filesystem write
failures, hard termination and power loss cannot guarantee a final failure record.
