# ROS bag / MCAP input

CloudAnalyzer can read robotics recordings directly when the optional
``[ros]`` extra is installed:

```bash
pip install "cloudanalyzer[ros]"
```

Supported file extensions: ``.bag``, ``.mcap``, ``.db3``.

Without the extra, bag paths fail with an install hint instead of a raw
``ImportError``.

## Inspect a recording

```bash
ca info run.mcap
ca info run.mcap --format-json
```

Lists every topic, message type, and message count.

## Trajectory QA from a bag

Compare an estimated odometry / pose / TF stream against a reference TUM file:

```bash
# nav_msgs/Odometry or geometry_msgs/PoseStamped
ca traj-evaluate run.mcap reference.tum --topic /odom

# tf2_msgs/TFMessage — pass the child frame id to track
ca traj-evaluate run.mcap reference.tum --topic /tf --frame base_link
```

When the bag contains exactly one supported trajectory topic, ``--topic``
can be omitted.

## SLAM from a bag

Extract ``sensor_msgs/PointCloud2`` scans and drive a SLAM pipeline:

```bash
pip install "cloudanalyzer[ros,slam]"

ca slam-run run.mcap /tmp/slam-out \
  --pointcloud-topic /points \
  --driver kiss-icp \
  --max-range 80
```

Each PointCloud2 message becomes ``/tmp/slam-out/.ingested_frames/frame_XXXXXX.pcd``.
Message header timestamps feed the TUM trajectory output.

## End-to-end regression gate

Put the pieces together with a config-driven gate:

```bash
# 1. Inspect topics
ca info run.mcap

# 2. Run SLAM from the bag
ca slam-run run.mcap qa/run --pointcloud-topic /points --driver kiss-icp

# 3. Score map + trajectory against references
ca run-evaluate qa/run/map.ply ref/map.pcd \
  qa/run/trajectory.tum ref/trajectory.tum \
  --output-json qa/summary.json

# 4. Or use cloudanalyzer.yaml + ca check for CI
ca check cloudanalyzer.yaml
```

For CI, pair ``ca check`` with the
[CloudAnalyzer GitHub Action](https://github.com/rsasaki0109/cloudanalyzer-action)
or the reusable workflows documented in [docs/ci.md](../ci.md).

## Experiments vs core

The adopted implementation lives in ``ca/core/bag_ingest.py``. Alternative
decode strategies remain under ``ca/experiments/bag_ingest/`` for benchmarking
only; CLI and library callers should import from ``ca.core``.
