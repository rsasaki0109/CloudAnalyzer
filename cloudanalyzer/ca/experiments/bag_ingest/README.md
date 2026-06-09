# ROS bag / MCAP ingest experiments

Compare how CloudAnalyzer should read robotics recordings before wiring
`ca info`, `ca traj-evaluate`, and `ca slam-run`.

## Strategies

| Module | Idea |
|---|---|
| `extract_all.py` | Read connection metadata and deserialize the first message on each topic to validate type coverage |
| `stream_decode.py` | Read connection metadata only (cheaper; no payload decode) |

## Status

Phase 1: `ca info run.mcap` lists topics/types/counts.
Phase 2: `ca traj-evaluate run.mcap ref.tum --topic /odom` reads Odometry / PoseStamped trajectories.
Phase 3: `ca slam-run run.mcap out --pointcloud-topic /points` extracts PointCloud2 scans.

Install optional dependencies:

```bash
pip install "cloudanalyzer[ros]"
```
