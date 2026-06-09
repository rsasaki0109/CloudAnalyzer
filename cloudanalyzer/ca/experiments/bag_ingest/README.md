# ROS bag / MCAP ingest experiments

Compare how CloudAnalyzer should read robotics recordings before wiring
`ca info`, `ca traj-evaluate`, and `ca slam-run`.

## Adopted core

The stable contract is **`ca/core/bag_ingest.py`**:

- `inspect_bag` — topic metadata for `ca info`
- `load_trajectory_from_bag` — Odometry / PoseStamped / TFMessage for `ca traj-evaluate`
- `materialize_pointcloud_bag` — PointCloud2 → PCD frames for `ca slam-run`

## Strategies (experiments only)

| Module | Idea |
|---|---|
| `extract_all.py` | Read connection metadata and deserialize the first message on each topic |
| `stream_decode.py` | Read connection metadata only (cheaper; no payload decode) |

Install optional dependencies:

```bash
pip install "cloudanalyzer[ros]"
```

Full usage tutorial: **[docs/commands/bag-ingest.md](../../../docs/commands/bag-ingest.md)**.
