# `ca mcp`: CloudAnalyzer for AI agents

`ca mcp` serves CloudAnalyzer's tools over the [Model Context Protocol](https://modelcontextprotocol.io)
(stdio), so an AI agent can fix and compare SLAM maps and evaluate point clouds on your machine,
with the same Rust core as the web app and the `ca` commands, and get JSON back.

## Install and register

```bash
pip install "cloudanalyzer[mcp]"
claude mcp add cloudanalyzer -- ca mcp        # Claude Code
```

Other MCP clients start the same command (`ca mcp`, no arguments) as a stdio server.

## Tools

| Tool | What it does |
|---|---|
| `session_layout(folder)` | Look at a SLAM session folder without loading it: poses file, poses and scans, whether they match, path length, IMU gravity folders nearby. Quick; call it first. |
| `slam_odometry(scans, out_dir, max_range?, voxel_size?, max_frames?, deskew?)` | Raw scans without poses: KISS-ICP odometry ([`ca slam-run`](slam-run.md)), writing `trajectory.tum` and `map.ply` (`pip install "cloudanalyzer[slam]"`) |
| `posegraph_fix(folder, out_dir?, poses?, keyframe_spacing?, gravity?, remove_dynamic?, truth?, voxel?, map_voxel?, find_loops?)` | [`ca posegraph-fix`](posegraph-fix.md): loops, IMU gravity, dynamic points, the fixed g2o / poses / map |
| `posegraph_compare(first, second, here, there, out_dir?, gravity_first?, gravity_second?, reach?, min_change?)` | [`ca posegraph-compare`](posegraph-compare.md): two drives joined, M3C2, the changed objects |
| `cloud_info(path)` | A cloud's size, bounds, centroid and density |
| `evaluate_map(candidate, reference, thresholds?)` | Chamfer, Hausdorff, F1 at thresholds, AUC |
| `evaluate_trajectory(estimate, reference, align_rigid?)` | ATE, RPE, drift, coverage of a timestamped trajectory |

Paths are on the machine the server runs on. The pose graph tools read every scan: seconds for a
short drive, a minute or so for a few thousand keyframes (KITTI 07, 1,101 scans: about 50 s), so
give the client a generous tool timeout. The maps they write open in the
[web app](https://rsasaki0109.github.io/CloudAnalyzer/app/) for a person to look at.

## From raw scans

With only the scans, an agent chains two tools: `slam_odometry` makes the trajectory, and
`posegraph_fix` takes the scans folder with `poses` pointing at it (and `keyframe_spacing` of about
1 m for 10 Hz scans). On KITTI 07's raw Velodyne scans, over MCP:

| Step | Time | Result |
|---|---|---|
| `slam_odometry` | 47 s | 1,106 poses, 697.5 m |
| `posegraph_fix`, every pose, OXTS gravity, dynamic removal | 12 s | 7 loops, ATE 2.147 → 1.150 m |
| `posegraph_fix`, a keyframe every metre (506) | 6 s | 7 loops, ATE 2.132 → 1.205 m |

## Example conversation

> Here is last night's drive in `runs/0929/`. Close the loops, use the IMU, take the traffic out, and
> tell me how far the end moved.

The agent calls `session_layout("runs/0929")` (1,101 poses, an `oxts` folder next to it), then
`posegraph_fix("runs/0929", "runs/0929/fixed", gravity="runs/oxts", remove_dynamic=True)`, and reads
the loops, the dynamic share and the files written from the report.
