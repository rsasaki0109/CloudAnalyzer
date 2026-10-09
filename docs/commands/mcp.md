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
| `start_mapping_run(source, out_dir, layout_hypothesis, max_attempts?, minimum_retained_fraction?, keyframe_spacing?, remove_dynamic?, pointcloud_topic?, imu_topic?)` | Start the calling agent's raw-log-to-both-maps loop with one fixed unverified layout; return observations and decision guidance ([workflow](mapping-run.md)) |
| `continue_mapping_run(finished_job_dir, out_dir, max_attempts, reason)` | Start a new bounded repair session from the exact delivered point/HD pair; retain previous repairs, fixed assumptions, immutable evidence and cumulative attempt history ([workflow](mapping-run.md#continue-from-a-delivered-map)) |
| `inspect_mapping_run(job_dir, offset?)` | Resume saved agent-run state, paged candidate observations, attempts and output paths without processing |
| `apply_supported_hd_plan(job_dir, plan_files, interval_ids, connect_endpoints, reason, expected_revision)` | Execute explicitly chosen, source-supported intervals through child drafting, exact endpoint inspection, four audits, retention checks and comparison; retain the root for a separate adoption decision ([workflow](mapping-run.md#apply-an-explicit-supported-plan)) |
| `advance_mapping_run(job_dir, action, reason, expected_revision)` | Inspect/refine/draft/connect, inspect gaps and retry point fusion, or inspect_patch/patch_gaps to add selected missing intervals while preserving existing lanes and connections; compare actual audited HD intervals/routes and explicitly deliver a retained draft |
| `start_mapping_job(source, out_dir, keyframe_spacing?, remove_dynamic?, max_attempts?, pointcloud_topic?, imu_topic?, minimum_retained_fraction?)` | Start an agent-controlled raw-recording job, generating the point-cloud map and corrected trajectory with hashes, processing reports and an explicit retained-extent goal ([workflow](mapping-job.md)) |
| `inspect_mapping_job(job_dir)` | Read persisted artifacts, attempts, source holds and remaining budget without loading clouds |
| `propose_mapping_corridors(job_dir, search_radius_m?)` | Generate cached low-surface corridor geometry and width/edge evidence before assigning lanes; no HD attempt is spent |
| `inspect_mapping_corridors(job_dir, candidate_id?, offset?)` | Read a paged candidate index or bounded original-frame geometry; verify saved input/report hashes without native processing |
| `generate_mapping_geometry(job_dir, decisions, reason)` | Adopt explicitly chosen source intervals as editable IR reference curves; record full-extent holds, infer no lane semantics and consume one shared HD attempt |
| `inspect_mapping_geometry(job_dir, candidate_id, offset?)` | Read saved geometry decisions and bounded curves without native processing; incomplete width/lane semantics and uncovered extent remain visible |
| `generate_mapping_corridor_lanes(job_dir, geometry_candidate_id, lane_specs, boundary_policy, reason)` | Export explicit driving-lane hypotheses within adopted source geometry, preserving outer curves and unresolved intervals; reject inadequate minimum widths, verify OSM reload and save both ground audits |
| `diagnose_mapping_candidate(job_dir, candidate_id)` | Explain saved per-lane/trace height mismatches, insufficient returns, endpoint holds and extent; verify artifacts without processing or spending attempts |
| `generate_mapping_candidate(job_dir, road_options, reason)` | Generate/audit one HD-map hypothesis against frozen point-map inputs; retain failures and the agent's reason |
| `select_mapping_candidate(job_dir, candidate_id, reason)` | Select an audited draft explicitly, retaining quality holds and unresolved deployment readiness |
| `session_layout(folder)` | Look at a SLAM session folder without loading it: poses file, poses and scans, whether they match, path length, IMU gravity folders nearby. Quick; call it first. |
| `slam_odometry(scans, out_dir, max_range?, voxel_size?, max_frames?, deskew?, pointcloud_topic?, imu_topic?, imu_to_lidar?)` | Raw scans without poses, as a folder or a ROS bag (`.bag`, `.mcap`, `.db3`, a rosbag2 folder): the Rust core's LiDAR odometry, writing `trajectory.tum` and `map.ply`; from a bag also the scans and, with an IMU topic, each scan's up direction (`pip install "cloudanalyzer[fast]"`) |
| `posegraph_fix(folder, out_dir?, poses?, keyframe_spacing?, gravity?, remove_dynamic?, truth?, voxel?, map_voxel?, find_loops?)` | [`ca posegraph-fix`](posegraph-fix.md): loops, IMU gravity, dynamic points, the fixed g2o / poses / map |
| `posegraph_compare(first, second, here, there, out_dir?, gravity_first?, gravity_second?, reach?, min_change?)` | [`ca posegraph-compare`](posegraph-compare.md): two drives joined, M3C2, the changed objects |
| `build_vector_map(cloud, trajectory, out_dir, forward_lanes?, backward_lanes?, left_hand_traffic?, lane_width?, speed_limit?, segment_length?, anchor_width_prior?, reference_map?, projection?, origin_lat?, origin_lon?)` | Draft Autoware Lanelet2 roads, coordinate metadata and an evidence/validation report in a new directory ([details](vectormap-build.md)) |
| `connect_vector_map_junctions(cloud, vector_map, out_dir, max_gap?, min_ground_support?, lane_pairs?, preview_only?)` | Preview or add ground-supported branching junction drafts, retaining existing IR geometry, rules and coordinates ([details](vectormap-connect.md)) |
| `measure_vector_map_signal(cloud, vector_map, out_dir, bounds, lanes, kind?, preview_only?)` | Measure a user-identified signal head from a 3D box and explicitly selected lanes; preview defaults to true ([details](vectormap-signal.md)) |
| `measure_vector_map_crosswalk(cloud, vector_map, out_dir, bounds, lanes?, candidate?, brightness_fraction?, preview_only?)` | Propose measured ground-paint bands, then add a user-confirmed crossing and lane assignment; preview defaults to true ([details](vectormap-crosswalk.md)) |
| `propose_vector_map_relations(vector_map, rule_id, candidate_key?, map_snapshot?, out_dir?)` | Read-only geometric target evidence, or explicit adoption of one current-map candidate ([details](vectormap-suggest.md)) |
| `view_link(paths)` | A link that opens results (a folder or files) in the web app for the person, served from this machine while the server runs ([`ca web-view`](web-view.md)) |
| `cloud_info(path)` | A cloud's size, bounds, centroid and density |
| `evaluate_map(candidate, reference, thresholds?)` | Chamfer, Hausdorff, F1 at thresholds, AUC |
| `evaluate_trajectory(estimate, reference, align_rigid?)` | ATE, RPE, drift, coverage of a timestamped trajectory |

Paths are on the machine the server runs on. The pose graph tools read every scan: seconds for a
short drive, a minute or so for a few thousand keyframes (KITTI 07, 1,101 scans: about 50 s), so
give the client a generous tool timeout. `view_link(out_dir)` then gives the person one link that
opens the maps and trajectories in the [web app](https://rsasaki0109.github.io/CloudAnalyzer/app/).

## Draft a vector map

For a surveyed cloud and a trajectory already in the same metre frame, call
`build_vector_map`. The trajectory must follow the outside forward lane. Review its
detected versus inferred boundary counts and validation issues. Use `existing_map` for
already aligned repeated passes, and `connect_vector_map_junctions` to preview junctions
before selecting branches. Ground support cannot establish driving permission or clearance.
`reference_map` copies coordinate metadata only. Use
`view_link([cloud, out_dir])` to overlay the resulting `.osm` on the cloud. For subsequent
editing register the separate server with
`claude mcp add vectormap -- vectormap mcp <out_dir>/lanelet2_map.osm`.

## From raw scans

With only the scans, an agent chains two tools: `slam_odometry` makes the trajectory, and
`posegraph_fix` takes the scans folder with `poses` pointing at it (and `keyframe_spacing` of about
1 m for 10 Hz scans). On KITTI 07's raw Velodyne scans, over MCP:

| Step | Time | Result |
|---|---|---|
| `slam_odometry` | 47 s | 1,106 poses, 697.5 m |
| `posegraph_fix`, every pose, OXTS gravity, dynamic removal | 12 s | 7 loops, ATE 2.147 → 1.150 m |
| `posegraph_fix`, a keyframe every metre (506) | 6 s | 7 loops, ATE 2.132 → 1.205 m |

## From a ROS bag

Given a bag, `slam_odometry` writes its `sensor_msgs/PointCloud2` scans to `out_dir/scans` (KITTI
`.bin`, with their intensity), and when
the bag has a `sensor_msgs/Imu` topic, each scan's up direction to `out_dir/gravity/gravity.txt`:
from the IMU's orientation, or from its mean acceleration over half a second when it gives none.
`imu_to_lidar` turns it into the LiDAR frame (identity when the axes agree, as for most built-in
IMUs). Its result names the `scans`, `trajectory` and `gravity` to pass on to `posegraph_fix`.

KITTI 07 recorded as a ROS1 bag (the Velodyne scans on `/velodyne_points`, the OXTS orientation on
`/imu/data`, 2.1 GB), over MCP: `slam_odometry` 82 s, then `posegraph_fix` with the bag's IMU
gravity and dynamic removal 13 s, 7 loops, ATE 2.147 → 1.150 m, the same as from the scan folder.

## Example conversation

> Here is last night's drive in `runs/0929/`. Close the loops, use the IMU, take the traffic out, and
> tell me how far the end moved.

The agent calls `session_layout("runs/0929")` (1,101 poses, an `oxts` folder next to it), then
`posegraph_fix("runs/0929", "runs/0929/fixed", gravity="runs/oxts", remove_dynamic=True)`, and reads
the loops, the dynamic share and the files written from the report.

## Measured signal heads

`measure_vector_map_signal` accepts a cloud, editable map, new `out_dir`, six
original-coordinate `bounds`, and user-confirmed `lanes`. Preview defaults to true;
after reviewing the measured housing, add with `preview_only=false` in another new
directory. See [signal measurement](vectormap-signal.md) for limits. This measures
an identified object's shape; signal classification, lamps and control relationships
are not inferred from the point cloud.
