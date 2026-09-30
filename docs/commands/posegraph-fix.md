# `ca posegraph-fix`

Fix a SLAM map from the command line: the web app's pose graph panel, without the browser. It finds
loops with ICP, ties the keyframes to IMU gravity, optimises the pose graph, leaves dynamic points
(traffic, pedestrians) out, and writes the fixed graph, poses and map, with a JSON report for
scripts, CI and AI agents.

It runs the same Rust core as the browser, natively on all cores and without the browser's
memory limit, so its results match the app's.

## Install

```bash
pip install 'cloudanalyzer[fast]'   # the Rust core (cloudanalyzer_core)
```

## Usage

```bash
ca posegraph-fix <session folder> [--out DIR] [--gravity OXTS_OR_FILE] [--remove-dynamic] [--truth GT] [--format-json]
```

The session folder holds a poses file and one scan per pose, as the web app's **Open folder…**
takes it:

- **poses**: a g2o graph (`VERTEX_SE3:QUAT`, `EDGE_SE3:QUAT`, `FIX`), or a KITTI (12 numbers a line) or
  TUM (`t x y z qx qy qz qw`) trajectory, preferring file names like `poses`, `traj`, `odom` or `gt`;
- **scans**: PCD, PLY, LAS/LAZ, XYZ or KITTI Velodyne `.bin`, named by frame number (`000042.pcd`), or
  one per pose in name order. A KITTI `calib.txt` in the folder moves them by its `Tr`.

No odometry yet? Give it the raw scans, a folder without a poses file or a ROS bag (`.bag`,
`.mcap`, `.db3`, a rosbag2 folder), with `--out`: the Rust core's LiDAR odometry (the same as the web
app's) places the scans first, into `<out>/odometry`. The core reads the recording itself: `.bag`, `.mcap`, `.db3`
and rosbag2 folders (lz4, zstd and bz2 compression), with no ROS install. A bag's
scans are written there as KITTI `.bin` with
their intensity, and when it has a `sensor_msgs/Imu` topic its up directions become the gravity
(`--imu-to-lidar` turns them into the LiDAR frame). Or pass an existing trajectory with `--poses`.

```bash
ca posegraph-fix drive.mcap --out fixed/ --remove-dynamic --keyframe-spacing 1
```

KITTI 07 recorded as a ROS1 bag (2.1 GB, Velodyne scans and the OXTS orientation as IMU): reading the
bag and the odometry 61 s, then 7 loops, the bag's IMU gravity and dynamic removal, ATE 2.67 → 1.76 m
(SE(3)-aligned 0.78 → 0.39 m), 78 s in all on a laptop.

| Option | Default | |
|---|---|---|
| `--poses FILE` | | The poses file when it is not in the folder (e.g. `trajectory.tum` from `ca slam-run`): one pose per scan, in order or by frame number |
| `--keyframe-spacing` | 0 | Keep one pose every this many metres of a trajectory (0: every pose); nodes keep their frame numbers |
| `--pointcloud-topic`, `--imu-topic` | | A bag's topics, when it has several |
| `--imu-to-lidar` | identity | The IMU-to-LiDAR rotation, 9 numbers row-major |
| `--max-range` | 80 | Odometry: drop points farther than this (m) |
| `--out DIR` | | Write `<poses>_fixed.g2o`, `<poses>_fixed_kitti.txt` (and `.tum` for a TUM input) and `<poses>_map.ply` (double coordinates, `intensity`, and `correction`: how far each point moved) |
| `--gravity PATH` | | IMU up directions: a KITTI OXTS folder (roll and pitch per frame, `calib_imu_to_velo.txt`) or a file of `frame ux uy uz` lines |
| `--no-imu-calibration` | off | Take the up directions as given (see below) |
| `--remove-dynamic` | off | Leave points other scans saw through out of the map, and write them to `<poses>_dynamic.ply` |
| `--no-loops` | off | Skip the loop search (e.g. gravity only) |
| `--voxel` | 0.4 | Thin each scan to one point per voxel (m) for registration and the map |
| `--map-voxel` | 0.2 | Thin the written map (m) |
| `--loop-radius` | 10 | Loop candidates at most this far apart (m), plus the drift allowance |
| `--drift` | 3 | Odometry drift allowed, in % of the path between two nodes |
| `--min-overlap` | 50 | Keep a loop when this % of the later scan overlaps the earlier |
| `--truth GT` | | Ground-truth poses (KITTI or TUM, same frames): report the ATE before and after |
| `--format-json` / `--output-json FILE` | | The report as JSON |

## IMU gravity on a real IMU

An IMU is seldom mounted square with the LiDAR, and its up direction is noisier than KITTI's
survey-grade OXTS. So `--gravity` first estimates the IMU's rotation into the scans' frame from the
drive itself: the rotation that makes every keyframe's measured up, seen in the world through its
pose, agree best (a drive that turns and tilts pins it down, upside-down IMUs included). It is used
only when it clearly helps: over 50 keyframes or more, halving their spread and taking a degree off
it (over a few keyframes, drift that follows the heading can pass for a skewed mount). The tie's
standard deviation is at least the IMU's noise, estimated from keyframes 1 and 2 apart so that
odometry drift does not count, so a noisy IMU levels the map without bending it while the drift
still goes. The report gives the spread, the rotation when used, and the noise. The web app's IMU
gravity does the same (the same Rust code), with a checkbox to turn it off.

| Drive | Up spread | Result |
|---|---|---|
| KITTI 07 (OXTS) | 0.49°, noise 0.05° | no rotation; ATE 1.162 m, as before |
| KITTI 09 (OXTS, hills) | 1.06°, noise 0.04° | no rotation; ATE 2.26 m, as before |
| hdl_graph_slam's `hdl_400.bag` (HDL-32E, a z-down GPS/IMU) | 8.60° → 2.19° with the rotation, noise 2.25° | height range 3.13 → 2.08 m, dynamic points 16.9 → 4.7 % |

## Example: KITTI 07

```bash
ca posegraph-fix kitti/07/velodyne --out fixed/ --gravity kitti/07/oxts --remove-dynamic \
  --truth kitti/07/gt_lidar.txt
```

```text
kiss_poses.txt: 1101 poses, 11,276,315 scan points
loops: 7 of 8 candidates added (0 implausible)
gravity: 1101 keyframes tied
dynamic: 57,137 of 11,276,315 points (0.51 %)
ATE: 2.135 -> 1.162 m (aligned 0.604 -> 0.471 m)
```

in about 50 s on a desktop, the same loops and trajectory as the web app. With `--format-json` the
report lists every loop (`[node, node, overlap]`), the optimisation costs, the timings and the files
written.

## From Python

```python
from ca.posegraph_fix import fix_session

report = fix_session("kitti/07/velodyne", "fixed/", gravity="kitti/07/oxts", remove_dynamic=True)
```

To look at the result: `ca web-view fixed/` opens the map and trajectory in the web app
([`ca web-view`](web-view.md)). Two drives through the same places: [`ca posegraph-compare`](posegraph-compare.md).

Or step by step with `cloudanalyzer_core.PoseGraph` (`from_poses`, `set_scan`, `find_loops`,
`set_gravity`, `optimize`, `detect_dynamic`, `map`, `to_g2o`).
