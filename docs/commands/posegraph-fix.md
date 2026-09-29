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

No odometry yet? `ca slam-run` makes one from the scans with KISS-ICP.

| Option | Default | |
|---|---|---|
| `--out DIR` | | Write `<poses>_fixed.g2o`, `<poses>_fixed_kitti.txt` (and `.tum` for a TUM input) and `<poses>_map.ply` (double coordinates, `intensity`, and `correction`: how far each point moved) |
| `--gravity PATH` | | IMU up directions: a KITTI OXTS folder (roll and pitch per frame, `calib_imu_to_velo.txt`) or a file of `frame ux uy uz` lines |
| `--remove-dynamic` | off | Leave points other scans saw through out of the map, and write them to `<poses>_dynamic.ply` |
| `--no-loops` | off | Skip the loop search (e.g. gravity only) |
| `--voxel` | 0.4 | Thin each scan to one point per voxel (m) for registration and the map |
| `--map-voxel` | 0.2 | Thin the written map (m) |
| `--loop-radius` | 10 | Loop candidates at most this far apart (m), plus the drift allowance |
| `--drift` | 3 | Odometry drift allowed, in % of the path between two nodes |
| `--min-overlap` | 50 | Keep a loop when this % of the later scan overlaps the earlier |
| `--truth GT` | | Ground-truth poses (KITTI or TUM, same frames): report the ATE before and after |
| `--format-json` / `--output-json FILE` | | The report as JSON |

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

or step by step with `cloudanalyzer_core.PoseGraph` (`from_poses`, `set_scan`, `find_loops`,
`set_gravity`, `optimize`, `detect_dynamic`, `map`, `to_g2o`).
