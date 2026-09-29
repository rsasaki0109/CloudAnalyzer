# `ca posegraph-compare`

Join two drives through the same places and list what changed between them: the web app's
**Join another graph** and **Compare parts (M3C2)**, from the command line, with a JSON report.

It runs the same Rust core as the browser, natively on all cores, so its results match the app's
(see also [`ca posegraph-fix`](posegraph-fix.md) for one drive).

## Usage

```bash
ca posegraph-compare <first session> <second session> --here N --there M [--out DIR] \
  [--gravity-first PATH] [--gravity-second PATH] [--format-json]
```

Each session folder holds a poses file and one scan per pose, as for `ca posegraph-fix`. `--here` and
`--there` name one node of each drive (vertex id or frame number) standing at the same place, a
few metres apart at most: their scans are registered with a yaw search, which places the whole
second drive. Then

1. loops are found in the first drive, and again across both once joined;
2. with `--gravity-first` / `--gravity-second` (KITTI OXTS folders or `frame ux uy uz` files) both
   drives are tied to their IMU's up direction;
3. the keyframes of each drive within `--reach` metres of the other's path are built into two maps
   (`--map-voxel`), and M3C2 measures the change from the first to the second (core points every 0.5 m,
   normal radius 1 m, projection radius 0.5 m, depth 2 m);
4. significant changes of at least `--min-change` metres group into changed objects, largest first.

| Option | Default | |
|---|---|---|
| `--out DIR` | | `joined.g2o`, `first_map.ply`, `second_map.ply`, and `m3c2.ply` (core points with `m3c2_distance`, `lod95`, `significant`, `change_object`) |
| `--no-loops` | off | Only the join |
| `--voxel` | 0.4 | Thin each scan (m) |
| `--map-voxel` | 0.3 | Thin the compared maps (m) |
| `--reach` | 50 | Compare keyframes within this distance of the other drive (m) |
| `--min-change` | 0.3 | Smallest change counted in a changed object (m) |
| `--format-json` / `--output-json FILE` | | The report as JSON: the join, loops, M3C2 statistics and the 50 largest changed objects (centroid, size, mean change) |

## Example: a campus in June and in December

NCLT sessions 2012-06-15 and 2012-12-01 (keyframes 200-2199), prepared with
`scripts/prepare_nclt.py`; summer keyframe 1219 and winter keyframe 850 stand at the same spot:

```bash
ca posegraph-compare nclt/2012-06-15/velodyne nclt/2012-12-01-part/velodyne --here 1219 --there 850 \
  --gravity-first nclt/2012-06-15/gravity --gravity-second nclt/2012-12-01-part/gravity
```

```text
joined at 1219 = 850: overlap 74 %
loops across the drives: 454 of 611 candidates
M3C2 at 1,534,545 core points: 1,223,176 measured, 399,185 significant (26.0 %)
3,688 changed objects; the largest:
  #1  122.9 x 70.0 x 16.7 m  -0.67 m  18,041 points  at [-83.482, 546.319, 1.89]
  ...
```

in about a minute (the tree crowns of June are the largest changes), where the browser takes
about five.

## From Python

```python
from ca.posegraph_fix import compare_sessions

report = compare_sessions("june/", "december/", "out/", here=1219, there=850)
```
