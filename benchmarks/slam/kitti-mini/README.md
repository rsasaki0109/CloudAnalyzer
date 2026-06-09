# kitti-mini

A locally-prepared SLAM benchmark suite built from the
[KITTI Odometry Benchmark](https://www.cvlibs.net/datasets/kitti/eval_odometry.php)
ground-truth bundle. The actual ground-truth poses and the reference map
are not committed (size + license); users download KITTI on their own
machine, then run a one-shot prep script that turns it into a
`ca benchmark eval`-ready suite under this directory.

> **License note.** The KITTI Odometry Benchmark is distributed under
> [CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/).
> CloudAnalyzer's MIT license applies only to the source code here —
> any prepared suite data inherits the upstream non-commercial,
> share-alike terms.

## Layout (after running the prep script)

```
benchmarks/slam/kitti-mini/
├── README.md                 # this file (committed)
├── suite.yaml.example        # canonical schema reference (committed)
├── .gitignore                # ignores data/ and the generated suite.yaml
├── suite.yaml                # generated; load it with `ca benchmark eval`
└── data/<sequence>/          # generated
    ├── map.pcd               # downsampled reference map
    └── trajectory.tum        # GT trajectory converted from KITTI 12-float format
```

## Step 1: download KITTI Odometry

Pick one sequence (00-10 ship with GT poses; 11-21 are the test set
without GT) and grab:

| File | Source | Notes |
|---|---|---|
| `poses/<seq>.txt` | Odometry ground-truth poses zip | 12-float 3x4 matrix per line |
| Reference map (`.pcd`) | Build it yourself | KITTI does **not** publish a survey-grade GT map; accumulate Velodyne scans into the GT frame, or use a map from a trusted SLAM run |

If you need a reference map and don't have one yet, a common recipe is:

```bash
# Accumulate Velodyne scans into the world frame using GT poses,
# then voxel-downsample to ~0.5 m. Any LiDAR mapping pipeline will do —
# e.g. accumulate point clouds in the global frame given the poses, then
# `pcl_voxel_grid` or Open3D's voxel_down_sample().
```

The script below treats the reference map as an **explicit argument**,
so any source works as long as you can point at one PCD/PLY file.

## Step 2: run the prep script

```bash
python scripts/prepare_kitti_mini.py \
  --kitti-poses    /data/kitti/odometry/poses/00.txt \
  --reference-map  /data/kitti/sequence_00/accumulated_map.pcd \
  --sequence sequence_00 \
  --voxel 0.50 \
  --max-poses 2000
```

What this does:

1. Converts the KITTI 12-float poses to TUM format internally
   (synthesizing timestamps at 10 Hz, the Velodyne rotation rate). The
   intermediate TUM file lives in a tempdir and is dropped after the
   suite is built.
2. Voxel-downsamples the reference map to 50 cm (KITTI outdoor scenes
   are larger and noisier than indoor datasets — coarser voxels keep
   the scoring cost reasonable).
3. Subsamples the GT trajectory evenly to at most 2000 poses (KITTI
   sequences contain 1000-5000 GT poses; 2000 is plenty for ATE/RPE).
4. Writes the processed files to `benchmarks/slam/kitti-mini/data/sequence_00/`.
5. Writes `benchmarks/slam/kitti-mini/suite.yaml` pointing at the new files.

If you already have a TUM-format trajectory (e.g. you ran your own
KITTI→TUM converter), use `--reference-trajectory` instead of
`--kitti-poses` — the script will pass it through without re-converting.

## Step 3: run `ca benchmark eval` against your SLAM output

```bash
ca benchmark eval benchmarks/slam/kitti-mini/suite.yaml \
  --map     outputs/my_slam_map.pcd \
  --trajectory outputs/my_slam_trajectory.tum \
  --sequence sequence_00 \
  --report qa/kitti-mini.html \
  --output-json qa/kitti-mini.summary.json
```

The Phase 2 gate (`min_auc`, `max_chamfer`, `max_ate`, `max_rpe`,
`max_drift`, `min_coverage`) ships as the suite's default; tighten or
relax per-run with `--gate key=value` overrides.

You can chain the result straight into a PR comment (Phase 5 / 10
reusable workflow) or pack it for retention (Phase 8 / 9
`ca bundle pack`/`diff`).

## Recalibrating the gate

The starter gate is intentionally loose: KITTI sequences span
500-2000 m, so absolute ATE / drift tolerances are larger than indoor
datasets. Once you've run a known-good SLAM pipeline against the suite
and seen the AUC / Chamfer / ATE distribution, re-run
`prepare_kitti_mini.py` with `--gate` overrides to bake the new
thresholds into `suite.yaml`:

```bash
python scripts/prepare_kitti_mini.py \
  --kitti-poses ... --reference-map ... \
  --gate min_auc=0.93 --gate max_ate=2.50 --gate max_drift=3.00
```

This will overwrite the existing `suite.yaml` and re-materialize the
data files.

## Leaderboard

To add `kitti-mini` rows to the public SLAM leaderboard after preparing
the suite locally:

```bash
python scripts/prepare_leaderboard_kitti.py \
  --velodyne-dir /path/to/velodyne \
  --kitti-poses /path/to/poses/00.txt

python scripts/build_leaderboard.py --include-optional --output docs/leaderboard
```

The prep script subsamples Velodyne frames into `scans/` (gitignored),
accumulates a coarse GT reference map, and writes `suite.yaml`.
