# newer-college-mini

A locally-prepared SLAM benchmark suite built from the
[Newer College Dataset](https://ori-drs.github.io/newer-college-dataset/)
ground-truth bundle. The actual reference map and ground-truth trajectory
are not committed (size + license); users download Newer College on
their own machine, then run a one-shot prep script that turns it into a
`ca benchmark eval`-ready suite under this directory.

> **License note.** The Newer College Dataset is distributed under
> [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
> CloudAnalyzer's MIT license applies only to the source code here —
> any prepared suite data inherits the upstream non-commercial,
> share-alike terms.

## Layout (after running the prep script)

```
benchmarks/slam/newer-college-mini/
├── README.md                 # this file (committed)
├── suite.yaml.example        # canonical schema reference (committed)
├── .gitignore                # ignores data/ and the generated suite.yaml
├── suite.yaml                # generated; load it with `ca benchmark eval`
└── data/<sequence>/          # generated
    ├── map.pcd               # downsampled reference map
    └── trajectory.tum        # subsampled GT trajectory
```

## Step 1: download Newer College

Pick one published sequence and grab the ground-truth bundle. The
dataset's website provides downloads for each release; you only need
the reference map and ground-truth trajectory, not the raw rosbags. For
example:

| Sequence            | What you need to grab from the dataset |
|---|---|
| `short_experiment`  | survey-grade map (`.pcd`) + GT poses (`.tum`) |
| `01_easy`           | same |
| `02_medium`         | same |

The exact file names and formats vary across the dataset's releases.
The prep script below treats the map / trajectory paths as **explicit
arguments**, so any layout works as long as you can point at the two
files.

## Step 2: run the prep script

```bash
python scripts/prepare_newer_college_mini.py \
  --reference-map  /data/newer-college/short_experiment/gt_map.pcd \
  --reference-trajectory /data/newer-college/short_experiment/gt_poses.tum \
  --sequence short_experiment \
  --voxel 0.10 \
  --max-poses 2000
```

What this does:

1. Voxel-downsamples the GT map to 10 cm (the published map is
   typically tens of millions of points — far more than `ca benchmark`
   needs to score a SLAM run).
2. Subsamples the GT trajectory evenly to at most 2000 poses (Newer
   College ground truth is recorded at survey-grade density).
3. Writes the processed files to `benchmarks/slam/newer-college-mini/data/short_experiment/`.
4. Writes `benchmarks/slam/newer-college-mini/suite.yaml` pointing at
   the new files.

The prep script is a thin wrapper around `ca benchmark init`, so its
exit code is non-zero if anything failed (missing input, malformed TUM,
empty map, etc.).

## Step 3: run `ca benchmark eval` against your SLAM output

Once `suite.yaml` exists, plug your SLAM pipeline's map + trajectory in:

```bash
ca benchmark eval benchmarks/slam/newer-college-mini/suite.yaml \
  --map     outputs/my_slam_map.pcd \
  --trajectory outputs/my_slam_trajectory.tum \
  --sequence short_experiment \
  --report qa/newer-college-mini.html \
  --output-json qa/newer-college-mini.summary.json
```

The Phase 2 gate (`min_auc`, `max_chamfer`, `max_ate`, `max_rpe`,
`max_drift`, `min_coverage`) ships as the suite's default; tighten or
relax per-run with `--gate key=value` overrides.

You can chain the result straight into a PR comment (Phase 5 / 10
reusable workflow) or pack it for retention (Phase 8 / 9
`ca bundle pack`/`diff`).

## Leaderboard

After preparing the suite, symlink or copy scan frames into `scans/`
(gitignored) and rebuild the public leaderboard:

```bash
python scripts/prepare_leaderboard_newer_college.py \
  --scans-dir /path/to/scans \
  --reference-map /path/to/gt_map.pcd \
  --reference-trajectory /path/to/gt_poses.tum

python scripts/build_leaderboard.py --include-optional --output docs/leaderboard
```

## Recalibrating the gate

The starter gate is intentionally loose. Once you've run a known-good
SLAM pipeline against the suite and seen the AUC / Chamfer / ATE
distribution, re-run `prepare_newer_college_mini.py` with `--gate`
overrides to bake the new thresholds into `suite.yaml`:

```bash
python scripts/prepare_newer_college_mini.py \
  --reference-map ... --reference-trajectory ... \
  --gate min_auc=0.97 --gate max_ate=0.30 --gate max_drift=0.80
```

This will overwrite the existing `suite.yaml` and re-materialize the
data files.
