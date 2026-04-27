# Analysis Commands

## ca info

Show basic metadata for a point cloud file.

```bash
ca info cloud.pcd
```

Output: points, colors, normals, bounding box, extent, centroid.

| Option | Description |
|---|---|
| `--format-json` | Print JSON to stdout |
| `--output-json` | Dump to JSON file |

## ca stats

Detailed statistics including density and point spacing distribution.

```bash
ca stats cloud.pcd
```

Output: points, volume, density (pts/unit³), spacing mean/median/min/max/std.

## ca diff

Quick distance statistics between two point clouds (no registration).

```bash
ca diff a.pcd b.pcd --threshold 0.05
```

| Option | Description |
|---|---|
| `--threshold` | Report how many points exceed this distance |
| `--format-json` | Print JSON to stdout |

## ca map-evaluate

Experimental MapEval-inspired map-to-map evaluation against a reference/GT map.

```bash
# GT-based threshold metrics
ca map-evaluate estimated.pcd reference.pcd --thresholds 0.2,0.1,0.05

# Apply a known initial transform and write colored PLY error maps
ca map-evaluate estimated.pcd reference.pcd \
  --align-mode initial \
  --initial-matrix "1,0,0,-0.1,0,1,0,0,0,0,1,0,0,0,0,1" \
  --artifact-dir qa/map-evaluate \
  --format-json
```

Output: Chamfer distance plus accuracy / completeness / F-score at each configured threshold. These are per-threshold nearest-neighbor metrics. They are intentionally separate from the `ca evaluate` metrics used by `ca batch`, `ca run-evaluate`, and `ca loop-closure-report`, which report an AUC / best-F1 curve over thresholds.

| Option | Default | Description |
|---|---|---|
| `--thresholds` | `0.2,0.1,0.08,0.05,0.01` | Comma-separated distance thresholds in meters for accuracy/completeness/F-score |
| `--align-mode` | `none` | Alignment mode: `none` or `initial` |
| `--initial-matrix` | `None` | 4x4 row-major transform applied when `--align-mode initial` is used |
| `--artifact-dir` | `None` | Optional directory for colored PLY error-map artifacts |
| `--format-json` | `false` | Print JSON to stdout |
| `--output-json` | `None` | Dump result as JSON |

## ca posegraph-validate

Validate common manual-loop-closure session artifacts without optimizing the graph.

```bash
ca posegraph-validate session/pose_graph.g2o \
  --tum session/optimized_poses_tum.txt \
  --key-point-frame session/key_point_frame \
  --format-json
```

Output: g2o vertex/edge/connectivity counts, optional TUM trajectory summary, optional keyframe count, and `summary.ok` with errors/warnings. This command is diagnostic: it exits nonzero for missing/malformed inputs that cannot be read, but a parsed session with `summary.ok: false` is reported in the payload instead of treated as a CLI failure. Use `ca loop-closure-report --require-posegraph-ok` when posegraph validity should participate in an automated quality gate.

| Option | Default | Description |
|---|---|---|
| `--tum` | `None` | Optional optimized poses in TUM format |
| `--key-point-frame` | `None` | Optional directory containing keyframe point clouds |
| `--format-json` | `false` | Print JSON to stdout |
| `--output-json` | `None` | Dump result as JSON |

## ca loop-closure-report

Compare before/after manual loop-closure outputs against one reference map, with optional trajectory and posegraph checks.

```bash
# Map-only before/after report
ca loop-closure-report before/map.pcd after/map.pcd reference/map.pcd

# Automated gate over map, trajectory, and validated posegraph sessions
ca loop-closure-report before/map.pcd after/map.pcd reference/map.pcd \
  --before-traj before/optimized_poses_tum.txt \
  --after-traj after/optimized_poses_tum.txt \
  --ref-traj reference/trajectory.tum \
  --before-session-root before \
  --after-session-root after \
  --min-auc-gain 0.01 \
  --max-after-chamfer 0.03 \
  --min-ate-gain 0.05 \
  --max-after-ate 0.5 \
  --require-posegraph-ok \
  --format-json
```

Output: before/after/delta map metrics from `ca.evaluate` (`AUC`, best-F1 curve, Chamfer, Hausdorff), optional trajectory before/after/delta metrics, optional posegraph session validation, and optional discovery metadata for `--before-session-root` / `--after-session-root`.

Exit code policy: the command exits with code `1` only when a configured quality gate fails. Map and trajectory gates are enabled by their threshold options. Posegraph validation is included in the report by default but affects the quality gate only when `--require-posegraph-ok` is set.

Metric note: `loop-closure-report` uses the existing `ca.evaluate` AUC/F1-curve metric set so before/after deltas line up with `ca batch` and `ca run-evaluate`. Use `ca map-evaluate` when you specifically need MapEval-style accuracy/completeness/F-score at fixed thresholds.

| Option | Default | Description |
|---|---|---|
| `--thresholds` | default evaluate thresholds | Comma-separated thresholds for the AUC/F1 curve |
| `--min-auc-gain` | `None` | Fail if `AUC(after)-AUC(before)` is below this value |
| `--max-after-chamfer` | `None` | Fail if after-map Chamfer exceeds this value |
| `--before-traj` | `None` | Optional trajectory before loop closure |
| `--after-traj` | `None` | Optional trajectory after loop closure |
| `--ref-traj` | `None` | Optional reference trajectory |
| `--traj-max-time-delta` | `0.05` | Max timestamp gap for trajectory matching/interpolation |
| `--traj-align-origin` | `false` | Align trajectory origins before scoring |
| `--traj-align-rigid` | `false` | Rigidly align trajectories to reference before scoring |
| `--min-ate-gain` | `None` | Fail if trajectory ATE RMSE improvement is below this value |
| `--max-after-ate` | `None` | Fail if after-trajectory ATE RMSE exceeds this value |
| `--before-g2o`, `--after-g2o` | `None` | Optional pose graph files |
| `--before-tum`, `--after-tum` | `None` | Optional optimized poses for posegraph session validation |
| `--before-key-point-frame`, `--after-key-point-frame` | `None` | Optional keyframe directories for posegraph session validation |
| `--before-session-root`, `--after-session-root` | `None` | Auto-discover map, g2o, TUM, and keyframe paths under session roots |
| `--session-map-name` | `map.pcd` | Map filename used during session-root discovery |
| `--require-posegraph-ok` | `false` | Fail the quality gate if any validated posegraph session has `summary.ok: false` |
| `--format-json` | `false` | Print JSON to stdout |
| `--output-json` | `None` | Dump report as JSON |

## ca batch

Run info on all point cloud files in a directory.

```bash
# Current directory
ca batch /path/to/pcds/

# Recursive scan
ca batch /path/to/pcds/ -r

# JSON output
ca batch /path/to/pcds/ --format-json | jq '.[].num_points'

# Evaluate every file against one reference
ca batch /path/to/results/ --evaluate reference.pcd

# Custom thresholds for batch evaluation
ca batch /path/to/results/ --evaluate reference.pcd -t 0.05,0.1,0.2

# Markdown / HTML report
ca batch /path/to/results/ --evaluate reference.pcd --report batch_report.md
ca batch /path/to/results/ --evaluate reference.pcd --report batch_report.html
# reports include inspection commands; HTML adds Copy buttons plus count-badged summary rows, quick actions, failed-first / recommended-first sort presets, and pass/failed/pareto/recommended controls

# Quality gate (exit code 1 if any file fails)
ca batch /path/to/results/ --evaluate reference.pcd --min-auc 0.95 --max-chamfer 0.02

# Compression benchmark workflow
ca batch decoded/ --evaluate reference.pcd \
  --compressed-dir compressed/ --baseline-dir original/
# report adds a quality-vs-size plot, Pareto candidates, a recommended point, clickable summary rows, quick actions, failed-first / recommended-first sort presets, and HTML filters
```

| Option | Default | Description |
|---|---|---|
| `-r`, `--recursive` | `false` | Scan subdirectories |
| `--evaluate` | `None` | Evaluate each file against the given reference point cloud |
| `--report` | `None` | Write batch evaluation report (`.md`, `.markdown`, `.html`) |
| `--min-auc` | `None` | Minimum AUC required to pass; exits with code 1 if any file fails |
| `--max-chamfer` | `None` | Maximum Chamfer distance allowed; exits with code 1 if any file fails |
| `--compressed-dir` | `None` | Directory containing compressed artifacts; matched by relative path or stem |
| `--baseline-dir` | `None` | Directory containing original uncompressed files for size ratio baseline |
| `-t`, `--thresholds` | default evaluate thresholds | Comma-separated distance thresholds for `--evaluate` |

## ca check

Run config-driven QA across artifact, detection, tracking, trajectory, ground, and integrated run checks.

```bash
# default config path
ca check

# explicit config path
ca check cloudanalyzer.yaml

# JSON summary for CI
ca check cloudanalyzer.yaml --format-json --output-json qa/summary.json
```

Minimal config:

```yaml
version: 1
defaults:
  report_dir: qa/reports
  json_dir: qa/results
checks:
  - id: mapping-postprocess
    kind: artifact
    source: outputs/map.pcd
    reference: baselines/map_ref.pcd
    gate:
      min_auc: 0.95
      max_chamfer: 0.02
  - id: localization-run
    kind: trajectory
    estimated: outputs/traj.csv
    reference: baselines/traj_ref.csv
    alignment: rigid
    gate:
      max_ate: 0.5
      max_rpe: 0.2
      max_drift: 1.0
      min_coverage: 0.9
```

`kind` supports `artifact`, `artifact_batch`, `trajectory`, `trajectory_batch`, `run`, and `run_batch`. `map` and `map_batch` are supported as aliases when you want mapping-oriented wording. Relative paths resolve from the config file location.
When one or more gated checks fail, `ca check` also prints a severity-first triage order. The JSON summary stores the same ranking under `summary.triage`.

| Option | Default | Description |
|---|---|---|
| `CONFIG` | `cloudanalyzer.yaml` | YAML/JSON config file for unified QA |
| `--format-json` | `false` | Print aggregated check results to stdout as JSON |
| `--output-json` | `None` | Dump aggregated check summary as JSON |

## ca init-check

Write a starter `cloudanalyzer.yaml` so a repo can adopt config-driven QA without hand-writing the contract first.

```bash
# integrated mapping / localization / perception starter
ca init-check

# mapping-only starter
ca init-check configs/mapping.cloudanalyzer.yaml --profile mapping

# overwrite an existing template
ca init-check --profile perception --force
```

`--profile` supports `mapping`, `localization`, `perception`, and `integrated`. The generated config is immediately runnable with `ca check ...`.

- `perception`: geometry artifact QA plus starter slices for 3D detection and 3D tracking JSON outputs
- `integrated`: mapping + localization + perception slices, plus one combined run gate

| Option | Default | Description |
|---|---|---|
| `OUTPUT` | `cloudanalyzer.yaml` | Destination path for the starter config |
| `--profile` | `integrated` | Template slice to scaffold |
| `--force` | `false` | Overwrite an existing file |

## ca baseline-decision

Decide whether a candidate `ca check` summary should promote, keep, or reject a baseline revision.

```bash
# compare a current candidate against prior baseline summaries
ca baseline-decision qa/current-summary.json \
  --history qa/baseline-2026-03-20.json \
  --history qa/baseline-2026-03-27.json

# machine-readable output
ca baseline-decision qa/current-summary.json \
  --history qa/baseline-summary.json \
  --format-json --output-json qa/baseline-decision.json
```

Input files are the JSON summaries emitted by `ca check --output-json ...`. The current stable strategy is `stability_window`: failed candidates are rejected immediately, strong candidates without enough history stay `keep`, and only stable improving windows become `promote`.

| Option | Default | Description |
|---|---|---|
| `CANDIDATE_JSON` | required | Candidate `ca check` summary JSON |
| `--history` | `[]` | Historical summary JSON files, oldest to newest |
| `--format-json` | `false` | Print the decision summary to stdout as JSON |
| `--output-json` | `None` | Dump the decision summary as JSON |

## ca detection-evaluate

Evaluate 3D object detections against reference annotations using class-aware 3D box matching. Supports both axis-aligned and oriented (yaw-rotated) boxes.

```bash
# basic evaluation
ca detection-evaluate est_det.json gt_det.json

# sweep IoU thresholds and enforce mAP / recall gates
ca detection-evaluate est_det.json gt_det.json \
  --iou-thresholds 0.25,0.5 --min-map 0.9 --min-recall 0.8 \
  --report qa/detection_report.html

# JSON output
ca detection-evaluate est_det.json gt_det.json \
  --format-json --output-json qa/detection_result.json
```

Input format:

```json
{
  "frames": [
    {
      "frame_id": "000001",
      "boxes": [
        {
          "label": "car",
          "center": [0.0, 0.0, 0.0],
          "size": [4.0, 1.8, 1.6],
          "yaw": 0.0,
          "score": 0.97
        }
      ]
    }
  ]
}
```

The stable contract keeps only the fields needed for evaluation: `label`, `center`, `size`, optional `yaw` (radians, Z-axis rotation, default 0.0), and optional `score`. When any box has a non-zero `yaw`, oriented 3D box IoU (BEV polygon intersection) is used automatically; otherwise axis-aligned IoU is used. The field alias `rotation_y` is also accepted for KITTI compatibility. The `kind: detection` check type is also available in `cloudanalyzer.yaml`.

Checked-in public example files are available under `demo_assets/public/rellis3d-frame-000001/object_eval/`.

| Option | Default | Description |
|---|---|---|
| `ESTIMATED` | required | Estimated detection sequence JSON |
| `REFERENCE` | required | Reference detection sequence JSON |
| `--iou-thresholds` | `0.25,0.5` | Comma-separated IoU thresholds used for AP / mAP |
| `--primary-iou-threshold` | `0.5` when available | Threshold used for precision / recall / F1 gate evaluation |
| `--min-map` | `None` | Minimum mean AP gate |
| `--min-precision` | `None` | Minimum precision gate at the primary IoU threshold |
| `--min-recall` | `None` | Minimum recall gate at the primary IoU threshold |
| `--min-f1` | `None` | Minimum F1 gate at the primary IoU threshold |
| `--report` | `None` | Write Markdown/HTML detection report |
| `--format-json` | `false` | Print JSON to stdout |
| `--output-json` | `None` | Dump result as JSON |

## ca ground-evaluate

Evaluate ground segmentation quality by comparing estimated ground/non-ground point clouds against reference labels.

```bash
# basic evaluation
ca ground-evaluate est_ground.pcd est_nonground.pcd ref_ground.pcd ref_nonground.pcd

# with quality gate
ca ground-evaluate est_ground.pcd est_nonground.pcd ref_ground.pcd ref_nonground.pcd \
  --min-f1 0.9 --min-iou 0.8 --voxel-size 0.5

# JSON output
ca ground-evaluate est_ground.pcd est_nonground.pcd ref_ground.pcd ref_nonground.pcd \
  --format-json --output-json qa/ground_result.json

# HTML / Markdown report
ca ground-evaluate est_ground.pcd est_nonground.pcd ref_ground.pcd ref_nonground.pcd \
  --min-f1 0.9 --min-iou 0.8 --report qa/ground_report.html
```

Both estimation and reference are provided as separate ground and non-ground point cloud files. Evaluation uses voxel-based confusion matrix comparison. The `kind: ground` check type is also available in `cloudanalyzer.yaml`, and ground checks can emit HTML/Markdown reports the same way as other QA checks.

| Option | Default | Description |
|---|---|---|
| `ESTIMATED_GROUND` | required | Estimated ground points (pcd/ply/las) |
| `ESTIMATED_NONGROUND` | required | Estimated non-ground points |
| `REFERENCE_GROUND` | required | Reference ground points |
| `REFERENCE_NONGROUND` | required | Reference non-ground points |
| `--voxel-size` | `0.2` | Voxel grid resolution (meters) |
| `--min-precision` | `None` | Minimum precision gate |
| `--min-recall` | `None` | Minimum recall gate |
| `--min-f1` | `None` | Minimum F1 gate |
| `--min-iou` | `None` | Minimum IoU gate |
| `--report` | `None` | Write Markdown/HTML ground segmentation report |
| `--format-json` | `false` | Print JSON to stdout |
| `--output-json` | `None` | Dump result as JSON |

## ca tracking-evaluate

Evaluate 3D multi-object tracking against reference tracks using class-aware frame-wise box matching plus ID-switch accounting.

```bash
# basic evaluation
ca tracking-evaluate est_track.json gt_track.json

# enforce recall / MOTA / ID-switch gates
ca tracking-evaluate est_track.json gt_track.json \
  --iou-threshold 0.5 --min-mota 0.8 --min-recall 0.9 --max-id-switches 2 \
  --report qa/tracking_report.html

# JSON output
ca tracking-evaluate est_track.json gt_track.json \
  --format-json --output-json qa/tracking_result.json
```

Input format:

```json
{
  "frames": [
    {
      "frame_id": "000001",
      "boxes": [
        {
          "label": "car",
          "track_id": "pred-17",
          "center": [0.0, 0.0, 0.0],
          "size": [4.0, 1.8, 1.6]
        }
      ]
    }
  ]
}
```

`ca tracking-evaluate` uses the same 3D box contract as detection (including optional `yaw`), but requires `track_id` on every box. When oriented boxes are present, oriented IoU is used for matching. The current stable output emphasizes `precision / recall / F1`, `MOTA`, `ID switches`, `track fragmentations`, and mean matched IoU. The `kind: tracking` check type is also available in `cloudanalyzer.yaml`.

Checked-in public example files are available under `demo_assets/public/rellis3d-frame-000001/object_eval/`. The tracking examples are deterministic synthetic sequences seeded from the public RELLIS-3D frame used by the detection examples.

| Option | Default | Description |
|---|---|---|
| `ESTIMATED` | required | Estimated tracking sequence JSON |
| `REFERENCE` | required | Reference tracking sequence JSON |
| `--iou-threshold` | `0.5` | IoU threshold used for frame-wise box matching |
| `--min-mota` | `None` | Minimum MOTA gate |
| `--min-recall` | `None` | Minimum recall gate |
| `--max-id-switches` | `None` | Maximum ID switches allowed |
| `--report` | `None` | Write Markdown/HTML tracking report |
| `--format-json` | `false` | Print JSON to stdout |
| `--output-json` | `None` | Dump result as JSON |

## ca convert-labels

Convert external label formats to CloudAnalyzer JSON box sequences.

```bash
# Convert a directory of KITTI label files
ca convert-labels --format kitti --input /path/to/kitti/label_2/ --output boxes.json

# Skip camera-to-lidar coordinate transform
ca convert-labels --format kitti --input labels/ --output boxes.json --no-camera-to-lidar
```

Currently supports KITTI 3D object detection label format. The converter reads `.txt` files from the input directory, parses 3D bounding box fields (dimensions, location, rotation_y), and writes a CloudAnalyzer-compatible JSON file with `yaw` populated from KITTI's `rotation_y`.

| Option | Default | Description |
|---|---|---|
| `--format` | required | Label format (`kitti`) |
| `--input` | required | Input label directory |
| `--output` | required | Output JSON path |
| `--no-camera-to-lidar` | `false` | Skip KITTI camera-to-lidar coordinate transform |

## ca traj-evaluate

Evaluate an estimated trajectory against a reference trajectory.

```bash
# CSV(timestamp,x,y,z) or TUM(timestamp x y z qx qy qz qw)
ca traj-evaluate est.csv gt.csv

# Report + quality gate
ca traj-evaluate est.csv gt.csv \
  --max-time-delta 0.05 --max-ate 0.5 --max-rpe 0.2 --max-drift 1.0 --min-coverage 0.9 \
  --report trajectory_report.html

# Ignore constant initial translation offset
ca traj-evaluate est.csv gt.csv --align-origin

# Fit a rigid transform (rotation + translation) before scoring
ca traj-evaluate est.csv gt.csv --align-rigid
```

Output: matched pose coverage, ATE RMSE/mean/max, translational RPE RMSE/mean/max, endpoint drift, duration coverage. Reports also emit sibling trajectory overlay and error timeline PNGs.

| Option | Default | Description |
|---|---|---|
| `--max-time-delta` | `0.05` | Max timestamp gap allowed for matching/interpolation in seconds |
| `--align-origin` | `false` | Translate the estimated trajectory so the first matched pose aligns to the reference |
| `--align-rigid` | `false` | Fit a rigid transform (rotation + translation) from estimated to reference positions |
| `--max-ate` | `None` | Maximum ATE RMSE allowed; exits with code 1 if exceeded |
| `--max-rpe` | `None` | Maximum translational RPE RMSE allowed; exits with code 1 if exceeded |
| `--max-drift` | `None` | Maximum endpoint drift allowed; exits with code 1 if exceeded |
| `--min-coverage` | `None` | Minimum matched-pose coverage ratio required (0-1); exits with code 1 if not met |
| `--report` | `None` | Write Markdown/HTML trajectory report |
| `--format-json` | `false` | Print JSON to stdout |
| `--output-json` | `None` | Dump result as JSON file |

## ca traj-batch

Evaluate all trajectory files in a directory against matched references in another directory.

```bash
# Match by relative path or stem
ca traj-batch runs/ --reference-dir gt/

# Quality gate + report
ca traj-batch runs/ --reference-dir gt/ \
  --max-time-delta 0.05 --max-ate 0.5 --max-rpe 0.2 --max-drift 1.0 --min-coverage 0.9 \
  --report traj_batch.html

# Apply origin alignment to every run before scoring
ca traj-batch runs/ --reference-dir gt/ --align-origin

# Apply rigid alignment to every run before scoring
ca traj-batch runs/ --reference-dir gt/ --align-rigid
```

Output: one row per trajectory with matched pose count, coverage, ATE RMSE, translational RPE RMSE, endpoint drift, alignment mode, optional pass/fail, plus inspection commands that jump into per-run `traj-evaluate`. HTML reports add pass/failed/low-coverage filters and ATE/RPE/coverage sorting. When `--min-coverage` is set, the low-coverage threshold in the report follows that value.

| Option | Default | Description |
|---|---|---|
| `--reference-dir` | required | Directory containing reference trajectories matched by relative path or stem |
| `-r`, `--recursive` | `false` | Scan subdirectories |
| `--max-time-delta` | `0.05` | Max timestamp gap allowed for matching/interpolation in seconds |
| `--align-origin` | `false` | Translate each estimated trajectory so the first matched pose aligns to the reference |
| `--align-rigid` | `false` | Fit a rigid transform (rotation + translation) from each estimated trajectory to its reference |
| `--max-ate` | `None` | Maximum ATE RMSE allowed; exits with code 1 if any file fails |
| `--max-rpe` | `None` | Maximum translational RPE RMSE allowed; exits with code 1 if any file fails |
| `--max-drift` | `None` | Maximum endpoint drift allowed; exits with code 1 if any file fails |
| `--min-coverage` | `None` | Minimum matched-pose coverage ratio required (0-1); exits with code 1 if any file fails |
| `--report` | `None` | Write Markdown/HTML trajectory batch report |
| `--format-json` | `false` | Print JSON to stdout |
| `--output-json` | `None` | Dump results as JSON file |

## ca run-evaluate

Evaluate one map output and one trajectory output together.

```bash
# Map + trajectory integrated QA
ca run-evaluate map.pcd map_ref.pcd traj.csv traj_ref.csv

# Combined quality gate + report
ca run-evaluate map.pcd map_ref.pcd traj.csv traj_ref.csv \
  --min-auc 0.95 --max-chamfer 0.02 \
  --max-ate 0.5 --max-rpe 0.2 --max-drift 1.0 --min-coverage 0.9 \
  --report run_report.html

# Apply origin alignment to the trajectory before scoring
ca run-evaluate map.pcd map_ref.pcd traj.csv traj_ref.csv --align-origin

# Apply rigid alignment to the trajectory before scoring
ca run-evaluate map.pcd map_ref.pcd traj.csv traj_ref.csv --align-rigid
```

Output: map Chamfer/Hausdorff/AUC/Best F1, trajectory matched coverage/ATE/RPE/drift/alignment, optional overall pass/fail, plus inspection commands for combined `ca web ... --trajectory ... --trajectory-reference ...`, `ca web --heatmap`, `ca heatmap3d`, and per-run `ca traj-evaluate`. Reports emit a map F1 curve together with trajectory overlay and error timeline PNGs.

| Option | Default | Description |
|---|---|---|
| `-t`, `--thresholds` | default evaluate thresholds | Comma-separated distance thresholds for map F1/AUC evaluation |
| `--max-time-delta` | `0.05` | Max timestamp gap allowed for trajectory matching/interpolation in seconds |
| `--align-origin` | `false` | Translate the estimated trajectory so the first matched pose aligns to the reference |
| `--align-rigid` | `false` | Fit a rigid transform (rotation + translation) from estimated to reference positions |
| `--min-auc` | `None` | Minimum map AUC required; contributes to the overall quality gate |
| `--max-chamfer` | `None` | Maximum map Chamfer distance allowed; contributes to the overall quality gate |
| `--max-ate` | `None` | Maximum trajectory ATE RMSE allowed; contributes to the overall quality gate |
| `--max-rpe` | `None` | Maximum trajectory translational RPE RMSE allowed; contributes to the overall quality gate |
| `--max-drift` | `None` | Maximum trajectory endpoint drift allowed; contributes to the overall quality gate |
| `--min-coverage` | `None` | Minimum trajectory matched-pose coverage ratio required; contributes to the overall quality gate |
| `--report` | `None` | Write Markdown/HTML combined run report |
| `--format-json` | `false` | Print JSON to stdout |
| `--output-json` | `None` | Dump result as JSON file |

## ca run-batch

Evaluate multiple map + trajectory runs together.

```bash
# Match map / trajectory / references by relative path or stem
ca run-batch maps/ \
  --map-reference-dir map_refs/ \
  --trajectory-dir trajs/ \
  --trajectory-reference-dir traj_refs/

# Combined quality gate + report
ca run-batch maps/ \
  --map-reference-dir map_refs/ \
  --trajectory-dir trajs/ \
  --trajectory-reference-dir traj_refs/ \
  --min-auc 0.95 --max-chamfer 0.02 \
  --max-ate 0.5 --max-rpe 0.2 --max-drift 1.0 --min-coverage 0.9 \
  --report run_batch.html

# Apply origin alignment to every trajectory before scoring
ca run-batch maps/ \
  --map-reference-dir map_refs/ \
  --trajectory-dir trajs/ \
  --trajectory-reference-dir traj_refs/ \
  --align-origin
```

Output: one row per run with map AUC/Chamfer and trajectory ATE/RPE/drift/coverage, optional overall pass/fail, plus inspection commands for a combined `ca web` run viewer, map heatmaps, and per-run `traj-evaluate`. Reports summarize mean map and trajectory quality across runs. HTML reports add pass/failed/map-issue/trajectory-issue filters and map/trajectory sorting.
When a quality gate is active, CLI/report summaries also break failures into map failures vs trajectory failures, and inspection commands include both a per-run `ca web ... --trajectory ... --trajectory-reference ...` viewer and `ca run-evaluate ...` drill-down command.

| Option | Default | Description |
|---|---|---|
| `--map-reference-dir` | required | Directory containing reference maps matched by relative path or stem |
| `--trajectory-dir` | required | Directory containing estimated trajectories matched to the map outputs |
| `--trajectory-reference-dir` | required | Directory containing reference trajectories matched by relative path or stem |
| `-r`, `--recursive` | `false` | Scan subdirectories |
| `-t`, `--thresholds` | default evaluate thresholds | Comma-separated distance thresholds for map F1/AUC evaluation |
| `--max-time-delta` | `0.05` | Max timestamp gap allowed for trajectory matching/interpolation in seconds |
| `--align-origin` | `false` | Translate each estimated trajectory so the first matched pose aligns to the reference |
| `--align-rigid` | `false` | Fit a rigid transform (rotation + translation) from each estimated trajectory to its reference |
| `--min-auc` | `None` | Minimum map AUC required; contributes to the overall quality gate |
| `--max-chamfer` | `None` | Maximum map Chamfer distance allowed; contributes to the overall quality gate |
| `--max-ate` | `None` | Maximum trajectory ATE RMSE allowed; contributes to the overall quality gate |
| `--max-rpe` | `None` | Maximum trajectory translational RPE RMSE allowed; contributes to the overall quality gate |
| `--max-drift` | `None` | Maximum trajectory endpoint drift allowed; contributes to the overall quality gate |
| `--min-coverage` | `None` | Minimum trajectory matched-pose coverage ratio required; contributes to the overall quality gate |
| `--report` | `None` | Write Markdown/HTML combined run-batch report |
| `--format-json` | `false` | Print JSON to stdout |
| `--output-json` | `None` | Dump results as JSON file |
