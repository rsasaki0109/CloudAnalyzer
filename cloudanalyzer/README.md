# CloudAnalyzer

AI-friendly CLI tool for point cloud analysis and evaluation.

For the full product overview, demos, and tutorials, see the [repository root README](https://github.com/rsasaki0109/CloudAnalyzer/blob/main/README.md).

## Install

From this directory (the Python package root):

```bash
pip install cloudanalyzer

# latest from source
git clone https://github.com/rsasaki0109/CloudAnalyzer.git
cd CloudAnalyzer/cloudanalyzer
pip install -e .

# or with Docker
docker build -t ca .
docker run ca info cloud.pcd
```

## Release Sanity Check

```bash
python3 -m pip install -e .[dev]
python3 -m build
python3 -m twine check dist/*
```

## Commands

There are **34** CLI subcommands (see `ca --help`). Summary:

### Analysis & Evaluation

| Command | Description |
|---|---|
| `ca compare` | Compare two point clouds with ICP/GICP registration |
| `ca diff` | Quick distance stats (no registration) |
| `ca evaluate` | F1, Chamfer, Hausdorff, AUC evaluation |
| `ca check` | Config-driven unified QA (`cloudanalyzer.yaml`) |
| `ca init-check` | Emit a starter `cloudanalyzer.yaml` profile |
| `ca detection-evaluate` | 3D object detection QA for axis-aligned boxes (mAP, precision/recall/F1, reports) |
| `ca ground-evaluate` | Ground segmentation QA (precision/recall/F1/IoU, optional gates) |
| `ca tracking-evaluate` | 3D multi-object tracking QA (MOTA, ID switches, precision/recall/F1, reports) |
| `ca traj-evaluate` | ATE, translational RPE, drift evaluation for trajectories |
| `ca traj-batch` | Batch trajectory benchmark with coverage, gate, and reports |
| `ca run-evaluate` | Combined map + trajectory QA for one run |
| `ca run-batch` | Combined map + trajectory benchmark across multiple runs |
| `ca info` | Point cloud metadata (points, BBox, centroid) |
| `ca stats` | Detailed statistics (density, spacing distribution) |
| `ca batch` | Run info on all files in a directory |

### Processing

| Command | Description |
|---|---|
| `ca downsample` | Voxel grid downsampling |
| `ca sample` | Random point sampling |
| `ca filter` | Statistical outlier removal |
| `ca merge` | Merge multiple point clouds |
| `ca align` | Sequential registration + merge |
| `ca split` | Split into grid tiles |
| `ca convert` | Format conversion (pcd/ply/las) |
| `ca normals` | Normal estimation |
| `ca crop` | Bounding box crop |
| `ca pipeline` | filter → downsample → evaluate in one step |

### Visualization

| Command | Description |
|---|---|
| `ca web` | Browser 3D viewer, with optional heatmap, reference overlay, and trajectory run overlay |
| `ca web-export` | Write a static browser viewer bundle (for demos and sharing) |
| `ca view` | Interactive 3D viewer |
| `ca density-map` | 2D density heatmap image |
| `ca heatmap3d` | 3D distance heatmap snapshot |

### Baseline history

| Command | Description |
|---|---|
| `ca baseline-save` | Save a QA summary JSON into a rotating history directory |
| `ca baseline-list` | List baselines saved in a history directory |
| `ca baseline-decision` | Promote / keep / reject a candidate baseline vs history |

### Utility

| Command | Description |
|---|---|
| `ca version` | Print CLI version |

## Usage Examples

```bash
# === Evaluation ===
# F1/Chamfer/Hausdorff evaluation with curve plot
ca evaluate source.pcd reference.pcd \
  -t 0.05,0.1,0.2,0.5,1.0 --plot f1_curve.png

# Trajectory evaluation with quality gate
ca traj-evaluate estimated.csv reference.csv \
  --max-time-delta 0.05 --max-ate 0.5 --max-rpe 0.2 --max-drift 1.0 --min-coverage 0.9 \
  --report trajectory_report.html
# report also writes sibling trajectory overlay and error timeline PNGs

# Ground segmentation evaluation with report output
ca ground-evaluate estimated_ground.pcd estimated_nonground.pcd \
  reference_ground.pcd reference_nonground.pcd \
  --min-f1 0.9 --min-iou 0.8 --report ground_report.html

# 3D object detection evaluation from JSON box sequences
ca detection-evaluate estimated_detection.json reference_detection.json \
  --iou-thresholds 0.25,0.5 --min-map 0.9 --report detection_report.html

# 3D multi-object tracking evaluation from JSON box sequences
ca tracking-evaluate estimated_tracking.json reference_tracking.json \
  --iou-threshold 0.5 --min-mota 0.8 --max-id-switches 2 \
  --report tracking_report.html

# Ignore constant initial translation offset
ca traj-evaluate estimated.csv reference.csv --align-origin

# Fit a rigid transform before scoring
ca traj-evaluate estimated.csv reference.csv --align-rigid

# Batch trajectory benchmark
ca traj-batch runs/ --reference-dir gt/ \
  --max-time-delta 0.05 --max-ate 0.5 --max-rpe 0.2 --max-drift 1.0 --min-coverage 0.9 \
  --report traj_batch.html
# HTML report adds copyable inspection commands plus pass/failed/low-coverage filters and ATE/RPE/coverage sorting
# low-coverage threshold follows --min-coverage when provided

# Combined run QA: map + trajectory in one report
ca run-evaluate map.pcd map_ref.pcd traj.csv traj_ref.csv \
  --min-auc 0.95 --max-chamfer 0.02 \
  --max-ate 0.5 --max-rpe 0.2 --max-drift 1.0 --min-coverage 0.9 \
  --report run_report.html
# inspection commands include a `ca web ... --trajectory ... --trajectory-reference ...` run viewer

# Combined run batch QA
ca run-batch maps/ \
  --map-reference-dir map_refs/ \
  --trajectory-dir trajs/ \
  --trajectory-reference-dir traj_refs/ \
  --min-auc 0.95 --max-chamfer 0.02 \
  --max-ate 0.5 --max-rpe 0.2 --max-drift 1.0 --min-coverage 0.9 \
  --report run_batch.html
# HTML report adds pass/failed/map-issue/trajectory-issue filters and map/trajectory sorting
# summary and CLI output also split map failures vs trajectory failures
# inspection commands include both a per-run `ca web ...` run viewer and `ca run-evaluate ...` drill-down command

# Full pipeline: filter → downsample → evaluate
ca pipeline noisy.pcd reference.pcd -o clean.pcd -v 0.2

# 3D distance heatmap
ca heatmap3d estimated.pcd reference.pcd -o heatmap.png

# Browser heatmap viewer with reference overlay and threshold filter
ca web estimated.pcd reference.pcd --heatmap

# Browser run viewer: map heatmap + trajectory overlay
ca web map.pcd map_ref.pcd --heatmap \
  --trajectory traj.csv --trajectory-reference traj_ref.csv
# With paired trajectories, the viewer also highlights the worst ATE pose and worst RPE segment.
# Clicking a marker or segment shows the timestamp and error summary in the inspection panel.
# The camera also moves toward the selected location, and Reset View returns to the full scene.
# The trajectory error timeline is shown in the same viewer and stays synced with 3D point selections.

# === Compare ===
ca compare source.pcd target.pcd \
  --register gicp --json result.json --report report.md \
  --snapshot diff.png --threshold 0.1

# Quick diff
ca diff a.pcd b.pcd --threshold 0.05

# === Processing ===
# Split large map into 100m tiles
ca split large_map.pcd -o tiles/ -g 100

# Downsample
ca downsample cloud.pcd -o down.pcd -v 0.05

# Filter outliers
ca filter raw.pcd -o clean.pcd -n 20 -s 2.0

# Align multiple scans
ca align scan1.pcd scan2.pcd scan3.pcd -o aligned.pcd -m gicp

# Batch info
ca batch /path/to/pcds/ -r

# Batch evaluation
ca batch /path/to/results/ --evaluate reference.pcd --format-json | jq '.[].auc'
ca batch /path/to/results/ --evaluate reference.pcd --report batch_report.html
# report includes inspection commands; HTML adds Copy buttons plus count-badged summary rows, quick actions, failed-first / recommended-first sort presets, and pass/failed/pareto/recommended controls
ca batch decoded/ --evaluate reference.pcd --compressed-dir compressed/ --baseline-dir original/
# report also emits a quality-vs-size scatter plot, Pareto candidates, a recommended point, failed-first / recommended-first sort presets, and HTML filters
ca batch /path/to/results/ --evaluate reference.pcd --min-auc 0.95 --max-chamfer 0.02

# Density heatmap
ca density-map cloud.pcd -o density.png -r 1.0 -a z
```

## Global Options

```bash
ca --verbose ...    # Debug output (stderr)
ca --quiet ...      # Suppress non-error output
```

## Output Options

- `--output-json <path>` — Dump result as JSON file
- `--format-json` — Print JSON to stdout for piping
- `--plot <path>` — F1 curve plot (evaluate only)
- `--report <path>` — Markdown/HTML report (`batch`, `detection-evaluate`, `ground-evaluate`, `tracking-evaluate`, `traj-evaluate`, `traj-batch`, `run-evaluate`, `run-batch`)

```bash
# Pipe JSON to jq
ca info cloud.pcd --format-json | jq '.num_points'
ca evaluate a.pcd b.pcd --format-json | jq '.auc'
```

## CI quality gate

Point cloud / trajectory / perception QA is usually driven by `ca check` and a `cloudanalyzer.yaml` config (see [docs/ci.md](https://github.com/rsasaki0109/CloudAnalyzer/blob/main/docs/ci.md) and the [map quality gate tutorial](https://github.com/rsasaki0109/CloudAnalyzer/blob/main/docs/tutorial-map-quality-gate.md)).

For object detection / tracking, the current stable contract is a JSON sequence:

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
          "score": 0.97,
          "track_id": "pred-17"
        }
      ]
    }
  ]
}
```

`ca detection-evaluate` uses `label`, `center`, `size`, and optional `score`; `ca tracking-evaluate` also requires `track_id`. Geometry is currently evaluated as **axis-aligned 3D boxes** and ignores yaw if present in the source JSON.

Checked-in public sample JSONs live under `demo_assets/public/rellis3d-frame-000001/object_eval/`:

- `detection_reference.json`
- `detection_estimated_good.json`
- `detection_estimated_regressed.json`
- `tracking_reference.json`
- `tracking_estimated_good.json`
- `tracking_estimated_regressed.json`

Those examples are generated from the public RELLIS-3D seed frame; the tracking files are deterministic synthetic 3-frame sequences seeded from that public frame so the contract can be demonstrated without bundling a full public MOT dataset.

In **this** GitHub repo, reusable workflows run the same gates in CI. Pin to a **tag or SHA** when calling them from another repository (not floating `@main`).

```yaml
jobs:
  qa:
    uses: rsasaki0109/CloudAnalyzer/.github/workflows/config-quality-gate.yml@main
    with:
      config_path: cloudanalyzer.yaml

  baseline:
    uses: rsasaki0109/CloudAnalyzer/.github/workflows/baseline-gate.yml@main
    with:
      config_path: cloudanalyzer.yaml
      history_dir: qa/history
```

The repo also ships a [manual quality-gate workflow](https://github.com/rsasaki0109/CloudAnalyzer/actions) that accepts source/reference paths and thresholds for ad-hoc runs.

## Python API

```python
from ca.evaluate import evaluate, plot_f1_curve
from ca.plot import plot_multi_f1, heatmap3d
from ca.pipeline import run_pipeline
from ca.split import split
from ca.info import get_info
from ca.diff import run_diff
from ca.downsample import downsample
from ca.filter import filter_outliers

# Evaluate
result = evaluate("estimated.pcd", "reference.pcd")
print(f"AUC: {result['auc']:.4f}, Chamfer: {result['chamfer_distance']:.4f}")
plot_f1_curve(result, "f1_curve.png")

# Compare multiple results
results = [evaluate(f"v{v}.pcd", "ref.pcd") for v in [0.1, 0.2, 0.5]]
plot_multi_f1(results, ["v0.1", "v0.2", "v0.5"], "comparison.png")

# Pipeline
result = run_pipeline("noisy.pcd", "reference.pcd", "clean.pcd", voxel_size=0.2)

# Split
result = split("large.pcd", "tiles/", grid_size=100.0)
```

## Supported Formats

- `.pcd` (Point Cloud Data)
- `.ply` (Polygon File Format)
- `.las` (LiDAR)
