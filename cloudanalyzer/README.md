# CloudAnalyzer

AI-friendly CLI tool for point cloud analysis and evaluation.

Docs: [command reference](https://github.com/rsasaki0109/CloudAnalyzer/tree/main/docs/commands),
[SLAM benchmark tutorial](https://github.com/rsasaki0109/CloudAnalyzer/blob/main/docs/tutorial-slam-benchmark.md),
[CI and quality gates](https://github.com/rsasaki0109/CloudAnalyzer/blob/main/docs/ci.md).
The repository also hosts a [browser point cloud viewer](https://rsasaki0109.github.io/CloudAnalyzer/app/).

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

There are **39** CLI subcommands (see `ca --help`). Summary:

### Analysis & Evaluation

| Command | Description |
|---|---|
| `ca compare` | Compare two point clouds with ICP/GICP registration |
| `ca scan-match-debug` | Debug one scan-to-map ICP/GICP attempt with before/after distances and artifacts |
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
| `ca web-view` | Open results (maps, trajectories) in the web app with one link, served from this machine ([docs](https://github.com/rsasaki0109/CloudAnalyzer/blob/main/docs/commands/web-view.md)) |
| `ca mcp` | Serve the pose graph and evaluation tools to AI agents over MCP ([docs](https://github.com/rsasaki0109/CloudAnalyzer/blob/main/docs/commands/mcp.md)) |
| `ca vectormap-build` | Draft Autoware Lanelet2 roads from a cloud and a trajectory, with detected/inferred counts and validation ([docs](https://github.com/rsasaki0109/CloudAnalyzer/blob/main/docs/commands/vectormap-build.md)) |
| `ca posegraph-compare` | Join two drives through the same places and list the changed objects (M3C2) ([docs](https://github.com/rsasaki0109/CloudAnalyzer/blob/main/docs/commands/posegraph-compare.md)) |
| `ca posegraph-fix` | Fix a SLAM map: loops by ICP, IMU gravity, dynamic points out; fixed poses, g2o and map ([docs](https://github.com/rsasaki0109/CloudAnalyzer/blob/main/docs/commands/posegraph-fix.md)) |
| `ca info` | Point cloud metadata (points, BBox, robust BBox, centroid) |
| `ca stats` | Detailed statistics (density, robust density, spacing distribution) |
| `ca batch` | Run info on all files in a directory |

`ca scan-match-debug --artifact-dir ...` shares before/after distance arrays
between metrics and PLY coloring. It performs two nearest-neighbour searches
instead of four, with unchanged ICP/GICP and output formats. The measured
diagnostic/artifact interval improves by 9.75–17.27% across six real-source
comparisons; loading/registration time is separate. [Measurements and memory
lifetime](../docs/cpu-scan-match-distance-reuse.md).

With a supporting Rust core, `scan-match-debug` also shares one working-map
tree across the two searches and releases it before artifact writing. The
diagnostic/statistics interval, including construction and cleanup, improves
by 20.55–24.45% in six real-source comparisons. Loading, registration and PLY
writing are outside that measurement. [Scoped lifetime and compatibility](../docs/cpu-scan-match-reference.md).

Multi-frame `slam-debug` also shares a lazy map snapshot and, without cropping,
one reference across selected attempts. A three-attempt real-source diagnostic
comparison is 17.47–44.66% shorter with 29/30 faster pairs and identical outputs.
Loading is included; ICP solving and artifacts are excluded. These repeated
attempts are not independent drive frames. [Scope and lifetime](../docs/cpu-slam-map-reuse.md).

GICP and artifact-producing ICP also share pre-solver target normals across
uncropped multi-frame attempts, with private working copies. The preceding measured
registration-preparation/result-copy interval is 21.05–36.37% shorter across
six real-source comparisons, with 18/18 faster pairs. Actual ICP iterations
are excluded. [Normal reuse measurements](../docs/cpu-registration-target-normals.md).

ICP diagnostics without PLY artifacts skip unused normal estimation. With
loading and distance/report assembly included, six real-source comparisons
are 72.16–81.47% shorter with 18/18 faster pairs. Actual ICP iterations are
excluded; GICP, artifact output and ordinary registration keep their normal
preparation. [Scope and verification](../docs/cpu-scan-icp-normals.md).

Verified little-endian binary PLY files containing only double XYZ coordinates
now use bulk loading. Six real-source comparisons shorten loading by
78.01–83.25% and diagnostic assembly by 23.08–31.70%, with 30/30 faster pairs
in both intervals. Actual ICP iterations are excluded; complex layouts retain
Open3D loading. [Supported layout and measurements](../docs/cpu-ply-xyz-loading.md).

Comparisons without a snapshot skip unused heatmap coloring. Six real-source
comparisons with registration disabled are 78.45–82.21% shorter, with 60/60
faster pairs and identical numeric/file reports. Snapshot colors and pixels
are preserved. [Conditions and measurements](../docs/cpu-compare-requested-colors.md).

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

`ca batch --evaluate` loads one reference snapshot per invocation and shares
its search tree across candidates when the Rust core supports it. Reference
edits are observed on the next batch. [Real-source measurements](../docs/cpu-batch-reference.md)
show 23.83–33.78% shorter whole batch functions for three candidates, including
loading, tree construction/destruction, both distance directions and statistics.

`ca loop-closure-report` shares one reference snapshot/tree for its before/after
map evaluations and releases it before optional trajectory/posegraph work.
[Actual-input report measurements](../docs/cpu-loop-reference.md) show
10.77–23.88% shorter map-only report functions, including loading, searches,
statistics and cleanup. The inputs use an archived actual ICP result; solving
and CLI serialization are outside those clocks.

`ca map-evaluate` parallelizes exact SciPy distance queries with at least 65,536
query points per direction. [Real-source measurements](../docs/cpu-map-nn-parallel.md)
show 16.57–26.35% shorter resident evaluations including default AWD/SCS, with
29/30 faster pairs and identical metrics/PLY outputs. Loading and CLI writes
are excluded. Large queries use all available CPU threads; small inputs keep
single-thread dispatch.

`map-evaluate` also indexes supported SCS error cells for exact cube-neighborhood
lookup. [Real-source measurements](../docs/cpu-map-scs-index.md) show 5.43–18.12%
shorter resident evaluations against the preceding parallel-NN implementation,
with identical full metrics/PLY outputs. Index setup/cleanup is included; file
loading and CLI writes are excluded. Small/radius-1/wide-span cases retain the
grid lookup, and the scoped index is released per structural evaluation.

Resident Gaussian covariance now accumulates six symmetric components in point
order. [Real-source measurements](../docs/cpu-map-covariance.md) show a further
5.13–17.64% shorter evaluation against the parallel-NN/indexed-SCS baseline,
with 27/30 faster pairs and identical full voxel summaries, metrics and PLYs.
The point-product temporary drops from 72 to 8 bytes per point, plus component
result arrays per voxel. Non-finite/extreme centered values retain scatter.

Resident voxel classification now uses bounded scalar integer codes for exact
grouping. [Real-source measurements](../docs/cpu-map-grouping.md) show a further
45.73–54.44% shorter evaluation against the preceding covariance implementation,
with 30/30 faster pairs and identical memberships, full Gaussians, metrics and
PLYs. Small inputs and ranges beyond signed-int64 capacity retain axis grouping.

Streaming AWD/SCS now groups each chunk once with stable point order, removing
full-chunk scans for every occupied voxel.
[Real-source measurements](../docs/cpu-map-stream-groups.md) show 69.87–82.36%
shorter complete streaming reduction with 18/18 faster pairs and identical
ordered update streams, Gaussians and reports. Fixed 100,000-point memory-map
chunks are timed; original-file loading/snapshot preparation are excluded.

Optional core-API centroid preprocessing now detects adjacent sorted groups,
batches singleton means and preallocates its output.
[Real-source API measurements](../docs/cpu-map-downsample-centroids.md) show
33.67–81.01% shorter complete evaluation with requested 0.2 m downsampling,
18/18 faster pairs and identical centroids, full metrics and PLYs. Loading and
CLI writes are excluded; this is not a disabled-downsampling speed claim.

Resident voxel means also use point-order coordinate reductions.
[Mean-sum measurements](../docs/cpu-map-mean-sums.md) show 65.02–74.36% shorter
aggregation with identical full outputs. Complete resident evaluation has a
smaller mixed result (−2.73% to +4.10%, 18/30 faster pairs); the component rate
does not describe whole-evaluation acceleration. Small/unsafe inputs keep scatter.

Supported Gaussian Wasserstein distances now use bounded matrix batches.
[Real-source measurements](../docs/cpu-map-wasserstein-batch.md) show
9.05–27.16% shorter complete resident evaluation, with 28/30 faster pairs and
identical per-voxel distances, full metrics and PLYs. Small/unsafe batches and
explicit NumPy error settings keep scalar calls; loading/CLI writes are excluded.

SCS also groups equal-length neighborhoods within bounded windows.
[SCS statistics measurements](../docs/cpu-map-scs-batch.md) show 10.38–31.92%
shorter SCS work with identical full outputs. Complete resident evaluation is
mixed (−2.30% to +5.99%, 16/30 faster pairs), so the component rate does not
describe a consistent whole-evaluation gain. Unsafe inputs retain scalar calls.

Large resident pairs also use compact midpoint point trees, preserving exact
distances and the existing worker policy.
[Point-tree measurements](../docs/cpu-map-nn-midpoint.md) show 7.92–17.04%
shorter complete resident evaluation, with 27/30 faster pairs and identical
full outputs. Both maps must have at least 65,536 points; loading/CLI writes
are excluded.

Eligible SCS indexes also use bounded vector queries with one worker.
[Query-batch measurements](../docs/cpu-map-scs-query-batch.md) show 14.47–26.73%
shorter complete SCS work, with 30/30 faster pairs and identical full outputs.
Whole resident evaluation is mixed (−5.08% to +9.51%, 18/30 faster pairs);
PandaSet regresses in both passes, so the component rate is not a universal
whole-evaluation improvement. Small or insufficient-capacity indexes retain
single-center queries.

Streaming voxel accumulation also constructs each cell's accumulator once.
[Allocation measurements](../docs/cpu-map-stream-accumulators.md) show
52.62–68.59% fewer constructions with identical Gaussian bits and full reports.
Whole streaming timings are mixed (−11.27% to +9.93%, 20/30 faster pairs),
without a consistent whole-evaluation speedup across sources.

Streaming accumulation also batches enough equal-size small-group moments
inside bounded windows. [Moment measurements](../docs/cpu-map-stream-moments.md)
show 1.66–28.90% shorter complete streaming AWD/SCS work, with 28/30 faster
pairs and identical Gaussian/bounds bits, ordered groups and full reports.
Loading/CLI writes are excluded; unsafe or insufficient groups use scalar updates.

[Expanded moment measurements](../docs/cpu-map-stream-expanded.md) accept
1–128-point groups with eight groups per size, retaining a 32,768-point gather
limit. Complete streaming work is a further 7.98–18.80% shorter, with 27/30
faster pairs and identical Gaussian/bounds bits, group inputs and reports.
These comparisons already include the preceding small-group optimization.

[Verified-scalar measurements](../docs/cpu-map-stream-finite.md) reuse the
existing whole-chunk guard to avoid repeated finite checks and coordinate
copies in remaining scalar groups. Complete streaming work is a further
6.88–16.46% shorter, with 25/30 faster pairs and identical full outputs.
Unsafe/small chunks keep filtering; loading/CLI writes are excluded.

[Moment-merge candidates](../docs/cpu-map-stream-merge-candidates.md) were
tested on the same real sources. All outputs match, but every candidate
regresses on at least one source; none is adopted.

The optional helper also batches sufficiently many 2–8-point means.
[Small-group API measurements](../docs/cpu-map-downsample-small-groups.md) show
55.24–70.38% shorter complete 0.2 m downsampled evaluation for eligible
PandaSet/Tokyo inputs, with 12/12 faster pairs and identical full outputs.
Autoware retains preceding means and has mixed timings (−16.50%, +2.62%);
no acceleration is claimed there. Input/error-policy guards preserve fallbacks.

`ca info` and `ca stats` also report p01-p99 robust extents and the number/ratio
of points outside that robust bounding box. These fields are useful for SLAM maps
where a small number of far outliers can dominate the ordinary axis-aligned
extent and hide whether the main map body is stable.

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
- `.las` / `.laz` (LiDAR)
- `.csv` (`x,y,z` or GLIM-style `x_m,y_m,z_m`)
