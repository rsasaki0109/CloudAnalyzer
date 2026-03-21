# CloudAnalyzer

AI-friendly CLI tool for point cloud analysis and evaluation.

## Install

```bash
pip install -e .

# or with Docker
docker build -t ca .
docker run ca info cloud.pcd
```

## Commands

### Analysis & Evaluation

| Command | Description |
|---|---|
| `ca compare` | Compare two point clouds with ICP/GICP registration |
| `ca diff` | Quick distance stats (no registration) |
| `ca evaluate` | F1, Chamfer, Hausdorff, AUC evaluation |
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
| `ca view` | Interactive 3D viewer |
| `ca density-map` | 2D density heatmap image |
| `ca heatmap3d` | 3D distance heatmap snapshot |

## Usage Examples

```bash
# === Evaluation ===
# F1/Chamfer/Hausdorff evaluation with curve plot
ca evaluate source.pcd reference.pcd \
  -t 0.05,0.1,0.2,0.5,1.0 --plot f1_curve.png

# Full pipeline: filter → downsample → evaluate
ca pipeline noisy.pcd reference.pcd -o clean.pcd -v 0.2

# 3D distance heatmap
ca heatmap3d estimated.pcd reference.pcd -o heatmap.png

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

```bash
# Pipe JSON to jq
ca info cloud.pcd --format-json | jq '.num_points'
ca evaluate a.pcd b.pcd --format-json | jq '.auc'
```

## CI Quality Gate

Use the `quality-gate.yml` workflow to fail CI if point cloud quality drops:

```yaml
# Triggers manually with source/reference paths and thresholds
# Fails if AUC < threshold or Chamfer > threshold
```

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
