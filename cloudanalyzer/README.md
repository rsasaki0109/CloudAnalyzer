# CloudAnalyzer

AI-friendly CLI tool for point cloud analysis.

## Install

```bash
pip install -e .
```

## Commands

### Analysis

| Command | Description |
|---|---|
| `ca compare` | Compare two point clouds with ICP/GICP registration |
| `ca diff` | Quick distance stats (no registration) |
| `ca info` | Point cloud metadata (points, BBox, centroid) |
| `ca stats` | Detailed statistics (density, spacing distribution) |
| `ca batch` | Run info on all files in a directory |
| `ca density-map` | 2D density heatmap image |

### Processing

| Command | Description |
|---|---|
| `ca downsample` | Voxel grid downsampling |
| `ca sample` | Random point sampling |
| `ca filter` | Statistical outlier removal |
| `ca merge` | Merge multiple point clouds |
| `ca align` | Sequential registration + merge |
| `ca convert` | Format conversion (pcd/ply/las) |
| `ca normals` | Normal estimation |
| `ca crop` | Bounding box crop |

### Visualization

| Command | Description |
|---|---|
| `ca view` | Interactive 3D viewer |

## Usage Examples

```bash
# Compare with GICP, full output
ca compare source.pcd target.pcd \
  --register gicp --json result.json --report report.md \
  --snapshot diff.png --threshold 0.1

# Quick diff with threshold
ca diff a.pcd b.pcd --threshold 0.05

# Batch info on a directory
ca batch /path/to/pcds/ -r

# Density heatmap
ca density-map cloud.pcd -o density.png -r 1.0 -a z

# Processing pipeline
ca filter raw.pcd -o clean.pcd -n 20 -s 2.0
ca downsample clean.pcd -o down.pcd -v 0.05
ca normals down.pcd -o with_normals.ply

# Align multiple scans
ca align scan1.pcd scan2.pcd scan3.pcd -o aligned.pcd -m gicp

# Format conversion
ca convert input.pcd output.ply

# Random sampling
ca sample cloud.pcd -o sampled.pcd -n 10000

# Crop by bounding box
ca crop cloud.pcd -o cropped.pcd \
  --x-min 0 --y-min 0 --z-min 0 \
  --x-max 10 --y-max 10 --z-max 5
```

## Global Options

```bash
ca --verbose ...    # Debug output (stderr)
ca --quiet ...      # Suppress non-error output
```

## Output Options

- `--output-json <path>` — Dump result as JSON file (all commands except view/version)
- `--format-json` — Print JSON to stdout for piping (info, diff, stats, batch)

```bash
# Pipe JSON to jq
ca info cloud.pcd --format-json | jq '.num_points'

# Save result for automation
ca diff a.pcd b.pcd --output-json result.json
```

## Python API

```python
from ca.info import get_info
from ca.diff import run_diff
from ca.compare import run_compare
from ca.stats import compute_stats
from ca.downsample import downsample
from ca.filter import filter_outliers
from ca.merge import merge
from ca.align import align
from ca.density_map import density_map

# Each function returns a dict
info = get_info("cloud.pcd")
print(info["num_points"])

result = run_diff("a.pcd", "b.pcd", threshold=0.1)
print(result["distance_stats"]["mean"])
```

## Supported Formats

- `.pcd` (Point Cloud Data)
- `.ply` (Polygon File Format)
- `.las` (LiDAR)
