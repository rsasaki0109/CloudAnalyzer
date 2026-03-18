# CloudAnalyzer

CloudAnalyzer is an AI-friendly CLI tool for analyzing and comparing point clouds.

## Install

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Compare two point clouds with GICP registration and full output
ca compare source.pcd target.pcd \
  --register gicp \
  --json result.json \
  --report report.md \
  --snapshot diff.png

# Compare without registration
ca compare a.pcd b.pcd --register none --report report.md

# ICP registration, JSON output only
ca compare a.ply b.ply --register icp --json result.json
```

## CLI Reference

```
ca compare SOURCE TARGET [OPTIONS]

Arguments:
  SOURCE  Path to source point cloud (pcd/ply/las)
  TARGET  Path to target point cloud (pcd/ply/las)

Options:
  --register TEXT   Registration method: icp, gicp, or 'none' to skip [default: gicp]
  --json TEXT       Output path for JSON report
  --report TEXT     Output path for Markdown report
  --snapshot TEXT   Output path for snapshot image (png)
```

## How It Works

1. Load source and target point clouds
2. (Optional) Align source to target via ICP/GICP registration
3. Compute nearest neighbor distances from source to target
4. Colorize source by distance (blue=close, red=far)
5. Output JSON metrics, Markdown report, and/or snapshot image
