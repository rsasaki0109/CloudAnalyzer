# ca compare

Compare two point clouds with optional ICP/GICP registration. Generates JSON, Markdown reports, and snapshot images.

## Usage

```bash
ca compare SOURCE TARGET [OPTIONS]
```

## Arguments

| Argument | Description |
|---|---|
| `SOURCE` | Source point cloud file (pcd/ply/las) |
| `TARGET` | Target point cloud file (pcd/ply/las) |

## Options

| Option | Default | Description |
|---|---|---|
| `--register` | `gicp` | Registration method: `icp`, `gicp`, or `none` |
| `--json` | — | Output path for JSON report |
| `--report` | — | Output path for Markdown report |
| `--snapshot` | — | Output path for snapshot image (png) |
| `--threshold` | — | Distance threshold to check |
| `--output-json` | — | Dump full result as JSON |

## Examples

```bash
# Full output with GICP registration
ca compare source.pcd target.pcd \
  --register gicp --json result.json --report report.md \
  --snapshot diff.png --threshold 0.1

# Without registration
ca compare a.pcd b.pcd --register none

# ICP registration, JSON only
ca compare a.ply b.ply --register icp --json result.json
```

## Pipeline

1. Load source and target point clouds
2. (Optional) Align source to target via ICP/GICP
3. Compute nearest neighbor distances
4. Colorize source by distance (blue=close, red=far)
5. Output JSON, Markdown report, and/or snapshot image
