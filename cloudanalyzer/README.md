# CloudAnalyzer

AI-friendly CLI tool for point cloud analysis.

## Install

```bash
pip install -e .
```

## Commands

### `ca compare` — Compare two point clouds

```bash
ca compare source.pcd target.pcd \
  --register gicp \
  --json result.json \
  --report report.md \
  --snapshot diff.png \
  --threshold 0.1 \
  --output-json full_result.json
```

Options: `--register` (icp/gicp/none), `--json`, `--report`, `--snapshot`, `--threshold`, `--output-json`

### `ca diff` — Quick distance stats (no registration)

```bash
ca diff source.pcd target.pcd --threshold 0.05
```

### `ca info` — Point cloud metadata

```bash
ca info cloud.pcd
# Points, BBox, extent, centroid, colors/normals
```

### `ca stats` — Detailed statistics

```bash
ca stats cloud.pcd
# Density, volume, spacing distribution
```

### `ca view` — Interactive 3D viewer

```bash
ca view cloud1.pcd cloud2.ply
```

### `ca downsample` — Voxel downsampling

```bash
ca downsample cloud.pcd -o down.pcd -v 0.05
```

### `ca sample` — Random point sampling

```bash
ca sample cloud.pcd -o sampled.pcd -n 10000
```

### `ca filter` — Statistical outlier removal

```bash
ca filter cloud.pcd -o filtered.pcd -n 20 -s 2.0
```

### `ca merge` — Merge multiple point clouds

```bash
ca merge a.pcd b.pcd c.pcd -o merged.pcd
```

### `ca align` — Sequential registration + merge

```bash
ca align scan1.pcd scan2.pcd scan3.pcd -o aligned.pcd -m gicp
```

### `ca convert` — Format conversion

```bash
ca convert input.pcd output.ply
```

### `ca normals` — Normal estimation

```bash
ca normals cloud.pcd -o with_normals.ply -r 0.5
```

### `ca crop` — Bounding box crop

```bash
ca crop cloud.pcd -o cropped.pcd \
  --x-min 0 --y-min 0 --z-min 0 \
  --x-max 10 --y-max 10 --z-max 5
```

### `ca version`

```bash
ca version
```

## Common Options

All commands (except `view` and `version`) support `--output-json <path>` to dump the result as a JSON file.

## Supported Formats

- `.pcd` (Point Cloud Data)
- `.ply` (Polygon File Format)
- `.las` (LiDAR)
