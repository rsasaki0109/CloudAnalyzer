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

## ca batch

Run info on all point cloud files in a directory.

```bash
# Current directory
ca batch /path/to/pcds/

# Recursive scan
ca batch /path/to/pcds/ -r

# JSON output
ca batch /path/to/pcds/ --format-json | jq '.[].num_points'
```

| Option | Default | Description |
|---|---|---|
| `-r`, `--recursive` | `false` | Scan subdirectories |
