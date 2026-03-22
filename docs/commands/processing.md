# Processing Commands

## ca downsample

Reduce point count using voxel grid filtering.

```bash
ca downsample cloud.pcd -o down.pcd -v 0.05
```

| Option | Default | Description |
|---|---|---|
| `-o`, `--output` | *required* | Output file path |
| `-v`, `--voxel-size` | `0.05` | Voxel grid cell size |

## ca sample

Randomly sample a fixed number of points.

```bash
ca sample cloud.pcd -o sampled.pcd -n 10000
```

| Option | Default | Description |
|---|---|---|
| `-o`, `--output` | *required* | Output file path |
| `-n`, `--num` | *required* | Number of points to keep |

## ca filter

Remove statistical outliers (Statistical Outlier Removal).

```bash
ca filter cloud.pcd -o clean.pcd -n 20 -s 2.0
```

| Option | Default | Description |
|---|---|---|
| `-o`, `--output` | *required* | Output file path |
| `-n`, `--neighbors` | `20` | Number of neighbors for mean distance |
| `-s`, `--std-ratio` | `2.0` | Std deviation multiplier threshold |

## ca merge

Merge multiple point clouds into one file.

```bash
ca merge a.pcd b.pcd c.pcd -o merged.pcd
```

## ca align

Align multiple point clouds sequentially using registration, then merge.

```bash
ca align scan1.pcd scan2.pcd scan3.pcd -o aligned.pcd -m gicp
```

| Option | Default | Description |
|---|---|---|
| `-o`, `--output` | *required* | Output file path |
| `-m`, `--method` | `gicp` | Registration method: `icp` or `gicp` |
| `--max-dist` | `1.0` | Max correspondence distance |

## ca split

Split a point cloud into grid tiles.

```bash
ca split large_map.pcd -o tiles/ -g 100
```

| Option | Default | Description |
|---|---|---|
| `-o`, `--output-dir` | *required* | Output directory |
| `-g`, `--grid-size` | *required* | Grid cell size (meters) |
| `-a`, `--axis` | `xy` | Split axes: `xy`, `xz`, or `yz` |

## ca convert

Convert between point cloud formats.

```bash
ca convert input.pcd output.ply
```

Supported: `.pcd`, `.ply`, `.las`

## ca normals

Estimate normals and save to file.

```bash
ca normals cloud.pcd -o with_normals.ply -r 0.5
```

| Option | Default | Description |
|---|---|---|
| `-o`, `--output` | *required* | Output file path |
| `-r`, `--radius` | `0.5` | Search radius |
| `--max-nn` | `30` | Max neighbors |

## ca crop

Crop point cloud to an axis-aligned bounding box.

```bash
ca crop cloud.pcd -o cropped.pcd \
  --x-min 0 --y-min 0 --z-min 0 \
  --x-max 10 --y-max 10 --z-max 5
```
