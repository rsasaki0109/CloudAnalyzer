# Visualization Commands

## ca view

Open an interactive 3D viewer. Supports multiple files.

```bash
# Single file
ca view cloud.pcd

# Multiple files (each gets a distinct color)
ca view scan1.pcd scan2.pcd scan3.pcd
```

## ca density-map

Generate a 2D density heatmap by projecting points onto a plane.

```bash
ca density-map cloud.pcd -o density.png -r 1.0 -a z
```

| Option | Default | Description |
|---|---|---|
| `-o`, `--output` | *required* | Output image path (png) |
| `-r`, `--resolution` | `0.5` | Grid cell size |
| `-a`, `--axis` | `z` | Projection axis: `x`, `y`, or `z` |

### Projection Axes

- `z` (default): Bird's eye view (X-Y plane)
- `y`: Front view (X-Z plane)
- `x`: Side view (Y-Z plane)

## ca heatmap3d

Render source point cloud colored by distance to a reference, saved as a snapshot image.

```bash
ca heatmap3d estimated.pcd reference.pcd -o heatmap.png
```

Colors: blue = close, red = far (jet colormap).
