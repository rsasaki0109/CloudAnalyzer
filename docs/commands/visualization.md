# Visualization Commands

## ca web

Open a browser-based Three.js viewer.

```bash
# Single file
ca web cloud.pcd

# Merge multiple files into one browser view
ca web scan1.pcd scan2.pcd scan3.pcd

# Heatmap mode: color source by distance to reference, overlay the reference cloud,
# and filter source points by error threshold in the browser
ca web estimated.pcd reference.pcd --heatmap

# Run viewer: overlay estimated/reference trajectories on top of the map heatmap
ca web map.pcd map_ref.pcd --heatmap \
  --trajectory traj.csv --trajectory-reference traj_ref.csv

# Apply trajectory alignment before display
ca web map.pcd map_ref.pcd --heatmap \
  --trajectory traj.csv --trajectory-reference traj_ref.csv \
  --trajectory-align-rigid
```

| Option | Default | Description |
|---|---|---|
| `-p`, `--port` | `8080` | HTTP port |
| `--max-points` | `2000000` | Maximum points displayed in browser |
| `--heatmap` | `false` | With 2 files, color the first by distance to the second, overlay the reference cloud, and enable threshold filtering |
| `--trajectory` | `None` | Estimated trajectory to overlay in the browser |
| `--trajectory-reference` | `None` | Reference trajectory to overlay alongside `--trajectory` |
| `--trajectory-max-time-delta` | `0.05` | Max timestamp gap used when matching `--trajectory` to `--trajectory-reference` |
| `--trajectory-align-origin` | `false` | Translate `--trajectory` so its first matched pose aligns to `--trajectory-reference` |
| `--trajectory-align-rigid` | `false` | Fit a rigid transform from `--trajectory` to `--trajectory-reference` before display |
| `--no-browser` | `false` | Don't auto-open the browser |

When trajectories are provided, long overlays are simplified for browser display through `ca.core.web_trajectory_sampling`, and the viewer reports displayed/original pose counts plus the active sampling strategy.

Large point clouds are also progressively loaded through `ca.core.web_progressive_loading`: the viewer opens with an initial subset, then appends deferred chunks in the background while keeping point picking and heatmap thresholding active.

When paired trajectories are provided, the viewer shows estimated/reference trajectory overlays, preserves inspection-critical anchors while simplifying, marks the worst ATE pose, highlights the worst RPE segment, renders a linked ATE/RPE error timeline, lets you click either the 3D overlays or timeline points to inspect timestamps and error summaries, and auto-focuses the camera on the selected pose/segment. Use `Reset View` to return to the full-scene view.

The viewer also supports point picking. Click a source point to inspect its XYZ coordinates, and in `--heatmap` mode the distance-to-reference value is shown as well. When a reference overlay is visible, the nearest displayed reference point is highlighted and its offset is shown too. Reference points can also be clicked directly.

## ca web-export

Write the same Three.js viewer as a static bundle for GitHub Pages or any other static host.

```bash
# Export a static heatmap bundle
ca web-export estimated.pcd reference.pcd \
  --heatmap \
  --output-dir docs/demo/local

# Export with trajectory overlays
ca web-export map.pcd map_ref.pcd \
  --heatmap \
  --trajectory traj.csv --trajectory-reference traj_ref.csv \
  --output-dir docs/demo/run-viewer
```

`ca web-export` shares the same visualization options as `ca web`, but writes `index.html`, `data.json`, and progressive chunk payloads into the target directory instead of starting a local server. The generated bundle uses relative asset paths, so it can be served from a repository subpath such as `https://<user>.github.io/<repo>/demo/...`.

## ca lidar-odometry-view

Open a LiDAR odometry trajectory directly in the browser. A map is optional, so this also works for odometry-only runs.

```bash
# Trajectory only
ca lidar-odometry-view trajectory.csv

# Trajectory over a map
ca lidar-odometry-view trajectory.csv --map map.pcd

# Estimated/reference trajectory comparison
ca lidar-odometry-view trajectory.csv \
  --map map.pcd \
  --slam-debug-report slam_debug_report.json \
  --trajectory-reference reference.tum \
  --trajectory-align-rigid
```

| Option | Default | Description |
|---|---|---|
| `--map`, `-m` | `None` | Optional map/point-cloud file behind the odometry trajectory. Repeatable |
| `--trajectory-reference` | `None` | Optional reference trajectory overlay |
| `--trajectory-max-time-delta` | `0.05` | Max timestamp gap used when matching to `--trajectory-reference` |
| `--trajectory-align-origin` | `false` | Translate the odometry trajectory so its first matched pose aligns to the reference |
| `--trajectory-align-rigid` | `false` | Fit a rigid transform from odometry trajectory to reference before display |
| `--slam-debug-report` | `None` | Optional `ca slam-debug --output-json` report; selected frames are marked on the trajectory |
| `-p`, `--port` | `8080` | HTTP port |
| `--max-points` | `2000000` | Maximum map points displayed in browser |
| `--no-browser` | `false` | Don't auto-open the browser |

## ca lidar-odometry-export

Write the LiDAR odometry viewer as a static bundle.

```bash
ca lidar-odometry-export trajectory.csv \
  --map map.pcd \
  --slam-debug-report slam_debug_report.json \
  --output-dir odometry_viewer
```

The exported directory contains `index.html` and `data.json`, so it can be opened by any static web server. When a SLAM debug report is supplied, suspicious frames are displayed as clickable trajectory markers with scan id, score, diagnosis, suggested action, and artifact paths. Existing SLAM debug artifacts referenced by the report are copied under `slam_debug_artifacts/` and linked from the marker inspection panel.

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
