# `ca slam-run`

Drive a LiDAR-odometry pipeline end-to-end on a sequence of input scans and
emit the artifacts that the rest of CloudAnalyzer's evaluation stack
(`ca run-evaluate`, `ca check`, `ca history`) already consumes.

This turns CloudAnalyzer from a "you bring the output, I'll score it" tool
into a "give me your scans, I'll run a SLAM and score it for you" tool. The
default driver wraps [KISS-ICP](https://github.com/PRBonn/kiss-icp), a small
scan-to-map LiDAR-odometry registration pipeline available on PyPI under
BSD.

## Install

```bash
pip install 'cloudanalyzer[slam]'
# or, if cloudanalyzer is already installed:
pip install kiss-icp kiss-slam small_gicp
```

The drivers are loaded lazily, so `ca` itself works without the SLAM
packages installed — `ca slam-run` only raises if the extra is missing
when you actually request that driver.

## Usage

```bash
ca slam-run <input> <output_dir> [--driver kiss-icp] [options]
```

`<input>` is either:

- A directory of LiDAR scans named so they sort into temporal order
  (`*.bin` KITTI Velodyne / `*.pcd` / `*.ply`); the first matching
  extension wins. Or
- A frames-list `.txt` with one scan path per line (relative paths are
  resolved against the list file). Or
- A ROS bag recording (`.bag` / `.mcap` / `.db3`) with
  `sensor_msgs/PointCloud2` scans — requires `pip install "cloudanalyzer[ros]"`.
  See [bag-ingest.md](bag-ingest.md).

`<output_dir>` will receive three files:

- `trajectory.tum` — estimated sensor poses (TUM format).
  Consumable by `ca traj-evaluate` and `ca run-evaluate`.
- `map.ply` — accumulated world-frame map. Consumable by `ca evaluate`
  / `ca map-evaluate` / `ca check`.
- `summary.json` — driver name, runtime, frame count, the driver config
  snapshot, and (when `--evaluate` is set) the nested `ca run-evaluate`
  block.

## Common options

| Option | Description |
|---|---|
| `--driver kiss-icp\|kiss-slam\|small-gicp` | SLAM driver to use. `kiss-icp` (default, adopted) is the upstream KISS-ICP scan-to-map LiDAR odometry. `kiss-slam` adds pose-graph optimization and MapClosures-based loop closures on top of the same odometry. `small-gicp` is scan-to-map VGICP via the `small_gicp` library's GaussianVoxelMap. All three pass the synthetic-figure8 gate; all but `kiss-icp` stay in experiments — see [What's adopted vs. what's experimental](#whats-adopted-vs-whats-experimental). |
| `--max-range 80` | Drop scan points farther than this from the sensor (meters). |
| `--voxel-size 0.5` | Local-map voxel grid (meters). Driver default kept if omitted. |
| `--deskew` | Enable KISS-ICP motion-deskew. Default off because `.bin/.pcd` dumps don't typically carry per-point timestamps. |
| `--max-frames N` | Cap how many frames are consumed. |
| `--frame-period 0.1` | Fallback per-frame time spacing in seconds. |
| `--evaluate` | After driving the SLAM, score the result against `--reference-map` and `--reference-trajectory` using `ca run-evaluate`. |
| `--format-json` | Print the summary to stdout as JSON (artifacts still land in `<output_dir>`). |

## Examples

### Drive KISS-ICP on a folder of KITTI Velodyne sweeps

```bash
ca slam-run /data/kitti/00/velodyne runs/seq00 \
    --driver kiss-icp \
    --max-range 80 \
    --voxel-size 0.5
```

Reads `*.bin` in lex order, drops the intensity column, and writes
`runs/seq00/{trajectory.tum,map.ply,summary.json}`.

### Drive + evaluate against a reference map / trajectory

```bash
ca slam-run scans/ runs/seq01 \
    --max-range 80 \
    --evaluate \
    --reference-map     ref/map.pcd \
    --reference-trajectory ref/poses.tum
```

The nested `evaluate` block in `summary.json` carries the same
Chamfer / AUC / ATE / RPE numbers as a standalone `ca run-evaluate` call,
so it slots straight into the existing `ca check` and
`ca report-pr-comment` flows.

### Drive on a frames-list

```bash
cat > frames.txt <<'EOF'
sweeps/00.pcd
sweeps/05.pcd
sweeps/10.pcd
EOF
ca slam-run frames.txt runs/sub --driver kiss-icp
```

Useful when you want to subsample a long sequence without renaming files.

### End-to-end dogfood on the bundled synthetic-figure8 suite

The `benchmarks/slam/synthetic-figure8/` fixture ships with raw scans
under `scans/`, so the full "scans → SLAM → benchmark eval" loop
reproduces from a clean checkout (no BYO data needed):

```bash
ca slam-run benchmarks/slam/synthetic-figure8/scans /tmp/figure8_run \
    --driver kiss-icp \
    --max-range 25 \
    --voxel-size 0.5 \
    --frame-period 0.1

ca benchmark eval benchmarks/slam/synthetic-figure8/suite.yaml \
    --map        /tmp/figure8_run/map.ply \
    --trajectory /tmp/figure8_run/trajectory.tum \
    --sequence default
```

On a clean run this passes the suite's default gate
(AUC=1.00, ATE≈1.6 mm, Coverage=100%). It is exercised in CI by
`tests/test_slam_run.py::test_cli_slam_run_then_benchmark_eval_passes_synthetic_figure8`
when the optional `[slam]` extra is installed. The suite's sensor
heading is tangent to the figure-8 trajectory (vehicle-style), so it
exercises both translation and rotation recovery. The `--driver
kiss-slam` output also passes the gate (its map is pulled from
kiss-icp's own multi-point-per-voxel local map representation),
verified by
`test_cli_slam_run_kiss_slam_passes_synthetic_figure8_gate`. The
`--driver small-gicp` output also passes the gate after the Phase 27
upgrade to scan-to-map VGICP (typically AUC≈0.99, Chamfer≈0.025,
ATE≈2 mm), verified by
`test_cli_slam_run_small_gicp_passes_synthetic_figure8_gate`. All
three real drivers now clear the synthetic-figure8 gate.

## What's adopted vs. what's experimental

- `KissICPSlamDriver` is the adopted driver — re-exported from
  `ca.core.slam_run`. The CLI imports only from `ca.core` for this one.
- `KissSLAMSlamDriver` (wraps the `kiss-slam` package: KISS-ICP odometry
  + pose-graph optimization + MapClosures loop closure) is in
  `ca.experiments.slam_run`. The CLI exposes it as `--driver kiss-slam`
  via a per-name lazy import — keeping the import path in `experiments`
  reflects that it is not the adopted core driver. On the short synthetic
  trajectories the slice's evaluator ships, KISS-SLAM degenerates to one
  round of pose-graph optimization over the KISS-ICP odometry chain (no
  loop closures fire because the sensor never travels far enough to
  cross the local-map splitting distance), so it produces effectively
  the same trajectory KISS-ICP does. Promotion to core is blocked on
  real-drift / revisit data that exercises the loop-closure path.
- `SmallGICPSlamDriver` (wraps `small_gicp`, MIT) is also in
  `ca.experiments.slam_run`. Exposed as `--driver small-gicp`. Does
  **scan-to-map VGICP** using `small_gicp.GaussianVoxelMap` as the
  registration target, with constant-velocity prediction for the
  initial guess. World-frame map is scan-stitched + voxel-downsampled
  (the voxel map's quantized centers don't satisfy the AUC / Chamfer
  gate on their own). After the Phase 27 upgrade from scan-to-scan it
  also passes the synthetic-figure8 default gate.
- `IdentityPassthroughSlamDriver` (under
  `ca.experiments.slam_run.identity_passthrough`) is a sentinel — always
  returns identity poses and concatenates the raw input frames as the
  "map". Useful failure floor for the slice's evaluator (any real driver
  should beat it by a wide margin on a curved trajectory). Not exposed
  through the CLI.

## Adding a third-party SLAM driver

External packages can publish their own driver under
`--driver <your-name>` without touching CloudAnalyzer itself. The CLI
resolves `--driver` through a registry in `ca.core.slam_run`; the
registry seeds the three built-in drivers at import time and folds in
anything published under the `cloudanalyzer.slam_run_drivers`
entry-point group on first lookup.

### 1. Implement the contract

Your driver needs to satisfy the `SlamRunDriver` Protocol —
a `name` attribute (the driver's *internal* string, snake_case by
convention) and a `run(request: SlamRunRequest) -> SlamRunResult`
method. Both types live in `ca.core.slam_run`.

```python
# my_slam_pkg/my_slam_driver.py
import numpy as np
from ca.core.slam_run import SlamRunRequest, SlamRunResult


class MySlamDriver:
    name = "my_slam"

    def run(self, request: SlamRunRequest) -> SlamRunResult:
        # ... your registration loop ...
        return SlamRunResult(
            driver=self.name,
            poses=poses,            # (N, 4, 4) ndarray
            timestamps_s=ts,        # (N,) ndarray
            map_points=map_world,   # (M, 3) ndarray
            runtime_s=runtime,
            frames_processed=n,
            metadata={"my_slam": {...}},
        )
```

### 2. Register via entry-point

Add this to your package's `pyproject.toml`:

```toml
[project.entry-points."cloudanalyzer.slam_run_drivers"]
my-slam = "my_slam_pkg.my_slam_driver:MySlamDriver"
```

The entry-point's left-hand side (`my-slam`) is the CLI string —
`ca slam-run --driver my-slam` after a `pip install my_slam_pkg`. The
right-hand side may resolve to either a callable factory or directly
to your driver class (we accept both).

### 3. Verify

```bash
pip install -e ./my_slam_pkg
ca slam-run scans/ runs/ --driver my-slam
```

The registry also accepts in-process registration via
`ca.core.slam_run.register_driver(name, factory)` — useful for
unit-testing your driver without going through `pip install`.

### Canonical working example

The repo ships a complete worked example under
[`plugins/cloudanalyzer-driver-example/`](../../plugins/cloudanalyzer-driver-example/):
a scan-to-scan point-to-point ICP driver via Open3D, depending only on
what CloudAnalyzer already requires (no `kiss-icp` / `kiss-slam` /
`small_gicp` extras). CI installs it in editable mode and exercises
`ca slam-run --driver example` end-to-end against the bundled
`synthetic-figure8` scans, so the entry-point pathway is locked in
against regressions.

## Related

- `ca traj-evaluate` — score just the trajectory against a reference TUM.
- `ca evaluate` / `ca map-evaluate` — score just the map against a reference cloud.
- `ca run-evaluate` — score both map + trajectory together.
- `ca benchmark eval` — wrap one `ca run-evaluate` call as a benchmark
  suite step (frozen reference + gate).
