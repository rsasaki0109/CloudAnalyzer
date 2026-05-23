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
  resolved against the list file).

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
| `--driver kiss-icp\|kiss-slam\|small-gicp` | SLAM driver to use. `kiss-icp` (default, adopted) is the upstream KISS-ICP scan-to-map LiDAR odometry. `kiss-slam` adds pose-graph optimization and MapClosures-based loop closures on top of the same odometry. `small-gicp` is pure scan-to-scan GICP via the `small_gicp` library (no local map; faster per frame but drifts more on long sequences). All three are experimental except `kiss-icp`, see [What's adopted vs. what's experimental](#whats-adopted-vs-whats-experimental). |
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
`--driver small-gicp` output recovers the trajectory to sub-cm ATE
but its scan-stitched map (genuine scan-to-scan registration, no
local map) falls slightly under the default AUC ≥ 0.95 / Chamfer ≤
0.05 gate (typically AUC≈0.92, Chamfer≈0.10) — a visible artifact of
the scan-to-map vs. scan-to-scan design choice.

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
  **scan-to-scan** GICP — no local map, no constant-velocity prediction.
  Faster per frame but drift accumulates unbounded; on the figure-8
  dogfood case it has ~10× the ATE of `kiss-icp`. It stays in
  experiments because the different speed / accuracy operating point is
  worth keeping visible — a future low-latency use case may prefer it
  over `kiss-icp`.
- `IdentityPassthroughSlamDriver` (under
  `ca.experiments.slam_run.identity_passthrough`) is a sentinel — always
  returns identity poses and concatenates the raw input frames as the
  "map". Useful failure floor for the slice's evaluator (any real driver
  should beat it by a wide margin on a curved trajectory). Not exposed
  through the CLI.

## Related

- `ca traj-evaluate` — score just the trajectory against a reference TUM.
- `ca evaluate` / `ca map-evaluate` — score just the map against a reference cloud.
- `ca run-evaluate` — score both map + trajectory together.
- `ca benchmark eval` — wrap one `ca run-evaluate` call as a benchmark
  suite step (frozen reference + gate).
