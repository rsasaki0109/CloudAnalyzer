# CloudAnalyzer

[![Test](https://github.com/rsasaki0109/CloudAnalyzer/actions/workflows/test.yml/badge.svg)](https://github.com/rsasaki0109/CloudAnalyzer/actions/workflows/test.yml)
[![Self QA](https://github.com/rsasaki0109/CloudAnalyzer/actions/workflows/self-qa.yml/badge.svg)](https://github.com/rsasaki0109/CloudAnalyzer/actions/workflows/self-qa.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Fix your SLAM map, clean it, measure it. In the browser.**

CloudAnalyzer is a point cloud viewer and analyzer that runs in your browser: open a drive's poses and scans,
close its loops, take out the cars that drove past, and compare the result with last week's map or the ground truth.
Everything runs locally in Rust compiled to WebAssembly, on every core: nothing to install, and your data
never leaves your machine.

<p align="center">
  <a href="https://rsasaki0109.github.io/CloudAnalyzer/app/?demo=posegraph-drive"><img src="docs/images/web/loop.gif" alt="A loop closed by hand: the drifted second lap glides onto the first" width="720"></a>
</p>

<p align="center">
  <b><a href="https://rsasaki0109.github.io/CloudAnalyzer/app/">Open the app</a></b> ·
  <a href="https://rsasaki0109.github.io/CloudAnalyzer/app/?demo=posegraph">SLAM demo</a> ·
  <a href="https://rsasaki0109.github.io/CloudAnalyzer/guide.html">Guide</a> ·
  <a href="#proven-on-real-data">Results on KITTI</a> ·
  <a href="#ca-the-ci-command-line">CLI</a>
</p>

## 1. Fix the map

Drop a folder with a trajectory (KITTI, TUM) or a g2o pose graph and one scan per pose.

<p align="center">
  <a href="scripts/fetch_pandaset.py"><img src="docs/images/web/odometry.gif" alt="A real drive through San Francisco replayed: Pandar64 scans build up along the street, colored by height" width="49%"></a>
  <a href="https://rsasaki0109.github.io/CloudAnalyzer/app/?demo=posegraph"><img src="docs/images/web/posegraph.jpg" alt="The corrected map colored by how far each point moved" width="49%"></a><br>
  <b>Replay the drive</b> scan by scan (PandaSet, San Francisco) · <b>See the correction</b>: every point colored by how far it moved
</p>

- **Close loops** between two keyframes you pick, or let it search the whole drive: candidates are found within
  the drift the odometry could have built up, registered with ICP on the worker pool, and kept only when their
  structure (not just the road) overlaps.
- **Level it** with the IMU's gravity (KITTI OXTS or a plain `index ux uy uz` file) or a floor plane.
- **Drag a keyframe** with a gizmo, fix it, undo anything; edges colored by their error show what disagrees.
- **Join sessions**: a second drive through the same streets is attached where they meet and tied in with loops.
- **Export** the graph as g2o, the poses as KITTI / TUM, and the map as a cloud.

## 2. Clean it and see what changed

<p align="center">
  <a href="scripts/fetch_pandaset.py"><img src="docs/images/web/dynamic.gif" alt="A real San Francisco street: the ghost trails the traffic left in the lanes turn red and leave the map" width="720"></a><br>
  A real street (PandaSet scene 019, 80 Pandar64 scans): the traffic's ghost trails turn red and leave the map
</p>

- **Remove dynamic objects**: points that nearby scans saw straight through (passing cars, pedestrians) leave the
  map as a red cloud of their own; parked cars, seen again from every side, stay.
- **Compare sessions or passes** with M3C2 and get the list of changed objects: a car that left, a new container.
- **Score a map against ground truth**: accuracy, completeness, F1, Chamfer, and the Wasserstein distance between
  voxel Gaussians (AWD / SCS, as in MapEval), which tells a shifted map from a bent one.
- **Ground and terrain**: extract the ground (CSF) and rasterize a DEM of the corrected map.

## 3. Measure any point cloud

The same app is a full point cloud workbench for survey, construction and mapping data.

<table>
  <tr>
    <td width="33%"><a href="https://rsasaki0109.github.io/CloudAnalyzer/app/?demo=c2c"><img src="docs/images/web/c2c.jpg" alt="Cloud-to-cloud distance"></a><br><b>C2C / C2M distance</b></td>
    <td width="33%"><a href="https://rsasaki0109.github.io/CloudAnalyzer/app/?demo=m3c2"><img src="docs/images/web/m3c2.jpg" alt="M3C2 change detection"></a><br><b>M3C2</b> change detection</td>
    <td width="33%"><a href="https://rsasaki0109.github.io/CloudAnalyzer/app/?demo=volume"><img src="docs/images/web/volume.jpg" alt="Cut and fill volume"></a><br><b>Cut / fill volume</b></td>
  </tr>
  <tr>
    <td><a href="https://rsasaki0109.github.io/CloudAnalyzer/app/?demo=ground"><img src="docs/images/web/ground.jpg" alt="Ground extraction"></a><br><b>Ground extraction</b> (CSF)</td>
    <td><img src="docs/images/web/profile.jpg" alt="Cross-section profile"><br><b>Cross-sections</b> along a polyline</td>
    <td><img src="docs/images/web/shapes.jpg" alt="RANSAC shape detection"><br><b>RANSAC shapes</b> and clustering</td>
  </tr>
</table>

Also: ICP and manual alignment, subsampling, outlier removal, normals, 2.5D meshing, DEM / DSM to GeoTIFF,
scalar fields with a calculator, lasso segmentation with undo, trajectory ATE / RPE, and an HTML QA report with
pass / fail gates. See the [guide](https://rsasaki0109.github.io/CloudAnalyzer/guide.html) for all of it.

**Opens** PLY, PCD, LAS / LAZ, E57, XYZ, KITTI `.bin`, OBJ / STL, Gaussian Splatting (3DGS PLY, `.splat`) and
COPC from a URL. LAS / LAZ larger than memory open thinned and fill in where you zoom; UTM and ECEF
coordinates stay exact to the millimetre. **Saves** PLY, LAS / LAZ, E57, CSV and session links.

## Proven on real data

Measured in the app on public datasets, with [KISS-ICP](https://github.com/PRBonn/kiss-icp) odometry as the input:

| KITTI trajectory ATE (RMSE) | Odometry | + loops | + IMU gravity |
|---|---|---|---|
| 00 &nbsp; 4,541 scans, 3.7 km | 22.3 m | 11.1 m | **5.8 m** |
| 07 &nbsp; 1,101 scans | 2.14 m | 1.65 m | **1.16 m** |
| 09 &nbsp; 1,591 scans, 38 m of hills | 16.4 m | 6.2 m | **2.3 m** |

| KITTI map against the ground-truth map | Odometry | + loops + IMU gravity |
|---|---|---|
| 07 &nbsp; F1 at 0.3 m | 0.45 | **0.70** |
| 09 &nbsp; F1 at 0.3 m | 0.16 | **0.42** |

- **Dynamic objects** on SemanticKITTI 07: F1 0.67 against the moving-object labels, 99.9 % of static points kept,
  1,101 scans in 8 s.
- **Across days**: KITTI 00 (Oct 3) joined with 07 (Sep 30) through 202 loops; the buildings agree to a mean of
  2 mm, and the changed objects are the parked cars.

## Try it

| Demo | |
|---|---|
| [SLAM loop closure](https://rsasaki0109.github.io/CloudAnalyzer/app/?demo=posegraph) | A drifting drive round a block: loops found, gravity tied, correction shown |
| [SLAM drive](https://rsasaki0109.github.io/CloudAnalyzer/app/?demo=posegraph-drive) | The same drive to fix yourself: replay, close a loop, remove cars |
| [Two LiDAR scans](https://rsasaki0109.github.io/CloudAnalyzer/app/?demo=c2c) | Cloud-to-cloud distance |
| [Landslide](https://rsasaki0109.github.io/CloudAnalyzer/app/?demo=m3c2) | M3C2 change detection |
| [Stockpile](https://rsasaki0109.github.io/CloudAnalyzer/app/?demo=volume) | Cut / fill volume |
| [Town](https://rsasaki0109.github.io/CloudAnalyzer/app/?demo=ground) | Ground extraction |

Or drop your own files on the [app](https://rsasaki0109.github.io/CloudAnalyzer/app/), or open one from a URL with
`?url=https://…/cloud.laz`.

## `ca`, the CI command line

`ca` turns SLAM, LiDAR, perception and 3DGS outputs into metrics, HTML reports and pass / fail gates for CI, on
the same Rust core.

```bash
pip install cloudanalyzer
ca evaluate candidate.pcd reference.pcd
```

Start with the [SLAM benchmark tutorial](docs/tutorial-slam-benchmark.md), then the
[command reference](docs/commands/), [CI and quality gates](docs/ci.md) and the
[SLAM leaderboard](https://rsasaki0109.github.io/CloudAnalyzer/leaderboard/).

## What's inside

| Path | |
|---|---|
| [`web/`](web/) | The browser app (TypeScript, three.js, a pool of WebAssembly workers) |
| [`rust/`](rust/) | The Rust core: registration, pose graph optimisation, M3C2, visibility, octrees, I/O; WebAssembly and Python bindings |
| [`cloudanalyzer/`](cloudanalyzer/) | `ca`, the Python CLI for quality gates in CI |

## Develop

```sh
cd web
npm install
npm run wasm   # build the Rust core to WebAssembly
npm run dev    # http://localhost:5173
```

Tests: `cargo test` in `rust/`, `npx playwright test` in `web/`. `npm run media` (after `npm run build`)
re-takes the README screenshots and GIFs.

## License

[MIT](LICENSE). Public demo data, sample data and derived images keep their upstream terms; see the
[image attribution](docs/images/ATTRIBUTION.md) and the [sample attribution](web/public/samples/ATTRIBUTION.md).
The odometry and dynamic-object GIFs are of [PandaSet](https://pandaset.org) scene 019 (Scale AI and Hesai,
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)), fetched with
[`scripts/fetch_pandaset.py`](scripts/fetch_pandaset.py); the other pose graph pictures are of a drive generated in the
browser. KITTI data is not redistributed here.
