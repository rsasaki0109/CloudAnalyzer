# CloudAnalyzer

[![Test](https://github.com/rsasaki0109/CloudAnalyzer/actions/workflows/test.yml/badge.svg)](https://github.com/rsasaki0109/CloudAnalyzer/actions/workflows/test.yml)
[![Self QA](https://github.com/rsasaki0109/CloudAnalyzer/actions/workflows/self-qa.yml/badge.svg)](https://github.com/rsasaki0109/CloudAnalyzer/actions/workflows/self-qa.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Fix your SLAM map, clean it, measure it. In the browser.**

CloudAnalyzer is a point cloud viewer and analyzer that runs in your browser: open a ROS bag or a drive's poses
and scans, close its loops, take out the cars that drove past, and compare the result with last month's map or
the ground truth. Everything runs locally in Rust compiled to WebAssembly, on every core: nothing to install, and
your data never leaves your machine.

<p align="center">
  <a href="scripts/prepare_nclt.py"><img src="docs/images/web/loop.gif" alt="A loop closed by hand on a real campus drive: two scans of the same place, metres apart, lined up by ICP, then the whole drive pulled together" width="720"></a><br>
  A loop closed by hand on a real drive (NCLT, University of Michigan): the same place seen twice a kilometre apart,
  lined up with ICP, and the drive pulled together
</p>

<p align="center">
  <b><a href="https://rsasaki0109.github.io/CloudAnalyzer/app/">Open the app</a></b> ·
  <a href="https://rsasaki0109.github.io/CloudAnalyzer/app/?demo=nclt">A real drive from a ROS bag</a> ·
  <a href="https://rsasaki0109.github.io/CloudAnalyzer/guide.html">Guide</a> ·
  <a href="#proven-on-real-data">Results on KITTI</a> ·
  <a href="#ca-the-ci-command-line">CLI</a>
</p>

## 1. Fix the map

Drop a recording as it is, a ROS 1 `.bag` or a ROS 2 `.mcap` (gigabytes are fine: it is read a slice at a time),
or a folder of scans without poses: LiDAR odometry places the scans in the browser, on the worker pool, and the
bag's IMU levels the result, its mounting estimated from the drive. Or drop a trajectory (KITTI, TUM) or a g2o
pose graph with one scan per pose.

<p align="center">
  <a href="scripts/fetch_pandaset.py"><img src="docs/images/web/odometry.gif" alt="A real drive through San Francisco replayed: Pandar64 scans build up along the street, colored by height" width="49%"></a>
  <a href="scripts/prepare_nclt.py"><img src="docs/images/web/posegraph.jpg" alt="A campus map after automatic loop closure, colored by how far each point moved" width="49%"></a><br>
  <b>Replay the drive</b> scan by scan (PandaSet, San Francisco) · <b>See the correction</b>: every point colored by how far it moved
</p>

- **Close loops** between two keyframes you pick, or let it search the whole drive: candidates are found within
  the drift the odometry could have built up, registered with ICP on the worker pool, and kept only when their
  structure (not just the road) overlaps.
- **Level it** with the IMU's gravity (KITTI OXTS or a plain `index ux uy uz` file) or a floor plane.
- **Drag a keyframe** with a gizmo, fix it, undo anything; edges colored by their error show what disagrees.
- **Join sessions**: a second drive through the same streets, as a folder or a bag, is attached where they meet
  and tied in with loops.
- **Export** the graph as g2o, the poses as KITTI / TUM, and the map as a cloud.
- **Build a vector map** over the point cloud: draft boundaries from a cloud and a trajectory with measured/inferred counts, draw roads with lanes in either direction, connect junctions,
  drag shared boundary vertices, add stop lines, traffic lights and crosswalks, and edit speed limits. The Lanelet2 panel uses
  [vectormap-rs](https://github.com/rsasaki0109/vectormap-rs), checks the map for Autoware, and saves
  `lanelet2_map.osm` with `map_projector_info.yaml`. Existing `.osm` maps can be opened and edited.
  [Map display](docs/vector-map-display.md) shows road surfaces, direction arrows, clipped crosswalk
  bands and signal faces; selecting a lane highlights its incoming and outgoing connections.
  [Automatic equipment search](docs/commands/vectormap-discover.md) finds paint and elevated
  panel proposals without feature boxes; inspect them and confirm their type and lane IDs.
  [Feature geometry editing](docs/vector-map-feature-editing.md) adjusts crossings, stop lines
  and signal housings with complete-map Undo while retaining observed paint and lamps.
  [Source quality checks](docs/vector-map-quality.md) expose lanes whose centres or boundaries
  lack nearby ground support; passing format validation does not establish map accuracy.
  [Source-footprint drafting](docs/vector-map-source-footprint.md) fits weak road candidates
  to supported low surfaces and reports missing extent explicitly, with actual before/after maps.
  [Physical boundary anchors](docs/vector-map-physical-anchors.md) optionally keep scan limits
  from shifting inferred lanes, with fixed-interval comparisons that expose gains and regressions.

To add a signal head from points, [measure an identified box](docs/commands/vectormap-signal.md)
and confirm its controlled lanes. For COPC surveys,
[read a full-density working box](docs/large-point-clouds.md#web-full-density-working-box) first.

<p align="center">
  <a href="docs/vector-map-media.md"><img src="docs/images/web/vector-map-hard-intersection.gif" alt="Original Tokyo points and recorded arterial poses with operator-traced branches generate a complex road draft, measured crossing and stop-marking candidates and signal housings; review, edit, Undo and reopen Lanelet2" width="800"></a><br>
  <b>Build a complex intersection from points. Review, edit, undo, export.</b><br>
  49 road lanes including connection drafts · 7 paint crossings · 2 stop-marking drafts · 4 signal housings.<br>
  No input map: recorded arterial segments and traced branches; lane widths, object types and lane links are operator inputs.<br>
  <a href="docs/vector-map-hard-intersection.md">Fixed source-only evaluation and limitations</a> ·
  <a href="docs/vector-map-media.md">Reproduce this GIF</a>
</p>

<details>
  <summary>Road boundaries drafted directly from real LiDAR and recorded poses</summary>
  <p align="center">
    <img src="docs/images/web/vector-map.gif" alt="Real PandaSet LiDAR and recorded poses become a continuous lane draft; a shared boundary is edited, undone and saved" width="800"><br>
    PandaSet scene 019: generated boundaries remain drafts requiring manual review.
  </p>
</details>

<details>
  <summary>Edit an existing surveyed intersection and draft its connections</summary>
  <p align="center">
    <img src="docs/images/web/vector-map-intersection.gif" alt="An imported surveyed intersection gains twelve reviewed connection drafts; shared vertices are edited and undone" width="800"><br>
    Approaches, crossings and signals are imported context. This example demonstrates connection drafting and existing-map editing.
  </p>
</details>

## 2. Clean it and see what changed

<p align="center">
  <a href="scripts/fetch_pandaset.py"><img src="docs/images/web/dynamic.gif" alt="A real San Francisco street: the ghost trails the traffic left in the lanes turn red and leave the map" width="720"></a><br>
  A real street (PandaSet scene 019, 80 Pandar64 scans): the traffic's ghost trails turn red and leave the map
</p>

- **Remove dynamic objects**: points that nearby scans saw straight through (passing cars, pedestrians) leave the
  map as a red cloud of their own; parked cars, seen again from every side, stay.
- **Compare sessions or passes** with M3C2 and get the list of changed objects: a car that left, a new container,
  or a whole season. From two bags of the same block, April and June, in the browser:

<p align="center">
  <a href="https://rsasaki0109.github.io/CloudAnalyzer/app/?demo=nclt-seasons"><img src="docs/images/web/bags.jpg" alt="Two ROS bags of the same campus block, April and June, opened with odometry, joined, and compared: the changes coloured on the map" width="720"></a><br>
  Two ROS 2 bags of the same block (NCLT, April and June 2012) opened in the app: odometry, the IMUs, 46 loops,
  and the changes between the months (<a href="https://rsasaki0109.github.io/CloudAnalyzer/app/?demo=nclt-seasons">try it</a>, 19 MB)
</p>

And the same campus trees half a year apart:

<p align="center">
  <a href="scripts/prepare_nclt.py"><img src="docs/images/web/seasons.gif" alt="The same campus trees in June and in December: full crowns, then bare branches" width="640"></a><br>
  The same campus trees in June and December 2012 (NCLT), two drives joined in the app:
  full crowns, then bare branches
</p>

- **Score a map against ground truth**: accuracy, completeness, F1, Chamfer, and the Wasserstein distance between
  voxel Gaussians (AWD) and its spread (SCS), which tells a shifted map from a bent one.
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

See [large point cloud IO limits](docs/large-point-clouds.md) for HTTP range
requirements, 64-bit counts, and the distinction between display LOD and full-density processing.
[`ca copc-tile`](docs/commands/copc-tile.md) saves full-density COPC tiles with
halo and resumable checkpoints; physical ten-billion-point processing remains unbenchmarked.

## Proven on real data

From recordings, in the app: the built-in odometry, then loops and the IMU.

| NCLT, a block on campus as ROS 2 bags (~250 m, 190 s each) | Odometry | + loops + IMU gravity |
|---|---|---|
| April, trajectory ATE / SE(3)-aligned | 1.25 m / 0.17 m | **1.07 m** / 0.17 m |
| April and June joined, SE(3)-aligned, each drive | | **0.17 m / 0.16 m** |

`hdl_400.bag` (2.4 GB, a Velodyne HDL-32E and a GPS/IMU, 1,263 scans) opens in the browser in about three and a
half minutes on a laptop: 248 keyframes, the IMU's mounting estimated (its up directions spread 8.2° before,
1.8° after) and 6 loops.

With poses from [KISS-ICP](https://github.com/PRBonn/kiss-icp) odometry as the input:

| KITTI trajectory ATE (RMSE) | Odometry | + loops | + IMU gravity |
|---|---|---|---|
| 00 &nbsp; 4,541 scans, 3.7 km | 22.3 m | 11.1 m | **5.8 m** |
| 07 &nbsp; 1,101 scans | 2.14 m | 1.65 m | **1.16 m** |
| 09 &nbsp; 1,591 scans, 38 m of hills | 16.4 m | 6.2 m | **2.3 m** |

| NCLT 2012-04-29 (Segway on campus, 2,777 keyframes, 3.2 km) | Odometry | + loops |
|---|---|---|
| Trajectory ATE, SE(3)-aligned | 4.83 m | **1.64 m** (168 loops found in 11 s) |

| KITTI map against the ground-truth map | Odometry | + loops + IMU gravity |
|---|---|---|
| 07 &nbsp; F1 at 0.3 m | 0.45 | **0.70** |
| 09 &nbsp; F1 at 0.3 m | 0.16 | **0.42** |

- **Dynamic objects** on SemanticKITTI 07: F1 0.67 against the moving-object labels, 99.9 % of static points kept,
  1,101 scans in 8 s.
- **Across days**: KITTI 00 (Oct 3) joined with 07 (Sep 30) through 202 loops; the buildings agree to a mean of
  2 mm, and the changed objects are the parked cars.
- **Across seasons**: NCLT June 2012 joined with December 2012 through 454 loops; 26 % of where they meet changed
  by more than the level of detection, largest first the trees.

## Try it

| Demo | |
|---|---|
| [Real drive from a ROS bag](https://rsasaki0109.github.io/CloudAnalyzer/app/?demo=nclt) | Once round a block on the NCLT campus, as a ROS 2 bag of scans and IMU (10 MB): odometry, gravity and loops |
| [Two seasons of a block](https://rsasaki0109.github.io/CloudAnalyzer/app/?demo=nclt-seasons) | The same block in April and in June, two bags (19 MB): the drives joined, loops between them, the changes between the seasons |
| [SLAM loop closure](https://rsasaki0109.github.io/CloudAnalyzer/app/?demo=posegraph) | A drifting drive round a block: loops found, gravity tied, correction shown |
| [SLAM drive](https://rsasaki0109.github.io/CloudAnalyzer/app/?demo=posegraph-drive) | The same drive to fix yourself: replay, close a loop, remove cars |
| [Two LiDAR scans](https://rsasaki0109.github.io/CloudAnalyzer/app/?demo=c2c) | Cloud-to-cloud distance |
| [Landslide](https://rsasaki0109.github.io/CloudAnalyzer/app/?demo=m3c2) | M3C2 change detection |
| [Stockpile](https://rsasaki0109.github.io/CloudAnalyzer/app/?demo=volume) | Cut / fill volume |
| [Town](https://rsasaki0109.github.io/CloudAnalyzer/app/?demo=ground) | Ground extraction |

Or drop your own files on the [app](https://rsasaki0109.github.io/CloudAnalyzer/app/), or open one from a URL with
`?url=https://…/cloud.laz`.

## `ca`: the command line, for CI and AI agents

`ca` runs the same Rust core natively, on all cores and without the browser's memory limit. It turns SLAM, LiDAR,
perception and 3DGS outputs into metrics, HTML reports and pass / fail gates for CI, and it fixes SLAM maps. It
reads ROS 1 bags, MCAP files, rosbag2 SQLite files and folders itself: no ROS install.

```bash
pip install "cloudanalyzer[fast]"
ca posegraph-fix drive.mcap --out fixed/ --remove-dynamic --keyframe-spacing 1   # 1. a recording in, a fixed map out
ca web-view fixed/                                                            # 2. look at it in the web app
ca posegraph-compare june/ december/ --here 1219 --there 850 --out changes/   # 3. what changed between two drives
```

1. From a ROS bag (or a folder of scans, or poses and scans), [`ca posegraph-fix`](docs/commands/posegraph-fix.md)
   makes the odometry (the same as the web app's), closes the loops, levels the map with the bag's IMU (its
   mounting estimated from the drive), leaves out what moved, and writes the fixed poses, g2o and map, with a JSON
   report.
2. [`ca web-view`](docs/commands/web-view.md) serves the results from your machine and opens them in the app with
   one link, to look at, measure and fix by hand.
3. [`ca posegraph-compare`](docs/commands/posegraph-compare.md) joins two drives through the same places and lists
   the changed objects.

On `hdl_400.bag` step 1 finds the same 6 loops, estimates the IMU's mounting (8.1° → 2.0°) and leaves out 5.1 %
of the points as dynamic in under three minutes on a laptop (the 2.4 GB bag is read in ten seconds); KITTI 07 as a
bag takes 78 s. [`ca mcp`](docs/commands/mcp.md) gives AI agents the same tools over MCP
(`claude mcp add cloudanalyzer -- ca mcp`), from `slam_odometry` to `view_link`.

[`ca vectormap-build`](docs/commands/vectormap-build.md) and the `build_vector_map` MCP tool
draft Autoware Lanelet2 roads from a surveyed cloud and a trajectory, with evidence and
validation reports. Review the draft with `ca web-view map.pcd draft-map` and edit it with
`vectormap mcp draft-map/lanelet2_map.osm`.
[`ca vectormap-connect`](docs/commands/vectormap-connect.md) and its MCP tool preview or
add ground-supported branching connections; the Web panel lets you select candidates
and undo the batch. Review turns, clearance and traffic rules before use.
[`ca vectormap-signal`](docs/commands/vectormap-signal.md) measures a user-identified
signal head from a point-cloud box. Preview its measured housing and explicitly
confirm controlled lanes; the Web panel and MCP tool use the same fitting.
[`ca vectormap-crosswalk`](docs/commands/vectormap-crosswalk.md) proposes measured
ground-paint bands from RGB or intensity. Preview the observed footprint, then
confirm the crossing and its lane IDs; Web additions support Undo and Lanelet2 export.
[`ca vectormap-discover`](docs/commands/vectormap-discover.md) searches generated road
corridors or supported ground surfaces without feature boxes. Review paint/panel
proposals, then explicitly confirm object types and lane assignments; the Web panel
supports inspection, rejection, measured additions, geometry editing and map Undo.
[Equipment association review](docs/vector-map-equipment-relations.md) links vehicle
signals to reviewed stop markings and pedestrian signals to crosswalks, retaining
physical geometry and unresolved controls through Undo and Lanelet2 reload.
[Geometric target suggestions](docs/vector-map-relation-proposals.md) show distance,
orientation and road context, highlight held alternatives, and require explicit adoption.
[Second-scene validation](docs/vector-map-cross-scene.md) reports boundary offsets,
missing equipment and held signal targets alongside a connected-movement fix.

For CI, `ca evaluate candidate.pcd reference.pcd` scores a map against a reference; start with the
[SLAM benchmark tutorial](docs/tutorial-slam-benchmark.md), then the
[command reference](docs/commands/), [CI and quality gates](docs/ci.md) and the
[SLAM leaderboard](https://rsasaki0109.github.io/CloudAnalyzer/leaderboard/).

## What's inside

| Path | |
|---|---|
| [`web/`](web/) | The browser app (TypeScript, three.js, a pool of WebAssembly workers) |
| [`rust/`](rust/) | The Rust core: LiDAR odometry, registration, pose graph optimisation, M3C2, visibility, octrees, ROS bag and point cloud I/O; WebAssembly and Python bindings |
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
The pose graph pictures are of real drives: [NCLT](http://robots.engin.umich.edu/nclt/) session 2012-04-29
(University of Michigan, [Open Database License](https://opendatacommons.org/licenses/odbl/1-0/), prepared with
[`scripts/prepare_nclt.py`](scripts/prepare_nclt.py)) and 2012-06-15 / 2012-12-01 for the loops, corrections and seasons,
with the ROS bag demos made from 2012-04-29 and 2012-06-15 by [`scripts/make_nclt_bag.py`](scripts/make_nclt_bag.py); and [PandaSet](https://pandaset.org)
scene 019 (Scale AI and Hesai, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), fetched with
[`scripts/fetch_pandaset.py`](scripts/fetch_pandaset.py)) for the odometry and dynamic objects,
and [`scripts/prepare_vector_map_pandaset.py`](scripts/prepare_vector_map_pandaset.py) for the vector map GIF.
KITTI data is not redistributed here.
