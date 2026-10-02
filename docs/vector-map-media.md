# Reproduce the vector map GIFs

The main README animation generates roads from real planning points and operator
paths, then reviews automatically discovered equipment. Another animation uses
recorded PandaSet poses. An expandable example edits imported survey context.
All geometry comes from the production app. Raw inputs stay external.

## Source-only equipment discovery

`docs/images/web/vector-map-equipment.gif` uses the Autoware planning point cloud
(1,757,841 points), with the copyright/source notice below. **No vector map or
surveyed feature geometry is loaded into the generator.** This sample has no
recorded drive: the three paths in `web/media/vector-map-operator-paths.json` were
specified by an operator over the points. They are not automatically recovered
trajectories. Lane counts, nominal widths and speed are priors. Points supply
ground height and supported boundaries; missing evidence remains inferred.

The capture builds six approach lanes and previews five point-supported junction
connections. The operator adds these drafts; ground support does not establish
permitted turns or clearance. Whole-ground search locates equipment without
feature boxes: 806 detected proposals, 129 retained (64 bars, 64 panels, one
repeated-paint pattern), 906 windows and 71 unsupported windows. The paint pattern
is a false crossing and is discarded. The capture adds a 36-point, approximately
2.994 m stop marking and a 47-point panel fitted to a 1.013 × 0.712 m housing.
Its inspection box contains 54 original points, including nearby returns excluded
from fitting. Types and lane 14 are explicit human reviews. The result has 11
road lanes, one stop marking and one signal face, with no invented lamps or sign.

A 0.02 m signal adjustment demonstrates editing; Undo restores byte-identical
OSM. Removing both additions restores the road-only map exactly. Export/reload
preserves measured provenance with the Local projector: original MGRS metre
coordinates stay in local_x/local_y without an assumed geographic origin.
Native/Web measured coordinates agree within 1e-7 m. These checks establish
workflow consistency, not classification or survey accuracy. There are many
uncertain panels, missed road-corridor heads and truncated proposals; see
[search methods and limitations](commands/vectormap-discover.md).

Prepare an updated native core (maturin or `cloudanalyzer[fast]`) and the external
cloud, then generate proof in a **new ignored directory**. Points are read in place:

```powershell
python scripts/prepare_vector_map_discovery_media.py demo_data/autoware/sample-map-planning/pointcloud_map.pcd notes/equipment-media
cd web
npm run wasm
npm run build
$env:VECTOR_MAP_PLANNING_DIR='../demo_data/autoware/sample-map-planning'
$env:VECTOR_MAP_DISCOVERY_DIR='../notes/equipment-media'
$env:PW_PORT='4174'
npm run media:equipment
```

Playwright operates the production app and closes its own server. Frames,
verification and timings stay ignored in `web/media-frames/vector-map-equipment/`.
ffmpeg crops/scales screenshots to an 800 × 528 looping GIF, about 22 seconds and
0.87 MB. The preparation script and capture never read a surveyed map, including
the separate post-hoc evaluation reference. The retired box-selected paint GIF
is removed because sampling patterns cannot be described as confirmed crossings.

## Intersection: surveyed context and new connection drafts

`docs/images/web/vector-map-intersection.gif` uses the official Autoware
`sample-map-planning` survey (1,757,841 points, MGRS 54SVE).
**Sample map: Copyright 2020 TIER IV, Inc.** See the official
[planning demo](https://docs.autoware.org/main/demos/planning-sim/) and
[download configuration](https://github.com/autowarefoundation/autoware/blob/main/ansible/roles/demo_artifacts/tasks/main.yaml).
The archive SHA256 is
`5536fce7bb8db7688fdf94ec004118b898637ad0d5b6175108b10989dd6e93b9`.
The sample-map archive has no separate license text; the code repository's
software license is not asserted to cover the survey.

The [junction ablation](vector-map-validation.md#branching-junction-connection-drafts)
removes 114 turn-labelled connector lanes, leaving 72 imported lanes and their
surveyed regulations. The animation focuses on one four-way intersection. It
shows twelve explicitly selected, ground-supported connection drafts, selected
lane topology, plan/3D inspection, a shared-boundary edit, exact Undo and MGRS
Lanelet2 export. The four crossings and signal geometries are imported surveyed
context; the demo does **not** automatically detect them or reconstruct the
entire map from points. Ground support does not establish legal turns, full-width
clearance, signal priority or surveyed boundary accuracy.

Obtain the official planning sample externally and use the existing evaluation
example to prepare an ignored, new output directory. Keep the original point
cloud in place; it does not need copying:

```sh
cargo run --manifest-path rust/Cargo.toml -p ca-wasm --example vector_map_junction_evaluate -- demo_data/autoware/sample-map-planning/pointcloud_map.pcd demo_data/autoware/sample-map-planning/lanelet2_map.osm notes/junction-media
cd web
npm ci
npx playwright install chromium
npm run wasm
npm run build
VECTOR_MAP_PLANNING_DIR=../demo_data/autoware/sample-map-planning VECTOR_MAP_JUNCTION_DIR=../notes/junction-media npm run media:intersection
```

PowerShell environment setup for the capture:

```powershell
$env:VECTOR_MAP_PLANNING_DIR='../demo_data/autoware/sample-map-planning'
$env:VECTOR_MAP_JUNCTION_DIR='../notes/junction-media'
$env:PW_PORT='4174'
npm run media:intersection
```

Playwright owns the preview server and closes it after capture. The input JSON
and report come from the ablation; reference topology is used only for its
evaluation, not by the Web proposal algorithm. The capture compares proposal
pairs with the native report and selects twelve pairs explicitly. It verifies
changed OSM after editing, byte-identical OSM after undoing the edit and the
connection batch, and twelve review-required tags after MGRS reimport. Shared
handles are chosen through visible pixels and UI hit feedback; failed selection
fails the capture. `verification.json`, source frames and timings are ignored in
`web/media-frames/vector-map-intersection/`. ffmpeg produces an 800 × 528 looping
animation (17 seconds, 19 encoded frames, approximately 1.04 MB). None of these
checks constitutes a held-out extraction accuracy test.

## Road drafting: real PandaSet points and recorded trajectory

The expandable README animation (`docs/images/web/vector-map.gif`) uses real
[PandaSet](https://github.com/scaleapi/pandaset-devkit) scene 019: original Pandar64
frames 0, 8, …, 72 in world coordinates, including intensity, and all 80 recorded
sensor positions. PandaSet is provided by Scale AI and Hesai under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The GIF is a cropped,
scaled capture with explanatory captions and a cursor ring. Raw inputs are not
included in this repository.

It shows generation with continuity tracking and trajectory-relative lateral fitting, manual movement
of an interior vertex on a shared boundary, one-step Undo, and export of
`lanelet2_map.osm` plus `map_projector_info.yaml`. Undo restores the generated draft
before the final export; the capture verifies byte-identical OSM before editing and after
Undo, and a changed OSM after dragging. The drive has a
Local projector. Captions describe actual UI actions; point cloud and map geometry
come from the app. The vertex move demonstrates editing, rather than a correction
validated against a surveyed marking. Generated lines remain a draft: see
[the real-data evaluation](vector-map-validation.md) for measured accuracy and limitations.

The current capture includes curb-profile checks, short missing-observation
anchors and trajectory-relative shape stabilization, retaining observed corners. It verifies 78 tracked source changes, 135 fitted vertices and 59 rejected
height-step candidates before editing. Rejection counts are not known false-detection
counts; the drive has no independent lane-boundary reference. The evaluation documents
the same-input boundary accuracy and reduced heading variation after stabilization.
PandaSet heading-change P95 decreased from 8.45° to 3.25°; it has no independent
lane-boundary reference, and both real inputs were used during development.

Prepare the public inputs in an ignored directory:

```sh
pip install numpy pandas remotezip
python scripts/prepare_vector_map_pandaset.py demo_data/pandaset-vector-map
```

The script downloads the public archive through range requests and writes
`map.pcd` and `trajectory.csv`; it refuses to overwrite either file. It keeps the
original world coordinates and intensity. CSV timestamps are the recorded LiDAR
timestamps relative to the first frame.

With Node, Rust/wasm-pack, Chromium and ffmpeg available:

```sh
cd web
npm ci
npx playwright install chromium
npm run wasm
npm run build
VECTOR_MAP_DEMO_DIR=../demo_data/pandaset-vector-map npm run media:vectormap
```

In PowerShell, set the environment variable before the last command:

```powershell
$env:VECTOR_MAP_DEMO_DIR='../demo_data/pandaset-vector-map'
npm run media:vectormap
```

Playwright starts the production preview server. Only the vector map capture runs;
the other README images stay unchanged. Source frames and their timing manifest
go to ignored `web/media-frames/vector-map/`. ffmpeg writes
`docs/images/web/vector-map.gif` at 800 × 528 pixels, looping for approximately
12 seconds (about 0.79 MB). The capture selects an interior point of shared boundary
2 through visible handles and panel feedback, so a failed selection fails the
capture instead of producing a misleading editing scene.

The captures are checked against the current map-display implementation
(2026-10-02), including topology highlights, crosswalk bands and saved signal faces.
Bounded COPC working-box and spatial signal workflows are described in
[large clouds](large-point-clouds.md) and [signal measurement](commands/vectormap-signal.md);
the road animation demonstrates drafting and editing on the recorded PandaSet
drive. It is not a large-source benchmark or automatic signal-classification demo.
