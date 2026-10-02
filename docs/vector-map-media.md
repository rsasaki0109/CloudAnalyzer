# Reproduce the vector map GIFs

The main README animation shows a real four-way intersection. The expandable
animation shows road drafting directly from PandaSet LiDAR and recorded poses.
Both are captures of the production Web app with captions and a cursor ring;
all point-cloud and map geometry is rendered by the app. Raw inputs are external
and are not redistributed in this repository.

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

It shows generation with continuity tracking and local curve fitting, manual movement
of an interior vertex on a shared boundary, one-step Undo, and export of
`lanelet2_map.osm` plus `map_projector_info.yaml`. Undo restores the generated draft
before the final export; the capture verifies byte-identical OSM before editing and after
Undo, and a changed OSM after dragging. The drive has a
Local projector. Captions describe actual UI actions; point cloud and map geometry
come from the app. The vertex move demonstrates editing, rather than a correction
validated against a surveyed marking. Generated lines remain a draft: see
[the real-data evaluation](vector-map-validation.md) for measured accuracy and limitations.

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
