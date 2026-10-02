# Reproduce the vector map GIFs

The main README animation shows a real four-way intersection. The expandable
animation shows road drafting directly from PandaSet LiDAR and recorded poses.
An additional animation measures paint and a signal housing in separate regions
of the same Autoware survey. All are captures of the production Web app with captions;
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

## Paint and signal housing measured from points

`docs/images/web/vector-map-features.gif` uses the same official planning survey
and copyright/source notice above. These are **separate regions from the main
intersection**: a paint box at `[3882, 73743, 18.5]`–`[3899, 73764, 20.5]` and a
signal box at `[3832.1, 73770.1, 24.74]`–`[3833.9, 73771.4, 25.27]`, in original
MGRS metre coordinates. Boxes, object identity and lane associations are explicit
operator inputs. Fits use cloud returns; reference paint corners and the imported
signal housing are not supplied to the measurement algorithms.

The capture imports the 72-approach ablation above, removing only surveyed signal
353 and its empty-lanes traffic-light rule 1008 to avoid duplication. Remaining
roads, stops, crossings and signals remain surveyed context. It retains RGB and
previews four observed bands from 7,434 box points / 5,568 ground returns with a
65% bright-point threshold. Observed paint determines the measured footprint,
including missing/worn ends; it is not filled with decorative stripes. Crossing
lanes 133/134 are explicitly selected drafts, not inferred legal priority.

A separate 61-point ROI supplies a user-identified signal housing's bottom edge
and height. Controlled lane 85 is explicitly selected and remains unverified.
Housing shape alone does not classify signals versus signs/background, and no
lamps, states, arrows or poles are synthesized. The capture changes a crossing
vertex by 0.1 m, then adjusts the signal's first vertex Z and housing height by
0.02 m; these demonstrate manual editing, not independently validated corrections.
Observed paint stays fixed, and both edits Undo to byte-identical previous OSM.
Export/reimport retains measured paint metadata and housing provenance.

After preparing the external planning survey and junction ablation above:

```powershell
cd web
$env:VECTOR_MAP_PLANNING_DIR='../demo_data/autoware/sample-map-planning'
$env:VECTOR_MAP_JUNCTION_DIR='../notes/junction-media'
$env:PW_PORT='4174'
npm run media:features
```

The capture uses full-density box inspection, disables EDL for sparse returns,
shows actual RGB paint and a 3D signal housing, then edits and saves through the
normal UI. Signal point size and camera zoom are increased for readability;
these display changes do not alter measurements. Captions are added over the
app capture. Source frames, verification and timing manifests stay ignored in
`web/media-frames/vector-map-features/`. The looping GIF is 800 × 528, about
16.4 seconds. It is a reproducible feature workflow, not a held-out classification,
survey-accuracy or traffic-rule benchmark. See
[paint measurement](commands/vectormap-crosswalk.md),
[signal measurement](commands/vectormap-signal.md) and
[geometry editing](vector-map-feature-editing.md).

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
