# Reproduce the vector map GIFs

The main README animation compares default and source-footprint roads in two
complex Tokyo junctions, then reviews supported connections and measured equipment.
Its [generation protocol and editable result](vector-map-supported-intersection.md)
retain deferred extent, connectivity warnings and unreviewed signal-stop links.
The earlier Tokyo animation below preserves its original geometry and quality
audit. Another animation uses recorded PandaSet poses. The expandable
surveyed-intersection example explicitly edits imported context. All captures
operate the production app; captions and crop/scale are added to real screenshots.

## Source-only hard intersection

This is the earlier 49-lane capture, retained for comparison and historical
reproduction. The main README now uses
`docs/images/web/vector-map-supported-intersection.gif`; see its separate
[comparison, verification and reproduction](vector-map-supported-intersection.md).

`docs/images/web/vector-map-hard-intersection.gif` is adapted from
[Hard Intersection Multimodal Samples](https://huggingface.co/datasets/dynamic-maps/hard-intersection-multimodal-sample),
Dynamic Map Platform Co., Ltd. (2026), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/),
revision `e8e8d2a5d49a8b9cb63b5b5ecef9b260ff48f39c`.
The input is 1,883,866 first-original points retained at 0.1 m voxels from the
35,668,990-point source LAS. XYZ, RGB and intensity are preserved; semantic
fields are cleared. **No reference map, semantic labels, feature boxes or
reference feature shapes are read by the media preparation or capture.**
See [data preparation, hashes and licensing](vector-map-hard-intersection.md).
Reuse prepared inputs in place rather than downloading or duplicating the clouds.

`web/media/vector-map-hard-intersection-inputs.json` records the explicit operator
choices. Three intervals of recorded drive 1 (`26047_Record050_260217-00.csv`)
retain 161, 41 and 115 original prepared poses and timestamps. They leave gaps at
the junctions; the interval endpoints are operator choices, not automatically
recognized intersections. The three branch paths are operator traces over source
intensity/geometry, **not recorded trajectories**. Their nominal Z is replaced
by ground fitting. Arterial lane counts are three in each direction, the east
branch two each and the west branches one each; 3.5 m widths and 40 km/h speeds
are priors. Supported boundary evidence and smoothing do not establish surveyed
lane counts or widths. The four original drives traverse the same arterial;
adding them all as separate roads produces overlapping drafts and is avoided here.

The actual UI workflow:

1. Fit 26 approach lanes from source points and the selected paths.
2. Preview 84 geometric connections with a 50 m gap limit; select 23 branches,
   yielding 49 road lanes. Endpoints and centre samples require source ground.
   These choices do not certify permitted turns or full-width clearance.
3. Search the full retained source: 328 detected proposals, 145 shown (64 bars,
   64 panels, 17 repeated-paint patterns), 148 windows, one unsupported window.
   Discard a doubtful paint pattern and two longitudinal bars without modifying
   the map. Longitudinal paint is not accepted as a stop marking.
4. Explicitly identify and associate seven paint-crossing candidates, two transverse
   branch marking drafts (67 points / 4.318 m and 224 points / 5.436 m) and four
   housing candidates (two vehicle and two pedestrian types). Inspect original
   housing points before adding. Object types and lane links are operator inputs.
   Measured stripe footprints and their opposing edges are retained, including
   partial paint footprints; missing bands or lamps are not invented.
5. Edit a housing vertex by 0.08 m in Z and restore the byte-identical map with Undo.
   Undo all 13 equipment additions back to the road-only map, then reopen the
   generated Lanelet2 file with measured provenance and no validation errors.

Native/Web geometry is compared for every generated road boundary, marking,
housing and crosswalk edge, allowing whole-way reversal. The captured maximum
absolute coordinate difference is 7.28 × 10⁻¹² m; the ignored verification report
also records exact Undo, successful reload and no page errors. The native core
and production WASM were built from `443e272485cedad47d9b088b5b213c0aa825fef1`
(reflected in main by PR #194). The compact
[capture verification record](vector-map-hard-intersection-media.json) freezes
binary/input/output/GIF hashes, counts and workflow checks; it is not an accuracy
test. The source is EPSG:6677 with
unchanged orthometric Z; Local projector export retains original metre coordinates
in local_x/local_y. No geographic origin or fitted alignment is assumed.
Zero validation errors and channel agreement establish workflow consistency, not
semantic correctness, survey accuracy or verified traffic rules.

Display clipping at about 35.53 m removes tall roof clutter; all fitting and
search use the full original retained cloud. The final view copies the same
original points inside this working box in browser memory, after fitting, so the
clipping-box wire can be removed. No point cloud is copied to disk. The exported
map remains byte-identical through this display operation.

The fixed 28-tile development audit and this whole-scene UI preview use different
window ownership and preview limits. The animation's seven operator-confirmed
paint objects are **not** a new 7/7 accuracy claim. Consult the separate fixed
[evaluation comparison](vector-map-hard-intersection.md#observed-paint-envelopes-and-clipped-window-refinement)
for misses, unmatched proposals and outline errors. This scene informed development;
no independent generalization result is claimed.

The generated README map has also undergone a separate
[source coverage and equipment geometry audit](vector-map-quality.md): nine of
49 lanes need source review despite zero structural errors. Crossing outline
errors and unmatched reviewed equipment remain. The animation demonstrates the
workflow; inspect these quality results before interpreting its visual appearance
as map accuracy.

Build the native core from the same checkout before preparing proof in a **new**
ignored directory. With the existing source preparation and updated native module:

```powershell
python scripts/prepare_hard_intersection_media.py notes/hard-intersection-prepared-v2 notes/hard-intersection-ui-proof --source-commit (git rev-parse HEAD)
cd web
npm run wasm
npm run build
$env:VECTOR_MAP_HARD_INTERSECTION_DIR='../notes/hard-intersection-prepared-v2'
$env:VECTOR_MAP_HARD_INTERSECTION_PROOF='../notes/hard-intersection-ui-proof'
$env:PW_PORT='4174'
npm run media:hard-intersection
```

The helper freezes CSV paths, reports, geometry, input/binary hashes and generated
OSM in the ignored proof directory. Playwright closes its own preview server;
frames, generated map and verification remain ignored under
`web/media-frames/vector-map-hard-intersection/`. ffmpeg creates an 800 × 528
looping GIF from the actual screenshots (24.2 seconds, 1,367,611 bytes). It replaces
the older planning-point-cloud equipment capture and its fixed candidate IDs.
Raw points, references and frame PNGs
are not redistributed in the README PR.

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
