# Reproduce the vector map GIF

The README animation is captured from the production Web app using real
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
12 seconds (about 0.8 MB). The capture selects an interior point of shared boundary
2 through visible handles and panel feedback, so a failed selection fails the
capture instead of producing a misleading editing scene.

The capture is checked against main `4d64557` (2026-10-02), including the bounded
COPC working-box and spatial signal workflows. Those workflows are described in
[large clouds](large-point-clouds.md) and [signal measurement](commands/vectormap-signal.md);
this animation demonstrates road drafting and editing on the recorded PandaSet
drive. It is not a large-source benchmark or automatic signal-classification demo.
