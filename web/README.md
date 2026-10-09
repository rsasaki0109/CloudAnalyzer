# CloudAnalyzer Web

A browser-based point cloud viewer and analyzer in the spirit of CloudCompare.
The analysis core is Rust ([`rust/crates/ca-core`](../rust/crates/ca-core)),
compiled to WebAssembly and run in a Web Worker; rendering uses three.js.
Files never leave the browser.

**Try it:** https://rsasaki0109.github.io/CloudAnalyzer/app/

## Features

- Open PLY (ascii / binary), PCD (ascii / binary / binary_compressed),
  LAS / LAZ 1.0–1.4 (point formats 0–10) and XYZ / TXT / CSV / PTS by
  drag-and-drop.
- Intensity and ASPRS classification from LAS/LAZ (also PLY, PCD and PTS):
  color by either, and show or hide individual classes.
- Eye-Dome Lighting (on by default, adjustable strength) so the shape of
  unlit clouds is easy to read.
- Level-of-detail rendering for large clouds: points are arranged in a
  nested octree at load time and only the nodes that matter on screen are
  drawn, up to an adjustable point budget (1M–20M).
- ICP registration (point-to-plane or point-to-point, overlap trimming,
  optional centroid pre-alignment) with the resulting 4×4 transform and undo.
- Save any cloud as binary PLY, LAS 1.4, LAZ or CSV, including its C2C/C2M
  distances as a scalar field (`scalar_C2C_distance` in PLY, a `float` extra
  byte field in LAS/LAZ; CloudCompare loads both directly).
- Clipping box with per-axis sliders and one-click X/Y/Z cross-sections; the
  point budget goes to what is inside, and **Crop** copies the points inside
  into new clouds.
- M3C2 change detection (Lague et al. 2013): the distance between two
  surveys along local surface normals, averaged over a cylinder, with a 95 %
  level of detection per core point; the result is a new cloud of core points
  (optionally one per voxel) with `m3c2_distance`, `lod95` and `significant`
  attributes, grey where too few points were found.
- Cut/fill volume (2.5D) between any two of: point clouds, meshes, or a
  constant height, on a grid with mean/min/max cell heights and optional
  filling of empty cells; the per-cell height difference is shown as a cloud.
- Filters that create a new cloud: one point per voxel, random subsampling,
  statistical outlier removal (SOR), and ground extraction with the Cloth
  Simulation Filter (classified copy, ground only, or non-ground only).
- Merge / split: merge the visible clouds into one (colors, intensity and
  classes kept; clouds without RGB take their display color; a `source`
  attribute remembers each point's cloud), split a cloud into one cloud per
  classification code, or split a merged cloud back into its files.
- Normals from the k nearest neighbours of each point (PCA), facing up or
  away from the centre, shown as direction colors or a hillshade and saved as
  `nx`, `ny`, `nz` in PLY (normals in PLY files are read too). Large clouds
  are split along the octree over the worker pool; points near a part's
  border use neighbours from their own part.
- Labels and images: **Label** (`L`) pins editable text to a picked point
  (kept in share links and sessions); **Image** saves the view as a PNG with
  labels, measurements, the colorbar and the profile plot drawn on it.
- Cross-section profiles: draw a polyline on the view (**Draw line**, click
  vertices, double-click or Enter to finish) and every visible cloud's points
  within the band are plotted as distance along the line against height, in
  the cloud's color, with a cursor readout, an optional 1:1 scale and CSV
  export (distance, x, y, z). Handy for comparing surveys; the line is part of
  shared links and sessions.
- Share and sessions: **Share** copies a link that reopens the current view:
  camera, display settings, how each cloud is shown (visibility, color mode,
  ICP transforms, hidden classes, clipping box) and C2C / C2M distances,
  which are recomputed. Clouds loaded from URLs (the sample, **Open URL**, or
  `?url=https://…/cloud.laz` links, repeatable) are downloaded again; local
  files are not uploaded anywhere, so whoever opens the link is asked to open
  them. **Save session** writes the same state as JSON; open it together with
  its files (or first, then the files). Remote servers must allow
  cross-origin requests (CORS).
- Display: fixed (pixels) or adaptive point size (as wide as the local point
  spacing, per drawn octree node, so near surfaces close up), background
  colour, and named camera views; all kept in share links and sessions.
- **Save workspace snapshot** writes a single `project.cloudanalyzer.zip` with
  all current loaded point clouds and mesh geometry, including filtered/cropped
  results, plus the edited HD map, lane notes and display settings. Open that ZIP
  to continue from the saved records without rerunning processing. PLY snapshots
  keep double coordinates and supported float/byte vertex attributes; transforms
  are baked once. The package is bounded to 64 MiB uncompressed, 127 clouds/meshes
  and 10 MiB project metadata. Original pose-graph/scan inputs and the verified
  generated-map review ZIP are included, with at most 2,048 ZIP members. The
  original review archive stays downloadable; its frozen audits do not validate
  later edits. Unloaded cloud detail stays external. Undo starts fresh.
  Browser autosave includes these records/inputs when its optional checkbox is
  enabled; otherwise it and **Save project** retain metadata/source references.
- Works on phones and tablets: the view fills the screen and the panels
  open as a bottom sheet (**Panels**); drag to orbit, pinch to zoom, two
  fingers to pan, tap to pick, double-tap (or double-click) to orbit around
  a point.
- Click a point to see its exact coordinates, color and C2C distance; press
  **Measure** (or `M`) and click two points for their distance and ΔX/ΔY/ΔZ.
- COPC (Cloud Optimized Point Cloud) files are read node by node: every
  octree level that fits "Max points per file" is loaded, so the density
  stays even and the rest of the file is never read. From a URL this uses
  HTTP range requests (adjacent nodes merged into one request), so a large
  public COPC opens without downloading it; nodes are decoded on the worker
  pool. Local COPC files are read the same way.
- Long loads and downloads show a progress bar with **Cancel** (a load stops
  between 16 MB slices); the status bar also shows the main worker's
  WebAssembly memory.
- Large files: LAS and binary PLY/PCD are streamed in 16 MB slices, so the
  file is never held in memory whole; files above the "Max points per file"
  setting (50M by default) keep every n-th point. LAZ is thinned while it is
  decompressed.
- Coordinates are kept in `f64`; georeferenced clouds get a shared global
  shift for rendering, like CloudCompare.
- Cloud-to-mesh (C2M) distance against OBJ / STL / PLY meshes, optionally
  signed by the mesh's face normals (points behind the surface are negative).
- Cloud-to-cloud (C2C) nearest-neighbour distance with summary statistics,
  color ramps, an adjustable display range and a colorbar.
- Large C2C jobs are split into spatially compact parts that run on a pool of
  WASM workers (no SharedArrayBuffer, so it works on static hosting such as
  GitHub Pages). Results are exact: each part receives every reference point
  that can be nearest to its queries.
- The octree index of clouds above 1M points and statistical outlier removal
  run on the same pool. The index is built in three steps (sort slices by
  level-2 node, build each node's subtree, stitch); SOR splits the cloud along
  the octree, and points near a part's border ask the workers still holding
  the neighbouring parts, so the result matches the single-threaded filter
  exactly. On 10M points with 8 workers, SOR takes 4.3 s instead of 17.9 s.
- The heavy WASM kernels run once on a small synthetic cloud when the page
  loads, so the browser has optimized them before the first real file
  (one long call would otherwise run unoptimized throughout).

## Validation

The core is compared with CloudCompare on a public LiDAR tile by
`scripts/validate_cloudcompare.py`; see [docs/validation.md](../docs/validation.md).

## Demos and guide

The empty viewer offers demos that load sample data and run an analysis; they
also start from a link, e.g. `?demo=volume`: `c2c` (two LiDAR scans), `volume`
(a stockpile), `ground` (CSF on a small town) and `m3c2` (a landslide). The
synthetic samples are generated by `node scripts/make-samples.mjs`. A user
guide is published next to the app (`docs/guide.html`).

## Development

Requires Rust with the `wasm32-unknown-unknown` target, `wasm-pack` and Node 22+.

```sh
cd web
npm install
npm run wasm   # build rust/crates/ca-wasm into src/wasm
npm run dev    # add `-- --host` to open it from other devices on the LAN
npm run build  # static site in dist/
```

End-to-end tests drive the production build in headless Chromium
(sample C2C, PLY/LAS/OBJ loading, classes, clipping, export):

```sh
npx playwright install chromium   # once
npm run build && npm run test:e2e
```

The Rust core is tested natively:

```sh
cd rust
cargo test --workspace
cargo run --release -p ca-core --example c2c -- compared.ply reference.pcd
```

Enable **Include current point and mesh records (64 MiB)** under browser autosave
for automatic recovery of processed geometry and computed attributes without
reselecting point sources. The default remains metadata-only recovery. Atomic
storage failures preserve the previous copy; **Download browser copy** exports
a saved record snapshot as a workspace ZIP. See [browser recovery](../docs/browser-projects.md#keep-processed-records-in-browser-recovery)
for capacity, external graph/audit inputs and backup limitations.
