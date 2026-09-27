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
- Save any cloud as binary PLY or CSV, including its C2C/C2M distances as a
  scalar field (`scalar_C2C_distance`, which CloudCompare loads directly).
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
- Works on phones and tablets: the view fills the screen and the panels
  open as a bottom sheet (**Panels**); drag to orbit, pinch to zoom, two
  fingers to pan, tap to pick, double-tap (or double-click) to orbit around
  a point.
- Click a point to see its exact coordinates, color and C2C distance; press
  **Measure** (or `M`) and click two points for their distance and ΔX/ΔY/ΔZ.
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
