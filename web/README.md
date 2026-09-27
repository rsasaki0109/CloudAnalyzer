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
- Level-of-detail rendering for large clouds: points are arranged in a
  nested octree at load time and only the nodes that matter on screen are
  drawn, up to an adjustable point budget (1M–20M).
- ICP registration (point-to-plane or point-to-point, overlap trimming,
  optional centroid pre-alignment) with the resulting 4×4 transform and undo.
- Click a point to see its exact coordinates, color and C2C distance; press
  **Measure** (or `M`) and click two points for their distance and ΔX/ΔY/ΔZ.
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

The Rust core is tested natively:

```sh
cd rust
cargo test --workspace
cargo run --release -p ca-core --example c2c -- compared.ply reference.pcd
```
