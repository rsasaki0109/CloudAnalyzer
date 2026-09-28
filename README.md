# CloudAnalyzer

Point cloud viewer and analyzer that runs in your browser.
Rust + WebAssembly. Your files stay on your machine.

**[Open the viewer](https://rsasaki0109.github.io/CloudAnalyzer/app/)** ·
[Guide](https://rsasaki0109.github.io/CloudAnalyzer/guide.html) ·
[Demos](#try-it)

<p align="center">
  <a href="https://rsasaki0109.github.io/CloudAnalyzer/app/?demo=ground">
    <img src="docs/images/web-viewer.png" alt="CloudAnalyzer web viewer: ground extraction on a small town" width="900">
  </a>
</p>

## Features

- **Open** PLY, PCD, LAS/LAZ, E57 (all scans, posed), XYZ and OBJ/STL meshes, including tens of millions of points,
  and COPC files straight from a URL (only the levels you need are downloaded).
- **Compare**: cloud-to-cloud and cloud-to-mesh distance, M3C2 change detection, cut/fill volume,
  trajectory ATE / RPE (TUM, KITTI, CSV; SE(3) / Sim(3) alignment, same numbers as the Python CLI).
- **Process**: ICP and manual alignment (point pairs, gizmo, matrix), subsampling, outlier removal, ground extraction (CSF), normals, merge/split,
  rasterize to a DEM / DSM (GeoTIFF or colored PNG), RANSAC shape detection (planes, cylinders, spheres)
  and Euclidean clustering, mesh a cloud (2.5D Delaunay).
- **Inspect**: clipping box, lasso segmentation with undo/redo, cross-section profiles, picking, measuring and labels; save the view as a PNG.
- **Report**: pass / fail gates on any result, saved as an HTML QA report or JSON in the `ca check` gate format.
- **Share**: links and session files that restore the view; export PLY, LAS/LAZ (distances as extra bytes), E57 or CSV,
  and meshes as PLY or OBJ.
- Works on phones and tablets.

## Try it

| Demo | What it shows |
|---|---|
| [Two LiDAR scans](https://rsasaki0109.github.io/CloudAnalyzer/app/?demo=c2c) | Cloud-to-cloud distance |
| [Stockpile](https://rsasaki0109.github.io/CloudAnalyzer/app/?demo=volume) | Cut / fill volume |
| [Town](https://rsasaki0109.github.io/CloudAnalyzer/app/?demo=ground) | Ground extraction |
| [Landslide](https://rsasaki0109.github.io/CloudAnalyzer/app/?demo=m3c2) | M3C2 change detection |

## What's inside

| Path | |
|---|---|
| [`web/`](web/) | The browser app (TypeScript, three.js) |
| [`rust/`](rust/) | The Rust core, its WebAssembly bindings and Python bindings (`cloudanalyzer_core`) |
| [`cloudanalyzer/`](cloudanalyzer/) | `ca`, a Python CLI for SLAM and point-cloud quality gates in CI (`pip install cloudanalyzer`) |

## Develop

```sh
cd web
npm install
npm run wasm   # build the Rust core to WebAssembly
npm run dev    # http://localhost:5173
```

Tests: `cargo test` in `rust/`, `npx playwright test` in `web/`.

## License

[MIT](LICENSE). Sample data keeps its upstream terms; see [attribution](web/public/samples/ATTRIBUTION.md).
