# CloudAnalyzer v0.6.0

Released 2026-10-23. **Repair maps, draft roads and inspect surface changes.**

[Open the browser app](https://rsasaki0109.github.io/CloudAnalyzer/app/) ·
[Guide](https://rsasaki0109.github.io/CloudAnalyzer/guide.html) ·
[Changes since v0.5.0](https://github.com/rsasaki0109/CloudAnalyzer/compare/v0.5.0...v0.6.0)

## Highlights

- **SLAM-map workspace:** LiDAR replay, loop inspection, pose editing, IMU-aware
  mapping and saved projects. Native registration and search optimizations have
  recorded real-data evidence; see the [README results](../README.md#proven-on-real-data).
- **Vector maps from points:** source-supported road drafts, shared-boundary
  editing, reviewed crossing/stop/signal proposals and Autoware Lanelet2 export.
  The [complex-intersection demonstration](vector-map-supported-intersection.md)
  separates observed support, inferred geometry, deferred ranges and unresolved
  equipment relations. Lane counts, object types and legal control need review.
- **Map quality:** [coverage and density](map-coverage.md), bidirectional
  [discrepancy regions and cell sections](distance-regions.md), scoped comparison
  counts and JSON/CSV reports.
- **Surface change and volume:** [Before/After cut and fill](surface-volume.md),
  signed cell heights, missing-coverage counts and input/settings provenance.
- **Terrain meshes:** [XY Delaunay meshing](terrain-mesh.md), faces/wireframe,
  OBJ/PLY and summary JSON, with compared After-cell and volume handoffs.
- **Large maps and performance:** [display versus processing scope](large-point-clouds.md),
  [resumable COPC tiles](commands/copc-tile.md), and measured CPU/WASM/WebGPU
  distance-processing improvements. Self QA smoke runs use explicit offline
  fixtures; historical benchmark packets keep their measured source hashes.
- **CLI and agents:** bridge cross-sections, member plans and damage candidates;
  source-audited mapping-run diagnostics and optional Rust Python acceleration.
  The [changelog](../CHANGELOG.md#060---2026-10-23) records the detailed changes.

![Terrain wireframe in the actual browser: synthetic ground, 1,681 vertices and 3,200 triangles](images/web/terrain-mesh-wireframe.png)

## Use

The browser app processes data locally with Rust/WebAssembly. Python users can
install the matching package from PyPI when its publish workflow completes:

```sh
python -m pip install --upgrade "cloudanalyzer==0.6.0"
```

For source builds, check out `v0.6.0`, then follow the
[Python installation guide](../cloudanalyzer/README.md) or [Web build instructions](../web/README.md).
The optional `cloudanalyzer-core` extension has its own version; it is not renamed
to 0.6.0 by this application release.

## Scope

Terrain meshing and cut/fill are 2.5D height-surface workflows. Crop or extract a
comparable ground surface first; vegetation, roofs, walls and stacked surfaces
need separate review. Edge limits do not guarantee preservation of every gap.
The distance review rectangle does not clip volume inputs. Units label existing
coordinates and do not convert them.

Physical processing of ten billion points has not been benchmarked. Loaded
previews, engine thinning, display LOD and tiled full-density processing must
not be treated as equivalent. Synthetic demonstrations validate geometry and
workflow, not independent survey accuracy.

The unmerged wgpu LiDAR AI draft and rejected RGB-boundary/VGICP experiments are
not part of this release.

## Validation

The release package builds as a source archive and a platform-independent wheel;
both pass strict `twine check`. The installed 0.6.0 wheel runs the checked-in
synthetic-figure8 benchmark with its quality gate passing and builds a static
leaderboard. Release-focused Python/CLI/documentation checks pass 131 tests.
The application code matches the preceding locally verified main revision:
TypeScript and production Web build passed, 130 Web unit tests passed and 19
related production-browser checks passed. These local checks were used for
release readiness; GitHub CI completion is not asserted here.
