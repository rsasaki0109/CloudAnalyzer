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
  Large core-point sets run on up to four CPU/WASM workers, with exact
  results and a conservative copying budget. Small core blocks are spread
  across workers to balance dense regions; see [CPU M3C2](../docs/cpu-m3c2.md).
  Exact trees are built once, transferred to pool lanes and reused for an
  unchanged input pair. The analysis worker joins the pool under the same
  memory estimate, admitting more large measured inputs; see
  [M3C2 tree reuse](../docs/cpu-m3c2-preparation.md) and the earlier
  [resident lane](../docs/cpu-m3c2-main-worker.md).
  A bounded [normal cache](../docs/cpu-m3c2-normals.md) speeds repeated
  projection changes. The first calculation fits fresh normals; matching
  repeated fitting settings can capture and reuse them under 64 MiB.
  Fresh calculations also [share reference radius searches](../docs/cpu-m3c2-shared-search.md)
  for normal fitting and projection, retaining exact neighbourhood order.
  Prepared fresh WASM fits use [paired normal eigensolvers](../docs/cpu-m3c2-pair.md),
  improving measured whole-call time by 1.42–10.84% while preserving outputs.
  An additional [short-neighbourhood guard](../docs/cpu-m3c2-short-pair.md)
  remains off after its initial gains failed to reproduce consistently.
  Repeated jobs also [reuse original core-point indices](../docs/cpu-m3c2-core-selection.md)
  under an 8 MiB allowance, improving measured whole-call time by 2.29–12.00%
  while taking fresh coordinate and attribute snapshots for every result.
  With reused normals, [ordered projection traversal](../docs/cpu-m3c2-projection.md)
  avoids temporary hit lists and repeated coordinate arithmetic while retaining
  distance, confidence and significance results.
- Map quality against ground truth: accuracy, precision, completeness, F1,
  matched Chamfer distance, and voxel Gaussian AWD/SCS. Large maps run both
  C2C directions on a shared CPU/WASM queue while voxel scores are computed.
  Unchanged voxel settings reuse three scalar scores; threshold-dependent
  nearest statistics and inlier metrics are recalculated. ICP also retains
  one immutable reference/tree/normal set under 128 MiB. See the
  [measured CPU reuse comparisons](../docs/cpu-compute-scratch.md).
  When compared clouds change against a fixed reference,
  [reference voxel statistics and covariance roots](../docs/cpu-voxel-reference.md)
  can also be reused under a bounded allowance.
  A [paired cross-matrix SIMD experiment](../docs/cpu-voxel-pair.md) preserves
  tested outputs but remains off after mixed whole-call measurements.
  Retained [voxel neighbour slots](../docs/cpu-voxel-neighbors.md) now replace
  repeated SCS hash lookups for owned references; measured voxel-score calls
  improve 16.67–34.09% under the updated retained-memory allowance.
  Results are assembled as slices finish, and forward summaries overlap
  remaining workers. Unused reverse statistics and duplicate coloring copies
  are skipped, with owned input snapshots and a copying budget; see
  [CPU Map quality](../docs/cpu-map-quality.md). Repeated evaluations reuse
  both directions' exact queries and trees, and dispatch warm searches before
  voxel computation; see [Map quality preparation](../docs/cpu-map-quality-preparation.md).
  Cold searches also queue behind preparation to overlap voxel work without
  waiting for a main-worker acknowledgement; see [CPU preparation dispatch](../docs/cpu-preparation-pipeline.md).
  An internal [voxel SIMD comparison](../docs/cpu-map-quality-simd.md) keeps
  scalar moments as the default because full-request gains were inconsistent.
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
- Workspace ZIPs prepare every point/mesh, map and original graph input before
  committing. Failed or cancelled preparation retains current work and Undo;
  success starts a fresh history. Preparation temporarily holds both workspaces.
- Reusable filter recipes apply up to eight ordered steps to sixteen selected
  point clouds, with one Undo for the whole batch. Results retain input/output
  point-record and WASM-build hashes in portable workspace snapshots. Failed,
  cancelled or stale preparation leaves sources and existing history available.
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
  C2C/C2M summaries select the exact median without fully sorting every
  distance, preserving the original reduction order; see
  [distance summaries](../docs/distance-summary.md).
  [Measured-source comparisons](../docs/c2c-real-source.md) cover PandaSet,
  Autoware and Tokyo survey geometry, with input hashes and CPU/GPU controls.
  [GPU opportunity measurements](../docs/gpu-opportunities.md) compare the
  current warm CPU/hybrid C2C paths and profile remaining heavy operations,
  including a separate C2M mesh-preparation probe.
  [Retained C2M mesh/BVH](../docs/cpu-c2m-preparation.md) reuses exact CPU
  preparation under a 1 GiB aggregate budget; repeated measured-source
  whole-operation times improved by 29–70% in the final comparisons.
  [Cached C2M traversal bounds](../docs/cpu-c2m-stack-bounds.md) further improve
  whole retained-CPU calls by 23.80–30.38% in two measured passes, preserving
  outputs and retained geometry sizes.
  An additional [C2M query-order cache](../docs/cpu-c2m-query-order.md) remains
  off after repeated regressions on one measured source.
  [Borrowed C2M query triples](../docs/cpu-c2m-query-borrow.md) also remain off:
  removing an additional WASM copy regressed on Autoware in both passes.
  [C2M missing-hit recovery](../docs/cpu-c2m-missing-hit.md) returns positive
  infinity for uncomputable hits and safely continues signed/mixed batches.
  [Native parallel C2M stack reuse](../docs/cpu-c2m-native-stack.md) reduces
  native API time by 1.48–9.88% in six measured comparisons; Python uses that
  core API when rebuilt. Browser worker timings are measured separately.
  [Native cached C2M traversal bounds](../docs/cpu-c2m-native-bounds.md) add
  median paired reductions of 0.58–13.43% against that adopted native baseline,
  with identical tested outputs in all six comparisons.
  [Native BVH preparation parallelism](../docs/cpu-c2m-native-preparation.md)
  preserves outputs/capacities and improves PandaSet/Autoware by 3.39–5.56%
  in both native passes; Tokyo results are mixed. Large native meshes use
  indexed centroid/corner work alongside the adopted traversal improvements.
  [Bottom-up C2M mesh bounds](../docs/cpu-c2m-bound-merge.md) avoid repeated
  internal-node vertex scans and further improve fresh native C2M calls by
  26.07–45.36% across both passes of three measured sources, with all 60 timed
  pairs faster. Shared native/WASM outputs and GPU buffer exports match;
  zero/nonfinite meshes retain the original bounds calculation.
  [Native parallel BVH construction](../docs/cpu-c2m-native-builder.md)
  builds independent child subtrees in disjoint preorder ranges, preserving
  node bits/order and capacities. Fresh native C2M calls improve by another
  14.49–33.45% across six comparisons against the bottom-up-bounds baseline,
  with all 60 timed pairs faster. Smaller meshes keep the sequential builder.
  [Native C2M seed-triangle deduplication](../docs/cpu-c2m-native-seed.md)
  preserves tested hit bits but remains off after mixed results in two passes:
  only 33/60 timed pairs improve, with direction changes on PandaSet/Autoware.
  [Native C2M Morton preparation](../docs/cpu-c2m-native-morton.md) preserves
  exact bound bits, equal-key order and capacities with parallel scans/key
  generation, but also remains off: only 30/60 timed pairs improve.
  [Native C2M compact temporary results](../docs/cpu-c2m-native-results.md)
  halve temporary query-result payload from 16 to 8 bytes per point on the
  measured 64-bit native build. Whole native calls improve by 1.68–4.86%
  across six comparisons, with 45/60 faster pairs and identical outputs.
  [Native C2M parallel result restoration](../docs/cpu-c2m-native-parallel-results.md)
  preserves tested bits with an inverse Morton map and disjoint output gathers,
  but remains off: all six comparisons regress by 0.10–5.06%, with only
  18/60 faster pairs. Compact results continue to use serial scatter.
  [Native C2M push-time pruning](../docs/cpu-c2m-native-eager-prune.md)
  preserves tested bits and avoids already-pruned stack entries, but remains
  off: Tokyo improves by 1.41–2.20%, while PandaSet/Autoware regress in both
  passes by 2.33–4.26%, with only 23/60 faster pairs overall.
  [Native C2M previous-hit triangle seeds](../docs/cpu-c2m-native-triangle-seed.md)
  reuse original corners/edges within query chunks and preserve tested bits,
  but remain off after mixed results: 24/60 faster pairs, direction changes
  on PandaSet/Tokyo and 1.60–2.48% regressions in both Autoware passes.
  [Native parallel centroid bounds](../docs/cpu-c2m-native-centroid-bounds.md)
  preserve tested node bits, split axes/order and capacities, but remain off:
  Autoware improves by 2.58–3.86%, PandaSet changes direction, and Tokyo
  regresses by 0.45–1.36% in both passes, with 35/60 faster pairs overall.
  [Native original-vertex leaf access](../docs/cpu-c2m-native-indexed-corners.md)
  omits a logical 60.8–121.6 MB corner array and preserves tested bits/order,
  but remains off: all six comparisons regress by 5.05–18.47%, with only
  4/60 faster pairs. Native C2M keeps copied ordered corners.
  [Native C2M stage profiling](../docs/cpu-c2m-native-stages.md) now records
  two passes on all three measured sources: queries account for 57.07–84.71%,
  BVH construction for 10.95–31.47%, and allocation/scatter for 0.40–1.34%.
  All 192 outputs match an independent production oracle. This diagnostic
  changes no production policy; further experiments should prioritize queries.
  [Native C2M sibling-bound SIMD](../docs/cpu-c2m-native-bound-pair.md) adopts
  two f64 SSE2 lanes with NaN handling folded into clamping: all six whole-call
  comparisons improve by 3.81–6.46%, with 60/60 faster timed pairs and complete
  output parity. The preceding explicit-mask variant stays off after mixed
  Autoware results. Other targets retain scalar sibling calculations.
  [Native C2M paired projection dots](../docs/cpu-c2m-native-projection-pair.md)
  preserve tested bits but stay off: PandaSet regresses by 6.78–12.24%, Tokyo
  by 6.14–7.29%, and Autoware changes direction, with only 9/60 faster pairs.
  The adopted clamped sibling-bound SIMD remains enabled.
  [Native retained parallel C2M](../docs/cpu-c2m-native-retained-par.md) adds
  `PreparedMesh::distances_par` for repeated queries on owned immutable
  geometry/BVHs. All six native comparisons improve by 14.09–35.21%, with
  60/60 faster pairs and complete output parity. Initial construction and
  approximately 111–223 MB of additional BVH payload per measured mesh are
  recorded separately. This Rust API is opt-in; browser/Python selection stays
  unchanged.
  [Python prepared-mesh C2M](../docs/cpu-c2m-python-prepared-mesh.md) exposes
  the retained native kernel via `cloudanalyzer_core.PreparedMesh`. Complete
  Python calls improve by 11.77–48.96% in six measured-source comparisons,
  with 58/60 faster pairs and complete unsigned native-oracle parity. Native
  geometry copies/BVH retain approximately 134–264 MB per measured mesh;
  setup and caller NumPy arrays are recorded separately. This API is opt-in.
  [Owned C2C reference queries](../docs/cpu-c2c-python-prepared-reference.md)
  add `cloudanalyzer_core.PreparedReference` for changed queries against one
  immutable reference tree. Whole Python calls improve by 54.68–61.28% across
  six comparisons, with 60/60 faster pairs and full preceding-wheel output
  parity. Setup and approximately 18–32 MB of native retained payload are
  separate costs. Existing CLI/browser/WASM/GPU selection stays unchanged.
  [GPU C2M triangle candidates](../docs/gpu-c2m-candidates.md) preserve tested
  outputs with four readback bytes per point, but remain internal and off
  after slower whole-operation comparisons against that retained CPU baseline.
  [GPU normal/SOR block hints](../docs/gpu-knn-block-hints.md) implement exact
  candidate-range verification, but remain internal and off by default after
  slower measured whole-operation timings.
  [GPU nearest-k candidates](../docs/gpu-knn-candidates.md) select point slots
  and use a lighter exact CPU radius check, but also remain off after slower
  measured timings and increased search/readback work.
- Optional [experimental WebGPU C2C](../docs/gpu-c2c.md): GPU candidate
  search with exact original-coordinate `f64` verification and CPU fallback.
  GPU preparation starts when selected; the device/pipeline survives reference
  changes. Unchanged references and GPU buffers are reused. Exact subtree boxes
  and f64 SIMD verify GPU candidates. Large jobs use larger query batches
  to reduce readbacks, within device limits. CPU workers remain the default.
  **WebGPU + CPU workers** additionally offers
  [parallel exact verification](../docs/gpu-c2c-verification.md), with cached
  f64 reference trees, snapshot-safe jobs and serial recovery on worker failure.
  [Validated exact-tree transfer](../docs/gpu-c2c-verifier-setup.md) avoids
  repeating reference-tree sorting in each verification worker.
  Imports also [reuse validated subtree ranges](../docs/gpu-c2c-import-bounds.md)
  to reduce verification-worker preparation.
  A [resident analysis lane comparison](../docs/gpu-c2c-main-worker.md) reduces
  reference replication, but remains opt-in after mixed warm timings.
  [Parallel batch sizing](../docs/gpu-c2c-parallel-batches.md) reduces readback
  and lane-message rounds for larger jobs with four or more workers.
  [Two readback slots](../docs/gpu-c2c-buffered-readback.md) can submit the next
  batch before mapping completes; this remains an internal opt-in after
  variable complete-call timings.
- Large C2C jobs are split into spatially compact parts that run on a pool of
  WASM workers (no SharedArrayBuffer, so it works on static hosting such as
  GitHub Pages). Results are exact: each part receives every reference point
  that can be nearest to its queries. Completed slices are joined while
  remaining workers calculate, without retaining all slice results; see
  [CPU C2C](../docs/cpu-c2c.md) for validation and measured comparisons.
  [Prepared CPU C2C](../docs/cpu-c2c-preparation.md) retains one unchanged
  source pair's partitioned queries and worker trees for repeated calculations.
  C2C and Map quality share one idle preparation group.
  Coordinate edits/removal invalidate preparation; every run still computes
  all distances and exact summaries.
- The octree index of clouds above 1M points and statistical outlier removal
  run on the same pool. The index is built in three steps (sort slices by
  level-2 node, build each node's subtree, stitch); SOR splits the cloud along
  the octree, and points near a part's border ask the workers still holding
  the neighbouring parts, so the result matches the single-threaded filter
  exactly. On 10M points with 8 workers, SOR takes 4.3 s instead of 17.9 s.
  Parallel normal results now scatter without per-point views, and SOR packs
  boundary queries directly and reuses candidate merge buffers. Measured-source
  comparisons show normal whole-call gains of about 4–12%; SOR's join phases
  improve, while whole-call gains vary. See [normal/SOR joins](../docs/cpu-knn-join.md).
  Unchanged geometry also shares partitioned trees and query orders between
  normal/SOR runs, with one idle preparation group under 128 MiB. Measured repeat
  calls improved by about 8–13% for normals and 5–9% for SOR; coordinate edits,
  removal and worker replacement invalidate preparation. See
  [normal/SOR preparation](../docs/cpu-knn-preparation.md).
  Exact k-neighbour candidate insertion now copies displaced hits once instead
  of swapping the new hit at every step. With both builds reusing preparation,
  measured k=24 calls improved by about 12–18% for normals and 7–16% for SOR;
  see [candidate insertion](../docs/cpu-knn-insert.md) for controls and limitations.
  Parallel SOR now joins only the required prefix of sorted candidate lists.
  Whole measured calls improved by about 4–10%, with exact retained points;
  see [SOR prefix joins](../docs/cpu-sor-prefix.md).
  Skipping unused SOR neighbour-index conversion was also measured, but
  remains disabled after mixed whole-call gains; see
  [SOR distance-only queries](../docs/cpu-sor-slots.md).
  Ordinary WASM normal fits now process two independent eigensolvers together.
  Whole measured calls improved by about 6–11%, preserving every normal bit;
  see [paired normal fits](../docs/cpu-normal-pair.md).
  Combining ancestral split-plane bounds was also measured, but remains off by
  default because whole-call gains were inconsistent. See
  [combined k-neighbour bounds](../docs/cpu-knn-bounds.md).
  A triangular point-to-plane ICP matrix was tested too. Whole-call gains were
  inconsistent, so the full matrix remains standard; see
  [ICP normal equations](../docs/cpu-icp-normal-equations.md).
  Optional paired WASM leaves for ICP correspondences preserve tested outputs
  but also remain disabled after inconsistent real-source timings; see
  [ICP correspondence SIMD](../docs/cpu-icp-simd.md).
  Prior-iteration ICP correspondence radii also remain disabled after checks
  against the preceding production build; see
  [ICP temporal radii](../docs/cpu-icp-temporal.md).
  A shared radius-state layout also remains disabled after slower whole-call
  measurements; see [ICP radius state](../docs/cpu-icp-radius-context.md).
  Paired columns in the full ICP matrix also remain disabled after gains
  varied across measured sources; see
  [ICP matrix SIMD](../docs/cpu-icp-plane-simd.md).
  Compact octree occupancy reduces its default scratch table from 8 MiB to
  256 KiB, but remains disabled after mixed whole-call measurements. See
  [octree occupancy](../docs/cpu-octree-occupancy.md).
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
