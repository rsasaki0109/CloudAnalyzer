# Changelog

All notable changes to CloudAnalyzer are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`ca slam-run`** (Phase 21) — drive a LiDAR-odometry pipeline
  end-to-end on a sequence of scans and emit the trajectory + accumulated
  map that the rest of CloudAnalyzer's evaluation stack already consumes.
  The default driver wraps the [KISS-ICP](https://github.com/PRBonn/kiss-icp)
  package; an `--evaluate` flag pipes the result straight into
  `ca run-evaluate` against a reference map + trajectory. This is the
  first step of CloudAnalyzer driving the SLAM itself rather than only
  scoring third-party outputs. Install with `pip install 'cloudanalyzer[slam]'`.
- New experiment slice `ca/experiments/slam_run/`: `KissICPSlamDriver`
  (adopted, re-exported from `ca/core/slam_run.py`) and
  `IdentityPassthroughSlamDriver` (sentinel — identity poses and
  concatenated input frames; sets the slice's failure floor).
- New core module `ca/core/slam_run.py` with the `SlamRunRequest`,
  `SlamRunResult`, `SlamRunDriver` Protocol, frame loaders (KITTI `.bin`
  + `.pcd` / `.ply` / `.csv`), TUM trajectory + PLY map writers, and a
  vectorized rotation-matrix → quaternion helper.
- **End-to-end dogfood on `synthetic-figure8`** (Phase 22) — the bundled
  `benchmarks/slam/synthetic-figure8/` suite now ships raw per-frame
  scans under `scans/` (200 frames, ~5.5 MB), regenerated deterministically
  by `scripts/build_synthetic_slam_suite.py`. The full "scans → `ca slam-run`
  → `ca benchmark eval`" loop now passes the suite's default gate
  (AUC=1.00, ATE≈1.4 mm) from a clean checkout, locked in by
  `tests/test_slam_run.py::test_cli_slam_run_then_benchmark_eval_passes_synthetic_figure8`.

### Changed

- `scripts/build_synthetic_slam_suite.py`: `_planar_map()` now emits a
  fully boxed room (four walls at x=±8 and y=±8 instead of only two).
  The two-wall layout left east/west translation under-constrained, so
  KISS-ICP recovered figure-8 motion in the wrong direction. This
  regenerates `reference/map.pcd` and the derived `sample_outputs/`.

## [0.3.0] - 2026-05-24

Fast follow-on to `v0.2.0` focused on performance hardening of the hot paths,
a meaningful new 3DGS evaluation feature, and an architecture cleanup that
graduates `map_evaluate` to the stable core.

### Added

- **`ca geometry-evaluate --splat-method ellipsoid`** — opt-in splat-aware
  ellipsoid surface sampling for 3D Gaussian Splatting PLY exports. Reads the
  standard `scale_0..2` (log-σ) and `rot_0..3` (`wxyz` quaternion) properties
  and surface-samples each splat using a Fibonacci unit-sphere lattice scaled
  per axis and rotated by the quaternion. Default remains `--splat-method
  centers` (backward-compatible). New `--splat-samples K` (default 8) sets
  points per splat. Fully vectorized via `np.einsum` so million-splat exports
  stay practical.
- **`benchmarks/3dgs/synthetic-room/` now ships the full 3DGS PLY schema**
  (`x, y, z, opacity, scale_0..2, rot_0..3`), so `--splat-method ellipsoid`
  can be exercised against the checked-in demo without external data.

### Changed

- **Performance: vectorize PLY vertex parsing and split tile bucketing**
  (Phase 17). ASCII PLY now reads the entire vertex block via one
  `np.loadtxt` call; binary little-endian PLY uses `np.frombuffer` with a
  structured dtype. `ca split` builds tile buckets with
  `np.unique(axis=0)` + `np.argsort`. Real 3DGS exports (1M+ splats) and
  city-scale `ca split` runs that previously took ~1 minute now finish in
  well under a second.
- **Performance: vectorize voxel-based `ca ground-evaluate`** (Phase 18).
  `_voxel_keys` no longer builds a Python `set` of tuples per call. It now
  returns the unique `(M, 3) int64` ndarray from `np.unique(axis=0)`, and a
  new `_voxel_intersection_size` helper runs `np.intersect1d` on a void view
  of each row. The four per-evaluation set comprehensions that dominated
  city-scale ground QA are gone.
- **Architecture: `map_evaluate` graduates to `ca.core`** (Phase 20). The
  request/result contract, shared helpers, and adopted
  `NNThresholdMapEvaluateStrategy` move to `ca/core/map_evaluate.py`. The
  reference-free `voxel_entropy` lane stays under `ca/experiments` as the
  orthogonal GT-free option. `ca map-evaluate` no longer reaches into
  experiments; the experiment-side modules remain as thin re-exports for
  backward compatibility.

### Fixed

- `_opacity_keep_mask` and `_sample_splat_ellipsoids` annotated to satisfy
  the stricter `ndarray` return-type checks under Python 3.10 mypy.

## [0.2.0] - 2026-05-23

This release consolidates roughly half a year of work since `v0.1.0`: 132 commits
that turn CloudAnalyzer into an end-to-end QA / benchmark / operations layer for
mapping, localization, and perception 3D pipelines.

### Added

#### Map / SLAM evaluation

- `ca map-evaluate` — MapEval-inspired accuracy / completeness @ threshold
  metrics for reconstructed maps against a GT map; supports MapEval-style
  initial alignment and optional colored error PLY artifacts.
- `ca posegraph-validate` — lightweight g2o / TUM / key_point_frame sanity
  check for manual loop-closure sessions.
- `ca loop-closure-report` — before/after/ref map + trajectory + posegraph QA
  in one command, with session-root auto-discovery and a quality-gate exit
  code. Available as a `kind: loop_closure` check in `cloudanalyzer.yaml`.
- `ca scan-match-debug` — scan-to-map ICP/GICP diagnosis with artifact bundle.
- `ca slam-debug` — SLAM debug report generator with primary and secondary
  diagnosis labels.
- `ca mme` — reference-free Mean Map Entropy via parallel scipy KDTree
  (contributed by @Taeyoung96 in #6).

#### Benchmarks

- `ca benchmark info` / `ca benchmark eval` — point a SLAM pipeline's map and
  trajectory at a frozen suite (reference + gate) for a one-command regression
  check; ships with a synthetic figure-8 suite under
  `benchmarks/slam/synthetic-figure8/`.
- `ca benchmark init` — build a `suite.yaml` from a local reference map and
  trajectory; works with arbitrary on-disk references.
- Newer College Dataset and KITTI Odometry wrappers
  (`scripts/prepare_newer_college_mini.py`, `scripts/prepare_kitti_mini.py`)
  with mini suite scaffolding under `benchmarks/slam/`.

#### Cross-representation geometry QA

- `ca geometry-evaluate` — same Chamfer / AUC / F1 metrics as `ca evaluate`,
  but first normalizes the source through a representation adapter so
  non-point-cloud inputs can be scored against a reference scan.
  - 3D Gaussian Splatting PLY adapter (opacity-aware filtering).
  - Triangle mesh adapter (OBJ / STL / GLB / GLTF / PLY-with-faces;
    surface-sampled, not vertex-only).
  - `--representation auto|point-cloud|gaussian-points|mesh` selector; output
    carries a `representation` block consumed by `ca report-pr-comment`.

#### CI / automation

- `ca report-pr-comment` — turn any CloudAnalyzer summary JSON
  (`ca check`, `ca run-evaluate`, `ca benchmark eval`) into a Markdown PR
  comment with worst-regression-first triage and `↑/↓` deltas against a baseline.
- `ca bundle pack` / `show` / `unpack` / `diff` — freeze one QA run (summary
  JSON, per-check reports, optional baseline) into a single `qa_bundle.zip`
  with a metadata header (project, commit, PR, notes). Reopenable without the
  original CI workspace.
- `ca history` — trend gate metrics across many bundles sorted by
  `metadata.created_at`; works on a rolling archive populated by
  `ca bundle pack` over time.
- New reusable workflows: `pr-comment.yml`, `self-qa.yml`,
  `config-quality-gate.yml`, `baseline-gate.yml`. PR comments idempotently
  update the same marker so re-runs don't stack duplicates.

#### Perception QA

- `ca ground-evaluate` — ground / non-ground segmentation QA
  (Precision / Recall / F1 / IoU).
- `ca detection-evaluate` — 3D bounding box detection QA from JSON box
  sequences with per-IoU mAP gate.
- `ca tracking-evaluate` — 3D multi-object tracking QA (MOTA, ID switches)
  from JSON box sequences.
- Checked-in public RELLIS-3D seed frame under
  `demo_assets/public/rellis3d-frame-000001/` with deterministic
  detection / tracking JSON examples.

#### Public demo

- Static perception batch report at
  `https://rsasaki0109.github.io/CloudAnalyzer/demo/perception/`, regenerated
  by `scripts/build_perception_demo.py` and gated by CI for no-drift.

### Changed

- **Performance: vectorize NN distance paths with `scipy.spatial.cKDTree`**
  (Phase 16). Both `ca.experiments.map_evaluate.nn_thresholds` and
  `ca.stats.compute_stats` previously walked an Open3D KD-tree one point at a
  time from Python, which made real-sized maps impractical. Both now use
  batched `cKDTree.query`; 100k-point NN finishes in well under a second.
  API and metric values are unchanged; `sampling_policy.nn_backend` in
  `nn_thresholds` shifts `"open3d_kdtree"` → `"scipy_ckdtree"`.
- README: rewritten to point at `docs/commands/*.md` and `docs/ci.md` for
  long examples (772 → 430 lines) and brought up to date with all new
  subcommands.

### Fixed

- `pr-comment.yml`: replace unsupported `jq test()` call with `contains()` to
  unblock the reusable workflow.
- Pre-existing mypy errors cleaned up so `mypy ca/ cloudanalyzer_cli/` stays
  the CI gate.
- `build_public_pack` made deterministic so the public benchmark pack tests
  pass without external state.

### Contributors

- @Taeyoung96 — `ca mme` (#6).

## [0.1.0] - 2026-04-05

Initial public release. Establishes the core "process, then evaluate
immediately" CLI (`ca evaluate`, `ca downsample`, `ca filter`,
`ca traj-evaluate`, `ca run-evaluate`, `ca check`, `ca web`, ...) and the
core / experiments split documented in `docs/architecture.md`.

[0.3.0]: https://github.com/rsasaki0109/CloudAnalyzer/releases/tag/v0.3.0
[0.2.0]: https://github.com/rsasaki0109/CloudAnalyzer/releases/tag/v0.2.0
[0.1.0]: https://github.com/rsasaki0109/CloudAnalyzer/releases/tag/v0.1.0
