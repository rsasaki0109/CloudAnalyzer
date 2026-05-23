# Changelog

All notable changes to CloudAnalyzer are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.2.0]: https://github.com/rsasaki0109/CloudAnalyzer/releases/tag/v0.2.0
[0.1.0]: https://github.com/rsasaki0109/CloudAnalyzer/releases/tag/v0.1.0
