# Development Roadmap

Status snapshot: 2026-08-04. The roadmap is organized around the product goal
in [VISION.md](../VISION.md): reproducible, CI-ready QA for 3D perception,
mapping, and reconstruction outputs.

## Product thesis

CloudAnalyzer should be the protocol and evidence layer above SLAM, Open3D,
MapEval, evo, gsplat, and dataset-specific tooling. The highest-value work is
making results comparable, reproducible, and scalable; adding isolated metrics
without those properties is lower priority.

## Implemented in the current development workspace

- Unified CI gates for point clouds, maps, trajectories, images, rendered 3DGS,
  structure, detection, tracking, and uncertainty.
- Headless snapshot fallback: Open3D when a display exists, deterministic
  Matplotlib projection otherwise.
- TUM/CSV quaternion preservation, rotational ATE/RPE, and opt-in distance RPE.
- Explicit `cloudanalyzer.mapeval_awd_scs.v1` parameters and serialized AWD/SCS
  protocol metadata.
- Fixed-commit MapEval parity harness with a source-compatible comparison lane,
  fixture hashes, tolerance/resource records, and CI-safe no-binary behavior.
- Chunked LAS/LAZ/CSV reading, streaming voxel moments, streaming AWD/SCS, and
  an optional PDAL-backed remote COPC adapter.
- `cloudanalyzer.rendered_eval.v1` manifests containing input hashes, camera
  matrices, image pairing, renderer settings, and runtime versions.
- `cloudanalyzer.protocol.v1` validation and a common `evaluation_protocol`
  result block for point-cloud, map, trajectory, image, rendered, geometry,
  and combined-run evaluation commands.
- Versioned benchmark metrics (`cloudanalyzer.metrics.v1`) and local report-bundle
  validation.

The implementation is pending commit/review. The working tree must remain
compatible with the existing MIT license and optional CUDA/PDAL backends.

## Priority plan

### Agent-controlled mapping — current product priority (2026-10-08)

- Connect raw recordings to both point-cloud maps and HD-road drafts through
  persistent MCP/CLI mapping jobs; first validation uses bundled NCLT.
- Retain attempts, input/native/artifact hashes, quality holds and agent decisions.
- Compare source support alongside retained road extent and unchanged semantic priors.
- Extend the loop to point-map quality comparisons, road topology, equipment and
  known georeferenced frames, with real-data evidence for each step.
- Evaluate automation by completed map outputs, unresolved issues and operator
  effort, rather than the number of editor controls.

### P0 — release quality and headless CI

- Keep no-display, Xvfb, and optional CUDA paths green.
- Reconcile release docs and tag `v0.5.0-alpha.1` only after benchmark and
  leaderboard smoke tests are reproducible.
- Pin or lock supported dependency combinations and remove avoidable warnings.

### P1 — trajectory protocol validation

- Rotational/distance RPE is now wired through `traj-batch`, `run-evaluate`,
  `run-batch`, benchmark gates, and `ca check`.
- Add golden fixtures and an evo subprocess oracle for development validation
  only; evo remains outside the MIT runtime dependency graph.
- Document frame, quaternion convention, timestamp association, and units.

### P2 — MapEval parity validation

- The fixed-commit fixture, source-compatible lane, tolerance/resource report,
  and no-binary CI smoke are implemented in
  [`docs/commands/mapeval-parity.md`](commands/mapeval-parity.md).
- Run the optional upstream executable comparison on a dependency-complete
  labeled runner and retain the JSON report as release evidence.
- Keep reference-free plane/MME proxies in a separate experimental metric lane.

### P3 — large-scale artifact validation

- `PointChunkReader`, streaming reducers, and the optional PDAL COPC adapter
  are implemented; wire them into more geometry/check paths as needed.
- Benchmark 1M/10M/100M-point inputs and publish a memory budget.

### P4 — 3DGS protocol and geometry extensions

- Renderer/camera/image conventions are now in the rendered-evaluation
  manifest.
- Add optional gsplat depth/hit-distance/LiDAR evaluation when the API is stable.
- Maintain CPU pre-rendered golden tests and separate GPU integration jobs.

### P5 — benchmark and release operations extensions

- Suite manifests now carry dataset source/license/preparation/hash metadata,
  and report bundles have a versioned metrics schema plus validation.
- Add multi-sequence baseline comparison and calibrated threshold histories.
- Verify every bundle before publishing a static leaderboard row.

## Explicitly deferred

- Learned metrics without model-weight/version provenance and threshold calibration.
- A full C++ rewrite of CloudAnalyzer.
- Reimplementing SLAM, reconstruction, training, or a general-purpose viewer.
- Making GPL-3.0 tools such as evo or OpenVINS runtime dependencies.
- Replacing the static report/leaderboard model with a large web platform.
