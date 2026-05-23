# Decisions

## web_point_cloud_reduction

### Adopted

`random_budget` is adopted directly as the current core strategy.

Reason: Best composite rank with quality weighted ahead of speed, readability, and extensibility.

### Not Adopted

- `hybrid_pipeline` remains experimental. Quality rank=2, runtime rank=2, readability rank=2, extensibility rank=1.
- `functional_voxel` remains experimental. Quality rank=3, runtime rank=3, readability rank=3, extensibility rank=2.

### Trigger To Re-run

- Browser point budget changes materially.
- New sampling strategy is proposed.
- `ca web` starts preserving additional attributes that change the reduction trade-off.

## web_trajectory_sampling

### Adopted

`turn_aware` is adopted directly as the current core strategy.

Reason: Best composite rank while preserving anchors and reducing geometric distortion.

### Not Adopted

- `uniform_stride` remains experimental. Quality rank=3, preserve rank=1, runtime rank=1, readability rank=1, extensibility rank=1.
- `distance_accumulator` remains experimental. Quality rank=2, preserve rank=2, runtime rank=3, readability rank=2, extensibility rank=2.

### Trigger To Re-run

- Browser trajectory overlay budget changes materially.
- `ca web` adds new trajectory inspection anchors or overlay modes.
- A new simplification strategy is proposed.

## web_progressive_loading

### Adopted

`distance_shells` is adopted directly as the current core strategy.

Reason: Best initial spatial coverage with acceptable planning cost and no extra chunk imbalance.

### Not Adopted

- `grid_tiles` remains experimental. Quality rank=2, chunk balance rank=1, runtime rank=3, readability rank=3, extensibility rank=3.
- `spatial_shuffle` remains experimental. Quality rank=3, chunk balance rank=2, runtime rank=2, readability rank=1, extensibility rank=1.

### Trigger To Re-run

- Initial browser payload budget changes materially.
- `ca web` adds camera-aware or view-dependent loading.
- A new chunk planning strategy is proposed.

## check_scaffolding

### Adopted

`static_profiles` is adopted directly as the current core strategy.

Reason: Literal profiles preserve full fidelity while keeping runtime and source complexity low for the current onboarding scope.

### Not Adopted

- `object_sections` remains experimental. Quality rank=2, runtime rank=2, compactness rank=3, readability rank=3, extensibility rank=2.
- `pipeline_overlays` remains experimental. Quality rank=3, runtime rank=3, compactness rank=1, readability rank=2, extensibility rank=1.

### Trigger To Re-run

- `ca init-check` gains user-supplied placeholders or path inference.
- New starter profiles are added beyond mapping, localization, perception, and integrated.
- Config generation needs structured customization beyond static scaffolds.

## check_regression_triage

### Adopted

`severity_weighted` is adopted directly as the current core strategy.

Reason: Best composite rank while preserving direct threshold-based ordering and the lowest runtime for CLI use.

### Not Adopted

- `signature_cluster` remains experimental. Quality rank=2, stability rank=3, runtime rank=2, readability rank=2, extensibility rank=3.
- `pareto_frontier` remains experimental. Quality rank=3, stability rank=2, runtime rank=3, readability rank=3, extensibility rank=1.

### Trigger To Re-run

- `ca check` starts triaging root-cause groups instead of individual failures.
- Batch checks expose richer aggregate metrics or per-item drill-down metadata.
- A new failure-ranking strategy is proposed.

## check_baseline_evolution

### Adopted

`stability_window` is adopted directly as the current core strategy.

Reason: Best composite rank by avoiding premature promotions while preserving perfect reject/promote accuracy on the shared scenarios.

### Not Adopted

- `threshold_guard` remains experimental. Quality rank=2, stability rank=1, runtime rank=2, readability rank=1, extensibility rank=3.
- `pareto_promote` remains experimental. Quality rank=3, stability rank=2, runtime rank=1, readability rank=3, extensibility rank=1.

### Trigger To Re-run

- Baseline promotion starts using historical cost, latency, or artifact size in addition to QA metrics.
- `ca check` emits richer per-check confidence or uncertainty metadata.
- A new promote / keep / reject strategy is proposed.

## ground_segmentation_evaluate

### Adopted

`voxel_confusion` is the stabilized core form of `nearest_neighbor`.

Reason: Best composite rank with the fastest runtime and robust voxel-level matching that avoids per-point distance computation.

### Not Adopted

- `voxel_confusion` remains experimental. Quality rank=2, stability rank=2, runtime rank=1, readability rank=1, extensibility rank=3.
- `height_band` remains experimental. Quality rank=3, stability rank=3, runtime rank=2, readability rank=3, extensibility rank=1.

### Trigger To Re-run

- Ground segmentation evaluation needs per-point rather than per-voxel resolution.
- Height-band diagnostics become a first-class output for slope analysis.
- A new matching or scoring strategy is proposed.

## map_evaluate

### Adopted

- `nn_thresholds` (reference-based, GT-aware MapEval-style accuracy/completeness@τ).
  Promoted to `ca/core/map_evaluate.py` as `NNThresholdMapEvaluateStrategy`.

### Not Adopted

- `voxel_entropy` (reference-free self-consistency proxy) stays under `ca/experiments`
  as the orthogonal GT-free lane until a single reference-free metric is settled.

### Trigger To Promote `voxel_entropy`

- Pick one GT-free metric (entropy / structure / MME) as the canonical lane.
- Define a stable failure-mode contract (when does a GT-free score block CI?).

## slam_run

### Adopted

- `kiss_icp` (`KissICPSlamDriver`). Wraps `kiss-icp` and is also
  exposed from `ca/core/slam_run.py` so `cloudanalyzer_cli` and
  `ca.benchmark` can depend only on `ca.core`.

### Not Adopted

- `kiss_slam` (`KissSLAMSlamDriver`). Wraps `kiss-slam` (KISS-ICP
  + pose-graph optimization + MapClosures loop closure). On the
  short synthetic trajectories the slice ships, sensor displacement
  from origin never crosses the local-map splitting distance, so
  KISS-SLAM degenerates to one round of PGO over the KISS-ICP
  odometry chain and produces the same trajectory KISS-ICP does.
  Held in experiments and re-evaluated once real-drift / revisit
  data lands.
- `small_gicp` (`SmallGICPSlamDriver`). Wraps `small_gicp`.
  Scan-to-map VGICP using a `GaussianVoxelMap` as the
  registration target. After the Phase 27 upgrade from
  scan-to-scan it also clears the synthetic-figure8 gate.
  Held in experiments alongside `kiss_slam` because real-data
  dogfood (KITTI / Newer-College drift / revisits) hasn't yet
  separated the three drivers on anything but synthetic
  geometry.
- `identity_passthrough` is a sentinel: it returns identity poses
  and concatenates the input frames as the 'map'. Its job is to
  fail loudly on any case that has non-trivial motion so that a
  regression in the real driver doesn't slip through.

### Triggers To Reconsider

- KITTI / Newer-College mini fixtures get wired through
  `ca slam-run` and KISS-SLAM's loop-closure / pose-graph kicks
  in. The KISS-ICP vs KISS-SLAM gap on those sequences flips the
  default driver.
- A latency-sensitive use case lands (e.g. a real-time CI
  budget) where `small_gicp`'s lower per-frame cost matters more
  than its higher drift. The slice could then promote driver
  selection from "single core driver" to "core driver per
  budget".
