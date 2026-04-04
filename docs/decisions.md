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

- `threshold_guard` remains experimental. Quality rank=2, stability rank=1, runtime rank=3, readability rank=1, extensibility rank=3.
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
