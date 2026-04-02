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
