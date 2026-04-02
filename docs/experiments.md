# Experiments

These comparisons keep stable `core/` contracts small and leave competing designs under `experiments/`.

## web_point_cloud_reduction

Reduce large point clouds for `ca web` without fixing the abstraction too early.

Stable code lives in `cloudanalyzer/ca/core/web_sampling.py`. Discardable variants live in `cloudanalyzer/ca/experiments/web_sampling`.

### Shared Inputs

| Dataset | Points | Budget | Purpose |
|---|---:|---:|---|
| structured_plane | 8100 | 1800 | Dense planar surface with low noise. |
| clustered_room | 7200 | 1600 | Separated spatial clusters with uneven density. |
| corridor_scan | 9500 | 2000 | Long corridor geometry with walls and floor. |

### Strategy Comparison

| Strategy | Design | Avg runtime ms | Avg chamfer | Avg coverage p95 | Readability | Extensibility | Composite rank |
|---|---:|---:|---:|---:|---:|---:|---:|
| random_budget | oop | 0.3548 | 0.123013 | 0.278898 | 81.70 | 50.00 | 1.300 |
| hybrid_pipeline | pipeline | 0.8648 | 0.204581 | 0.262417 | 69.70 | 60.12 | 1.850 |
| functional_voxel | functional | 12.1023 | 0.214913 | 0.240124 | 54.25 | 55.75 | 2.850 |

### Notes

- Geometry quality uses original-to-reduced coverage plus reduced-to-original fidelity.
- Readability and extensibility scores are heuristic and generated from AST/source-shape metrics.
- The selected stable strategy is extracted only after comparing concrete implementations.


## web_trajectory_sampling

Reduce `ca web` trajectory overlays without losing inspection-critical poses.

Stable code lives in `cloudanalyzer/ca/core/web_trajectory_sampling.py`. Discardable variants live in `cloudanalyzer/ca/experiments/web_trajectory_sampling`.

### Shared Inputs

| Dataset | Points | Budget | Preserve | Purpose |
|---|---:|---:|---:|---|
| straight_corridor | 1600 | 120 | 1 | Mostly straight trajectory where aggressive decimation is acceptable. |
| right_angle_turn | 1898 | 140 | 2 | Single sharp turn that should remain visible after simplification. |
| switchback | 2073 | 150 | 2 | Repeated turns where turn preservation matters more than pure stride. |

### Strategy Comparison

| Strategy | Design | Avg runtime ms | Avg mean error | Avg p95 error | Preserve ratio | Readability | Extensibility | Composite rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| turn_aware | oop | 0.6766 | 0.000000 | 0.000000 | 1.0000 | 44.00 | 38.70 | 1.900 |
| uniform_stride | functional | 0.1636 | 0.000549 | 0.000000 | 1.0000 | 95.85 | 59.40 | 2.000 |
| distance_accumulator | pipeline | 4.1397 | 0.000543 | 0.000000 | 1.0000 | 51.35 | 58.70 | 2.100 |

### Notes

- Quality uses reconstructed position error together with path-length preservation.
- Preserve ratio tracks whether inspection-critical anchors survive simplification.
- Readability and extensibility scores are heuristic and generated from AST/source-shape metrics.


## web_progressive_loading

Load `ca web` point clouds with a small initial payload and bounded deferred chunks.

Stable code lives in `cloudanalyzer/ca/core/web_progressive_loading.py`. Discardable variants live in `cloudanalyzer/ca/experiments/web_progressive_loading`.

### Shared Inputs

| Dataset | Points | Initial | Chunk | Purpose |
|---|---:|---:|---:|---|
| corridor_run | 2560 | 180 | 220 | Long corridor where initial payload should cover the full extent. |
| clustered_yard | 972 | 160 | 180 | Separated clusters where front-loading one region is visibly bad. |
| multi_level_room | 2095 | 220 | 260 | Ground plane plus elevated structure that benefits from spatial coverage. |

### Strategy Comparison

| Strategy | Design | Avg runtime ms | Initial coverage p95 | Progressive coverage AUC | Chunk std | Readability | Extensibility | Composite rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| distance_shells | radial | 0.7776 | 0.881581 | 0.400205 | 38.165523 | 62.10 | 38.60 | 1.500 |
| grid_tiles | grid | 1.0763 | 1.596086 | 0.661457 | 38.165523 | 60.50 | 34.50 | 2.200 |
| spatial_shuffle | functional | 0.9923 | 1.596086 | 0.661457 | 38.165523 | 68.50 | 55.60 | 2.300 |

### Notes

- Quality is dominated by how well the initial payload covers the whole cloud.
- Progressive coverage AUC tracks how quickly spatial holes disappear as chunks arrive.
- Chunk size standard deviation penalizes visibly uneven deferred loads.


## check_scaffolding

Generate starter `cloudanalyzer.yaml` files without locking config authoring into a single large abstraction.

Stable code lives in `cloudanalyzer/ca/core/check_scaffolding.py`. Discardable variants live in `cloudanalyzer/ca/experiments/check_scaffolding`.

### Shared Inputs

| Profile | Expected checks | Purpose |
|---|---:|---|
| mapping | 1 | Single artifact QA slice for map post-processing. |
| localization | 1 | Single trajectory QA slice for localization runs. |
| perception | 1 | Single artifact QA slice for 3D reconstruction output. |
| integrated | 4 | Combined mapping, localization, perception, and integrated run gate. |

### Strategy Comparison

| Strategy | Design | Avg runtime ms | Fidelity | Avg yaml lines | Readability | Extensibility | Composite rank |
|---|---:|---:|---:|---:|---:|---:|---:|
| static_profiles | functional | 0.0042 | 1.0000 | 26.50 | 67.70 | 43.03 | 1.350 |
| object_sections | oop | 0.0265 | 1.0000 | 26.50 | 25.10 | 81.40 | 2.200 |
| pipeline_overlays | pipeline | 0.8375 | 1.0000 | 26.00 | 48.76 | 89.90 | 2.450 |

### Notes

- Fidelity is measured by parsing rendered YAML through `load_check_suite` and checking expected ids, kinds, and output paths.
- Runtime covers template rendering only, not file writing.
- Readability and extensibility scores are heuristic and generated from AST/source-shape metrics.

