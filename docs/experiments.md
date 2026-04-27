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
| random_budget | oop | 0.4479 | 0.123013 | 0.278898 | 81.70 | 50.00 | 1.300 |
| hybrid_pipeline | pipeline | 1.1354 | 0.204581 | 0.262417 | 69.70 | 60.12 | 1.850 |
| functional_voxel | functional | 17.4749 | 0.214913 | 0.240124 | 54.25 | 55.75 | 2.850 |

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
| turn_aware | oop | 0.9043 | 0.000000 | 0.000000 | 1.0000 | 44.00 | 38.70 | 1.900 |
| uniform_stride | functional | 0.1817 | 0.000549 | 0.000000 | 1.0000 | 95.85 | 59.40 | 2.000 |
| distance_accumulator | pipeline | 5.4872 | 0.000543 | 0.000000 | 1.0000 | 51.35 | 58.70 | 2.100 |

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
| distance_shells | radial | 1.1460 | 0.881581 | 0.400205 | 38.165523 | 62.10 | 38.60 | 1.500 |
| grid_tiles | grid | 1.4921 | 1.596086 | 0.661457 | 38.165523 | 60.50 | 34.50 | 2.050 |
| spatial_shuffle | functional | 1.5301 | 1.596086 | 0.661457 | 38.165523 | 68.50 | 55.60 | 2.450 |

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
| static_profiles | functional | 0.0039 | 1.0000 | 26.50 | 67.70 | 43.03 | 1.350 |
| object_sections | oop | 0.0209 | 1.0000 | 26.50 | 25.10 | 81.40 | 2.200 |
| pipeline_overlays | pipeline | 1.1249 | 1.0000 | 26.00 | 48.76 | 89.90 | 2.450 |

### Notes

- Fidelity is measured by parsing rendered YAML through `load_check_suite` and checking expected ids, kinds, and output paths.
- Runtime covers template rendering only, not file writing.
- Readability and extensibility scores are heuristic and generated from AST/source-shape metrics.


## check_regression_triage

Rank failed mapping, localization, and perception checks so `ca check` surfaces the most informative regression first.

Stable code lives in `cloudanalyzer/ca/core/check_triage.py`. Discardable variants live in `cloudanalyzer/ca/experiments/check_triage`.

### Shared Inputs

| Dataset | Failed checks | Expected top order | Purpose |
|---|---:|---|---|
| integrated_cascade | 3 | integrated-run, localization-run, mapping-postprocess | Integrated run failure should outrank milder single-artifact and trajectory regressions. |
| batch_tradeoff | 3 | run-batch, trajectory-batch, artifact-batch | Run-batch with several moderate failures should outrank a single-dimension collapse. |
| duplicate_geometry_regressions | 3 | mapping-postprocess, perception-output, localization-run | Near-duplicate geometry failures should stay below the single most severe one. |

### Strategy Comparison

| Strategy | Design | Avg runtime ms | Avg NDCG | Top1 hit | Stability | Diversity | Readability | Extensibility | Composite rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| severity_weighted | functional | 0.0281 | 0.9872 | 1.0000 | 1.0000 | 0.8889 | 77.50 | 38.42 | 1.100 |
| signature_cluster | pipeline | 0.0326 | 0.9744 | 1.0000 | 1.0000 | 0.8889 | 63.30 | 34.10 | 2.250 |
| pareto_frontier | oop | 0.0466 | 0.8846 | 0.6667 | 1.0000 | 0.8889 | 53.02 | 55.45 | 2.650 |

### Notes

- Quality is scored against expected failure orderings using a small NDCG variant plus top-1 hit rate.
- Stability checks whether the top-ranked failure stays stable under small metric perturbations.
- Diversity is tracked but not weighted heavily in the final ranking because current CLI use favors severity-first triage.


## check_baseline_evolution

Decide whether a candidate QA summary should promote, keep, or reject a baseline revision.

Stable code lives in `cloudanalyzer/ca/core/check_baseline_evolution.py`. Discardable variants live in `cloudanalyzer/ca/experiments/check_baseline_evolution`.

### Shared Inputs

| Dataset | Expected decision | History size | Purpose |
|---|---|---:|---|
| stable_improvement_window | promote | 2 | Candidate should be promoted after a stable passing window with stronger margins. |
| candidate_failure_reject | reject | 1 | Candidate that fails the quality gate should be rejected immediately. |
| insufficient_history_keep | keep | 1 | A strong candidate without enough history should stay in keep mode. |
| recent_failure_keep | keep | 2 | A recovering candidate should not be promoted immediately after a recent failure. |
| mixed_tradeoff_keep | keep | 2 | Candidate with a stronger mean margin but worse weakest margin should stay keep. |

### Strategy Comparison

| Strategy | Design | Avg runtime ms | Decision match | Critical match | Stability | False promote | False reject | Readability | Extensibility | Composite rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stability_window | pipeline | 0.0211 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 46.11 | 39.54 | 1.650 |
| threshold_guard | functional | 0.0206 | 0.6000 | 0.6000 | 1.0000 | 0.4000 | 0.0000 | 64.98 | 36.22 | 1.700 |
| pareto_promote | oop | 0.0312 | 0.6000 | 0.6000 | 1.0000 | 0.4000 | 0.0000 | 45.85 | 48.79 | 2.650 |

### Notes

- Decision quality compares each strategy against shared promote / keep / reject expectations.
- Stability checks whether small metric perturbations preserve the same decision.
- False promote is weighted more heavily than raw confidence because baseline drift is more damaging than delayed promotion.


## ground_segmentation_evaluate

Evaluate ground segmentation quality by comparing estimated ground/non-ground points against reference labels using precision, recall, F1, and IoU.

Stable code lives in `cloudanalyzer/ca/core/ground_evaluate.py`. Discardable variants live in `cloudanalyzer/ca/experiments/ground_evaluate`.

### Shared Inputs

| Dataset | Est ground pts | Ref ground pts | Expected min F1 | Purpose |
|---|---:|---:|---:|---|
| flat_ground_perfect | 200 | 200 | 0.99 | Flat ground with objects above; identical estimation and reference. |
| noisy_boundary | 185 | 200 | 0.70 | Ground segmentation with 10% ground leak and 5% nonground leak near boundary. |
| sloped_terrain_miss | 174 | 200 | 0.70 | Sloped ground where upper slope region is missed by the estimator. |

### Strategy Comparison

| Strategy | Design | Avg runtime ms | Avg F1 | Avg IoU | Avg precision | Avg recall | Stability | Readability | Extensibility | Composite rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nearest_neighbor | oop | 5.2420 | 0.9552 | 0.9160 | 0.9910 | 0.9233 | 1.0000 | 84.31 | 55.68 | 1.500 |
| voxel_confusion | functional | 0.4303 | 0.9539 | 0.9138 | 0.9907 | 0.9213 | 0.9969 | 95.50 | 40.22 | 1.850 |
| height_band | pipeline | 0.7086 | 0.9539 | 0.9138 | 0.9907 | 0.9213 | 0.9959 | 66.99 | 75.77 | 2.650 |

### Notes

- Quality compares each strategy against expected F1 thresholds on synthetic ground segmentation scenarios.
- Stability checks whether F1 stays close under small positional perturbations.
- Voxel-based approaches are faster but coarser; nearest-neighbor is precise but slower.


## map_evaluate

Evaluate point-cloud maps either against a reference (GT-based distance/coverage) or without GT (self-consistency proxies).

Experimental code lives in `cloudanalyzer/ca/experiments/map_evaluate/`.

### Shared Inputs

| Dataset | Has GT | Purpose |
|---|---:|---|
| gt_drift | true | Estimated map has small rigid drift relative to reference. |
| gt_incomplete | true | Estimated map misses a region; completeness should drop. |
| no_gt_self_consistency | false | No GT; fused map from two noisy overlapping scans. |

### Notes

- This slice is MapEval-inspired: it keeps a threshold list (accuracy levels) and separates GT-based vs GT-free evaluation.
- Current implementations are lightweight proxies; real-map scale should switch to KD-trees and richer outputs.

