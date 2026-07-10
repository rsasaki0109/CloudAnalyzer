# `ca plane-consistency` (experimental)

Evaluate local plane repeatability in one point-cloud map without a ground
truth map:

```bash
ca plane-consistency map.pcd --voxel-size 1.0 --format-json
```

The command reports two deterministic, lower-is-better proxies:

- `plane_normal_dispersion`: sign-invariant dispersion of PCA normals from
  locally planar voxel patches.
- `coplanar_offset_rmse`: weighted residual of parallel patch offsets after
  grouping nearby planes.

These metrics are inspired by the PNE/CPV topology direction described by
Ouyang et al., *A Novel Topology Metric for Indoor Point Cloud SLAM* (2023),
and its 2025 EAAI abstract. They are deliberately named differently: the full
2025 formulation is not publicly available, so CloudAnalyzer does not claim a
faithful PNE/CPV reproduction. MME remains available through `ca mme`.

```yaml
checks:
  - id: indoor-structure
    kind: structure
    source: output/map.pcd
    gate:
      voxel_size: 1.0
      max_plane_normal_dispersion: 0.15
      max_coplanar_offset_rmse: 0.05
```

The check participates in severity-aware triage and PR-comment metric deltas.
Sparse maps with no qualifying planar patches report unavailable (`NaN`)
metrics and fail any configured metric gate.

The example thresholds are illustrative only. Metric scale depends on voxel
size, point density, sensor noise, and scene structure; calibrate gates from a
representative accepted baseline captured with the same settings before using
them in CI.
