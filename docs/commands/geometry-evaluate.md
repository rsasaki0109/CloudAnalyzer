# `ca geometry-evaluate`

Cross-representation geometry QA. Runs the same Chamfer / AUC / F1 pipeline as `ca evaluate`, but first normalizes the source artifact through a *representation adapter* so non-point-cloud inputs (3D Gaussian Splatting PLY exports, triangle meshes) score against a reference scan without forcing manual conversion.

```bash
ca geometry-evaluate exported_gaussians.ply reference_scan.pcd \
  --opacity-threshold 0.1 \
  --voxel 0.05 \
  --plot geometry_f1.png \
  --output-json geometry_qa.json
```

## Supported representations

| Value | When to use | Filters applied |
|---|---|---|
| `auto` (default) | Let the loader detect from file content | Same as the detected representation below |
| `point-cloud` | Plain PCD / PLY / LAS / LAZ / CSV | None — passes through `ca.io.load_point_cloud` |
| `gaussian-points` | 3DGS PLY exports with an `opacity` property | Extracts Gaussian centers (xyz); applies sigmoid to opacity and (optionally) filters splats below `--opacity-threshold`. With `--splat-method ellipsoid`, surface-samples each splat using `scale_*`/`rot_*` for a closer proxy to the rendered surface |
| `mesh` | Triangle mesh (OBJ / STL / GLB / GLTF, or PLY with a `face` element) | Surface-samples `--mesh-samples` points via Open3D (default: uniform; `poisson_disk` available) |

Auto-detection precedence:

1. `.obj` / `.stl` / `.glb` / `.gltf` → `mesh`
2. `.ply` with an `opacity` property → `gaussian-points`
3. `.ply` with a `face` element → `mesh`
4. Everything else → `point-cloud`

The mesh adapter samples points from the **surface**, not the vertex set, so flat faces get represented proportionally to their area. That's the difference that makes Chamfer / AUC meaningful for triangle meshes — sampling only vertices misses large quads and biases the score toward whichever face happens to be heavily subdivided.

The reference is always loaded as a point cloud — `--representation` only describes the *source* artifact.

## Options

| Option | Purpose |
|---|---|
| `--representation <kind>` | Override auto-detection (`auto` / `point-cloud` / `gaussian-points` / `mesh`) |
| `--opacity-threshold <0..1>` | Drop splats whose rendered alpha (sigmoid of opacity) is below this. Ignored for `point-cloud` / `mesh` |
| `--voxel <meters>` | Voxel-downsample the adapted source before scoring |
| `--mesh-samples <N>` | Number of points to sample from the mesh surface (default `100000`). Mesh representation only |
| `--mesh-method <name>` | Mesh sampling strategy: `uniform` (fast, default) or `poisson_disk` (slower, more uniform spread) |
| `--splat-method <name>` | 3DGS sampling strategy: `centers` (default, splat centers only) or `ellipsoid` (surface-sample each splat using `scale_*`/`rot_*`). `gaussian-points` only |
| `--splat-samples <K>` | Points sampled per splat in `ellipsoid` mode (default `8`). Ignored unless `--splat-method=ellipsoid` |
| `--thresholds 0.05,0.1,0.5` | Override the F1/AUC distance thresholds |
| `--plot <file.png>` | Save the F1 curve plot |
| `--output-json <file>` | Dump the full result dict (incl. `representation` block) |
| `--format-json` | Print the result dict to stdout |

## Result shape

`ca geometry-evaluate` returns the same dict as `ca evaluate` plus a `representation` block:

```json
{
  "source_path": "exported_gaussians.ply",
  "auc": 0.9731,
  "chamfer_distance": 0.0412,
  ...,
  "representation": {
    "requested": "auto",
    "detected": "gaussian-points",
    "original_count": 1248975,
    "final_count": 412330,
    "applied_filters": ["opacity>=0.1 kept=412330", "voxel=0.05 kept=412330"],
    "opacity_threshold": 0.1,
    "voxel_size": 0.05
  }
}
```

`ca report-pr-comment` already understands this shape — pipe the JSON straight into it for a PR Markdown blob that surfaces the representation alongside the Chamfer / AUC numbers.

## Demo data

A tiny synthetic scene ships at `benchmarks/3dgs/synthetic-room/`, regenerated deterministically by `scripts/build_synthetic_3dgs_demo.py`. It contains:

| File | Purpose |
|---|---|
| `reference.pcd` | Planar room with two walls; the scoring reference |
| `gaussians.ply` | Mixed-alpha 3DGS export (~half "high alpha", half "low alpha" noise) |
| `gaussians_dense.ply` | Same scene without the low-alpha noise; sanity-check input |

```bash
# See all splats — low-alpha noise hurts the score.
ca geometry-evaluate benchmarks/3dgs/synthetic-room/gaussians.ply \
                    benchmarks/3dgs/synthetic-room/reference.pcd

# Filter; score noticeably improves.
ca geometry-evaluate benchmarks/3dgs/synthetic-room/gaussians.ply \
                    benchmarks/3dgs/synthetic-room/reference.pcd \
                    --opacity-threshold 0.5
```

## Mesh examples

```bash
# Score an OBJ export against a reference scan; defaults to 100k uniform samples.
ca geometry-evaluate my_reconstruction.obj reference_scan.pcd

# Tighten samples for higher-fidelity scoring (slower); use Poisson disk for
# more uniform coverage on highly anisotropic meshes.
ca geometry-evaluate my_reconstruction.obj reference_scan.pcd \
  --mesh-samples 500000 --mesh-method poisson_disk \
  --thresholds 0.01,0.05,0.10

# A PLY with face elements is auto-detected as mesh.
ca geometry-evaluate textured_mesh.ply reference_scan.pcd --output-json mesh_qa.json
```

The result's `representation` block records `mesh_samples` and `mesh_method` so downstream consumers (PR comments, dashboards) can show which sampling settings produced the numbers.

## 3DGS ellipsoid sampling

By default the `gaussian-points` adapter uses splat centers only (`--splat-method centers`). For thin/elongated splats this underestimates how much surface each one covers. Opt in to `--splat-method ellipsoid` to surface-sample each splat using the standard 3DGS PLY properties `scale_0..2` (log-σ) and `rot_0..3` (`wxyz` quaternion):

```bash
ca geometry-evaluate exported_gaussians.ply reference_scan.pcd \
  --splat-method ellipsoid --splat-samples 12 \
  --opacity-threshold 0.1
```

`--splat-samples K` sets how many quasi-uniform points (Fibonacci lattice on the unit sphere, then scaled by `exp(scale)` and rotated by the quaternion) are emitted per splat. Total source-side points become `K × number_of_surviving_splats`, so keep `K` modest (default `8`) on million-splat exports and pair with `--voxel` if needed.

The `representation` block in the JSON output records `splat_method` and (when ellipsoid) `splat_samples`.

## What's deliberately out of scope (today)

- **Reproducible mesh seeds**. Open3D ≤0.19 doesn't expose a seed for `sample_points_uniformly` / `sample_points_poisson_disk`, so two runs of the same mesh produce slightly different point sets. The *surface* matches; the Chamfer / AUC delta between runs is well under any meaningful gate threshold, but bit-reproducibility is on hold until Open3D adds the parameter.
- **Color / SH coefficients**. Only geometry is scored; colors and spherical-harmonic coefficients are ignored.
