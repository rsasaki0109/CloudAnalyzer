# `ca geometry-evaluate`

Cross-representation geometry QA. Runs the same Chamfer / AUC / F1 pipeline as `ca evaluate`, but first normalizes the source artifact through a *representation adapter* so non-point-cloud inputs (Gaussian Splatting PLY exports today, mesh vertex / depth-derived clouds later) score against a reference scan without forcing manual conversion.

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
| `gaussian-points` | 3DGS PLY exports with an `opacity` property | Extracts Gaussian centers (xyz); applies sigmoid to opacity and (optionally) filters splats below `--opacity-threshold` |

Detection is intentionally conservative: a PLY only flips to `gaussian-points` when the header advertises an `opacity` property. Plain vertex-only PLYs stay as `point-cloud`.

The reference is always loaded as a point cloud — `--representation` only describes the *source* artifact.

## Options

| Option | Purpose |
|---|---|
| `--representation <kind>` | Override auto-detection (`auto` / `point-cloud` / `gaussian-points`) |
| `--opacity-threshold <0..1>` | Drop splats whose rendered alpha (sigmoid of opacity) is below this. Ignored for `point-cloud` |
| `--voxel <meters>` | Voxel-downsample the adapted source before scoring |
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

## What's deliberately out of scope (today)

- **Ellipsoid sampling**. Gaussian splats are oriented ellipsoids (xyz + scale + rotation), but the current adapter treats each splat as the center point only. Sampling additional points along the ellipsoid surface using `scale`/`rot` is a future enhancement; for cross-representation regression tracking the center-only proxy already captures most drift.
- **Mesh adapter**. Mesh vertex extraction is straightforward and will follow once the API stabilizes here.
- **Color / SH coefficients**. Only geometry is scored; colors and spherical-harmonic coefficients are ignored.
