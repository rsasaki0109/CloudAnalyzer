# `ca rendered-evaluate`

Render a 3D Gaussian Splatting PLY at supplied camera poses, score the
images photometrically (PSNR / SSIM / LPIPS / FCM), and optionally run geometry
QA against a reference point cloud — one command for the full 3DGS
regression gate.

```bash
ca rendered-evaluate scene.ply references/ \
  --cameras transforms.json \
  --reference-pointcloud reference.pcd \
  --metrics psnr,ssim,lpips \
  --report rendered-report.html
```

Requires the optional gsplat backend:

```bash
pip install "cloudanalyzer[gs]"
```

## Pipeline

```text
3DGS PLY
→ gsplat renderer (optional [gs] extra)
→ rendered PNGs
→ ca image-evaluate (PSNR / SSIM / LPIPS)
→ optional ca geometry-evaluate
→ combined HTML / JSON report
```

## Camera formats

| Input | Layout |
|---|---|
| nerfstudio / Instant-NGP | `transforms.json` with global `w`, `h`, `fl_x` or `camera_angle_x`, plus per-frame `transform_matrix` and `file_path` |
| COLMAP text | `cameras.txt` + `images.txt` in the same directory (pass either file or the directory) |

Pass `--cameras` as a file path or a directory containing one of the
layouts above.

## Common options

| Option | Description |
|---|---|
| `--cameras PATH` | Camera bundle (required) |
| `--reference-pointcloud PATH` | Optional reference scan for Chamfer / AUC / F1 geometry QA |
| `--metrics psnr,ssim,lpips,dreamsim_distance,frequency_consistency` | Image metrics; learned metrics need `[perceptual]`, FCM uses NumPy/SciPy (rendering itself needs `[gs]`) |
| `--opacity-threshold 0.5` | Drop low-alpha splats before rendering |
| `--geometry-opacity-threshold` | Separate opacity filter for geometry (defaults to render threshold) |
| `--geometry-voxel 0.05` | Voxel-downsample before geometry QA |
| `--rendered-dir PATH` | Keep rendered PNGs instead of using a temp directory |
| `--report PATH.html` | Combined photometric + geometry report |
| `--output-json PATH` | Full result payload |
| `--format-json` | Print JSON to stdout |

## Demo data

The bundled `benchmarks/3dgs/synthetic-room/` scene ships with
`transforms.json`, reference PNGs, and a reference scan. Regenerate with:

```bash
pip install -e './cloudanalyzer[gs]'
python scripts/build_synthetic_3dgs_demo.py --output benchmarks/3dgs/synthetic-room
```

Smoke test:

```bash
ca rendered-evaluate benchmarks/3dgs/synthetic-room/gaussians_dense.ply \
  benchmarks/3dgs/synthetic-room/reference \
  --cameras benchmarks/3dgs/synthetic-room/transforms.json \
  --reference-pointcloud benchmarks/3dgs/synthetic-room/reference.pcd \
  --metrics psnr,ssim
```

Rendering `gaussians_dense.ply` against its own reference views should
return SSIM ≈ 1.0 (PSNR = +∞ on identical pairs).

## CI integration

Add a gated check to `cloudanalyzer.yaml` (requires `cloudanalyzer[gs]` on the runner):

```yaml
checks:
  - id: splat-qa
    kind: rendered
    splat: outputs/scene.ply
    cameras: outputs/transforms.json
    reference_dir: baselines/renders/
    reference_pointcloud: baselines/reference.pcd
    metrics: psnr,ssim,lpips,frequency_consistency
    gate:
      min_psnr: 28.0
      min_ssim: 0.85
      max_frequency_consistency: 0.12
      max_chamfer: 0.15
```

Photometric and geometry means flow into `ca report-pr-comment` like other check kinds.
See [../ci.md](../ci.md).

Frequency consistency follows the paper-defined grayscale 5x5 LoG,
zero-padding, and normalized-Frobenius formulation. See
[Frequency Consistency Metric](frequency-consistency.md) for its exact
`[0, 1]` range and flat-reference policy.

For CPU-only CI runners (no gsplat/CUDA), set `skip_render: true` and point
`rendered_dir` at pre-rendered PNGs; geometry QA still runs against the splat PLY.

```yaml
checks:
  - id: splat-qa
    kind: rendered
    splat: outputs/scene.ply
    cameras: outputs/transforms.json
    rendered_dir: baselines/renders/
    reference_dir: baselines/renders/
    reference_pointcloud: baselines/reference.pcd
    skip_render: true
    gate:
      min_ssim: 0.99
      max_chamfer: 0.15
```

## Related

- [`ca image-evaluate`](image-evaluate.md) — score two existing image directories
- [`ca geometry-evaluate`](geometry-evaluate.md) — cross-representation geometry QA
- [Public 3DGS demo](https://rsasaki0109.github.io/CloudAnalyzer/demo/3dgs/) — browser walkthrough
