# `ca rendered-evaluate`

Render a 3D Gaussian Splatting PLY at supplied camera poses, score the
images photometrically (PSNR / SSIM / LPIPS), and optionally run geometry
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
| `--metrics psnr,ssim,lpips` | Photometric metrics (`lpips` needs `[gs]`) |
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

## Related

- [`ca image-evaluate`](image-evaluate.md) — score two existing image directories
- [`ca geometry-evaluate`](geometry-evaluate.md) — cross-representation geometry QA
- [Public 3DGS demo](https://rsasaki0109.github.io/CloudAnalyzer/demo/3dgs/) — browser walkthrough
