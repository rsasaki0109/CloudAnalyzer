# synthetic-room (3DGS demo)

A tiny Gaussian Splatting-style sample to smoke-test ``ca geometry-evaluate`` and
``ca rendered-evaluate`` without external data. Regenerate deterministically with
``scripts/build_synthetic_3dgs_demo.py`` (requires ``pip install 'cloudanalyzer[gs]'``
for reference PNG generation).

| File | Purpose |
|---|---|
| `reference.pcd` | Reference scan for geometry QA |
| `gaussians.ply` | Mixed-opacity 3DGS export |
| `gaussians_dense.ply` | High-opacity-only sanity case |
| `transforms.json` | nerfstudio camera poses for rendering |
| `reference/` | Ground-truth PNG renders of ``gaussians_dense.ply`` |

Example commands:

```bash
# Geometry-only QA
ca geometry-evaluate benchmarks/3dgs/synthetic-room/gaussians_dense.ply \
                    benchmarks/3dgs/synthetic-room/reference.pcd

# Render + photometric + geometry combined report
ca rendered-evaluate benchmarks/3dgs/synthetic-room/gaussians_dense.ply \
    benchmarks/3dgs/synthetic-room/reference \
    --cameras benchmarks/3dgs/synthetic-room/transforms.json \
    --reference-pointcloud benchmarks/3dgs/synthetic-room/reference.pcd \
    --metrics psnr,ssim,lpips --report /tmp/rendered-report.html

# CI gate (kind: rendered) — dogfooded in rendered-self-qa.yml
ca check benchmarks/3dgs/synthetic-room/configs/suite-rendered.cloudanalyzer.yaml
```
