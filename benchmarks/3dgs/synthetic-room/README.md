# synthetic-room (3DGS demo)

A tiny Gaussian Splatting-style sample to smoke-test ``ca geometry-evaluate``
without external data. Both PLY files are deterministic; rerun
``scripts/build_synthetic_3dgs_demo.py`` to reproduce them byte-for-byte.

| File | Shape | Purpose |
|---|---|---|
| `reference.pcd` | planar room + two walls | Reference scan that all geometry evaluations score against |
| `gaussians.ply` | reference centers + small noise; mixed opacity | Half the splats are "high alpha" (rendered alpha ≥ 0.6); the rest are "low alpha" (≤ 0.2) so the opacity filter has something to drop |
| `gaussians_dense.ply` | only the high-alpha half | Sanity case: should score very close to the reference even without opacity filtering |

Example commands:

```bash
# Auto-detect representation; no filtering — sees all splats.
ca geometry-evaluate benchmarks/3dgs/synthetic-room/gaussians.ply \
                    benchmarks/3dgs/synthetic-room/reference.pcd

# Filter out low-alpha splats before scoring (much closer to the reference).
ca geometry-evaluate benchmarks/3dgs/synthetic-room/gaussians.ply \
                    benchmarks/3dgs/synthetic-room/reference.pcd \
                    --opacity-threshold 0.5
```
