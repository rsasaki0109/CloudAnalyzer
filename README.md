# CloudAnalyzer

[![Test](https://github.com/rsasaki0109/CloudAnalyzer/actions/workflows/test.yml/badge.svg)](https://github.com/rsasaki0109/CloudAnalyzer/actions/workflows/test.yml)
[![Self QA](https://github.com/rsasaki0109/CloudAnalyzer/actions/workflows/self-qa.yml/badge.svg)](https://github.com/rsasaki0109/CloudAnalyzer/actions/workflows/self-qa.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Catch SLAM and 3D perception regressions before they ship.**

CloudAnalyzer turns SLAM, LiDAR, point-cloud, perception, and 3DGS outputs into
**CI-grade QA evidence**: metrics, reports, and pass/fail gates.

<!-- Regenerate with `scripts/build_readme_gif.sh` (requires vhs and `ca`). -->
<p align="center">
  <img src="docs/images/readme-demo.gif" alt="CloudAnalyzer terminal demo" width="800">
</p>

<p align="center">
  <a href="#quick-start">Quick start</a> ·
  <a href="https://rsasaki0109.github.io/CloudAnalyzer/">Live demos</a> ·
  <a href="https://rsasaki0109.github.io/CloudAnalyzer/app/">Web viewer (beta)</a> ·
  <a href="docs/commands/">Docs</a>
</p>

<p align="center">
  <img src="docs/images/readme-workflow.svg" alt="CloudAnalyzer workflow" width="900">
</p>

## Why teams use CloudAnalyzer

- Compare candidate maps, trajectories, point clouds, and renders with a frozen reference.
- Export metrics JSON, an HTML report, and a deterministic CI gate.
- Keep provenance so every result can be reviewed and reproduced.

> **Checked-in proof:** `PASS` · Map AUC `1.0000` · Chamfer `0.0145 m` ·
> Trajectory ATE RMSE `0.0016 m` · [open the report](docs/leaderboard/runs/kiss-slam__synthetic-oval/report.html)

## Examples

<p align="center">
  <a href="https://rsasaki0109.github.io/CloudAnalyzer/demo/perception/">
    <img src="docs/images/readme-pointcloud-triptych.png" alt="Point-cloud comparison" width="1000">
  </a>
</p>

<table>
  <tr>
    <td width="50%"><a href="https://rsasaki0109.github.io/CloudAnalyzer/demo/perception/"><img src="docs/pr/perception_summary_card.png" alt="Perception report" width="100%"></a></td>
    <td width="50%"><a href="https://rsasaki0109.github.io/CloudAnalyzer/demo/compare/"><img src="docs/images/density_hdl_localization_map.png" alt="Point-cloud density map" width="100%"></a></td>
  </tr>
  <tr>
    <td width="50%"><a href="docs/commands/geometry-evaluate.md"><img src="docs/images/f1_hdl_localization_v0_5.png" alt="Geometry metrics" width="100%"></a></td>
    <td width="50%"><a href="https://rsasaki0109.github.io/CloudAnalyzer/demo/3dgs/"><img src="docs/demo/3dgs/samples/view_00.png" alt="3DGS render" width="48%"><img src="docs/demo/3dgs/samples/view_04.png" alt="3DGS render" width="48%"></a></td>
  </tr>
</table>

## Quick start

```bash
pip install cloudanalyzer
ca evaluate candidate.pcd reference.pcd
```

From this repository:

```bash
pip install -e ./cloudanalyzer
```

## Golden path: SLAM benchmark

The bundled synthetic Figure-8 suite checks a map and trajectory against frozen
references:

```text
inputs:   dataset suite + baseline/reference + candidate outputs
outputs:  metrics JSON + HTML report + pass/fail gate + leaderboard-ready result
```

```bash
ca benchmark info benchmarks/slam/synthetic-figure8/suite.yaml
ca benchmark eval benchmarks/slam/synthetic-figure8/suite.yaml \
  --map benchmarks/slam/synthetic-figure8/sample_outputs/map_pass.pcd \
  --trajectory benchmarks/slam/synthetic-figure8/sample_outputs/trajectory_pass.tum \
  --out qa/synthetic-figure8
```

The same path runs in the [SLAM benchmark smoke workflow](.github/workflows/slam-benchmark-smoke.yml)
and is explained in the [SLAM tutorial](docs/tutorial-slam-benchmark.md).

## Live demos

- [Demo hub](https://rsasaki0109.github.io/CloudAnalyzer/)
- [Web viewer (beta)](https://rsasaki0109.github.io/CloudAnalyzer/app/): Rust/WASM point cloud viewer with cloud-to-cloud distance, in your browser ([source](web/))
- [Point-cloud comparison](https://rsasaki0109.github.io/CloudAnalyzer/demo/compare/)
- [SLAM leaderboard](https://rsasaki0109.github.io/CloudAnalyzer/leaderboard/)
- [3DGS evaluation](https://rsasaki0109.github.io/CloudAnalyzer/demo/3dgs/)
- [Perception report](https://rsasaki0109.github.io/CloudAnalyzer/demo/perception/)

## Docs

- [Command reference](docs/commands/)
- [CI and quality gates](docs/ci.md)
- [Map quality-gate tutorial](docs/tutorial-map-quality-gate.md)
- [Unified run quality-gate tutorial](docs/tutorial-run-quality-gate.md)
- [Public benchmark packs](benchmarks/public/README.md)
- [Architecture](docs/architecture.md)

## License

CloudAnalyzer source code is under the [MIT License](LICENSE). Public demo data
and derived images retain their upstream terms; see [image attribution](docs/images/ATTRIBUTION.md).
