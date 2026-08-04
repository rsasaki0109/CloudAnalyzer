# CloudAnalyzer

[![Test](https://github.com/rsasaki0109/CloudAnalyzer/actions/workflows/test.yml/badge.svg)](https://github.com/rsasaki0109/CloudAnalyzer/actions/workflows/test.yml)
[![Self QA](https://github.com/rsasaki0109/CloudAnalyzer/actions/workflows/self-qa.yml/badge.svg)](https://github.com/rsasaki0109/CloudAnalyzer/actions/workflows/self-qa.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Catch SLAM and 3D perception regressions before they ship.**

CloudAnalyzer is the CI-grade QA layer for teams building SLAM, LiDAR,
perception, and 3DGS pipelines. It turns candidate maps, trajectories, rendered
images, and point clouds into **metrics JSON, browsable reports, and deterministic
pass/fail gates**.

Turn SLAM, mapping, perception, and reconstruction outputs into CI-grade QA evidence.

<!-- Regenerate with `scripts/build_readme_gif.sh` (requires vhs and `ca`). -->
<p align="center">
  <img src="docs/images/readme-demo.gif" alt="CloudAnalyzer terminal demo comparing a downsampled point cloud and exporting a browser viewer" width="960">
</p>

<p align="center">
  <a href="#golden-path-slam-benchmark">30-second check</a> ·
  <a href="https://rsasaki0109.github.io/CloudAnalyzer/">Live demos</a> ·
  <a href="#install">Install</a> ·
  <a href="docs/commands/">Command reference</a>
</p>

<p align="center">
  <img src="docs/images/readme-workflow.svg" alt="CloudAnalyzer workflow from robotics artifacts through metrics and reports to a CI pass or fail gate" width="960">
</p>

```text
inputs:   dataset suite + baseline/reference + candidate outputs
outputs:  metrics JSON + HTML report + pass/fail gate + leaderboard-ready result
```

## Why teams use CloudAnalyzer

| The problem | The outcome |
|---|---|
| Regressions hide in screenshots and ad-hoc numbers | Versioned metrics and explicit quality gates |
| Different runs are difficult to compare fairly | Frozen references, protocol metadata, and input provenance |
| A failed CI job does not explain what changed | HTML reports, error artifacts, and copyable inspection commands |

> **Proof from a checked-in benchmark run:** `PASS` · Map AUC `1.0000` ·
> Chamfer `0.0145 m` · Trajectory ATE RMSE `0.0016 m` ·
> [open the full report](docs/leaderboard/runs/kiss-slam__synthetic-oval/report.html).

## See It in Action

Every image below is a checked-in CloudAnalyzer output. Click a panel to open the
corresponding live demo or command guide.

<p align="center">
  <a href="https://rsasaki0109.github.io/CloudAnalyzer/demo/perception/">
    <img src="docs/images/readme-pointcloud-triptych.png" alt="Three-dimensional point-cloud comparison showing the RELLIS-3D reference, a passing deep baseline, and a failing non-deep baseline" width="1100">
  </a>
</p>

<p align="center"><strong>Point-cloud QA in one glance:</strong> same reference scene, different artifact density, one deterministic gate.</p>

<table>
  <tr>
    <td width="50%" valign="top">
      <a href="https://rsasaki0109.github.io/CloudAnalyzer/demo/perception/">
        <img src="docs/pr/perception_summary_card.png" alt="Perception comparison summary with pass and fail quality gates" width="100%">
      </a>
      <p><strong>Perception QA</strong><br>Batch comparisons with metrics and deterministic gates.</p>
    </td>
    <td width="50%" valign="top">
      <a href="https://rsasaki0109.github.io/CloudAnalyzer/demo/compare/">
        <img src="docs/images/density_hdl_localization_map.png" alt="Point-cloud density map from the HDL localization sample" width="100%">
      </a>
      <p><strong>Point-cloud comparison</strong><br>See spatial density and map-quality changes.</p>
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <a href="docs/commands/geometry-evaluate.md">
        <img src="docs/images/f1_hdl_localization_v0_5.png" alt="F1, precision, and recall curves for a point-cloud evaluation" width="100%">
      </a>
      <p><strong>Geometry evidence</strong><br>Chamfer, F1, AUC, and threshold curves ready for review.</p>
    </td>
    <td width="50%" valign="top">
      <a href="https://rsasaki0109.github.io/CloudAnalyzer/demo/3dgs/">
        <img src="docs/demo/3dgs/samples/view_00.png" alt="Rendered 3D Gaussian Splatting sample view" width="48%">
        <img src="docs/demo/3dgs/samples/view_04.png" alt="Second rendered 3D Gaussian Splatting sample view" width="48%">
      </a>
      <p><strong>Rendered evaluation</strong><br>Photometric and geometry checks for 3DGS outputs.</p>
    </td>
  </tr>
</table>

It complements PCL, Open3D, CloudCompare, and SLAM/LIO stacks: those tools create
or process 3D data; CloudAnalyzer verifies the resulting artifacts and catches
regressions across a whole run.

## Live Demos

- [CloudAnalyzer demo hub](https://rsasaki0109.github.io/CloudAnalyzer/)
- [Point-cloud comparison](https://rsasaki0109.github.io/CloudAnalyzer/demo/compare/)
- [Live SLAM leaderboard](https://rsasaki0109.github.io/CloudAnalyzer/leaderboard/)
- [3DGS rendered evaluation](https://rsasaki0109.github.io/CloudAnalyzer/demo/3dgs/)
- [Perception batch report](https://rsasaki0109.github.io/CloudAnalyzer/demo/perception/)

## Install

Run once without installing:

```bash
uvx cloudanalyzer evaluate before.pcd after.pcd
```

Or install from PyPI:

```bash
pip install cloudanalyzer
```

Install optional ROS/bag support with `pip install "cloudanalyzer[ros]"`. For
development, clone this repository and run `pip install -e ./cloudanalyzer`.

## Golden Path: SLAM Benchmark

The bundled synthetic Figure-8 suite is a reproducible smoke test. From the
repository root:

```bash
pip install -e ./cloudanalyzer

ca benchmark info benchmarks/slam/synthetic-figure8/suite.yaml
ca benchmark eval benchmarks/slam/synthetic-figure8/suite.yaml \
  --map benchmarks/slam/synthetic-figure8/sample_outputs/map_pass.pcd \
  --trajectory benchmarks/slam/synthetic-figure8/sample_outputs/trajectory_pass.tum \
  --out qa/synthetic-figure8
```

The evaluation checks the candidate map and trajectory against frozen references
and quality gates. It writes `metrics.json`, `summary.md`, `report.html`, a locked
manifest, and provenance. Replace the two `sample_outputs` paths with outputs from
your own SLAM system.

This exact path is exercised by
[the SLAM benchmark smoke workflow](.github/workflows/slam-benchmark-smoke.yml).

To run the full raw-scans-to-report workflow with a supported SLAM driver, follow
the [SLAM benchmark tutorial](docs/tutorial-slam-benchmark.md).

## Quick Point-Cloud QA

Compare one candidate with its reference:

```bash
ca evaluate candidate.pcd reference.pcd
```

Or process and evaluate in one operation:

```bash
ca downsample map.pcd -o down.pcd -v 0.2 --evaluate
```

`--evaluate` reports how much quality changed, including Chamfer distance and
F1/AUC metrics. See the [evaluation](docs/commands/evaluate.md) and
[processing](docs/commands/processing.md) references for thresholds, plots, and
supported operations.

The animated terminal walkthrough at the top is generated from the same CLI path
shown above. It can be rebuilt locally with `scripts/build_readme_gif.sh`.

## Built for 3D robotics teams

- **SLAM/LIO teams:** prove map quality, trajectory accuracy, drift, and loop-closure gains.
- **Mapping and point-cloud teams:** catch regressions after filtering, registration,
  downsampling, compression, or format conversion.
- **Perception teams:** gate ground segmentation, 3D detection, and multi-object tracking.
- **Reconstruction and 3DGS teams:** compare geometry and rendered images across representations.
- **Platform and CI teams:** publish machine-readable evidence, browser reports, baselines,
  PR summaries, and CI exit codes.

The distinguishing workflow is **process, evaluate, report, and gate** through one
CLI. CloudAnalyzer is an output-verification layer, not another low-level geometry
library or desktop viewer.

## Commands and Guides

Run `ca --help` or open the focused references:

- [Command reference](docs/commands/)
- [Benchmark suites](docs/commands/benchmark.md)
- [MapEval parity and scale validation](docs/commands/mapeval-parity.md)
- [Map and trajectory analysis](docs/commands/analysis.md)
- [Geometry evaluation](docs/commands/geometry-evaluate.md)
- [Image and rendered evaluation](docs/commands/image-evaluate.md)
- [Plane consistency without ground truth](docs/commands/plane-consistency.md)
- [Visualization and static browser exports](docs/commands/visualization.md)
- [CI and quality gates](docs/ci.md)
- [Map quality-gate tutorial](docs/tutorial-map-quality-gate.md)
- [Unified run quality-gate tutorial](docs/tutorial-run-quality-gate.md)
- [Public benchmark packs](benchmarks/public/README.md)
- [Architecture](docs/architecture.md) and [project vision](VISION.md)

SLAM integrators can also consult the [driver plugin contract](docs/driver-plugin.md).

## Public Data and Attribution

The documentation includes examples derived from public datasets; these assets
retain their upstream terms and are not relicensed by CloudAnalyzer's MIT license.

- README map figures and the map-viewer demo use the
  [`hdl_localization` sample map](https://github.com/koide3/hdl_localization/blob/master/data/map.pcd),
  published by AISL at Toyohashi University of Technology under
  [BSD-2-Clause](https://github.com/koide3/hdl_localization/blob/master/LICENSE).
- The perception demo uses public RELLIS-3D material. Checked-in derived assets
  remain subject to the upstream CC BY-NC-SA 3.0 terms.
- Exact source URLs, file-level attribution, and regeneration details are in
  [image attribution](docs/images/ATTRIBUTION.md) and
  [perception-demo attribution](docs/demo/perception/ATTRIBUTION.md).

## License

CloudAnalyzer source code is available under the [MIT License](LICENSE). Public
demo data and derived assets retain the licenses documented in the attribution
files above.
