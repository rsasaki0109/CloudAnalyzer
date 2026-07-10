# CloudAnalyzer

[![Test](https://github.com/rsasaki0109/CloudAnalyzer/actions/workflows/test.yml/badge.svg)](https://github.com/rsasaki0109/CloudAnalyzer/actions/workflows/test.yml)
[![Self QA](https://github.com/rsasaki0109/CloudAnalyzer/actions/workflows/self-qa.yml/badge.svg)](https://github.com/rsasaki0109/CloudAnalyzer/actions/workflows/self-qa.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Turn SLAM, mapping, perception, and reconstruction outputs into CI-grade QA evidence.**

CloudAnalyzer compares candidate maps, trajectories, rendered images, and point
clouds with frozen references. It produces metrics JSON, browsable reports, and
deterministic pass/fail gates for local development and CI.

```text
inputs:   dataset suite + baseline/reference + candidate outputs
outputs:  metrics JSON + HTML report + pass/fail gate + leaderboard-ready result
```

It complements PCL, Open3D, CloudCompare, and SLAM/LIO stacks: those tools create
or process 3D data; CloudAnalyzer verifies the resulting artifacts and catches
regressions across a whole run.

## Public Demos

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

<!-- Regenerate with `scripts/build_readme_gif.sh` (requires vhs). -->
![CloudAnalyzer terminal demo](docs/images/readme-demo.gif)

## Where It Fits

- **SLAM and localization:** map and trajectory evaluation, ATE/RPE/drift,
  loop-closure QA, benchmark suites, and run-level reports.
- **Mapping operations:** regression checks after downsampling, filtering,
  registration, splitting, compression, or format conversion.
- **Perception:** ground segmentation, 3D detection, and multi-object tracking
  metrics with config-driven gates.
- **Reconstruction and 3DGS:** geometry and rendered-image comparisons across
  representations, including perceptual metrics.
- **Automation:** machine-readable results, browser reports, baseline history,
  PR summaries, and CI exit codes.

The distinguishing workflow is **process, evaluate, report, and gate** through one
CLI. CloudAnalyzer is an output-verification layer, not another low-level geometry
library or desktop viewer.

## Commands and Guides

Run `ca --help` or open the focused references:

- [Command reference](docs/commands/)
- [Benchmark suites](docs/commands/benchmark.md)
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
