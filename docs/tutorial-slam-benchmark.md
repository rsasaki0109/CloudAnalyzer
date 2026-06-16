# Tutorial: SLAM Benchmark Golden Path

This tutorial shows the smallest repeatable CloudAnalyzer workflow for a SLAM
or LIO result:

```text
benchmark suite + candidate map + candidate trajectory
  -> metrics JSON
  -> HTML report
  -> CI pass/fail gate
  -> leaderboard-ready row
```

CloudAnalyzer does not implement your SLAM estimator. It gives you a stable
artifact contract for checking whether an estimator output is good enough to
ship.

## Scenario

You have a candidate SLAM run with:

- a reconstructed map, for example `outputs/map.ply`
- an estimated trajectory, for example `outputs/trajectory.tum`

You want to compare that run against a frozen benchmark suite that contains:

- a reference map
- a reference trajectory
- gate thresholds for map quality, trajectory error, drift, and coverage

The bundled `synthetic-figure8` suite is tiny and checked in, so it works as a
clone-after-install smoke test.

## Step 1: Inspect The Suite

```bash
ca benchmark info benchmarks/slam/synthetic-figure8/suite.yaml
```

The manifest freezes the reference data and gate:

```yaml
gate:
  min_auc: 0.95
  max_chamfer: 0.05
  max_ate: 0.30
  max_rpe: 0.20
  max_drift: 0.50
  min_coverage: 0.90
```

Interpretation:

- `min_auc` / `max_chamfer` check the map against the reference map.
- `max_ate` / `max_rpe` check trajectory accuracy.
- `max_drift` checks endpoint drift.
- `min_coverage` checks how much of the reference trajectory could be paired.

## Step 2: Evaluate A Candidate Run

The suite includes a known-good sample output. Use it first to verify the
workflow:

```bash
mkdir -p qa

ca benchmark eval benchmarks/slam/synthetic-figure8/suite.yaml \
  --map benchmarks/slam/synthetic-figure8/sample_outputs/map_pass.pcd \
  --trajectory benchmarks/slam/synthetic-figure8/sample_outputs/trajectory_pass.tum \
  --out qa/synthetic-figure8
```

Expected result: the overall gate passes and `qa/synthetic-figure8/` contains
the standard report bundle:

```text
qa/synthetic-figure8/
├── manifest.lock.yaml
├── metrics.json
├── provenance.json
├── report.html
├── report_*.png
└── summary.md
```

CI gates should read `qa/synthetic-figure8/metrics.json`; reviewers usually open
`qa/synthetic-figure8/report.html` first.

## Step 3: Swap In Your SLAM Output

Keep the same suite and replace only the candidate artifacts:

```bash
ca benchmark eval benchmarks/slam/synthetic-figure8/suite.yaml \
  --map outputs/my_slam_map.ply \
  --trajectory outputs/my_slam_trajectory.tum \
  --out qa/my-run
```

This is the core CloudAnalyzer contract. The candidate source can be KISS-ICP,
KISS-SLAM, FAST-LIO, LIO-SAM, a proprietary mapping stack, or a plugin driver.
CloudAnalyzer only needs the output artifacts.

## Step 4: Use It In CI

`ca benchmark eval` exits non-zero when the overall quality gate fails. A minimal
GitHub Actions step can be as small as:

```yaml
- name: SLAM benchmark gate
  run: |
    mkdir -p qa
    ca benchmark eval benchmarks/slam/synthetic-figure8/suite.yaml \
      --map outputs/my_slam_map.ply \
      --trajectory outputs/my_slam_trajectory.tum \
      --out qa/slam-benchmark
```

Upload `qa/slam-benchmark/` as a workflow artifact so reviewers can inspect
failures without rerunning the benchmark locally.

CloudAnalyzer dogfoods the checked-in sample-output path in
`.github/workflows/slam-benchmark-smoke.yml`.

## Step 5: Run A Built-In SLAM Driver

If you want CloudAnalyzer to run a SLAM driver from raw scans before evaluating,
install the SLAM extra:

```bash
pip install -e './cloudanalyzer[slam]'
```

Then run the bundled scans through the default KISS-ICP driver:

```bash
ca slam-run benchmarks/slam/synthetic-figure8/scans qa/kiss-icp \
  --driver kiss-icp \
  --max-range 25 \
  --voxel-size 0.5 \
  --frame-period 0.1

ca benchmark eval benchmarks/slam/synthetic-figure8/suite.yaml \
  --map qa/kiss-icp/map.ply \
  --trajectory qa/kiss-icp/trajectory.tum \
  --out qa/kiss-icp-benchmark
```

Third-party drivers can register with the `cloudanalyzer.slam_run_drivers`
entry point. See
[`plugins/cloudanalyzer-driver-example`](../plugins/cloudanalyzer-driver-example/)
for a minimal package.

## Step 6: Build A Public Dataset Suite

For a real dataset, keep CI light and make the expensive preparation explicit:

1. Download or mount the dataset locally.
2. Build a suite with `ca benchmark init` or a dataset-specific prep script.
3. Commit the manifest and small metadata.
4. Keep large raw data outside the repo.
5. Run the heavy benchmark in an optional or scheduled workflow.

Examples:

```bash
python scripts/prepare_kitti_mini.py --help
python scripts/prepare_newer_college_mini.py --help
```

## What Not To Put Here

- Do not add a new SLAM estimator to CloudAnalyzer core.
- Do not run huge public datasets in every PR.
- Do not add a metric unless it also has report output, gate semantics, and a
  reproducible fixture.
- Do not treat the browser viewer as the source of truth; the JSON metrics and
  suite manifest are the contract.
