# v0.5.0-alpha Release Preflight

This alpha should prove that CloudAnalyzer is now a CI-grade evidence layer for
SLAM / mapping / reconstruction artifacts, not just a collection of evaluators.

## Scope

The release is ready when a clean checkout can take a frozen benchmark suite plus
candidate outputs and deterministically produce:

- `metrics.json`
- `summary.md`
- `report.html`
- `manifest.lock.yaml`
- `provenance.json`
- a CI gate result
- a static leaderboard row

## Included Track

- Benchmark report bundle output via `ca benchmark eval --out <dir>`.
- Public SLAM benchmark smoke workflow.
- Unified `ca check` gate severity policy:
  `fail`, `warn`, `soft_fail`, `skip`, `not_applicable`.
- Static `ca leaderboard build` site generation.
- SLAM driver conformance helper for plugin authors.
- Portable public reports with scrubbed local paths.
- CPU-safe synthetic 3DGS fixture verification in CI.

## Must Pass Before Tagging

- GitHub Actions `Test` on `main`.
- GitHub Actions `Pages` on `main`.
- GitHub Actions `SLAM Benchmark Smoke` on the release candidate commit.
- GitHub Actions `Public Benchmark Pack` on the release candidate commit.
- Local or CI package build:

```bash
cd cloudanalyzer
python -m build
python -m twine check dist/*
```

## Golden Path Smoke

```bash
ca benchmark eval benchmarks/slam/synthetic-figure8/suite.yaml \
  --map benchmarks/slam/synthetic-figure8/sample_outputs/map_pass.pcd \
  --trajectory benchmarks/slam/synthetic-figure8/sample_outputs/trajectory_pass.tum \
  --out qa/synthetic-figure8

ca leaderboard build qa/synthetic-figure8 --out /tmp/cloudanalyzer-leaderboard
```

Expected outputs:

- `qa/synthetic-figure8/metrics.json`
- `qa/synthetic-figure8/summary.md`
- `qa/synthetic-figure8/report.html`
- `qa/synthetic-figure8/manifest.lock.yaml`
- `qa/synthetic-figure8/provenance.json`
- `/tmp/cloudanalyzer-leaderboard/results.json`
- `/tmp/cloudanalyzer-leaderboard/index.html`

## Versioning

The current package version is `0.4.0`. For a PyPI alpha, use PEP 440 form
`0.5.0a1` in `cloudanalyzer/ca/__init__.py`; the matching Git tag can be
`v0.5.0-alpha.1`.

Do not publish this as `v0.5.0` until the benchmark/report bundle schema has
settled enough that downstream CI users can reasonably depend on it.

## Release Blockers

- A benchmark report bundle contains machine-local absolute paths.
- `ca check --format-json` omits `gate_summary`.
- `ca benchmark eval --out` produces nondeterministic checked-in fixture diffs.
- `ca leaderboard build` silently mixes incomparable dataset/gate rows.
- The main `Test` workflow requires CUDA or live rendering to pass.
- The README no longer shows the benchmark gate as the first workflow.

## Non-Goals

- New SLAM/LIO estimator implementation.
- New low-level point cloud algorithm catalog.
- SaaS dashboard, Slack, or Notion integration.
- Heavy public dataset execution in default CI.
- Stable v1 schema guarantees.
