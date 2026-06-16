# `ca leaderboard`

Build static leaderboard sites from benchmark report bundles.

The input is one or more directories produced by:

```bash
ca benchmark eval benchmarks/slam/synthetic-figure8/suite.yaml \
  --map outputs/map.ply \
  --trajectory outputs/trajectory.tum \
  --out qa/kiss-icp__synthetic-figure8
```

Build a self-contained static site:

```bash
ca leaderboard build qa/*__synthetic-figure8 --out site/leaderboard
```

Output layout:

```text
site/leaderboard/
├── index.html
├── results.json
└── runs/
    └── <bundle-id>/
        ├── manifest.lock.yaml
        ├── metrics.json
        ├── provenance.json
        ├── report.html
        ├── report_*.png
        └── summary.md
```

`results.json` uses `schema_version:
cloudanalyzer.leaderboard.v0.1` and contains one row per bundle:

```json
{
  "id": "kiss-icp__synthetic-figure8",
  "method": "kiss-icp",
  "dataset": "synthetic-figure8",
  "sequence": "default",
  "suite_version": 1,
  "gate_status": "pass",
  "metrics": {
    "ate_rmse_m": 0.08,
    "rpe_rmse_m": 0.11,
    "drift_m": 0.09,
    "coverage_ratio": 1.0,
    "map_auc": 0.997,
    "chamfer_m": 0.031
  }
}
```

The builder copies each input bundle into `runs/<bundle-id>/`, so the generated
site can be uploaded directly to GitHub Pages or a workflow artifact.

Warnings:

- Rows sharing the same dataset/sequence but using different suite versions or
  gates are marked with `warnings[].code == "incomparable_rows"`.
- Bundle read errors are recorded in `errors[]`; the CLI exits with code `1`
  when any bundle could not be read.
