# `ca benchmark`

SLAM benchmark suite runner. A *benchmark suite* freezes a reference map, reference trajectory, and a quality gate so users can plug their SLAM output in for a one-command regression check.

CloudAnalyzer ships one synthetic suite (`benchmarks/slam/synthetic-figure8/`) for smoke-testing; real public-data suites (Newer College, KITTI, Hilti, ...) follow the same manifest shape.

## Subcommands

### `ca benchmark info <suite.yaml>`

Print the suite metadata (sequences, references, gate).

```bash
ca benchmark info benchmarks/slam/synthetic-figure8/suite.yaml
```

With `--format-json`, emits the same payload as a JSON object for tooling.

### `ca benchmark eval <suite.yaml> --map ... --trajectory ...`

Evaluate one SLAM run against the suite's frozen reference + gate. Exit code is `1` when `overall_quality_gate` fails, matching `ca run-evaluate`.

```bash
ca benchmark eval benchmarks/slam/synthetic-figure8/suite.yaml \
  --map outputs/my_slam_map.pcd \
  --trajectory outputs/my_slam_trajectory.tum \
  --report qa/run_report.html
```

Useful options:

| Option | Purpose |
|---|---|
| `--sequence <name>` | Pick a sequence when the suite has more than one (default: first sequence) |
| `--gate key=value` | Override the suite gate (repeatable). `key=none` disables a gate metric |
| `--thresholds 0.1,0.2,0.5` | Override the F1/AUC distance thresholds for the map score |
| `--align-origin` / `--align-rigid` | Trajectory alignment knobs forwarded to `run-evaluate` |
| `--report <file>` | Write a Markdown / HTML run report |
| `--output-json <file>` | Dump the full result dict (incl. `benchmark` block) |
| `--format-json` | Print the result as JSON to stdout |

## Suite manifest schema

```yaml
version: 1
name: synthetic-figure8
description: ...
license: MIT (synthetic)
sequences:
  default:
    description: ...
    reference_map: reference/map.pcd
    reference_trajectory: reference/trajectory.tum
sample_outputs:       # optional — used by docs / tests
  default:
    map: sample_outputs/map_pass.pcd
    trajectory: sample_outputs/trajectory_pass.tum
gate:
  min_auc: 0.95
  max_chamfer: 0.05
  max_ate: 0.30
  max_rpe: 0.20
  max_drift: 0.50
  min_coverage: 0.90
```

All paths in the manifest are resolved relative to the manifest file. Allowed gate keys are the same six accepted by `ca run-evaluate`: `min_auc`, `max_chamfer`, `max_ate`, `max_rpe`, `max_drift`, `min_coverage`.

## Adding your own suite

Drop a `suite.yaml` next to your reference data:

```
benchmarks/slam/my-suite/
├── suite.yaml
└── reference/
    ├── map.pcd
    └── trajectory.tum
```

Then point `ca benchmark` at it. There is no global registry yet — suites are addressed by manifest path.

## Result shape

`ca benchmark eval` returns the same dict as `ca run-evaluate` plus a `benchmark` block:

```json
{
  "map": { ... run-evaluate map block ... },
  "trajectory": { ... },
  "overall_quality_gate": { "passed": true, "reasons": [] },
  "benchmark": {
    "suite": "synthetic-figure8",
    "version": 1,
    "sequence": "default",
    "source_path": "/abs/path/to/suite.yaml",
    "gate": { "min_auc": 0.95, "max_chamfer": 0.05, ... },
    "reference": { "map": "...", "trajectory": "..." }
  }
}
```

The `benchmark` block lets downstream consumers (dashboards, PR comments, baseline history) recover which calibration produced the numbers.
