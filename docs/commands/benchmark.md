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

### `ca benchmark init <suite_dir> --reference-map ... --reference-trajectory ...`

Build a `suite.yaml` (plus copied/downsampled data files) from raw reference inputs on disk. Use this to turn a downloaded public SLAM dataset into a `ca benchmark eval`-ready suite without hand-writing the manifest.

```bash
ca benchmark init /tmp/my-suite \
  --name my-suite \
  --description "My SLAM regression suite" \
  --reference-map  /data/site/gt_map.pcd \
  --reference-trajectory /data/site/gt_poses.tum \
  --sequence default \
  --voxel 0.10 \
  --max-poses 2000 \
  --gate min_auc=0.95 --gate max_ate=0.30
```

What it does:

- Voxel-downsamples the reference map (`--voxel 0` disables).
- Subsamples the reference trajectory to an evenly-spaced subset (`--max-poses 0` keeps everything).
- Writes `suite.yaml` + `data/<sequence>/{map.pcd,trajectory.tum}` under `<suite_dir>`.
- Optionally bakes in a passing sample output (`--sample-map`/`--sample-trajectory`).

Useful options:

| Option | Purpose |
|---|---|
| `--sequence <name>` | Sequence name written to the manifest (default: `default`) |
| `--sequence-description <text>` | Per-sequence description (defaults to the suite description) |
| `--license <text>` | License string for the suite manifest |
| `--voxel <m>` | Voxel size for map downsampling (default: `0.0`, plain copy) |
| `--max-poses <N>` | Subsample the trajectory to at most N evenly-spaced poses (default: unlimited) |
| `--sample-map` / `--sample-trajectory` | Bundle a known-good SLAM output as a smoke-test reference |
| `--gate key=value` | Bake a gate threshold into the manifest (repeatable; unknown keys are dropped) |

For public datasets that need a license-aware download step, prefer the dedicated wrapper scripts under `scripts/` (e.g. `scripts/prepare_newer_college_mini.py`) — they call this same code path with dataset-specific defaults.

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

Two ways:

1. **Hand-write the manifest.** Drop a `suite.yaml` next to your reference data:

   ```
   benchmarks/slam/my-suite/
   ├── suite.yaml
   └── reference/
       ├── map.pcd
       └── trajectory.tum
   ```

2. **Let `ca benchmark init` build it** for you from raw ground-truth files (see above). This is the recommended path for public datasets that ship multi-million-point reference maps and need voxel downsampling.

Then point `ca benchmark` at it. There is no global registry yet — suites are addressed by manifest path.

## Public-dataset wrappers

| Suite | Wrapper script | Notes |
|---|---|---|
| `synthetic-figure8` | `scripts/build_synthetic_slam_suite.py` | Tiny synthetic data; checked in under `benchmarks/slam/synthetic-figure8/`. |
| `newer-college-mini` | `scripts/prepare_newer_college_mini.py` | Local prep from a Newer College Dataset (CC-BY-NC-SA 4.0) download. See `benchmarks/slam/newer-college-mini/README.md`. |

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
