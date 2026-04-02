# CI / Quality Gate

## Test CI

Every push to `main` runs:

1. **mypy** type check on `ca/` and `cli/`
2. **pytest** with `xvfb-run` (for Open3D offscreen rendering)

See `.github/workflows/test.yml`.

## Quality Gate

The `quality-gate.yml` workflow evaluates point cloud quality and fails if thresholds are not met.

### Parameters

| Input | Default | Description |
|---|---|---|
| `source` | *required* | Path to source (estimated) point cloud |
| `reference` | *required* | Path to reference point cloud |
| `auc_threshold` | `0.9` | Minimum AUC (F1) to pass |
| `chamfer_threshold` | `0.1` | Maximum Chamfer Distance to pass |

### Usage

Trigger manually via GitHub Actions UI or API:

```bash
gh workflow run quality-gate.yml \
  -f source=path/to/estimated.pcd \
  -f reference=path/to/reference.pcd \
  -f auc_threshold=0.95 \
  -f chamfer_threshold=0.05
```

### Pass/Fail Logic

```
PASS if: AUC >= auc_threshold AND Chamfer <= chamfer_threshold
FAIL otherwise
```

The evaluation result JSON is uploaded as a build artifact.

### Integration Example

To use in a mapping pipeline:

1. Build a new map from sensor data
2. Run quality gate against a known-good reference map
3. Gate deployment on the result

```bash
# Build map
ca align scan1.pcd scan2.pcd scan3.pcd -o new_map.pcd -m gicp

# Evaluate quality
ca evaluate new_map.pcd reference_map.pcd --format-json | jq '.auc'

# Automated check
AUC=$(ca evaluate new_map.pcd reference_map.pcd --format-json | jq -r '.auc')
if (( $(echo "$AUC < 0.9" | bc -l) )); then
  echo "FAIL: AUC $AUC < 0.9"
  exit 1
fi

# Batch quality gate over multiple outputs
ca batch outputs/ --evaluate reference_map.pcd --min-auc 0.95 --max-chamfer 0.02
```

## Config-Driven QA

When one pipeline produces multiple artifacts, keep the gate in `cloudanalyzer.yaml` and run it with one command:

```bash
ca check cloudanalyzer.yaml
```

Example config:

```yaml
version: 1
summary_output_json: qa/summary.json
defaults:
  report_dir: qa/reports
  json_dir: qa/results
checks:
  - id: mapping-postprocess
    kind: artifact
    source: outputs/map.pcd
    reference: baselines/map_ref.pcd
    gate:
      min_auc: 0.95
      max_chamfer: 0.02
  - id: localization-run
    kind: trajectory
    estimated: outputs/traj.csv
    reference: baselines/traj_ref.csv
    alignment: rigid
    gate:
      max_ate: 0.5
      max_rpe: 0.2
      max_drift: 1.0
      min_coverage: 0.9
```

`ca check` writes per-check reports / JSON when `report_dir` and `json_dir` are configured, and it exits with code `1` when any gated check fails.
