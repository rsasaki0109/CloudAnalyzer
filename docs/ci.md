# CI / Quality Gate

## Test CI

Every push to `main` runs:

1. **mypy** type check on `ca/` and `cli/`
2. **pytest** with `xvfb-run` (for Open3D offscreen rendering)
3. **Asset sync checks** for `docs/demo/perception/`, `benchmarks/slam/synthetic-figure8/`, and `benchmarks/3dgs/synthetic-room/` — each generator is re-run and `git diff --exit-code` enforces no drift between the script and the checked-in bytes

See `.github/workflows/test.yml`.

## Self QA (dogfood)

Every pull request also runs `.github/workflows/self-qa.yml`. It builds the gitignored `benchmarks/public/stanford-bunny-mini` pack, then runs the bundled **[cloudanalyzer-action](https://github.com/rsasaki0109/cloudanalyzer-action)** (`@v1`) against `suite-pass.cloudanalyzer.yaml`. The action runs `ca check`, posts (or idempotently updates) a PR comment marked `cloudanalyzer-self-qa`, uploads QA artifacts, and fails the job when the gate fails.

This dogfoods the public three-line Action entry point, gives every PR a living example of what downstream users get, and detects accidental damage to the bundled QA pack — if a PR changes a baseline `.pcd` without updating the expected summary, the gate fails and the PR turns red.

Fork PRs are skipped (the `GITHUB_TOKEN` from a fork PR doesn't have `pull-requests: write`); the underlying `ca check` still runs in the regular `Test` workflow.

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
ca init-check --profile integrated
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
  - id: manual-loop-closure
    kind: loop_closure
    before_session_root: runs/before-loop
    after_session_root: runs/after-loop
    reference_map: baselines/map_ref.pcd
    before_traj: runs/before-loop/optimized_poses_tum.txt
    after_traj: runs/after-loop/optimized_poses_tum.txt
    ref_traj: baselines/trajectory_ref.tum
    gate:
      min_auc_gain: 0.01
      min_ate_gain: 0.05
      require_posegraph_ok: true
  - id: rendered-views
    kind: image
    rendered_dir: outputs/renders/seq00
    reference_dir: baselines/references/seq00
    gate:
      min_psnr: 28.0
      min_ssim: 0.85
```

`ca check` writes per-check reports / JSON when `report_dir` and `json_dir` are configured, and it exits with code `1` when any gated check fails.

The `image` check kind wraps `ca image-evaluate`: it pairs rendered and reference
images by filename, computes per-pair PSNR / SSIM, and gates on the aggregate
**mean** (`min_psnr` is in dB, `min_ssim` in `[0, 1]`). Bit-identical pairs
(PSNR = +∞) are excluded from the mean and trivially satisfy `min_psnr`; a gate
fails if zero pairs match across the two directories. PSNR / SSIM means flow into
`ca report-pr-comment` with up/down deltas against a baseline like every other kind.

### GitHub Actions

**Marketplace Action (recommended).** [`rsasaki0109/cloudanalyzer-action@v1`](https://github.com/rsasaki0109/cloudanalyzer-action) runs `ca check`, renders the PR comment, posts it idempotently, and fails on gate errors:

```yaml
permissions:
  contents: read
  pull-requests: write

jobs:
  qa:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v6

      - uses: rsasaki0109/cloudanalyzer-action@v1
        with:
          config: cloudanalyzer.yaml
          baseline: qa/baseline-summary.json   # optional
          project: my-mapping-pipeline         # optional
```

**Reusable workflow.** Use `.github/workflows/config-quality-gate.yml` when the repository already contains `cloudanalyzer.yaml` and you want to compose jobs manually:

```bash
gh workflow run config-quality-gate.yml \
  -f config_path=cloudanalyzer.yaml \
  -f artifact_name=cloudanalyzer-check-results
```

The workflow runs `ca check`, uploads `summary.json`, and also uploads each generated report / JSON file declared by the config.

### QA Bundles (artifact retention)

`ca bundle pack` freezes a single QA run (summary JSON + per-check reports + optional baseline + metadata header) into a `qa_bundle.zip` that is reopenable from any future runner. Pair it with `actions/upload-artifact@v6` in CI for long-term retention:

```yaml
- name: Pack QA bundle
  run: |
    ca bundle pack qa/summary.json \
      --output qa/bundle.zip \
      --baseline qa/baseline-summary.json \
      --project "${{ github.repository }}" \
      --commit "${{ github.sha }}" \
      --pr-number "${{ github.event.pull_request.number }}" \
      --runner-id "${{ github.run_id }}"

- name: Upload QA bundle
  uses: actions/upload-artifact@v6
  with:
    name: qa-bundle
    path: qa/bundle.zip
    retention-days: 90
```

The bundle format is documented at [`docs/commands/bundle.md`](commands/bundle.md). It is the OSS layer the hosted retention / dashboard story sits on top of.

### PR Comment From `summary.json`

After `ca check` writes the suite summary (or `ca benchmark eval` / `ca run-evaluate` writes a single-run JSON), use `ca report-pr-comment` to turn that artifact into a Markdown blob. There are two ways to wire it in CI.

**Reusable workflow.** `pr-comment.yml` handles checkout, install, rendering, and (idempotent) posting via `gh` when you already ran `ca check` in a separate job:

```yaml
jobs:
  qa:
    uses: rsasaki0109/CloudAnalyzer/.github/workflows/config-quality-gate.yml@main
    with:
      config_path: cloudanalyzer.yaml

  pr-comment:
    needs: qa
    if: ${{ github.event_name == 'pull_request' }}
    permissions:
      pull-requests: write
      contents: read
    uses: rsasaki0109/CloudAnalyzer/.github/workflows/pr-comment.yml@main
    with:
      summary_path: qa/summary.json
      baseline_summary_path: qa/baseline-summary.json   # optional
      project: my-mapping-pipeline                       # optional header label
      marker: my-pipeline-qa                             # optional; used for idempotent updates
      dry_run: false                                     # set true to log without posting
```

Re-runs find the previous comment via the marker (default `cloudanalyzer-qa`) and *update it in place* instead of stacking duplicates on every force-push. The rendered Markdown is also uploaded as the `cloudanalyzer-pr-comment` artifact for inspection.

**Inline (when you already have a job set up).** Call the CLI directly and pipe to `gh`:

```yaml
- name: Render PR comment
  run: |
    ca report-pr-comment qa/summary.json \
      --baseline qa/baseline-summary.json \
      --output qa/pr-comment.md

- name: Post PR comment
  if: github.event_name == 'pull_request'
  env:
    GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
  run: gh pr comment "${{ github.event.pull_request.number }}" --body-file qa/pr-comment.md
```

`--baseline` is optional; without it the comment is rendered without ↑/↓ delta annotations. See [`docs/commands/report-pr-comment.md`](commands/report-pr-comment.md) for the full schema and output format.

### Reusable Workflow From Another Repository

CloudAnalyzer can also be consumed as a reusable workflow:

```yaml
name: CloudAnalyzer QA

on:
  pull_request:
  workflow_dispatch:

jobs:
  qa:
    uses: rsasaki0109/CloudAnalyzer/.github/workflows/config-quality-gate.yml@main
    with:
      config_path: cloudanalyzer.yaml
      artifact_name: cloudanalyzer-check-results
```

この workflow は caller repository を checkout して、その中の `cloudanalyzer.yaml` を評価する。caller repo 自体に CloudAnalyzer source が無い場合は、`cloudanalyzer_repository` と `cloudanalyzer_ref` から install する。

外部利用時は `@main` ではなく tag か commit SHA への pin を推奨。

### Caller Repo Examples

Mapping repo:

```yaml
checks:
  - id: mapping-postprocess
    kind: artifact
    source: outputs/map.pcd
    reference: baselines/map_ref.pcd
    gate:
      min_auc: 0.95
      max_chamfer: 0.02
```

Localization repo:

```yaml
checks:
  - id: localization-run
    kind: trajectory
    estimated: outputs/trajectory.csv
    reference: baselines/trajectory_ref.csv
    alignment: rigid
    gate:
      max_ate: 0.5
      max_rpe: 0.2
      max_drift: 1.0
      min_coverage: 0.9
```

Perception repo:

```yaml
checks:
  - id: perception-output
    kind: artifact
    source: outputs/reconstruction.pcd
    reference: baselines/reconstruction_ref.pcd
    gate:
      min_auc: 0.95
      max_chamfer: 0.02
  - id: detector-regression
    kind: detection
    estimated: outputs/detections.json
    reference: baselines/detections_ref.json
    thresholds: [0.25, 0.5]
    gate:
      min_map: 0.9
      min_recall: 0.8
  - id: tracker-regression
    kind: tracking
    estimated: outputs/tracks.json
    reference: baselines/tracks_ref.json
    thresholds: [0.5]
    gate:
      min_mota: 0.8
      min_recall: 0.8
      max_id_switches: 2
```
