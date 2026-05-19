# `ca report-pr-comment`

Render a Markdown blob suitable for a GitHub PR comment from any CloudAnalyzer summary JSON. Auto-detects whether it received a `ca check` suite summary or a single-run `ca run-evaluate` / `ca benchmark eval` JSON, so the same command works at the gate-orchestrator level **and** at the per-run level.

```bash
ca report-pr-comment qa/summary.json \
  --baseline qa/baseline-summary.json \
  --output qa/pr-comment.md
```

## Inputs

| Source | What to pass | Notes |
|---|---|---|
| `ca check` | `summary_output_json:` file from `cloudanalyzer.yaml` | Includes per-check failed reasons + triage |
| `ca run-evaluate --output-json` | The JSON file | Single map + trajectory run |
| `ca benchmark eval --output-json` | The JSON file | Same as run-evaluate plus the `benchmark` block (suite identity in the header) |

`--baseline` accepts a JSON of the *same shape* as the current summary. Each metric then shows ``(was 0.987 ↓)``-style deltas; arrows track value direction so the reader can compare against the gate sitting right next to the metric.

## Options

| Option | Purpose |
|---|---|
| `--baseline <file>` | Baseline JSON of the same shape; renders metric deltas |
| `--project <label>` | Override the project label shown in the header |
| `--output <file>` / `-o` | Write Markdown to a file instead of stdout |

## Reusable workflow

The simplest way to wire this in CI is the bundled reusable workflow
`pr-comment.yml`. It handles install, render, and idempotent comment
update on re-runs (no duplicate comments after a force-push).

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
      project: my-mapping-pipeline                       # optional
      marker: my-pipeline-qa                             # optional, default cloudanalyzer-qa
      dry_run: false                                     # set true to log without posting
```

Inputs:

| Input | Required | Notes |
|---|---|---|
| `summary_path` | yes | Path to summary JSON, relative to the caller repo root |
| `baseline_summary_path` | no | Baseline JSON of the same shape for ↑/↓ deltas |
| `project` | no | Header label override |
| `marker` | no | Hidden HTML comment used to find / update the previous comment |
| `pr_number` | no | Override the PR number (defaults to the triggering pull_request) |
| `dry_run` | no | If true, print the rendered Markdown to the job log and skip posting |
| `cloudanalyzer_repository` / `cloudanalyzer_ref` | no | Install source when the caller repo doesn't ship CloudAnalyzer itself |

The rendered Markdown is uploaded as the `cloudanalyzer-pr-comment`
workflow artifact even when `dry_run: true`, so you can preview the
output without granting `pull-requests: write` first.

## Inline GitHub Actions wiring

```yaml
- name: Run gate
  run: ca check cloudanalyzer.yaml

- name: Render PR comment
  run: ca report-pr-comment qa/summary.json --baseline qa/baseline-summary.json --output qa/pr-comment.md

- name: Post PR comment
  if: github.event_name == 'pull_request'
  env:
    GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
  run: gh pr comment "${{ github.event.pull_request.number }}" --body-file qa/pr-comment.md
```

The same flow works for a SLAM benchmark by replacing `ca check` with:

```yaml
- name: Run SLAM benchmark
  run: ca benchmark eval benchmarks/slam/synthetic-figure8/suite.yaml \
       --map outputs/map.pcd --trajectory outputs/traj.tum \
       --output-json qa/benchmark.json

- name: Render PR comment
  run: ca report-pr-comment qa/benchmark.json --output qa/pr-comment.md
```

## Output format

The renderer surfaces the same triage that `ca check` already computes (severity-weighted): the worst regression appears first under "Recommended triage". Failure rows include both the metric value and the gate that broke, so a reviewer does not have to open the underlying report just to know which knob moved.
