# CloudAnalyzer GitHub Action

Run [`ca check`](https://github.com/rsasaki0109/CloudAnalyzer) on a config file and optionally post an idempotent PR comment.

## Usage

```yaml
- uses: rsasaki0109/cloudanalyzer-action@v1
  with:
    config: cloudanalyzer.yaml
```

With baseline comparison and a custom project label:

```yaml
- uses: rsasaki0109/cloudanalyzer-action@v1
  with:
    config: cloudanalyzer.yaml
    baseline: qa/baseline-summary.json
    project: my-pipeline
```

## Inputs

| Input | Required | Default | Description |
|---|---|---|---|
| `python_base_image` | no | `python:3.10-slim` | Python 3.10-compatible image used for the Docker build; override for registry availability |
| `config` | yes | — | Path to `cloudanalyzer.yaml` relative to the repository root |
| `baseline` | no | `""` | Baseline summary JSON for ↑/↓ deltas |
| `comment` | no | `true` | Post or update a PR comment on pull requests |
| `fail_on_gate` | no | `true` | Fail the job when a gated check fails |
| `project` | no | `""` | Project label in the PR comment header |
| `marker` | no | `cloudanalyzer-qa` | Hidden HTML marker for idempotent comment updates |

## Outputs

| Output | Description |
|---|---|
| `summary_json` | Path to the generated summary JSON |
| `passed` | `true` when all gated checks passed |
| `worst_check` | First failed check id, when available |
| `comment_path` | Path to the rendered Markdown comment |

## Permissions

```yaml
permissions:
  contents: read
  pull-requests: write   # required when comment=true on pull_request
```

## Notes

- Open3D runs under `xvfb-run` inside the Docker image.
- Comment posting is skipped with a warning on non-PR events.
- QA artifacts (`summary.json`, rendered comment) upload as `cloudanalyzer-action-results`.
- When the caller repository is CloudAnalyzer itself, the action editable-installs the checkout for dogfooding.
- Internal self-QA selects a digest-pinned GHCR Python 3.10 / Debian bookworm image to avoid anonymous Docker Hub HTTP 429. Other callers retain the existing default unless they explicitly set `python_base_image`; custom images need Python/pip and Debian-compatible apt packages.

## Examples

See [`examples/`](examples/).

## Publishing

Published at **[github.com/rsasaki0109/cloudanalyzer-action](https://github.com/rsasaki0109/cloudanalyzer-action)** (`@v1`).

To list on GitHub Marketplace: open the action repository → **Settings** → **Actions** → **Publish to GitHub Marketplace** (requires accepting the GitHub Marketplace Developer Agreement once per account).

This directory in the CloudAnalyzer monorepo is the development source; copy or subtree-sync changes here before tagging a new release on the standalone repository.
