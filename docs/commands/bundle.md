# `ca bundle`

Pack, unpack, and inspect CloudAnalyzer QA result bundles. A bundle is a single ZIP that freezes one QA run — the summary JSON, the per-check reports the summary points at, an optional baseline summary, and a versioned metadata header — so the result is reopenable later without going back to the original CI workspace.

This is the OSS layer the hosted retention / dashboard story sits on top of: bundles are the contract.

## Subcommands

### `ca bundle pack <summary> --output <bundle.zip>`

Build a bundle from a summary JSON.

```bash
ca bundle pack qa/summary.json \
  --output qa/bundle.zip \
  --baseline qa/baseline-summary.json \
  --project my-mapping-pipeline \
  --commit "$GITHUB_SHA" \
  --pr-number "$PR_NUMBER" \
  --runner-id "$GITHUB_RUN_ID" \
  --note dataset=newer-college-mini --note voxel=0.05
```

| Option | Purpose |
|---|---|
| `--output <file>` / `-o` | Output bundle ZIP path (required) |
| `--baseline <file>` | Optional baseline summary JSON of the same shape; pulled into the bundle for reproducible delta computation later |
| `--project <label>` | Project label written into metadata (defaults to the summary's `project` field when present) |
| `--commit <sha>` | Git commit SHA to record |
| `--pr-number <num>` | PR number to record |
| `--runner-id <id>` | CI runner identifier (e.g. `${{ github.run_id }}`) |
| `--note key=value` | Free-form metadata (repeatable) — e.g. dataset hashes, voxel sizes, SLAM version strings |

The packer accepts both `ca check` suite summaries (`summary` + `checks`) and single-run JSON from `ca run-evaluate` / `ca benchmark eval` (`overall_quality_gate` + `map`/`trajectory`). For check suites it walks `summary["checks"]`, copies each `report_path` / `json_path` into the bundle under `reports/<check_id>/`, and silently skips ones the runner cannot find (the original artifact may have been on a different machine). For single-run summaries the bundle just carries the JSON.

### `ca bundle unpack <bundle.zip> --output <dir>`

Extract a bundle into a directory.

```bash
ca bundle unpack qa/bundle.zip --output qa/restored/
```

Rejects archive entries with absolute paths or `..` traversal up front, so it is safe to point at bundles from untrusted sources.

### `ca bundle diff <old.zip> <new.zip>`

Compare two bundles and render a Markdown report. Reuses the same metric / delta layout as `ca report-pr-comment`, so the diff drops straight into a PR comment or a dashboard table.

```bash
ca bundle diff qa/baseline.zip qa/bundle.zip --output qa/diff.md
ca bundle diff qa/baseline.zip qa/bundle.zip --format-json | jq '.warnings'
```

| Option | Purpose |
|---|---|
| `--output <file>` / `-o` | Write Markdown to a file instead of stdout |
| `--format-json` | Emit the structured diff dict (`old`, `new`, `warnings`) for tooling |

Convention: `<old>` is the baseline (e.g. last release), `<new>` is the current run. The Markdown header shows a metadata-comparison table (project / commit / PR / runner / dataset notes) with `⚠️` next to each field whose value differs; then a `**Metadata divergence**` list highlights the mismatches; then the `## CloudAnalyzer QA:` block (identical to what `ca report-pr-comment` renders) shows the per-metric / per-check deltas.

`ca bundle diff` rejects diffs across different `summary_kind` values (a `ca check` bundle vs. a single-run bundle) because the metric layouts are not comparable. Metadata mismatches (different commit, different PR, different notes) are surfaced as **warnings** rather than errors so reviewers can decide whether the comparison is apples-to-apples.

### `ca bundle show <bundle.zip>`

Print metadata and table of contents without extracting.

```bash
ca bundle show qa/bundle.zip
ca bundle show qa/bundle.zip --format-json | jq '.metadata.notes'
```

`--format-json` emits the full structured payload so downstream tools (dashboards, retention indexers, PR-comment formatters) can pick what they need.

## Bundle layout

```
qa_bundle.zip
├── metadata.json              # bundle_version / created_at / project / commit / pr / runner / notes / artifact index
├── summary.json               # the original ca check / ca run-evaluate / ca benchmark eval JSON
├── baseline-summary.json      # only when --baseline was supplied
└── reports/<check_id>/        # per-check report.html / report.md / report.json copied from the summary
```

`metadata.json` shape (`bundle_version: 1`):

```json
{
  "bundle_version": 1,
  "created_at": "2026-05-19T11:51:34+00:00",
  "cloudanalyzer_version": "0.1.0",
  "summary_kind": "check_suite",
  "project": "my-mapping-pipeline",
  "git_commit": "abc1234",
  "pr_number": "42",
  "runner_id": "9876543210",
  "notes": {
    "dataset": "newer-college-mini",
    "voxel": "0.05"
  },
  "has_baseline": true,
  "artifacts": [
    {
      "check_id": "mapping-postprocess",
      "field_name": "report_path",
      "archive_path": "reports/mapping-postprocess/mapping-postprocess.html",
      "source_path": "/abs/path/on/runner/...",
      "size_bytes": 21456
    }
  ]
}
```

`source_path` is preserved as a forensic clue (which runner produced it) but is not used for extraction — `archive_path` is the canonical in-bundle location.

## CI wiring

Pair `ca bundle pack` with `actions/upload-artifact@v6` so the QA bundle lives alongside the regular test artifacts:

```yaml
- name: Run QA
  run: ca check cloudanalyzer.yaml --output-json qa/summary.json

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

## Future-proofing

`bundle_version: 1` is the only version that exists today; the loader rejects anything else explicitly so older / mismatched CloudAnalyzer installs surface that as a clear error rather than silently misinterpreting fields. New fields land as additive `metadata.json` keys — existing tooling that ignores unknown keys keeps working.
