# `ca history`

Build a time-series view of QA gate metrics across many bundles.

`ca bundle diff` answers *"is run X better or worse than run Y?"*; `ca history` answers *"how has the gate moved across the last N runs?"* by reading any number of `qa_bundle.zip` archives, sorting them by their metadata's `created_at` stamp, and rendering a per-metric trend table.

## Usage

```bash
# Trend across explicit bundles (any order — sorted by metadata.created_at).
ca history runs/2026-05-15.zip runs/2026-05-17.zip runs/2026-05-19.zip

# Discover every *.zip in a directory and trend them.
ca history --from-dir runs/

# Tighten the glob if you only want some bundles.
ca history --from-dir runs/ --pattern 'ci-*.zip'

# Emit JSON for dashboards / scripting.
ca history --from-dir runs/ --format-json > trend.json

# Write Markdown to a file (e.g. for a PR comment or wiki page).
ca history --from-dir runs/ --output qa/history.md
```

## Output shape

`ca history` supports both bundle summary shapes (`check_suite` and `single_run`); mixing them across the input set is rejected because the metric columns wouldn't line up.

### `check_suite` bundles

Renders one section per `check_id`, with that check's relevant metrics across the timeline. Columns depend on the check `kind`:

- `artifact`: `auc`, `chamfer_distance`, `hausdorff_distance`, `f1`, `precision`, `recall`
- `trajectory`: `ate_rmse`, `rpe_translation_rmse`, `drift_endpoint`, `coverage_ratio`
- `run`: union of the above

```markdown
## CloudAnalyzer QA history

**Bundles**: 3 (check_suite)
**Project**: `cloudanalyzer-self-qa`

### `mapping-postprocess` (artifact)

| When | Commit | Status | auc | chamfer_distance | ... |
|---|---|---|---|---|---|
| 2026-05-15T01:02:03+00:00 | abc12345 | ✅ | 1.0000 | 0.0015 | ... |
| 2026-05-17T01:02:03+00:00 | def67890 | ✅ | 0.9997 | 0.0019 | ... |
| 2026-05-19T01:02:03+00:00 | 11223344 | ❌ | 0.9750 | 0.0210 | ... |
```

### `single_run` bundles (`ca run-evaluate` / `ca benchmark eval`)

Renders one table covering map + trajectory metrics:

```markdown
| When | Commit | PR | Status | Map AUC | Map Chamfer | Map Hausdorff | Map Best F1 | Traj ATE | Traj RPE | Traj Drift | Coverage |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-05-15T01:02:03+00:00 | abc12345 | #42 | ✅ | 0.9500 | 0.0500 | 0.1200 | 0.8800 | 0.1000 | 0.0200 | 0.0500 | 94.0% |
| 2026-05-17T01:02:03+00:00 | def67890 | #43 | ✅ | 0.9450 | 0.0510 | 0.1250 | 0.8780 | 0.1100 | 0.0210 | 0.0530 | 93.5% |
```

## `--format-json` shape

```json
{
  "history_version": 1,
  "summary_kind": "single_run",
  "entries": [
    {
      "bundle_path": "/abs/path/to/qa.zip",
      "created_at": "2026-05-15T01:02:03+00:00",
      "summary_kind": "single_run",
      "project": "cloudanalyzer-self-qa",
      "git_commit": "abc12345",
      "pr_number": "42",
      "overall_passed": true,
      "metrics": {
        "map.auc": 0.95,
        "trajectory.ate.rmse": 0.10,
        ...
      },
      "per_check_passed": {},
      "per_check_metrics": {},
      "per_check_kind": {}
    }
  ]
}
```

For `check_suite` bundles, `per_check_passed` / `per_check_metrics` / `per_check_kind` are populated and `metrics` is empty; vice versa for `single_run`.

## Tips

- Pair with `ca bundle pack` from CI to keep a rolling archive — `ca history --from-dir` then turns the archive into a graphable trend without manual JSON wrangling.
- For dashboards that want a single time-series number per commit, `--format-json` is the stable contract (`history_version: 1`); the Markdown layout may evolve.
- For two-snapshot comparison with metric deltas, use [`ca bundle diff`](bundle.md) instead — `ca history` deliberately doesn't compute deltas because the "previous" is ambiguous across a long timeline.
