# mapping-scorecard

Read a saved mapping run's generation and measurement summary as JSON:

```sh
ca mapping-scorecard runs/current
ca mapping-scorecard runs/current --baseline runs/previous > comparison.json
```

The command reads `run.json`, `job.json` and optional `measurement.json`. It does
not run mapping, start Docker or modify those files. The existing
`python -m ca.mapping_scorecard` entry point remains available.

A single scorecard includes recorded generation extent, unresolved traces,
action count and available performance measurements. Human intervention counts
and independent accuracy remain unknown unless supplied by another workflow;
recorded actions are not a measure of human effort.

A baseline must have identical source and layout hashes. Comparison deltas are
**current minus baseline**. `performance_comparison` explains whether performance
deltas are eligible and why they may be omitted. Known matching generation stages,
successful current measurements, the supported protocol and matching sampling
intervals are required. This does not establish identical hardware or system load.

Missing, stale or damaged optional measurement evidence leaves generation
summaries available. Unavailable metrics are `null`; `measurement_status`,
`measurement_report_error` and `measurement_diagnostics` explain the recorded
state. A successful command does not mean that mapping succeeded or its outputs
are accurate.

Metadata must use unambiguous JSON: duplicate object keys (including nested or
escaped duplicates), `NaN`, `Infinity` and floating-point overflow such as `1e999`
are rejected. Invalid optional measurements produce `invalid_report` while
preserving the generation summary; invalid required run/job metadata stops the
command. Nonnegative integer metrics retain their exact integer values.
Source/layout identities and present diagnostic sections must be objects;
`history`, when present, must be an array of objects. Invalid structures return
an error naming the field instead of producing misleading action counts. Missing
or null output/diagnostic sections are supported for incomplete runs and leave
their metrics unknown. Missing history retains the default count of zero.

The run hash and measurement binding use the exact bytes parsed for each
metadata file, without reopening those files to compute identities. A receipt
for a later version cannot supply metrics for an earlier parsed summary.
This is not a transaction across files or a guarantee that files remain unchanged
after reading; use a completed, inactive run for a stable comparison.

Exit 0 means the summary/comparison was produced, including summaries of failed
runs or invalid optional measurements. Missing/invalid required metadata or an
incompatible baseline returns a nonzero exit status. Use the JSON fields, rather
than the command exit code, to assess generation and measurement outcomes.

Create measurement receipts with [mapping-measure](mapping-measure.md).
See [mapping-run](mapping-run.md) and the
[measurement workflow](../mapping-priority-work.md#measure-a-fresh-real-log-run).
