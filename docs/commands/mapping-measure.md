# mapping-measure

Measure a synchronous mapping command and save its execution receipt:

```sh
pip install 'cloudanalyzer[benchmark]'
ca mapping-measure --run-dir runs/april --timeout 600 -- \
  python scripts/mapping_e2e_coverage.py \
  web/public/samples/nclt-2012-04-29.mcap runs/april 2.5
ca mapping-scorecard runs/april
```

Put wrapper options before `--` and the executable/arguments after it. Arguments
are passed directly, with no shell expansion. Quote paths containing spaces.
The wrapper does not append the run directory to the child command: pass the
same directory explicitly if the child requires it. Existing output directories
are rejected. The child creates its normal output; the wrapper saves
`measurement.json` afterward, even if the child fails to create the directory.

`--interval` controls process-tree RSS sampling (0.01–1 second, default 0.05).
`--timeout` accepts positive finite seconds; omitted means unlimited. The limit
is checked between samples, so blocking process creation/monitoring can delay
its observation. Termination waits may add up to ten seconds. Only synchronous
work through the direct child's observed exit is measured; detached work is not.

Child stdout/stderr are inherited. The final receipt is also printed to stdout,
which may therefore contain child logs before the JSON. For automation, read
`RUN_DIR/measurement.json` directly. Exit 0 means the child succeeded; failure,
timeout, interruption or invalid options return nonzero. Check `status`,
`exit_code`, `error` and `cleanup_error` for details. Missing output-size evidence
is null with `output_size_error`; it does not erase independent timing metrics.

The receipt records command/script identities, platform, elapsed time, sampled
RSS and output bytes. RSS may miss short peaks or double-count shared pages.
These observations are not accuracy measurements or controlled hardware benchmarks.
Only runs with valid `run.json` and `job.json` can subsequently be summarized by
[mapping-scorecard](mapping-scorecard.md).

The existing `python -m ca.mapping_measure` entry point remains supported.
