# Single-lookup target multiplicities

Target-balanced odometry now counts exact target coordinates directly from the
accepted pair records. The first pass assigns a dense coordinate ID and increments
its frequency. A second pass replaces each ID in the output buffer with that
frequency. This removes the temporary target-coordinate vector and the second
hash lookup for every pair. The diagnostic coordinate-slice interface uses the
same counter.

Coordinate equality, signed-zero normalization, accepted pairs, their order,
robust weights, 256-pair H/b reductions and registration policies are unchanged.
The additional frequency vector holds one entry per unique target. Target
balancing remains opt-in; its accuracy and stopping policy are unchanged.

## Observed costs

The [comparison packet](../benchmarks/slam/flatwall-count-lookup/README.md) retains
all native outputs from the same real recordings and predetermined states used in
the preceding [stage study](slam-flatwall-balance-stages.md). Ratios below are
candidate/reference elapsed time; less than 1 means less elapsed time. These are
single-host observations with uncontrolled CPU placement and frequency.

For the 60 alignment-eligible fixed states, take the median of ten measured
32-evaluation blocks per state, thread count and implementation, then sum those
state medians. Four original empty-source controls are retained separately.

| Fixed-state operation | 1 worker | 4 workers |
| --- | ---: | ---: |
| Exact count stage | 0.752682 | 0.762010 |
| Whole target-balanced H/b | 0.955550 | 0.902984 |
| Unchanged default H/b control | 1.002076 | 0.962183 |

The integrated count stage took about 24% less time. Whole fixed-state balanced
H/b took about 4.4% less time with one worker and 9.7% less with four. Stage
instrumentation, code layout and scheduling also affect the observations; the
unchanged control exposes some of that variation.

Complete recording registration did **not establish an overall speedup**:

| Complete native registration | Cases | Balanced ratio | Default control ratio |
| --- | ---: | ---: | ---: |
| All original conditions | 16 | 1.007763 | 1.038606 |
| Legacy registration sampling | 8 | 1.008272 | 0.990802 |
| Fine registration sampling | 8 | 1.007689 | 1.046521 |

Each case/mode/implementation has six measured complete recording totals, from two
fresh processes with one warmup and three measured rounds each. Processes run in
reference/candidate/candidate/reference order. For each case take its six-sample
median, then sum case medians. The balanced sum was 17.0216 s for the reference and
17.1538 s for the candidate, about 0.8% more elapsed time. Five of sixteen case
medians decreased; individual ratios ranged from 0.935259 to 1.056059. No result
was dropped or replaced. The unchanged default control increased about 3.9% in
aggregate, so this experiment does not isolate a whole-registration benefit.

The registration clocks include filtering, ICP, solving, learned thresholds and
map insertion/pruning on preloaded scans. They exclude bag decoding, gyro deskew,
input preparation, validation, serialization, application/UI and GPU work. The
fixed-state clocks cover H/b construction only. These results do not establish
whole-application, GPU or accuracy improvements.

## Numerical checks and retained records

- All 92,160 replayed frames match the original poses, iterations, termination
  reasons and learned thresholds; floating-point poses and thresholds match bits.
- Both builds match all 4,000 capped-step systems and updates, and all 6,020 final
  pairs from the eight original capped traces, bit for bit.
- The eight fixed-state processes check 512 state outputs and 257,328 ordered pair
  rows against the independent original-data search/matrix oracle. Counts and
  native matrices also match bits across implementations at each worker count.
- All 2,560 measured fixed-state blocks and 512 warmup blocks are retained. One
  empty-control block recorded 0 ns for search and counting. Those original values
  remain intact. Only empty-control component durations may be zero; whole calls,
  outer stage clocks and every eligible-state component must remain positive.

The original positive-duration validator rejected that empty control. Its source,
failure, invocation and the explicit correction are retained under
`prior-validation/` and `validation-note-01.json`. The selection, native binaries,
timings and frozen protocol were not changed or rerun to replace a sample.

The packet contains deterministic gzip copies of 30 original native JSON outputs,
execution logs, build inventories and receipts, the frozen protocol, analysis
source, all per-state/per-case summaries, and source aliases for both 84-file
build inventories. Public source projections normalize newlines only; the raw
executed-source hashes and original private snapshots remain preserved. Earlier
studies still resolve their unchanged executed source through the existing stage
archive. Their original metadata and affected test source are retained in
`history/`.

Local validation: the 24 odometry release tests pass (the explicit-input benchmark
is ignored), and parallel all-target clippy passes with warnings denied. A broader
Windows release library run has three NaN sign/payload bit-comparison failures in
unchanged ICP/mesh tests. Both independently frozen builds reproduce exactly that
same failure set: reference 507 pass/3 fail, candidate 509 pass/3 fail. These
pre-existing failures were retained and are not reported as a clean release suite.

Run `python benchmarks/slam/flatwall-count-lookup/verify.py` to hash-check the
packet, verify source/build/execution bindings, replay every recorded comparison
and recompute the summaries. Regression tests are in
`cloudanalyzer/tests/test_flatwall_count_lookup_packet.py`.
