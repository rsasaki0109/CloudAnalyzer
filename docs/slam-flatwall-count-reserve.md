# Reserve exact target-count storage

The opt-in target-balancing path reserves its coordinate hash table and dense
frequency vector for at most one unique coordinate per accepted pair. This
removes repeated capacity growth and rehashing during counting. Coordinate keys,
signed-zero normalization, dense IDs, one-entry lookups, pair order, counts,
weights and floating-point arithmetic have identical source. Search, worker
dispatch, chunks/reductions, solver, stopping, map updates and defaults are
unchanged. The default unbalanced H/b kernel does not use these allocations.
Explicit final-pose diagnostic capture also uses this counting helper; its
numeric output is unchanged.

This upper-bound reservation can hold more unused capacity on duplicate-heavy
inputs. It trades allocation overhead for space; no peak-memory improvement is
claimed. There is no measured-data-specific capacity threshold or new option.

## Whole native registration

The [packet](../benchmarks/slam/flatwall-count-reserve/README.md) retains all
16 original real cases: two recordings, both prediction policies, raw/gyro and
legacy/fine sampling. Four fresh native processes use ABBA order, four workers,
one warmup and three measured rounds. Each case/mode/implementation uses the
median of six complete measured recording totals; the aggregate sums those case
medians. Ratios below 1 mean less candidate elapsed time.

| Conditions | Cases | Balanced ratio | Unchanged default control |
| --- | ---: | ---: | ---: |
| All conditions | 16 | 0.928349 | 0.992696 |
| Legacy sampling | 8 | 0.915743 | 1.014960 |
| Fine sampling | 8 | 0.930446 | 0.988767 |

Balanced case ratios range from 0.834822 to
1.023219; 15/16 decrease. Default control
ratios range from 0.900050 to 1.136636;
8/16 decrease. Every warmup, recording total, frame clock and
regression remains available. Default control variation is not a separate
optimization claim. These are reused development recordings, without a blind
holdout, on one host with uncontrolled frequency and placement.

Clocks include complete native automatic register calls on preloaded scans:
filtering, sampling, prediction, dispatch, ICP, solving, insertion and pruning.
Bag decoding, gyro preparation, validation and serialization stay outside clocks.
No GPU, whole-application or accuracy improvement is established.

## Fixed states and numerical proof

Eight additional processes retain all 64 frozen real states: 60 eligible and
four empty-source controls. Each worker count uses ABBA order, one warmup and
five measured blocks of 32 repetitions. Sum per-state medians of ten measured
blocks over the eligible states; these diagnostic kernel windows are separate
from the whole-register metric above.

| Workers | Count ratio | Balanced H/b ratio | Default H/b control |
| ---: | ---: | ---: | ---: |
| 1 | 0.727974 | 0.964321 | 0.994457 |
| 4 | 0.749586 | 0.930481 | 0.979814 |

All 512 fixed-state validation records match native pair/count/matrix bits within
each worker count and check independent original arrays: 257,328 ordered pair
rows checked. All stage clocks and empty controls remain available. Existing
unit tests cover signed zeros, empty inputs, nearby distinct coordinates,
resizing and repeated targets. No test asserts an implementation-specific
capacity. All 81,920 replay frames preserve original pose and learned-threshold
bits, iteration counts and termination reasons. Both binaries also match complete
numerical frame records. The source proof permits only capacity reservations.

The baseline reuses PR317 candidate executables after checking all 84 source
files against current main, identical commands and toolchain. Original build
inputs, receipts, logs and binary hashes remain bound to those exact bytes.
The new candidate has a fresh 84-file snapshot. Targeted release tests pass
(29 baseline and 29 candidate), as do serial/parallel all-target clippy and
formatting. This does not claim a clean broad local Windows release suite;
previous unchanged NaN bit-test failures remain documented in the prior study.
The installed extension is never replaced; raw bags are not redistributed.
