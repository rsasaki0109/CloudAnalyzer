# Reuse voxel-face distance bounds

Each nearest-point query now computes its six squared distances to voxel faces
once and reuses them across the existing 26-neighbor traversal. The three-axis
sum keeps its original order. Voxel division/floor/casts, center-first dx/dy/dz
order, strict comparisons and point order retain their definitions. Target
counts, weights, H/b arithmetic and chunks, worker dispatch, solving, stopping,
map updates and defaults have identical source. Both serial and parallel builds
use the change; there is no new option or worker pool.

## Complete native registration

The [packet](../benchmarks/slam/flatwall-face-gaps/README.md) retains all 16 real
recording cases in both modes: two recordings, raw/gyro inputs, two prediction
policies and legacy/fine sampling. Four fresh native processes run in
reference/candidate/candidate/reference order, with four workers, one warmup and
three measured rounds each. For each case and mode, take the median of six
complete measured recording totals per implementation, then sum case medians.
Ratios below 1 mean less candidate elapsed time.

| Registration conditions | Cases | Balanced ratio | Default ratio |
| --- | ---: | ---: | ---: |
| All conditions | 16 | 0.906989 | 0.896907 |
| Legacy sampling | 8 | 0.910800 | 0.961228 |
| Fine sampling | 8 | 0.906424 | 0.886750 |

Balanced case ratios range from 0.859094 to
0.981808; 16 of 16 decrease. Default case
ratios range from 0.834363 to 1.041056;
13 of 16 decrease. All warmups, frame clocks, recording totals
and slower outcomes remain available. This reused development cohort has no
blind holdout. Host placement and frequency are uncontrolled.

Clocks cover whole native register calls on preloaded scans, including filtering,
sampling, prediction, dispatch, all ICP steps, solving and map insertion/pruning.
Bag decoding, gyro preparation of inputs, checks and serialization stay outside
the clocks. These measurements establish no GPU, application-wide or accuracy
gain.

## Fixed real states and numerical checks

The same 64 previously frozen real states remain: 60 eligible states and four
empty-source controls. At each worker count, four fresh native processes use
the same comparison order, with one warmup and five measured blocks of 32
repetitions. Sum per-state medians of ten measured blocks over the 60 eligible
states. The default H/b column uses these same fixed states, with balancing off.
It is separate from the complete default trajectory timings above.

| Workers | Balanced H/b ratio | Default H/b ratio |
| ---: | ---: | ---: |
| 1 | 0.908873 | 0.862130 |
| 4 | 0.892550 | 0.833153 |

Every fixed-state pair, count and matrix is checked against independent original
state arrays. Within each worker count, complete native validation records also
match bit for bit across both binaries and repeats: 512 states and 257,328
ordered pair rows checked in total. All stage clocks, including unchanged count
and accumulation work and empty controls, are retained; their observed variation
is not a separate optimization claim.

All 81,920 whole-registration replay frames preserve original pose and learned
threshold bits, iteration counts and termination reasons. The independent
verifier compares original checked-in trajectories, plus the complete numerical
frame records of both binaries. Two native unit tests additionally compare
nearest targets and distance bits to an exhaustive ordered 27-cell search,
including sparse cells, boundaries, ties, signed zero, empty maps and NaN queries.
Analytic points are used only for correctness tests; all timings use real data.

Both native binaries are freshly built from 84-file source snapshots with the
same commands and toolchain. The baseline is current main, including the earlier
comment correction; the preceding timed binary is not substituted for it.
Archived sources, build logs/receipts, protocols and lossless output bytes bind
the recorded executions. The source proof restricts production edits to the six
reused face squares. Targeted release tests pass (27 baseline, 29 candidate),
as do serial/parallel all-target clippy and formatting. This does not claim a
clean broad local Windows release suite; earlier unchanged NaN bit-test failures
remain documented in the preceding study. The installed Python extension is
never replaced, and no raw bag is redistributed.
