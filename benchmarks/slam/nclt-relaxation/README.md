# Fixed-gain ICP convergence experiment

This tests a concrete way to reduce the number of ICP iterations: multiply the
normal-equation right-hand side by 1.5 before solving. The linear solve therefore
proposes approximately 1.5 times the ordinary update. Apply it using the existing
SE3 update around the prediction center and keep the existing translation/rotation
convergence tolerances (0.1 mm / 0.01 mrad). The convergence test sees the enlarged
update; no tolerance relaxation or premature stop is introduced.

This is an experimental example, **not a change to production defaults**. Fixed
extrapolation can overshoot or oscillate. Lower trajectory error alone cannot
prove map consistency, and prior repository notes report worse map consistency
when tolerances were loosened. The candidate therefore retains the original
thresholds and is screened against both accuracy and convergence failures.

## Prospective protocol

`protocol.json`, the example and runner were committed as `5cce95c` before any
candidate result was inspected. Gains 1 and 1.5, both bundled April/June NCLT
recordings, all 474 scans, three repetitions per arm, alternating order and one
Rayon thread. Rejection criteria were fixed beforehand: more than 5% worse ATE
or either RPE RMS on either recording, new iteration-limit/numeric failures, or
no median runtime improvement. Passing is only a development screening result;
independent map-quality and held-out evaluation are required before changing
default behavior. No post-result gain search is performed.

Both arms use the same stepped Rust implementation and capture each update's
translation magnitude at the prediction center and rotation angle. Registration
time includes this trace overhead. It excludes bag decoding, common range
cropping and output serialization; process wall time includes these. The first
scan and every difficult frame remain in the output. All estimator processes
finish before the runner loads any reference CSV. Parameters do not depend on
reference errors. Measurements use the shared VM, not a controlled timing host.

The data are the previously examined, sparse, motion-compensated demonstration
cohort, not the pending full-rate sessions. Reference poses come from original
NCLT body-pose rows, transformed to body-z-up. Scores use rigid position-fit ATE
without scale and consecutive SE3 RPE, with a maximum 50 ms interpolation bracket.
The independent verifier uses the existing separate matrix-based SE3 evaluator,
checks every score and compares gain 1 with the prior production replay.

## Results and decision

| Recording | Gain | Median registration (s) | Iterations | ATE RMS (m) | Translation RPE RMS (m) | Rotation RPE RMS (deg) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2012-04-29 | 1.0 | 17.162 | 12998 | 0.165450 | 0.167944 | 2.038093 |
| 2012-04-29 | 1.5 | 11.491 | 8991 | 0.165665 | 0.168353 | 2.038517 |
| 2012-06-15 | 1.0 | 18.509 | 14788 | 0.154142 | 0.141100 | 1.730891 |
| 2012-06-15 | 1.5 | 12.684 | 10368 | 0.154228 | 0.141535 | 1.730031 |

All 12 runs completed. Each arm produced identical poses across its three
repetitions. All noninitial frames converged; no iteration-limit or numeric
failures occurred. Median registration time fell **33.04% / 31.47%**, and
iterations fell **30.83% / 29.89%** (April / June). ATE increased by
0.216 / 0.086 mm; the largest RPE RMS increase was 0.31%. Endpoint error
changed from 0.3433 to 0.3508 m (April) and 0.3666 to 0.3611 m (June).

The candidate **passes the predefined development screen**. Retain it as an
experimental executable; do not promote it to production defaults yet. The
next adoption gate is map consistency and independent data, not additional
gain tuning on these two recordings. This is a measured candidate speedup,
not an established general speedup or a SOTA result.

Gain 1 reproduces the previous production replay with identical iteration
counts and pose matrices within 1e-11 absolute tolerance (floating-point
roundoff from the separately compiled stepped harness). Both scoring methods
independently reproduce all 12 trajectory scores. `summary.json` retains
full-precision metrics and observed timing ranges.

## Reproduction

```sh
cargo build --release --manifest-path rust/Cargo.toml -p ca-core \
  --features parallel --example bench_bag_relaxation
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python benchmarks/slam/nclt-relaxation/run.py \
  --binary rust/target/release/examples/bench_bag_relaxation \
  --out /path/to/new-results
python benchmarks/slam/nclt-relaxation/verify.py /path/to/new-results
```

The example requires explicit gain 1 or 1.5 and a new output file. Compressed
per-run artifacts retain all poses, update traces, iterations and timings.
SHA256 identities of binary, sources, inputs, reference CSVs and outputs accompany
the results. The binary is not committed. A failed subprocess aborts the runner
with prior artifacts retained; an incomplete run set is not a completed study.

NCLT-derived contents retain ODbL 1.0 / DBCL 1.0 terms. Source: University of
Michigan NCLT, Carlevaris-Bianco, Ushani and Eustice, IJRR 2016.
