# Native odometry: skip structural zeros in normal equations

The existing NCLT development recordings expose 12,998 / 14,788 ICP iterations
across 236 / 238 frames. The slowest frames need roughly 150–200 iterations.
Each matched point contributes three Jacobian rows, each with only three
structurally nonzero coefficients. The previous dense loop visits 108 matrix
terms per point; the candidate visits 27, retaining row order and both matrix
triangles. Correspondence search, weights, sampling, convergence, map retention
and iteration limits are unchanged. This is a computational optimization, not a
new registration algorithm or accuracy claim.

## Experiment

Baseline: `437223a406108986337ebc090804e449fd2eeace` core source.
Candidate: the sparse row-index loop in this change. Both use the new
`bench_bag_odometry` example, the same release compiler configuration and
`parallel` feature, with `RAYON_NUM_THREADS=1`. Rust/Cargo 1.99.0; shared cloud VM.
Both executables were built before measurement; no compilation overlapped timing.
The candidate executable predates only test additions and rustfmt formatting.
Binary/input SHA256 identities are recorded in `results/identities.json`.

Use every bundled April/June scan, with the same strict 1.5–80 m crop as the
previous comparison. Three repetitions per arm and recording, alternating
baseline/candidate order between repetitions. All frame records are retained,
including poses, timestamps, iteration counts, termination, registration point
counts and duration. The runner verifies exact equality of every non-timing
field across every run against the first baseline run.

Registration timing includes native begin/alignment/map update but excludes bag
decoding, the common crop and JSON writing. Whole-process wall time includes
those operations and startup. These measurements are native-only and must not
be mixed with earlier Python timings. No reference trajectory enters estimation.
This is the existing sparse development cohort, not the pending full-rate cohort
or independent held-out evidence. Observed timing ranges on a shared VM are not
confidence intervals.

## Results

| Recording | Baseline median (s) | Candidate median (s) | Reduction | Baseline range (s) | Candidate range (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2012-04-29 | 17.908 | 17.265 | 3.59% | 17.840–17.980 | 17.157–17.725 |
| 2012-06-15 | 19.708 | 19.462 | 1.25% | 19.131–20.769 | 18.615–19.697 |

All 12 runs completed. Every pose, timestamp, iteration count, termination and
registration point count matched exactly across arms and repetitions. The
candidate was faster in all six adjacent same-session pairs, but the timing
ranges overlap in June and the median benefit there is small. This supports a
modest reduction in work, not a large performance or accuracy advance.
The remaining 12,998 / 14,788 iterations are unchanged: reducing the cost per
iteration alone does not address slow convergence.

Validation: 25 native odometry tests passed in release mode with `parallel`.

## Reproduce

Build `bench_bag_odometry` at baseline and candidate source revisions using:

```sh
cargo build --release --manifest-path rust/Cargo.toml -p ca-core \
  --features parallel --example bench_bag_odometry
```

The baseline revision predates the example; copy the candidate's unchanged
example source into a detached baseline checkout before building it. Preserve
both executables outside the build directory, then run:

```sh
python benchmarks/slam/nclt-sparse-equations/run.py \
  --baseline /path/to/baseline --candidate /path/to/candidate \
  --out /path/to/new-results
```

The output directory must be new. Failed executions abort without replacing
existing artifacts. The Rust unit test independently computes the dense
Jacobian equations for multiple centers and correspondence thresholds, including
zero accepted correspondences. Existing odometry tests cover nonfinite systems,
scan/motion behavior, diagnostics, iteration termination and worker dispatch.

NCLT-derived data retain ODbL 1.0 / DBCL 1.0 terms; source is the University of
Michigan NCLT dataset (Carlevaris-Bianco, Ushani and Eustice, IJRR 2016).
