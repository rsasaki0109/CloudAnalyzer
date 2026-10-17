# Flatwall transported-shape gradient and update experiments

This packet validates a separate unit-weight native gradient API and reports
two experimental update policies that were **not adopted** in registration.
The data are the same 36 fixed real problems, each with caller-prepared and
native-prepared shapes (72 histories), from the
[immutable-target study](../flatwall-gicp-target/README.md). Dataset attribution
and frozen input provenance are retained there. No synthetic performance data
or reference trajectory is used.

## Objective and decision

The initial [protocol](protocol.json) declares
`F=0.5 sum rᵀ(Ct+R Cs Rᵀ)⁻¹r`, exact Euclidean nearest assignment, unit weights,
the original 30-iteration budget, original `1e-4` / `1e-5` full-direction
tolerances and six trial scales. Its exact transported-shape gradient is
preconditioned by the original frozen SPD residual normal matrix. Acceptance
requires strict decrease and Armijo `1e-4`; a short trial cannot declare
convergence. Normalized surface shapes are not calibrated Gaussian covariance;
this score omits log determinant and robust-weight derivatives.

The first policy rematches target points/shapes at each trial and forbids source
support changes. Its convergence regressions motivated a separately frozen
[held-assignment variant](fixed-variant-protocol.json): source/target IDs and
target shapes stay fixed during inner trials while source shapes are rotated.
It rematches at each outer iteration. No scales, tolerances or iteration limits
were changed between variants.

| Outcome | Original | Rematched trial | Held-assignment trial |
| --- | ---: | ---: | ---: |
| `converged_step` | 54 | 22 | 54 |
| `max_iterations` | 10 | 0 | 10 |
| `no_descent` | 0 | 42 | 0 |
| `insufficient_correspondences` | 8 | 8 | 8 |
| Accepted steps | 686 | 372 | 764 |

The rematched policy stops at `no_descent` in 34 previously converged histories.
The held-assignment policy changes six previously limited histories to
converged, while six previously converged histories reach the limit. Equal
aggregate convergence counts conceal this exchange. Its accepted-step count
also increases. Reported final residual scores are no higher in 58 rematched
histories and 52 held-assignment histories, including eight empty cases; these
counts do not establish better physical alignment.

The [scope decision](scope-decision.json) therefore adopts neither optimizer in
the native/default/frontend path. Both complete outcome sets are retained in
[rematched frames](rematched/frames.json) and [held-assignment frames](fixed/frames.json).
Their protocols, receipts and hashes preserve the original experiments and
controllers. Raw input/optimizer arrays remain private.

## Native diagnostic API

`gicp_transported_linearization` returns the unchanged original frozen factor
fields plus `metric_gradient_rotation`, `objective_gradient` and
`objective_right_hand_side`. For `e=C⁻¹r` and `S=R Cs Rᵀ`, the additional
rotation gradient is `sum(e × (S e))`. The normal matrix remains the frozen
residual matrix, not the exact transported Hessian. See the
[API and frame contract](../../../docs/slam-gicp-transported.md).

The [native audit summary](audit/summary.json) covers 216 original/experimental
histories and 3,728 baseline/trial states. The independent NumPy gradient and
native result differ by at most `1.836042429204099e-11`. Serial/parallel
gradient bits and every original frozen factor field agree exactly. All 72
default registrations and all 72 cycle-enabled registrations equal the saved
results bit for bit. Native nearest IDs agree on 944 rematched candidate checks.

The native package [identity](native-artifact.json) and
[113 build-input hashes](native-build-inputs.json) pin a complete private build;
the installed extension is not replaced. Native controls cover the closed-form
`9/22` metric gradient, all six finite-difference components, frame/lever-arm
changes, strides/ownership, empty and isotropic inputs, first errors and a new
derivative overflow that leaves the original API valid. The same package is
used for the full native suite and unchanged motion/frontend regression checks.

## Reproduction and limits

The two reference modules are `scripts/gicp_transported_reference.py` and
`scripts/gicp_transported_fixed_reference.py`. They are independent audit
experiments. `scripts/audit_flatwall_gicp_transported.py` checks the native
gradient on saved original and experimental arrays; `--help` lists inputs.
It requires fresh output paths, the complete private native package on
`PYTHONPATH`, `RAYON_NUM_THREADS=4` and `OPENBLAS_NUM_THREADS=1`.

Run `python benchmarks/slam/flatwall-gicp-transported/verify.py` to verify copied
JSON bytes and reported outcome/state accounting. This check does not rerun raw
real data or independently certify the optimizer. Every report, including
rejected outcomes, uses explicit LF checkout rules. No convergence-repair,
physical accuracy, observability, GPU or whole-SLAM speed claim is made.
