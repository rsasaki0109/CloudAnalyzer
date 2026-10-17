# Iterative GICP registration

Rust `gicp_registration::register` and Python `gicp_register` align points to
an immutable target using caller-prepared SPD shapes. The API is opt-in;
shape preparation and online frontend adoption remain separate work.

```python
from cloudanalyzer_core import gicp_register

result = gicp_register(
    source, target, source_shapes, target_shapes, initial_pose, center_world,
    max_correspondence_distance=2.0, capture_history=True,
)
pose = result["pose"]  # last accepted source-to-world transform, (4,4)
reason = result["termination"]
```

Points are float64 `(N,3)` / `(M,3)` arrays; shapes are `(N,3,3)` / `(M,3,3)`.
Source points and shapes use source axes, target points and shapes world axes.
`initial_pose` is proper rigid and homogeneous; `center_world` is a finite
`(3,)` fixed center throughout the call. Inputs are copied before releasing
the GIL. Outputs and optional history arrays own their storage. Normalized
surface shapes do not imply calibrated measurement covariance.

## Matching and optimization

Every outer iteration transforms all source points and queries the global
float64 target k-d tree. A squared distance must be strictly below
`max_correspondence_distance**2`. Equal-distance ties use the existing tree's
deterministic seed/traversal and strict improvement rule, without a
minimum-original-index guarantee. Each query starts without a previous-query
guess. This global search differs from the odometry voxel stencil; differences
in results cannot be attributed solely to the residual metric.

Weights default to one. Optional `robust_scale=s` is a positive distance and
sets `w=(s²/(s²+d²))²` from baseline Euclidean distance. Matches, weights and
`C=target_shape+R source_shape R.T` remain fixed throughout each inner search.
The [factor kernel](slam-gicp-linearization.md) builds `H`, `b` and half-squared
Mahalanobis cost. Parameters are translation xyz, then left world rotation xyz
about the center. Trials use `Q=exp_so3(scale*delta_rotation)`, `R'=Q R`,
`t'=Q(t-center)+center+scale*delta_translation`.

At least three positive-weight matches and positive finite weighted RMS point
radius `r` about the center are required. With
`D=diag(1,1,1,1/r,1/r,1/r)`, solve the undamped system
`A=D sym(H) D/sum_w`, `b'=D b/sum_w`, `delta=D solve(A,b')`.
Each Cholesky Schur pivot must exceed `1e-12 * max(diag(A))`. This numerical
rank guard applies no eigenvalue floor, damping or alternative-model fallback.

Convergence requires both norms of the **full solved step** to meet the
translation and rotation tolerances, default `1e-4` distance units and `1e-5`
radians. That final small step is not applied. Otherwise try scales
`[1,.5,.25,.125,.0625,.03125]` and adopt the first finite proper candidate
with a strictly smaller frozen cost. A shortened step alone cannot signal
convergence. Rematching and metric transport start another surrogate; costs
across outer iterations need not decrease. The default budget is 30 iterations.

`gicp_frozen_cost(source, target, combined_shapes, candidate_pose,
weights=weights)` returns `cost` and unweighted `mahalanobis_squared` using
supplied baseline metrics. It performs no shape transport, matching or weight
refresh. Rust `FrozenMetric` owns pairs, weights and prepared Cholesky factors
for repeated evaluations.

## Results and failures

| Termination | Meaning |
| --- | --- |
| `converged_step` | Supported full step meets both tolerances. |
| `insufficient_correspondences` | Fewer than three positive-weight matches. |
| `singular_system` | Zero radius or failed numerical rank guard. |
| `nonfinite` | Runtime arithmetic or every trial candidate is invalid. |
| `no_descent` | Finite candidates exist but no declared scale decreases cost. |
| `max_iterations` | Iteration budget exhausted. |

Only `converged_step` sets `converged=True`. All outcomes return the last
accepted pose, possibly the initial pose. `iterations` counts outer attempts;
`accepted_steps` counts adopted candidates. The API rematches and relinearizes
at the returned pose to report `final_cost`, `matched_source_indices`,
`matched_target_indices`, `squared_distances` and `weights`. This refreshed
cost is separate from inner frozen costs. Empty support gives zero cost and a
failure status. Runtime numeric failure can leave `final_cost=None`;
`failure_reason` gives detail when matching or linearization fails.

`capture_history=True` retains baseline/accepted poses, match indices,
distances, weights, combined shapes, `H`, `b`, full step, attempted scales and
trial costs. Invalid trial costs are `None`. Rejected states are never adopted.
History defaults to an empty list to avoid retaining all iteration data.

Malformed inputs raise before matching, including invalid unmatched shapes.
Finite/proper-pose and strict SPD validation follow the factor kernel. Source
and target shapes are checked in input order. Distances, tolerances and optional
robust scale must be finite and positive; squared scales must be representable.
Empty point sets are valid inputs with explicit insufficient support.

With the Rust `parallel` feature, matching, factor and trial rows use indexed
Rayon work followed by serial source-order sums. Serial and parallel outputs
retain identical bits, including history and first errors. `parallel=False`
remains the default. This slice makes no timing claim.

## Verification

Native tests compare iterations with an independent NumPy/SciPy solver,
including frozen trial costs, robust weights, rigid recovery, world/source
frame changes, a source lever arm, strict thresholds, numeric failures,
strides and ownership. Closed-form controls are unit tests.
[`audit_flatwall_gicp_registration.py`](../scripts/audit_flatwall_gicp_registration.py)
checks real fixed-map registrations with exhaustive reference matching.
Its [evidence packet](../benchmarks/slam/flatwall-gicp-registration/README.md)
records the frozen protocol and limits. Subsequent maps use the original
captured poses. These local trial poses do not establish trajectory accuracy,
physical observability or successful IMU initialization. Existing ICP,
odometry, initialization gates and defaults are unchanged.

An explicit [`GicpCycleGuard`](slam-gicp-cycles.md) can diagnose repeated outer
states separately from convergence. It is disabled by default.
