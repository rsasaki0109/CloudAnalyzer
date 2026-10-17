# Costs between GICP outer iterations

`scripts/audit_flatwall_gicp_transition.py` diagnoses saved registration
histories. It evaluates costs at the original captured poses without running
an optimizer, accepting a pose or changing a termination reason.

The [registration API](slam-gicp-registration.md) freezes matching IDs,
weights and the combined shape during inner trials. The next outer iteration
updates matching IDs and transports source shapes again. A decrease in one
frozen trial cost need not be a decrease after that refresh.

## Fixed support and cost cube

Take the intersection of source IDs matched before and after an accepted
step, in original source order. On this fixed support define

`F(a,r,p) = 0.5 * sum residual(p,a).T * C(a,r)^-1 * residual(p,a)`

where `residual(p,a) = R_p * source + t_p - target[a]` and
`C(a,r) = target_shape[a] + R_r * source_shape * R_r.T`.

The eight cells independently select baseline (`0`) or accepted (`1`) state:

| Axis | Selection |
| --- | --- |
| `a` | Captured target ID, including its world point **and world shape** |
| `r` | Pose rotation used to transport the source shape |
| `p` | Pose used to transform the source point in the residual |

All cells use the same common source IDs and unit weights. Off-diagonal
combinations are diagnostic counterfactuals on real inputs and captured poses;
they are not new estimator states. Target shapes keep world axes and source
shapes keep source axes. Shape canonicalization uses the existing symmetry
tolerance and strict SPD checks; no identity shape or eigenvalue floor is used.

At the accepted residual pose, one accounting order is:

```text
motion             = F(0,0,1) - F(0,0,0)
source_metric      = F(0,1,1) - F(0,0,1)
assignment         = F(1,1,1) - F(0,1,1)
common_delta       = motion + source_metric + assignment
full_refreshed_delta = common_delta + added_cost - removed_cost
```

`added_cost` evaluates newly matched source IDs at the accepted pose with
their new target IDs and refreshed shapes. `removed_cost` evaluates lost
source IDs at the baseline pose with their old IDs and shapes. These remain
separate; changing-support sums are not treated as one fixed objective.
Empty common support has a zero cost cube and no per-point comparison.

The tool also reports assignment-first accounting and the interaction
between assignment and source-metric refresh. These effects depend on the
chosen accounting order. In particular, an assignment change also selects a
different target shape; it does not isolate target coordinate changes alone.

## Checks and interpretation

Each cube cell is evaluated independently with NumPy and compared with the
unchanged native factor/frozen-cost APIs, including per-point residual costs.
Full baseline, frozen trial and refreshed costs are checked against saved
history values. Complete arrays, source/native/controller identities and
input hashes accompany the audit; the public packet retains reported
summaries and limited-case tails.

This diagnostic accepts captured unit weights only. It does not approximate
robust weight refresh, select a matching policy or infer physical accuracy.
Repeated correspondence IDs and pose recurrence are different observations.
The existing iteration limit and convergence tolerances retain their meanings.
