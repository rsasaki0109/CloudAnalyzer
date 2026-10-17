# GICP outer-state cycle guard

`GicpCycleGuard` enables bounded numerical recurrence detection for
`gicp_register` and `GicpTarget.register`. Registration defaults to
`cycle_guard=None`; its original results and histories retain the same keys
and floating-point bits. The frontend does not enable this option.

```python
from cloudanalyzer_core import GicpCycleGuard

result = target.register(
    source, source_shapes, initial_pose, center_world,
    max_correspondence_distance=2.0,
    cycle_guard=GicpCycleGuard(),
)
if result["termination"] == "cycle_detected":
    period = result["cycle"]["period"]
    # result["pose"] is the last accepted pose; result["converged"] is False.
```

Rust callers set `RegistrationParams.cycle_guard` to
`Some(gicp_cycle::CycleGuardParams::default())`.

## Witness contract

The default checks periods 2 through 8. A period `p` requires three consecutive
passes through all `p` phases: each of the last `2p` baseline states must agree
with its state `p` iterations earlier. A single returning phase is insufficient.
The smallest verified period is reported. At least one adjacent pose pair in
the last period must exceed the original translation or rotation convergence
tolerance. Stationary states and drift entirely within that envelope do not
provide a cycle witness.

Every compared pair must have identical **literal ordered source and target
IDs**, including the complete matched-source support. Pose, combined shapes
and weights must also agree under the following bounds:

| Configuration | Default | Comparison |
| --- | ---: | --- |
| `max_period` | 8 | Integer in 2..16, inclusive |
| `confirmations` | 2 | Complete returns; integer in 2..4 |
| `translation_tolerance` | 1e-9 m | Center-aware left relative translation norm |
| `rotation_tolerance` | 1e-10 rad | Principal relative rotation angle |
| `relative_shape_tolerance` | 1e-9 | Per-row Frobenius difference / larger shape norm |
| `weight_tolerance` | 1e-12 | Absolute difference of each same-ID weight |

All tolerances must be finite and positive. Both pose bounds must be at most
one hundredth of the corresponding registration convergence tolerances. For
tighter registration tolerances, supply explicitly smaller cycle bounds.

For poses `a` and `b` and the fixed world center `c`, comparison uses
`Rrel = Ra Rbᵀ` and `dt = (ta-c) - Rrel (tb-c)`. Its rotation angle uses
`atan2(norm(vee(Rrel-Rrelᵀ)/2), (trace(Rrel)-1)/2)`.
Evaluation subtracts rotations and translations first: `D=(Ra-Rb)Rbᵀ`,
`dt=(ta-tb)-D(tb-c)` and the angle denominator is `1+trace(D)/2`.
For rotations these are equivalent expressions; identical input poses give
exactly zero without magnifying an approximate `Ra Raᵀ-I` by a source lever arm.
These motion norms are invariant to a source coordinate change, including
translation of the source origin, and to a rigid change of world coordinates
in exact arithmetic. Shape Frobenius comparisons are invariant to world basis
rotation. Large coordinate origins can still limit floating-point resolution.
No metric repair, floor, hash-only identity, or tolerance adaptation is used.

The per-call buffer holds at most `max_period * (confirmations+1)` baseline
snapshots, each storing the pose, ordered IDs, combined shapes and weights.
Storage is proportional to this bound times matched rows. History capture is
independent; shared immutable targets keep no guard state between calls.

## Stopping and output

Matching and linearization errors, support checks, radius checks and the
undamped solver retain priority. The existing full Gauss–Newton step
convergence test runs before cycle detection. Only an eligible baseline is
buffered, before trying any candidate at that iteration. A short accepted
trial or small period-return displacement cannot declare convergence.

A witnessed cycle returns `termination="cycle_detected"`, `converged=False`,
and `failure_reason="repeated nonstationary GICP outer states"`. The current
pose is the last pose accepted by an earlier iteration. The detecting history
row contains the full step and unchanged before/after pose, with no trials or
accepted scale. Final matches, weights and refreshed cost are recomputed at
the returned pose as usual; a final rematch error still takes priority.

Only a witnessed cycle adds a `cycle` dictionary. It contains `period`,
`confirmations`, one-based inclusive `first_iteration` / `last_iteration`,
`compared_state_pairs`, and maxima `max_translation`, `max_rotation`,
`max_relative_shape`, `max_weight` over all witnessing pairs. Enabled calls
without a witness retain the original output keys. Nonfinite comparison
arithmetic reports the existing `nonfinite` termination with an explicit
reason.

This is a bounded approximate recurrence diagnosis, not a proof of an exact
mathematical cycle. Periods outside the bound, incomplete returns, changes in
support, or drift above any bound remain unclassified. It does not recover
convergence, change the objective, adopt a frontend state, or establish physical
accuracy. See the [real-input evidence](../benchmarks/slam/flatwall-gicp-cycle/README.md)
and the [outer-transition diagnostic](slam-gicp-transitions.md).
