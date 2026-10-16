# Fixed-correspondence GICP factor

`cloudanalyzer_core.gicp_linearization` assembles a GICP residual system for
explicit point pairs and shape matrices. Rust exposes `ca_core::gicp::linearize`
with an array of `Correspondence` values. This is an independent calculation
kernel: existing odometry, matching, solvers, initialization limits and defaults
do not call it implicitly.

```python
from cloudanalyzer_core import gicp_linearization

system = gicp_linearization(
    source, target, source_shapes, target_shapes, pose, center_world,
    weights=weights, parallel=True,
)
H, b, cost = system["normal_matrix"], system["right_hand_side"], system["cost"]
```

| Input | Shape and frame |
| --- | --- |
| `source` | `(N,3)` float64 source points in sensor axes |
| `target` | `(N,3)` float64 corresponding target points in world axes |
| `source_shapes` | `(N,3,3)` float64 symmetric positive definite matrices in source axes |
| `target_shapes` | `(N,3,3)` float64 symmetric positive definite matrices in world axes |
| `pose` | `(4,4)` proper rigid sensor-to-world transform |
| `center_world` | `(3,)` explicit fixed center for world rotation increments |
| `weights` | Optional `(N,)` finite nonnegative scalar weights; omitted means one |
| `parallel` | Optional indexed per-pair CPU work; default `False` |

The center is independent of the source-frame origin. When reparameterizing the
source frame with a lever arm, transport points/shapes and adjust the pose while
retaining the same physical rotation center. Python copies inputs, supports
strided arrays and releases the GIL for native work. Outputs own their arrays.

At the supplied pose, the kernel computes
`C = C_target + R C_source R.T`, `q = pose(source)` and `residual = q-target`.
Parameter order is **translation xyz, then world rotation xyz about center_world**.
For `arm = q-center_world`, `J = [I, -[arm]x]`. Cholesky whitening supplies
`H = sum(w J.T C^-1 J)`, `b = -sum(w J.T C^-1 residual)` and
`cost = 0.5 sum(w residual.T C^-1 residual)`, without constructing an inverse.

**Correspondences, external weights and transported `C` are held fixed** while
differentiating the residual. The returned system is a fixed-metric surrogate,
not the full Hessian or gradient of a changing-match or changing-`C(R)` objective.
It does not apply a state update, solve, damping, correspondence inference or
fallback to another residual model.

Shapes can describe normalized surface geometry; they are not automatically
calibrated measurement uncertainty. Their declared scale affects the objective.
Comparing objective values across different shape/residual models does not
establish trajectory accuracy. No covariance or observability gate is inferred.

Each input matrix must be finite and symmetric within `1e-12 * max(abs(matrix))`.
Within that tolerance it is canonicalized by a symmetric average. Each individual
shape and the transported sum must have strictly positive Cholesky pivots.
The kernel applies no eigenvalue floor or repair. Pose orthogonality and
determinant tolerance is `1e-10`; Python requires a homogeneous last row within
`1e-12`. Nonfinite arithmetic and accumulated overflow are rejected explicitly.
Zero-weight pairs are still validated. Errors identify the first input pair;
common pose/center errors have no pair index.

The result contains `normal_matrix` `(6,6)`, `right_hand_side` `(6,)`, half-squared
`cost`, `weight_sum`, `positive_weight_points`, `source_world` `(N,3)`,
`combined_shapes` `(N,3,3)` and **unweighted** `mahalanobis_squared` `(N,)`.
An empty batch returns zero system/support and correctly shaped empty arrays.
The six-parameter normal matrix can be rank deficient even with valid shapes.
Parallel execution computes independent rows and then sums in original input
order; serial and parallel results/error indices are checked for bit equality.
No speed claim follows from enabling this option.

The formulation was checked against the primary
[small_gicp factor](https://github.com/koide3/small_gicp/blob/master/include/small_gicp/factors/gicp_factor.hpp).
The real-data audit declares normalized surface eigenvalues `[0.001,1,1]`,
informed by its [feature estimator](https://github.com/koide3/small_gicp/blob/master/include/small_gicp/util/normal_estimation.hpp).
Our explicit center/order and unsupported-input policy are documented above.

The [real-recording packet](../benchmarks/slam/flatwall-gicp-linearization/README.md)
reuses the [captured actual matches](slam-odometry-linearization.md), constructs
declared local shapes and checks the native factor against independent NumPy
algebra. It compares conditional geometry on identical retained pairs/weights,
without producing or adopting a new trajectory. Integrating neighborhood
preparation, matching and iterative registration remains a separate step.
