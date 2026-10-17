# Transported-shape GICP gradient audit

`gicp_transported_linearization` differentiates the residual score with respect
to both transformed points and rotated source shapes at **explicit fixed
pairs**. It is a separate unit-weight diagnostic API. Existing registration,
frozen factors, cycle handling and frontend defaults do not call it implicitly.

```python
from cloudanalyzer_core import gicp_transported_linearization

factor = gicp_transported_linearization(
    source_pairs, target_pairs, source_shapes, target_shapes,
    pose, center_world, parallel=False,
)
gradient = factor["objective_gradient"]  # translation xyz, left world rotation xyz
```

Points/shapes use the same source/world frames, float64 array layouts, proper
pose and fixed world center as the [original factor](slam-gicp-linearization.md).
Rust uses `gicp_transported::linearize(pairs, pose, center_world, parallel)`.
Original factor validation runs first, including every row; then every external
Rust pair weight must equal one. The Python function exposes no weights or
robust scale argument. Zero-weight or robust-weight derivatives are not
silently omitted. Source shapes may describe normalized surface geometry;
this API does not infer calibrated covariance.

For each declared pair, let

```text
r = R source + t - target
S = R source_shape Rᵀ
C = target_shape + S
e = C⁻¹r
F = 0.5 sum(rᵀ C⁻¹r)
```

The original frozen residual gradient is `-right_hand_side`. Differentiating
the transported source shape adds
`metric_gradient_rotation = sum(e × (S e))` to its three rotation entries.
`objective_gradient` includes that term, and `objective_right_hand_side` is
its negative. The transport, Cholesky solves and input-order sums use float64;
no eigenvalue floor, pose repair, damping or fallback is introduced.

Every original output field, including `normal_matrix`, `right_hand_side`,
cost, transported shapes and per-row residuals, retains the original frozen
definition and bits. The returned normal matrix **is not the exact Hessian of
F**. A caller cannot assume that solving the old right-hand side optimizes the
new score. Serial/parallel derivative results and first-error order agree.
Arrays own their storage. A new derivative overflow is an explicit error and
does not change the behavior of the original API.

The score omits a log determinant and weight derivatives. It is a declared
unit-weight geometric residual score, not a Gaussian likelihood. No nearest
search, optimizer or state update occurs in this API.

## Update experiments and adoption decision

An independent NumPy/SciPy study tested two policies on all 72 frozen real
histories: 36 fixed problems with two shape variants. Both use the exact
transported gradient, the original SPD residual normal matrix as a
preconditioner, the same 30-iteration budget, `1e-4` / `1e-5` full-direction
tolerances and six trial scales. Acceptance requires strict decrease and an
Armijo coefficient of `1e-4`. Shortened steps cannot declare convergence.

The first policy rematches target points/shapes at each trial and rejects any
matched-source support change. It stops at `no_descent` in 34 histories that
previously converged. The second holds declared source/target IDs and target
shapes fixed during inner trials while retransporting source shapes; it
rematches at the next outer iteration. Six previously limited histories
converge, while six previously converged histories reach the limit. It also
uses more accepted steps overall. Its outer refreshed costs need not decrease.

Neither policy is added to native registration or the frontend. A lower shape
score and a changed stopping label do not establish better physical alignment.
The [evidence packet](../benchmarks/slam/flatwall-gicp-transported/README.md)
reports every outcome, including regressions. The separate reference
controllers are audit experiments rather than application algorithms.
