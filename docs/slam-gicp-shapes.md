# Native GICP surface shapes

Rust `gicp_shapes::estimate` and Python `gicp_shapes` prepare normalized f64
surface shapes for the [iterative GICP API](slam-gicp-registration.md).
This is an opt-in preparation API. Existing normals, ICP, odometry and
initialization defaults retain their behavior.

```python
import numpy as np
from cloudanalyzer_core import gicp_shapes, gicp_register

s = gicp_shapes(source)
t = gicp_shapes(target)
si, ti = s["supported"], t["supported"]
result = gicp_register(
    source[si], target[ti], s["shapes"][si], t["shapes"][ti],
    initial_pose, center_world, max_correspondence_distance=2.0,
)
# Registration indices refer to the filtered inputs.
original_source_ids = np.flatnonzero(si)[result["matched_source_indices"]]
```

Inputs are float64 `(N,3)` points; shapes use the same coordinate frame as
those points. Source shapes therefore use source axes, target-map shapes use
world axes. The GICP registration API handles source-to-world transport.
Inputs are copied before releasing the GIL. Every output array owns its data.

## Neighborhoods and algebra

`neighbors=20` is the default and must be at least five. Each query is an input
point; all input points, including that point, are eligible neighbors. The
exact f64 k-d tree selects `min(neighbors,N)` points in distance order, using
its existing deterministic traversal on equal distances. A minimum original
index tie policy is not promised; duplicate coordinates can select another
eligible point instead of the query's own index.

For each neighborhood the implementation sums query-relative offsets in
neighbor order, divides once by count, then forms centered population scatter
in that same order. It constructs the lower triangle and reflects it, scales
by the maximum absolute entry for a f64 Jacobi eigensolve, rescales eigenvalues
and sorts them by value with an axis-index tie break. The least-spread vector
is normalized in f64; its sign cancels in
`shape=I+(.001-1)*n*n.T`. Supported shapes have normalized eigenvalues
`[.001,1,1]`. They describe surface geometry, not calibrated sensor covariance.

Support requires at least five neighbors, finite scatter eigenvalues,
`largest>0` and `middle>1e-12*largest`. Unsupported rows have a zero shape and
`supported=False`. Filter them explicitly before registration. No identity
fallback, planarity selection or additional eigengap rejection is applied.
For repeated minimum eigenvalues the selected normal is an arbitrary
deterministic direction in the minimum eigenspace. A single matrix comparison
across independent solvers is then inappropriate; eigenpair residuals and the
normalized shape spectrum provide the numerical controls.

## Results and validation

| Key | Shape / meaning |
| --- | --- |
| `shapes` | `(N,3,3)` float64; zero when unsupported. |
| `supported` | `(N,)` bool, with explicit input index identity. |
| `scatter_eigenvalues` | `(N,3)` signed ascending float64 population-scatter values. |
| `neighbor_counts` | `(N,)` uint64, actual clipped neighborhood size. |
| `neighbor_indices` | `(N,K)` uint64 with `capture_neighbors=True`; otherwise `(N,0)`. |
| `neighbor_squared_distances` | `(N,K)` float64 when captured; otherwise `(N,0)`. |

Here `K=min(neighbors,N)`. Fewer than five neighbors produce zero eigenvalues
and unsupported shapes. Empty input is valid. Nonfinite input points or
unrepresentable distance/mean/scatter/eigensystem arithmetic raise an indexed
error. Input validation and runtime failures identify the first affected row
in source order. No numeric repair or automatic point removal occurs.

`parallel=False` is the default. With the core `parallel` feature, independent
64-query chunks reuse per-chunk search scratch. A serial input-order collection
preserves every output bit and first error relative to serial execution.
`capture_neighbors=False` avoids retaining full neighborhood arrays.

The [real-recording audit](../benchmarks/slam/flatwall-gicp-shapes/README.md)
checks exact neighbor distance ranks against exhaustive search, PCA against
NumPy and registration against an independent iterative solver. The original
caller-shape registration outputs remain bit-identical on the new binary.
Native shapes introduce their own arithmetic and neighbor tie policy; old
SciPy shapes and resulting poses are compared separately. No new trajectory,
physical accuracy, initializer acceptance, GPU or whole-SLAM speed claim follows
from these numerical checks. Prepared target caching and online integration
remain separate work.
