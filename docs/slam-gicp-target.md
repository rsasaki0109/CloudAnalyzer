# Reusing a GICP target

Python `GicpTarget` and Rust `gicp_registration::PreparedTarget` own copied
world-frame target points, explicit SPD shapes and an exact f64 k-d tree.
Reuse an instance while the target remains fixed to avoid target array copies,
whole-target validation and tree construction on every registration.

```python
from cloudanalyzer_core import GicpTarget

target = GicpTarget(target_world, target_shapes_world)
for source_sensor, source_shapes_sensor, prediction, center_world in scans:
    result = target.register(
        source_sensor, source_shapes_sensor, prediction, center_world,
        max_correspondence_distance=2.0, capture_history=True,
    )
```

Points are float64 `(M,3)` and shapes float64 `(M,3,3)`. Every target row must
be finite and its shape symmetric and strictly positive definite under the
[existing factor checks](slam-gicp-linearization.md), even if never matched.
The constructor checks row count, then rows in input order, then tree index
capacity. Invalid input raises `ValueError` with the first failing target row.
An empty target is valid. `len(target)` reports the original target row count.

Original coefficients are stored unchanged, preserving the exact shape
canonicalization in the factor kernel. Construction does not estimate shapes,
drop unsupported rows, insert fallback shapes or reorder public target IDs.
When using [gicp_shapes](slam-gicp-shapes.md), the caller explicitly selects
supported rows before construction and retains any required ID mapping.

Constructor arrays may be changed or deleted after construction. The object
exposes no mutable buffers or map update operation; create a new object when
the map changes. It retains both original-order points/shapes and tree-owned
point/index/node storage, so holding several targets increases memory use.

`register` takes source-frame float64 `(N,3)` points and `(N,3,3)` SPD shapes,
a proper homogeneous source-to-world `(4,4)` initial pose and finite `(3,)`
fixed world center. It copies source inputs before releasing the GIL. Every
call validates the pose/parameters and all source rows. Constructor target
errors occur separately, before any per-call source validation. The standalone
`gicp_register` retains its original common/source/target validation order.

Results, matching ties, strict threshold, frozen metric trials, arithmetic
order, statuses, default serial mode and optional parallel/history behavior
are the same as [gicp_register](slam-gicp-registration.md). Both APIs share the
same registration loop. No mutable scratch is kept in the target: Rust targets
are `Send + Sync` and the frozen Python object permits shared read-only calls
from threads while computation releases the GIL. Each returned result and
history array owns its storage. Altering one result cannot affect later calls.

Target caching only removes preparation at the registration API boundary.
Matched factors still validate/transport shapes, and correspondences and
weights are recomputed at each outer iteration. This opt-in API does not
change the online frontend, initialization acceptance, solver or convergence
policy. A fixed map is the caller's declared frame and lifetime; no physical
accuracy or production speed follows from reuse alone.
