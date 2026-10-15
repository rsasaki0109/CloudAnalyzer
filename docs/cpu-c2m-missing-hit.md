# C2M missing-hit recovery

C2M now returns positive infinity for a query with no computable triangle hit
and continues the batch. It resets the guessed triangle before the next query
and skips signing a missing hit. This applies to fresh and prepared CPU
distances, candidate-hinted distances, experimental query-order reuse and
native Rayon C2M. A later call can reuse and free the same prepared WASM mesh.

## Failure and correction

BVH searches initialize `MeshHit.triangle` to `usize::MAX` and its squared
distance to positive infinity. NaN/infinite query coordinates, squared-distance
overflow from finite coordinates, or uncomputable mesh geometry can leave
that initial hit unchanged. The preceding batch code stored this sentinel as
the next query's guess, then indexed a triangle with it. Signed calls could
also index the sentinel while obtaining the closest triangle's normal.
Native code panicked; a WASM trap could leave the exported mesh borrowed and
make subsequent use/free fail.

The correction checks the existing missing-hit sentinel after the search.
It writes positive infinity, clears the guess and continues without indexing
the mesh or computing a sign. Single unsigned missing hits already returned
positive infinity; signed and mixed batches now follow that result convention.
Infinity means no computable hit, rather than a measured geometric distance
or an assigned side of a triangle.

Queries that find a triangle retain all preceding distance arithmetic,
Morton order, triangle visitation, strict insertion, sign handling and guess
policy. No coordinate filtering, mesh repair or triangle-index validation
policy is added. Explicit invalid guesses passed directly to low-level BVH
APIs are outside this batch fix. Empty meshes still return the preceding
error/`None`; malformed WASM flat arrays retain their existing errors.

The [cached traversal bounds](cpu-c2m-stack-bounds.md) remain enabled.
The [query-order](cpu-c2m-query-order.md) and
[borrowed-query](cpu-c2m-query-borrow.md) experiments remain disabled by
default. Geometry storage, memory allowances and worker scheduling are
unchanged. The correction adds no GPU dispatch or speedup claim.

## Validation

A new regression failed on the preceding implementation with a triangle
index of `usize::MAX`, then passed after the fix. Native cases cover NaN
payloads and both infinities on every coordinate axis, intervening positive
and negative finite queries, reversed batches, finite squared-distance
overflow, all-NaN mesh geometry, signed/unsigned calls, repeated cache hits
and recovery on the same prepared mesh. A 9,000-point batch crosses several
Rayon chunks. Analytic plane distances provide exact expected f64 bits;
missing hits must be positive infinity. Retained mesh bytes stay unchanged.

397 native tests pass with the parallel feature, along with native/WASM
Clippy denying warnings, formatting and release WASM/TypeScript/Vite builds.
Thirteen browser tests pass; three opt-in physical-GPU scenarios skip.
The new WASM regression exercises fresh, prepared, hinted and cached-order
paths, preserves caller coordinate bits, checks signed missing single hits
and mixed batches, recovers finite results on subsequent calls, and frees
all prepared objects. Independent archived WASM oracles continue to verify
preceding finite f64 results/signs across 240 geometry/count/scale/sign
conditions, repeated batches and earlier traversal/order-cache cases.
Production worker tests also cover overlapping requests, source changes,
obsolete responses and failed-worker recovery. Generated fixtures are for
correctness; no new performance measurement is reported.

- [Native regressions](../rust/crates/ca-core/tests/mesh_preparation.rs)
- [WASM regression](../web/e2e/mesh-missing-hit.spec.ts)
- [Batch distances](../rust/crates/ca-core/src/distance.rs)
- [Hinted distances](../rust/crates/ca-core/src/mesh.rs)
