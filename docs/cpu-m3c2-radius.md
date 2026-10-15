# M3C2 radius pruning

M3C2 can combine the lower bounds from ancestral k-d tree split planes
when collecting neighbours. This reduces unnecessary radius traversal in
the normal fit and both cylinder searches, in the serial path and each
CPU/WASM worker. It does not retain extra tree data or copy additional
coordinates. The existing core partitioning, worker limit and conservative
input-copy budget remain in effect.

This is an experimental comparison option, disabled by default. The measured
runs below did not show a consistent end-to-end improvement. The separate
[resident analysis lane](cpu-m3c2-main-worker.md) supplies the adopted CPU
parallelization change.

## Arithmetic and traversal

The original radius search visits the near child first, and visits the far
child when the squared distance to that split plane is within the squared
radius. It tests one plane at a time. A far child can consequently satisfy
each individual plane test while its entire region lies outside the sphere.

The new traversal carries three squared lower bounds, initially zero.
For a far child, it keeps the larger of the existing bound and the current
split-plane squared distance on that axis. If `lower_x + lower_y + lower_z`
exceeds the squared radius, the child has no accepted points and is skipped.
Near children inherit the existing bounds. The original individual plane
test is retained. There are no stored bounding boxes or new per-tree arrays.

Each finite nonnegative term is a lower bound on the corresponding squared
coordinate difference of every point in the subtree. The sum uses the same
left-associated x/y/z order as the scalar leaf distance. The comparison is
strictly greater than the radius squared, so equality stays eligible.
Accepted leaf points and near/far traversal retain their original order;
M3C2 sums, PCA normals, cylinder selection, variance, LoD95 and significance
use the existing arithmetic. Native tests include underflow, overflow,
signed zero, nonfinite inputs and radii at adjacent representable boundary
values, and compare unsorted neighbour indices as well as result bits.

`KdTree::within` preserves its existing single-plane behavior for other
algorithms. M3C2 selects `within_with_bounds` through its computation state.
The native `m3c2_with_radius_bounds` function and optional final WASM
arguments to `m3c2Distances` and `computeM3c2` retain the old traversal for
comparison. The internal production request field `radiusBounds: false`
uses that control in both serial and parallel execution. Returned
`M3c2Output.radiusBounds` records the selected mode. Omitted controls use
the original traversal. No UI setting is needed.

## Measured inputs and timing

`web/scripts/benchmark-m3c2-radius.mjs` requires two input files; it has no
synthetic fixture mode. The saved runs use the existing measured-source
PandaSet, Autoware planning and Tokyo Takanawadai pairs. Each pair contains
unchanged XYZ coordinates at alternating original source indices. They are
disjoint subsets of one measured cloud, with possible geometric duplicates,
not independent surveys or change-detection ground truth. Embedded input
manifests contain source provenance, attribution, coordinates, point counts
and SHA-256 hashes.

The saved radius runs load both full inputs with `keepEvery: 1`. Core spacing selects
original measured points; it does not average or generate coordinates.
Each mode rebuilds its plan and both trees and computes fresh results.
Five warm-up pairs precede alternating sequential timed comparisons.
Timings include core selection, snapshots and input copying, tree builds,
f64 searches and accumulation, joining, attributes, result LOD indexing and
RPC delivery. Input loading, pool startup, parity checks and exports are
outside the timer. Complete PLY bytes and displayed distance, LoD95 and
significance bits must match on every warm-up and timed pair. Both modes
must use the same number of lanes and cores. Source, manifest, production
worker/WASM and benchmark-script hashes are checked before and after the run.
Reports refuse to overwrite an existing file.

The current script explicitly sets `mainWorker: false` to retain the old
pool-only scheduling policy. An optional `--max-points` budget can select a
smaller measured subset; reports record original and loaded counts and
stride. None of the saved radius runs use that option.

The automatic scheduler may select one worker for large inputs under the
existing copy budget. The radius pruning applies in that case too; a
one-worker result is not evidence of a parallel scheduling speedup. Other
applications, CPU power/clocks and garbage collection are uncontrolled.

## Results and decision

All runs use normal radius 0.4 m, projection radius 0.25 m, half-depth 0.5 m
and core spacing 0.25 m. Each row has five warm-up pairs and ten timed pairs.
All use one CPU worker under the original input-copy budget, including the
PandaSet automatic case. The explicit serial case is a second scheduling
control, not a distinct parallel configuration.

| Input / scheduling | Selected cores | Plane control median, ms | Combined bounds median, ms | Median paired reduction | Faster pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| [PandaSet / automatic](../benchmarks/cpu/m3c2/real-source/windows-pandaset019-radius.json) | 212,541 | 1,447.00 | 1,364.65 | -2.54% | 4/10 |
| [PandaSet / serial](../benchmarks/cpu/m3c2/real-source/windows-pandaset019-radius.json) | 212,541 | 1,500.65 | 1,492.35 | -0.47% | 5/10 |
| [Autoware / automatic](../benchmarks/cpu/m3c2/real-source/windows-autoware-planning-radius.json) | 696,739 | 2,534.40 | 2,515.70 | -2.62% | 4/10 |
| [Tokyo / automatic](../benchmarks/cpu/m3c2/real-source/windows-tokyo-takanawadai-radius.json) | 299,755 | 2,395.70 | 2,478.65 | -2.45% | 3/10 |

Paired reduction is the median of `100 * (1 - bounded_i / planes_i)`,
not the ratio of independently calculated medians. Positive values mean
less elapsed time. All paired medians are negative, and bounded traversal
is faster in only 16 of 40 timed pairs. The 40 timed and 20 warm-up pairs
all preserve displayed result bits and complete PLY bytes exactly. Reducing
search visits does not guarantee a shorter complete request; carrying
the bounds adds traversal work, and plan/tree/result construction is timed.
No point-count heuristic or default activation follows from these results.

These historical measurements precede the resident analysis lane. Their
common production artifacts are:

| Artifact | SHA-256 |
| --- | --- |
| `worker-BhWg_9vN.js` | `406df497ee85ddf75ea7abded873b30fbce01fbc488d1da1a4ab5c7ae821bfc9` |
| `c2c-worker-3-QwxsA6.js` | `3cb74a6bec56c3fe4490c8bf1f71dc9a122cb5bafa864210d8ff51057a5fab74` |
| `ca_wasm_bg-j5QX2KYG.wasm` | `5d81d7c2c982be0ed1e969780b65bcf330070aab03f292c26653cf34518e9ef2` |

The initial PandaSet script hash is
`b71c4dea9adb895fab765317ab6452cf393129b4e8113d421d890df255289766`;
Autoware and Tokyo use
`31e570c0ae9042d6ab11f6bf115cb95b0a9a0e7045e2beff6ab9c300042aa1ec`.
The latter added optional measured-input stride selection and explicit
loaded-count validation, while using the same full-input default. The
current script also fixes the pool-only control explicitly. Saved reports
retain the hashes of the code actually measured.

```powershell
cd web
node scripts/benchmark-m3c2-radius.mjs --reference ../notes/c2c-real-pandaset/reference.ply --compared ../notes/c2c-real-pandaset/compared.ply --input-manifest ../notes/c2c-real-pandaset/inputs.json --core-spacing 0.25 --scheduling both --repeats 10 --output ../benchmarks/cpu/m3c2/real-source/windows-pandaset019-radius-repeat.json
```
