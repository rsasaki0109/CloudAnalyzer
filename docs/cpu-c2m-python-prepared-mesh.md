# Python C2M with a prepared mesh

`cloudanalyzer_core.PreparedMesh` exposes the existing retained parallel Rust
query path to Python. Repeated unsigned queries on the same complete measured
TIN shorten by 11.77–48.96% in these six comparisons. All six paired medians
improve, with 58/60 faster timed pairs. This is an opt-in API with a one-time
constructor and persistent native storage.

```python
import cloudanalyzer_core as cc

vertices, triangles = cc.read_mesh("design.ply")
mesh = cc.PreparedMesh(vertices, triangles)
first = mesh.distances(cc.read("scan.ply")["positions"])  # signed=True
next_batch = mesh.distances(other_points, signed=False)
native_capacity_bytes = mesh.retained_bytes
```

Construction owns copies of float64 vertices and uint32 faces, validates faces
and builds the existing BVH. Original array mutation/deletion does not affect
the object. Empty geometry or faces with no valid indices raise `ValueError`;
shape and dtype errors follow the existing NumPy binding contract. Strided
arrays are accepted. Each query copies its input points and returns its own
one-dimensional float64 NumPy array in input order. Native construction and
query work release the GIL; Rayon runs the query kernel in parallel. Multiple
Python threads may share the immutable object. Queries keep independent
order/guesses/scratch; changing a batch or the signed setting retains no query
state. Recreate the object when geometry changes. Releasing its last reference
releases the owned native vectors. No global mesh cache is added.

The class uses the same adopted native preparation/build policies, copied
corners, clamped sibling-bound SIMD and scalar triangle projection as
`cloud_to_mesh`. Precision, signs, strict ties and missing-hit behavior remain
the existing f64 behavior. The existing stateless function remains available.

## Measurements and scope

[Real-source preparation](c2c-real-source.md) supplies unchanged measured f64
coordinates and original source-index subsets. Geometry is the complete
production Delaunay Auto TIN archived by the
[native retained comparison](cpu-c2m-native-retained-par.md). Geometry, query
PLY, manifests and original sources are fingerprinted before and after every
run. No new thinning, generated coordinates, noise, displacement or meshing
variant is used. These are subsets of one measured source, not independent
surveys or accuracy ground truth. Tokyo keeps its preceding original-point
selection. Small generated fixtures are used only in correctness tests.

| Source | Query points | TIN vertices | TIN triangles |
| --- | ---: | ---: | ---: |
| PandaSet 019 | 531,908 | 501,097 | 844,324 |
| Autoware planning | 878,920 | 873,771 | 1,688,821 |
| Tokyo Takanawadai | 941,933 | 893,804 | 1,466,179 |

Windows x86_64, 12 logical CPUs, Python 3.12.10 and NumPy 2.5.4 use a frozen
release abi3 wheel built with maturin 1.15.0, Rust 1.98.1, optimization level 3,
LTO and one codegen unit. The wheel is installed only in an isolated workspace
venv. The native core/configuration fingerprints match the preceding retained
comparison; the binding adds the class. `RAYON_NUM_THREADS=12` is set before
loading the extension. Fresh and retained calls alternate in the same process,
one call at a time. Each source/pass uses a new process and prepared object.
An excluded first-use pair and five warm-up pairs precede ten timed pairs.
Passes run PandaSet → Autoware → Tokyo, then reverse the source order.

Clocks cover the complete Python API call: NumPy-to-Rust copies, native work,
query-local cleanup and NumPy result creation. Fresh calls also copy/validate
the geometry and build/drop its BVH. Retained queries reuse owned geometry/BVH.
Loading, TIN construction, constructor setup, signed correctness queries,
statistics, serialization, hashing and final object destruction are excluded
from repeated-call timings. No builds, tests, scans or edits run alongside them.

| Source/pass | Fresh median, ms | Prepared median, ms | Median paired shortening | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| PandaSet initial | 382.45 | 244.58 | +36.91% | 10/10 |
| PandaSet final | 463.48 | 315.48 | +31.29% | 9/10 |
| Autoware initial | 678.06 | 420.06 | +37.66% | 10/10 |
| Autoware final | 1,289.12 | 726.07 | +48.96% | 10/10 |
| Tokyo initial | 1,766.00 | 1,418.75 | +15.18% | 9/10 |
| Tokyo final | 1,615.96 | 1,380.03 | +11.77% | 10/10 |

Shortening is `median(100 * (1 - prepared_ms / fresh_ms))`. All observations
remain, including the slower prepared pair for final PandaSet (520.60 / 538.69
ms) and initial Tokyo (1,895.20 / 1,950.56 ms). Absolute times vary substantially,
especially final Autoware; scheduling, clocks, power, other applications,
allocator/page retention and cache effects are uncontrolled. These are measured
whole-call effects, not isolated BVH savings or a guarantee for another workload.
Native Rust results and these Python results have different conversion/lifetime
costs; their percentages are not interchangeable or additive. Signed speed is
not measured here. Browser/WASM/GPU dispatch and Python CLI selection stay
unchanged; GPU alternatives should include this opt-in CPU baseline for retained
references and account for setup, transfers, verification and lifetime.

## Setup and retained storage

| Source/pass | Constructor, ms | Fresh first call, ms | Constructor + prepared first call, ms | Estimated calls to amortize setup |
| --- | ---: | ---: | ---: | ---: |
| PandaSet initial | 127.37 | 388.94 | 383.53 | 1 |
| PandaSet final | 139.35 | 541.85 | 529.78 | 1 |
| Autoware initial | 238.56 | 692.60 | 679.52 | 1 |
| Autoware final | 297.07 | 760.95 | 710.36 | 1 |
| Tokyo initial | 201.90 | 1,654.35 | 1,532.86 | 1 |
| Tokyo final | 188.47 | 1,568.86 | 1,531.89 | 2 |

Each setup is one observation including NumPy geometry copies, validation and
BVH construction. The fresh first call precedes construction, then the prepared
first call follows it. These are warmed-process first observations, not a cold
process benchmark. The estimate is `ceil(setup_ms / median(fresh_ms - prepared_ms))`,
using repeated-call pairs; it can differ from observed first totals and is not
a promised break-even point. Final object destruction remains outside both the
estimate and the repeated prepared-query clock.

| Source | Native object payload capacity, bytes | Caller geometry arrays, bytes |
| --- | ---: | ---: |
| PandaSet | 133,609,096 | 22,158,216 |
| Autoware | 264,160,784 | 41,236,356 |
| Tokyo | 232,581,128 | 39,045,444 |

`retained_bytes` counts owned native geometry plus BVH node/order/corner vector
capacities. Python construction copies caller geometry, so the native object
adds this entire persistent payload while caller arrays remain alive. Dropping
caller geometry arrays after preparation can release their separate buffers.
This differs from the native Rust benchmark, which moves existing geometry
vectors into the object. The byte count excludes headers, allocator overhead,
pages, query/result arrays and temporary buffers; it is not process RSS or a
memory cap. The prepared object keeps the same reported capacity across queries.

## Correctness and reproduction

All 192 unsigned outputs (first-use, warm-up and timed calls) match the preceding
independent native oracle's full f64 bytes and canonical binary scalar PLY bytes.
Sequential f64 reductions reproduce all six oracle statistics; identical raw
distances establish the same statistics for subsequent calls. PLY serialization
is reconstructed in the benchmark outside clocks using the native writer's
XYZ-double/scalar-float layout; this does not add a Python production writer.
Twelve additional full signed outputs agree between fresh and prepared calls
in f64 bits, all six statistics and canonical scalar PLY bytes. Signed checks
are internal parity, while the preceding independent oracle is unsigned.

Twenty-nine Python regressions cover empty/boundary/Morton-sized batches, changed
queries, signed defaults, strided and read-only arrays, owned input copies,
independent results, invalid faces/shapes/dtypes, large f64 coordinates, reversed
face ties/degenerate faces, missing/overflow hits and concurrent readers. Release
wheel tests and ca-py Clippy with warnings denied pass. Core behavior is already
covered by the preceding native regressions and is unchanged in this step.

[Benchmark script](../rust/crates/ca-py/scripts/benchmark_prepared_mesh.py) accepts
`--geometry`, `--compared`, `--input-manifest`, `--native-report`, `--oracle-f64`,
`--source-root` and a new `--output` path, plus `--threads 12 --repeats 10`.
Use the full native geometry and unsigned oracle corresponding to the supplied
report in a release-wheel environment. The script checks installed Python
sources, source/configuration and extension fingerprints, original measured
source provenance, every result and immutable capacity. Existing output paths,
duplicate/unknown options, invalid thread/repeat counts and mismatched inputs
fail. Reports preserve full raw timings and metadata:

- PandaSet: [initial](../benchmarks/cpu/c2m/real-source/windows-pandaset019-python-prepared-mesh-initial.json), [final](../benchmarks/cpu/c2m/real-source/windows-pandaset019-python-prepared-mesh-final.json)
- Autoware: [initial](../benchmarks/cpu/c2m/real-source/windows-autoware-planning-python-prepared-mesh-initial.json), [final](../benchmarks/cpu/c2m/real-source/windows-autoware-planning-python-prepared-mesh-final.json)
- Tokyo: [initial](../benchmarks/cpu/c2m/real-source/windows-tokyo-takanawadai-python-prepared-mesh-initial.json), [final](../benchmarks/cpu/c2m/real-source/windows-tokyo-takanawadai-python-prepared-mesh-final.json)
