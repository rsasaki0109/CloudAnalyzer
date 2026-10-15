# Compact midpoint point trees for resident MapEval

Large resident map pairs now use compact midpoint SciPy trees for exact NN
queries. Complete resident evaluation improves by 7.92–17.04% across six
real-source comparisons, with 27/30 faster timed pairs, against the preceding
[SCS-batch implementation](cpu-map-scs-batch.md). Both fresh point trees/queries,
threshold statistics, Gaussian construction, Wasserstein and SCS index
construction/destruction are included. Input loading, imports, parity tracing,
PLY writing and CLI startup/serialization are excluded.

The two-direction point-search component improves by 5.13–20.33%, with 29/30
faster pairs. Component and whole-evaluation percentages have different scopes
and are not additive. All complete distance arrays and evaluation outputs
match the preceding implementation by their bytes.

The later [bounded SCS queries](cpu-map-scs-query-batch.md) add a separate
component improvement. Reproduce this point-tree delta with its frozen
sources; current production includes both changes.

## Tree policy and scope

When both query and target maps have at least 65,536 points, point NN tree
construction passes `balanced_tree=False` to `cKDTree`. This selects midpoint
partitioning instead of the preceding median partitioning. Compact node bounds
remain enabled and leaf size remains 16. The experiment changes tree layout
and its construction cost; the distance metric and exact query are preserved.

Smaller or asymmetric map pairs retain preceding tree construction. The
existing worker policy is independent: query arrays of at least 65,536 points
use `workers=-1`, and smaller queries use one worker. Queries remain `k=1`
with the preceding exact Euclidean/default-epsilon parameters. Empty-query and
empty-reference behavior, non-finite input errors and caller-array ownership
are unchanged. The two directions still construct independent fresh trees.

No persistent reference cache or new worker pool is added. The supported-error
cell SCS tree uses its existing constructor. Structural moments, Gaussian
distance formulas, SCS order/reductions, protocols, thresholds and PLY writing
are unchanged by this step. Performance measurements cover resident evaluation;
the streaming structural API does not perform these point searches.

The 65,536-point cutoff conservatively limits the new policy to the measured
large-query/large-reference regime. It is not an optimum cutoff established
across all distributions. Tree layout can change native node counts and depth;
no peak-RSS measurement or memory improvement is claimed. Both policies retain
their normal tree/input/output allocations and release trees per invocation.

## Real-source comparison

[Source preparation and attribution](c2c-real-source.md) supply fixed original
XYZ coordinates without new thinning, displacement, noise or generated points.
Tokyo retains its preceding original-point selection. These are disjoint-index
subsets of one measured source with possible geometric duplicates, rather than
independent surveys or accuracy ground truth. Generated fixtures are used only
for correctness tests.

| Source | Estimated / reference points |
| --- | ---: |
| PandaSet 019 | 531,908 / 531,908 |
| Autoware planning | 878,920 / 878,921 |
| Tokyo Takanawadai | 941,933 / 941,933 |

Requests use 3 m structural voxels, minimum 100 points in both maps per
supported cell, neighbor radius 5 and thresholds `(0.2,0.1,0.08,0.05,0.01)` m.
Alignment and downsampling are disabled. Both versions include all preceding
MapEval optimizations; only the large-pair point-tree partitioning changes.

Windows, Python 3.12.10, NumPy 2.3.4, SciPy 1.17.1 and Open3D 0.19.0 use 12
logical CPUs, `OMP_NUM_THREADS=12` and `OPENBLAS_NUM_THREADS=1` before imports.
Production does not set those environment variables. Sixteen source/config/
test/controller fingerprints, the preceding baseline and compiled SciPy tree
are recorded with archived copies.

Each source/pass starts a fresh process. Full Gaussian/distance/PLY parity,
an excluded first timing pair and two warm-up pairs precede five alternating
timed pairs. Sources run PandaSet → Autoware → Tokyo, then reverse that order.
Tracing hooks are disabled during timing. No edits, builds, tests or scans run
during comparisons, and all observations remain.

| Source/pass | Median-partition policy median, ms | Midpoint policy median, ms | Median paired shortening | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| PandaSet initial | 1,075.77 | 938.09 | +7.92% | 5/5 |
| PandaSet final | 1,908.93 | 1,743.42 | +9.23% | 4/5 |
| Autoware initial | 1,956.11 | 1,800.18 | +9.42% | 5/5 |
| Autoware final | 2,299.57 | 1,907.64 | +17.04% | 4/5 |
| Tokyo initial | 1,940.53 | 1,602.07 | +9.06% | 4/5 |
| Tokyo final | 1,891.84 | 1,634.12 | +10.20% | 5/5 |

After complete timing and structural parity capture, the same process measures
just both point-search directions. These component clocks include fresh tree
construction, both exact queries, tree destruction and distance-output arrays.
Threshold statistics, Gaussian/Wasserstein/SCS work and checking/hashing are
excluded. The component has its own excluded first pair, two warm-ups and five
alternating timed pairs. Every returned array matches its full parity hash.

| Source/pass | Median-partition NN median, ms | Midpoint NN median, ms | Median paired shortening | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| PandaSet initial | 731.16 | 612.94 | +15.61% | 5/5 |
| PandaSet final | 936.67 | 877.47 | +5.13% | 4/5 |
| Autoware initial | 982.21 | 804.36 | +19.34% | 5/5 |
| Autoware final | 1,439.97 | 1,174.86 | +18.04% | 5/5 |
| Tokyo initial | 1,347.65 | 1,122.42 | +20.33% | 5/5 |
| Tokyo final | 1,154.93 | 907.52 | +16.33% | 5/5 |

Shortening is `median(100 * (1 - midpoint_ms / balanced_ms))`, rather than
the ratio of marginal medians. All six whole medians improve, but three
individual whole pairs and one point-search pair regress. Scheduling, power,
other applications, allocator state and warmed pages/caches are uncontrolled;
passes vary materially. No causal diagnosis of that variation is established.
Results describe these measured map pairs and do not establish a universal
tree-layout improvement. No full CLI/cold-file, streaming, browser/WASM or GPU
speed is claimed.

## Validation and reproduction

All 108 complete parity/first/warm-up/timed reports match. All 24 full f64 NN
distance arrays, 24 complete PLY outputs and 24 Gaussian maps match bytes,
including 192,880 saved voxel records. Twelve full Gaussian-distance arrays
contain 28,464 matching records, and twelve full SCS-term arrays contain
28,436 matching records. Reports/distances/PLYs and Gaussian-distance hashes
also match independently recorded preceding SCS-batch results.

The independent audit recomputes twelve serial, median-partition point queries,
six structural evaluations, Gaussian summaries and six Gaussian-distance
sequences. It verifies 14,232 ordered neighborhoods against integer cube
enumeration, derives every scalar SCS term and checks every PLY coordinate/RGB
value. Source/library fingerprints, complete report hashes, SCS metadata, pair
ordering and all timing summaries are checked. Twelve negative controls reject
wrong sources/baselines, invalid counts/voxel sizes, duplicate arguments and
existing destinations without replacing data.

388 related Python tests pass. The 29 new cases cover exact distance bits with
contiguous/strided/reversed inputs and tiny/large magnitudes, duplicate hits
and ties, input/result ownership, the 65,535/65,536/65,537-point dispatch boundary
for both maps, empty/asymmetric maps, existing worker selection, duplicate/
line/plane/anisotropic/uneven distributions, non-finite error propagation,
complete report equality and PLY bytes. The existing Windows Unix `resource`
upstream-harness import limitation is unchanged; that separate harness is
excluded.

The [controller](../cloudanalyzer/scripts/benchmark_map_nn_midpoint.py) requires
`--source-root`, `--reference`, `--compared`, `--input-manifest`, `--baseline`,
`--baseline-sha256` and new `--work-dir`/`--output` destinations. Defaults are
`--repeats 5 --structure-voxel-size 3`; all six records use them. Full timings,
reports, input/source/library fingerprints and parity paths remain in the
records. Gaussian files use `<4q12d` (128 bytes), Gaussian distances `<3qd`
(32 bytes), and SCS terms `<3qQd` (40 bytes). Reproduce preceding deltas with
their respective frozen sources.

- PandaSet: [initial](../benchmarks/cpu/map-evaluate/real-source/windows-pandaset019-nn-midpoint-initial.json), [final](../benchmarks/cpu/map-evaluate/real-source/windows-pandaset019-nn-midpoint-final.json)
- Autoware: [initial](../benchmarks/cpu/map-evaluate/real-source/windows-autoware-planning-nn-midpoint-initial.json), [final](../benchmarks/cpu/map-evaluate/real-source/windows-autoware-planning-nn-midpoint-final.json)
- Tokyo: [initial](../benchmarks/cpu/map-evaluate/real-source/windows-tokyo-takanawadai-nn-midpoint-initial.json), [final](../benchmarks/cpu/map-evaluate/real-source/windows-tokyo-takanawadai-nn-midpoint-final.json)
