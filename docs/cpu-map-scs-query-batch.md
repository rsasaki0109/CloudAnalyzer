# Bounded vector queries for SCS

MapEval now groups eligible SCS spatial queries into bounded batches. Complete
SCS work, including fresh index construction, queries and statistics, is
14.47–26.73% shorter across six real-source comparisons, with 30/30 faster
timed pairs, against the preceding [midpoint point-tree implementation](cpu-map-nn-midpoint.md).
All ordered neighborhoods, per-center terms and complete evaluation outputs
match the preceding implementation by their bytes.

Whole resident evaluation is mixed: paired medians range from −5.08% to
+9.51%, with 18/30 faster pairs. Autoware improves in both passes (+3.50%,
+9.51%); PandaSet regresses in both passes and Tokyo has one improving and one
regressing pass. The SCS component improvement is adopted. These results do
not establish a consistent whole-evaluation gain across all three sources.

## Query policy and arithmetic

The integer spatial index remains unchanged. It sorts voxel keys
lexicographically and builds the tree with exactly represented local integer
coordinates. Query batching applies only to eligible indexes with at least
1,024 supported error cells. For `V` indexed cells, its calculated capacity is
`min(256, floor(65,536 / V))`. Capacities below two retain single-center calls.
Thus the current vector-query regime is `1,024 <= V <= 32,768`; a final partial
batch may contain one center. Unsupported index cases retain their existing
lookup, including radius 1, nonstandard key/radius types and unsafe spans.

Each center can return at most `V` indices, including itself. The capacity
therefore bounds returned indices before the vector query, even for dense
neighborhoods. Queries request one worker, the same exact infinity-norm radius
search and sorted slots. Existing point-NN parallel workers are unchanged.
Small-query-batch thread setup was unhelpful in the exploratory comparison;
the selected path introduces no new worker pool.

Centers are consumed in dictionary insertion order with bounded `islice`
blocks. Each result retains the preceding lexicographic neighbor order and
excludes its own center. Batch query/result arrays are released before the
next vector query. The Gaussian distances, SCS statistics, zero-error rule,
final `nanmean`, support counts, protocols and artifacts remain unchanged.
No persistent cache or all-center neighbor table is added.

| Source | Supported error cells | Calculated query capacity | Worst-case returned indices per full batch | Coordinate payload per full batch, bytes |
| --- | ---: | ---: | ---: | ---: |
| PandaSet 019 | 1,202 | 54 | 64,908 | 1,296 |
| Autoware planning | 3,948 | 16 | 63,168 | 384 |
| Tokyo Takanawadai | 1,966 | 33 | 64,878 | 792 |

The index-count bound constrains vector-query results. Single-center fallback
can retain one larger neighborhood, as in the preceding implementation.
Python integer/list objects, the result object array, returned value lists,
existing statistic windows, index storage and allocator state add memory;
query results can coexist with statistic windows. The figures above are
payload/count bounds, not measured peak RSS or a total memory cap.

The helper is shared by resident and streaming structural evaluation. Tests
cover both; the following timings cover resident evaluation and its SCS
component, without establishing a streaming-speed result.

## Real-source comparison

[Source preparation and attribution](c2c-real-source.md) supply fixed original
XYZ coordinates without new thinning, displacement, noise or generated points.
Tokyo retains its preceding original-point selection. The pairs are
disjoint-index subsets of one measured source with possible geometric
duplicates, rather than independent surveys or accuracy ground truth.
Generated fixtures are used only for correctness tests.

| Source | Estimated / reference points | Nonempty SCS centers |
| --- | ---: | ---: |
| PandaSet 019 | 531,908 / 531,908 | 1,200 |
| Autoware planning | 878,920 / 878,921 | 3,943 |
| Tokyo Takanawadai | 941,933 / 941,933 | 1,966 |

Requests use 3 m structural voxels, minimum 100 points in each map per
supported cell, neighbor radius 5 and thresholds `(0.2,0.1,0.08,0.05,0.01)` m.
Alignment and downsampling are disabled. Both versions include all preceding
MapEval optimizations; only SCS query dispatch changes.

Windows, Python 3.12.10, NumPy 2.3.4, SciPy 1.17.1 and Open3D 0.19.0 use 12
logical CPUs, `OMP_NUM_THREADS=12` and `OPENBLAS_NUM_THREADS=1` before imports.
Production does not set those variables. Seventeen source/config/test/controller
fingerprints, the preceding baseline and compiled SciPy tree are recorded
with archived copies.

Each source/pass starts a fresh process. Full Gaussian/distance/PLY parity,
an excluded first timing pair and two warm-up pairs precede five alternating
timed pairs. Sources run PandaSet → Autoware → Tokyo, then reverse that order.
Tracing hooks are disabled during timing. No edits, builds, tests or scans run
during comparisons, and all observations remain.

Complete clocks include fresh point NN trees/queries, threshold statistics,
voxel grouping, means/covariances, Wasserstein and SCS index construction/
destruction. Loading, imports, hashing, parity tracing, PLY writing and CLI
startup/serialization are excluded.

| Source/pass | Single-query policy median, ms | Batch-query policy median, ms | Median paired shortening | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| PandaSet initial | 1,473.09 | 1,474.42 | −5.08% | 2/5 |
| PandaSet final | 2,244.07 | 2,336.56 | −1.82% | 2/5 |
| Autoware initial | 2,773.82 | 2,712.93 | +3.50% | 3/5 |
| Autoware final | 2,390.16 | 2,236.82 | +9.51% | 5/5 |
| Tokyo initial | 2,085.44 | 1,949.79 | +6.88% | 4/5 |
| Tokyo final | 2,102.30 | 2,044.67 | −1.06% | 2/5 |

After complete timing and structural parity capture, the same process measures
SCS with the full Gaussian-distance map. Component clocks include fresh SCS
index construction/queries, statistic guards/reductions, ordered term-array
creation and final `nanmean`. Gaussian/distance setup, file loading and output
checking/hashing are excluded. This has its own excluded first pair, two
warm-ups and five alternating timed pairs.

| Source/pass | Single-query SCS median, ms | Batch-query SCS median, ms | Median paired shortening | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| PandaSet initial | 147.84 | 121.42 | +14.47% | 5/5 |
| PandaSet final | 296.01 | 246.44 | +15.39% | 5/5 |
| Autoware initial | 409.46 | 309.00 | +20.90% | 5/5 |
| Autoware final | 429.32 | 304.92 | +26.73% | 5/5 |
| Tokyo initial | 308.61 | 263.89 | +21.27% | 5/5 |
| Tokyo final | 328.62 | 254.56 | +16.95% | 5/5 |

Shortening is `median(100 * (1 - batch_ms / single_ms))`, rather than the ratio
of marginal medians. Component and whole percentages have separate scopes and
are not additive. Scheduling, power, other applications, allocator state and
warmed pages/caches are uncontrolled; passes vary materially. No causal
diagnosis of the mixed whole timings is established. No full CLI/cold-file,
streaming, browser/WASM or GPU speed is claimed.

## Validation and reproduction

All 108 complete parity/first/warm-up/timed reports match. All 24 full f64 NN
distance arrays, 24 complete PLY outputs and 24 Gaussian maps match bytes,
including 192,880 saved voxel records. Twelve full Gaussian-distance arrays
contain 28,464 matching records; twelve SCS-term arrays contain 28,436 matching
records, including center keys, neighbor counts, order and f64 terms. Reports/
distances/PLYs and Gaussian-distance hashes also match independently recorded
preceding midpoint-point-tree results.

An independent audit recomputes twelve serial point queries, six structural
evaluations, Gaussian summaries and six Gaussian-distance sequences. It checks
14,232 ordered neighborhoods against integer cube enumeration and independently
derives every scalar SCS term. Every PLY coordinate/RGB value, source/library
fingerprint, complete report hash, SCS index metadata, pair order and timing
summary is verified. Twelve negative controls reject wrong sources/baselines,
invalid counts/voxel sizes, duplicate arguments and existing destinations
without replacing data.

410 related Python tests pass. The 22 new cases cover dispatch at 1,023/1,024/
1,025 cells, the 32,768/32,769-cell capacity boundary, dense worst-case hit
counts, radius 1/2/5, huge absolute integer origins, safe/unsafe local spans,
center/neighbor order, exact cube boundaries, center exclusion, scratch-array
release before the next query, backend errors and complete resident/streaming
reports and PLY bytes. The existing Windows Unix `resource` upstream-harness
import limitation is unchanged; that separate harness is excluded.

The [controller](../cloudanalyzer/scripts/benchmark_map_scs_query_batch.py)
requires `--source-root`, `--reference`, `--compared`, `--input-manifest`,
`--baseline`, `--baseline-sha256` and new `--work-dir`/`--output` destinations.
Defaults are `--repeats 5 --structure-voxel-size 3`; all six records use them.
Full timings/reports, fingerprints and parity paths remain in the records.
Gaussian files use `<4q12d` (128 bytes), Gaussian distances `<3qd` (32 bytes),
and SCS terms `<3qQd` (40 bytes). Reproduce preceding deltas with their frozen
sources.

- PandaSet: [initial](../benchmarks/cpu/map-evaluate/real-source/windows-pandaset019-scs-query-batch-initial.json), [final](../benchmarks/cpu/map-evaluate/real-source/windows-pandaset019-scs-query-batch-final.json)
- Autoware: [initial](../benchmarks/cpu/map-evaluate/real-source/windows-autoware-planning-scs-query-batch-initial.json), [final](../benchmarks/cpu/map-evaluate/real-source/windows-autoware-planning-scs-query-batch-final.json)
- Tokyo: [initial](../benchmarks/cpu/map-evaluate/real-source/windows-tokyo-takanawadai-scs-query-batch-initial.json), [final](../benchmarks/cpu/map-evaluate/real-source/windows-tokyo-takanawadai-scs-query-batch-final.json)
