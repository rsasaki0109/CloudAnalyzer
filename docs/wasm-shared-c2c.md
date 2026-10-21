# Shared-memory WASM C2C

The standalone [shared C2C experiment](../web/experiments/shared-c2c/README.md)
uses one immutable k-d tree, query snapshot and cached Morton order across
Rayon Web Workers. It builds serial and shared-memory variants without changing
the application's current CPU/WebGPU backends or Pages registration.

`PreparedC2c::distances_par` is an additive, opt-in `ca-core` API. It keeps the
existing scalar f64 nearest-neighbour arithmetic, partitions the retained query
order into chunks with independent guesses, and restores every result to input
order. Existing call sites are unchanged. Historical native benchmark manifests
resolve the prior `distance.rs` through its existing executed-source archive;
their executed source hashes and measurements remain unchanged.

## Measured inputs and protocol

The inputs are the existing PandaSet scene 019 map and surveyed Autoware planning
sample described in [real-source C2C](c2c-real-source.md). Reference/compared
contain 531,908/531,908 PandaSet points and 878,921/878,920 Autoware points.
Preparation preserves every selected decoded XYZ bit, selecting even/odd source
indices without generated points or coordinate changes. These subsets are not
independent surveys or SLAM accuracy ground truth. Original data stays outside
the public packet; provenance, licensing notes and hashes are in the reports.

The production worker loads/indexes the inputs once. Its binary f64 PLY exports
give both standalone builds exactly the same coordinates and point order.
An untimed diagnostic compares every authoritative f64 distance bit across the
serial build and both modes of the threaded build. Every timed request also
checks all displayed f32 distance bits and ordered f64 summary fields against
the production worker. All checks passed, including across repeats and counts.

Each configuration has two runs of five warmup rounds and ten timed rounds.
Within a round, stable serial, same-binary serial, production pool and shared
parallel rotate order. All eight final runs are retained in the
[measurement packet](../benchmarks/wasm-shared-c2c/manifest.json).
Pool capacities match the shared lane count; five/eight lanes also exercise the
existing query-count policy. Module/thread startup, cold preparation and first
requests are recorded separately. Warm request time includes calculation,
ordered summaries, output copies and RPC. Production uses spatial partitions;
the shared alternative uses a full tree, so the comparison includes that design
difference as well as threading.

The host was an i7-9750H with 12 logical CPUs, approximately 32 GiB RAM and
Windows, with compiler, browser and Node versions recorded per report. Other
applications, CPU clocks, power and GC were uncontrolled; one Autoware rerun
showed substantial host-time variation. These are local Chromium measurements.

## Warm request results

Times are separate medians in milliseconds. The last column is the median of
each paired `(shared / pool - 1) * 100`, not the ratio of the timing medians.
Negative percentages mean shorter shared calls. Separate timing medians and
the paired statistic can differ in direction when timings vary.

| Source | Lanes | Run | Same-binary serial | Existing pool | Shared | Paired vs pool |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PandaSet | 4 | 1 | 183.14 | 60.95 | 63.31 | -2.50% |
| PandaSet | 4 | 2 | 211.88 | 68.45 | 67.97 | -6.19% |
| Autoware | 4 | 1 | 252.17 | 90.93 | 84.27 | -5.68% |
| Autoware | 4 | 2 | 328.04 | 117.58 | 116.32 | -0.34% |
| PandaSet | 5 | 1 | 173.76 | 53.61 | 50.70 | -3.42% |
| PandaSet | 5 | 2 | 179.38 | 54.59 | 53.36 | +0.21% |
| Autoware | 8 | 1 | 287.40 | 68.13 | 68.92 | -1.03% |
| Autoware | 8 | 2 | 379.48 | 79.95 | 76.84 | -4.77% |

The median paired speedup over same-binary serial ranges from 2.91x to 4.59x.
Against the already parallel production path, gains are small, the five-lane
PandaSet direction changes, and the host variation limits a general conclusion.
The shared path remains experimental; no default backend or worker policy changes.

## Memory and static-host checks

Retained tree/query/order payload is exactly 32,978,320 bytes for PandaSet at
both four and five lanes, and 54,493,116 bytes for Autoware at both four and eight.
Adding lanes does not duplicate those immutable vectors. Committed shared WASM
pages after repeated calls are approximately 81.13/83.19 MiB for PandaSet and
127.81/136.06 MiB for Autoware at the lower/higher counts. TLS, stacks and transient
allocations still affect committed memory.

These page counts are a high-water allocation, not RSS. Production pool pages
and retained-input bytes are also recorded, with different contents. Both
alternatives coexist during paired timing; these counters do not establish
whole-application memory savings or peak process memory.

The runner starts a header-free static server below a Pages-style subpath.
A scoped service worker enables isolation after one reload. Actual shared
memory and responding lane indices are checked. An unisolated worker rejects
threading and can initialize the serial build. Failed snapshot replacement and
release are tested as well. This validates local Chromium compatibility;
the optional build is not deployed in the normal Pages app. CI builds both
variants and checks the existing RELLIS-derived demo pair as a compatibility
smoke, without treating its single trial as performance evidence.
