# Reusing parallel CPU C2C preparation

Ordinary parallel CPU C2C now retains preparation for one unchanged compared/
reference pair. Repeated calls reuse the spatial partitions, exact query
coordinates, Morton query order, worker reference trees and original output
indices. Every call computes every nearest distance and the exact summaries
again; distances and statistics are not cached. Scalar leaves remain the
default. Small jobs keep the existing serial path.

The pool and partition thresholds are unchanged: at least two parts with
100,000 prospective queries each, using `hardwareConcurrency - 1` capped at
eight. The internal `cpuCache: false` request retains the previous path,
including planning, coordinate transfers and worker tree construction on
every call. `C2cOutput.cpuCached` and `cpuCacheReused` distinguish retained
preparation and reuse. `cpuCacheInputBytes` records reference/query XYZ and
retained output-index element bytes, excluding derived tree/Morton storage.

## Exact arithmetic, ownership and recovery

The main worker captures all split coordinates and output indices before
yielding. A lane's `C2cPrepared` owns its exact query coordinates and reference
tree in WASM. `PreparedC2c` and the original uncached core routine share the
same search helper. The tree construction, Morton ordering, guessed-hit
sequence, square root and output order are unchanged. Each call starts with
a fresh guessed hit; scalar/SIMD controls can share the immutable preparation.

Each lane begins searching as soon as its own preparation completes, while
other lanes may still be preparing. Progressive joining consumes results as
they arrive and does not retain an array of distance slices. The original
output indices stay in the prepared pair. Each call allocates independent
output, so concurrent readers cannot overwrite one another's distances.
The collected-joining comparison remains available.

Ordinary C2C still waits for cold preparation acknowledgements by default.
The [cold CPU dispatch evaluation](cpu-preparation-pipeline.md) enables queued
searches for Map quality's independent voxel phase; C2C comparisons retain
an internal opt-in because their measured benefit was inconsistent.

Only one idle CPU preparation group is retained across C2C and
[Map quality](cpu-map-quality-preparation.md). Switching workflows replaces
the idle group; Map quality retains both directions together. A different
pair/partition count, coordinate edit or removal retires preparation. Retired pairs
remain leased until their active snapshot jobs finish, then release only
their own worker state. Query edits invalidate the pair as well as reference
edits. In-flight jobs retain their original coordinate/order snapshots.

Preparation or verification failure closes the affected job to further
output writes and drains all submitted lanes before retirement. No partial
result is published. A later call rebuilds preparation. Cleanup targets live
workers and does not create replacements solely to release vanished state.

The pool now observes worker loss while idle and identifies each replacement
WASM instance by a generation. CPU pair reuse and cached GPU verification
references validate these generations. The next operation therefore rebuilds
missing CPU preparation or reimports an exact GPU reference, rather than
trusting an earlier promise from the lost worker. GPU candidate/verification
arithmetic and query-staging defaults are unchanged.

## Retention cost

The three measured workloads retain at least 38,713,496, 54,592,624 and
60,764,660 bytes of reference/query coordinate and output-index elements
across the prepared lanes/main worker. These are payload lengths, not
process-memory measurements or full allocation totals. Reference slot
indices, tree nodes, Morton order, vector spare capacity and allocator
metadata add storage. Source clouds, per-call distance arrays, other cached
GPU references and concurrent retired pairs are separate.

WASM memory capacity can remain at its high-water mark after vectors are
freed. Keeping preparation also prevents those live allocations from being
reused for other operations until the pair is retired. The demonstrated
speedup trades retained preparation memory for less repeated work.

## Reproduce measured-source comparisons

Use the unchanged inputs and provenance in
[measured-source C2C](c2c-real-source.md). These are disjoint subsets of
measured geometry, not independent surveys or accuracy ground truth.
From `web/`, after release WASM and production builds:

```powershell
node scripts/benchmark-c2c.mjs --compare-cache --reference ../notes/c2c-real-pandaset/reference.ply --compared ../notes/c2c-real-pandaset/compared.ply --input-manifest ../notes/c2c-real-pandaset/inputs.json --output ../notes/cpu-cache-new.json
```

The script excludes loading, rendering and the already-started CPU pool
warm-up. It records the first uncached and first cached call separately,
including all preparation, always in that order. These single observations
are not alternating first-use comparisons. Five warm-up pairs then precede
ten alternating timed pairs. Both paths use identical inputs, partition and
worker counts, scalar leaves, progressive joining and adaptive summaries.
Every timed cached call must reuse preparation. Timing includes all per-call
work, transfers/results, joining and summaries; the uncached path repeats
planning and worker preparation within that time.

Both controls run in the same pool; the prepared pair remains resident
during uncached calls. The comparisons do not isolate reduced preparation
work from allocation, memory-locality and host-workload effects.

Every comparison requires identical displayed f32 distance bits and reported
f64 statistics. Reports retain source/manifest/script/build hashes and refuse
to overwrite existing files. Other CPU kernel/join/summary comparisons
explicitly disable preparation caching to isolate their controls. Backend
comparisons now use the current CPU default with warm prepared-pair reuse;
older backend reports retain their earlier uncached CPU policy and hashes.

## Initial measurements

Paired reduction is the median of `100 * (uncachedMillis - cachedMillis) /
uncachedMillis`, distinct from the difference between two wall medians.

| Source / report | Workers | Uncached wall median | Cached wall median | Paired reduction | Faster pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| [PandaSet](../benchmarks/cpu/c2c/real-source/windows-pandaset019-preparation-cache.json) | 5 | 426.10 ms | 125.60 ms | 71.14% | 10/10 |
| [Autoware](../benchmarks/cpu/c2c/real-source/windows-autoware-planning-preparation-cache.json) | 8 | 596.35 ms | 197.75 ms | 65.45% | 10/10 |
| [Tokyo](../benchmarks/cpu/c2c/real-source/windows-tokyo-takanawadai-preparation-cache.json) | 8 | 371.05 ms | 85.00 ms | 76.49% | 10/10 |

First uncached/cached observations were 545.00/390.80, 713.40/709.10 and
468.20/423.40 ms respectively. They include preparation, vary with run order
and do not establish a first-use improvement. The demonstrated benefit is
repeated calculation with the same source identities and coordinates.

## Repeated and final-build measurements

After adding idle-worker loss detection and generation validation, two more
rounds used the tested final production build:

| Source | Repeated paired reduction | Final uncached wall median | Final cached wall median | Final paired reduction | Final faster pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| PandaSet | [70.03%](../benchmarks/cpu/c2c/real-source/windows-pandaset019-preparation-cache-repeat.json) | 165.20 ms | 51.90 ms | [68.75%](../benchmarks/cpu/c2c/real-source/windows-pandaset019-preparation-cache-final.json) | 10/10 |
| Autoware | [73.03%](../benchmarks/cpu/c2c/real-source/windows-autoware-planning-preparation-cache-repeat.json) | 234.60 ms | 64.35 ms | [72.71%](../benchmarks/cpu/c2c/real-source/windows-autoware-planning-preparation-cache-final.json) | 10/10 |
| Tokyo | [75.27%](../benchmarks/cpu/c2c/real-source/windows-tokyo-takanawadai-preparation-cache-repeat.json) | 277.95 ms | 63.00 ms | [76.80%](../benchmarks/cpu/c2c/real-source/windows-tokyo-takanawadai-preparation-cache-final.json) | 10/10 |

Every one of the 90 timed cache pairs across all nine reports was faster with
reuse. Absolute times fell substantially for both paths between initial and
later runs; these measurements do not identify the cause. Percentage
reductions compare paired calls within each run, without combining the
initial uncached timings with the later cached timings.

Final first uncached/cached observations were 181.20/163.10, 262.80/236.80
and 316.70/284.00 ms. The repeated round recorded 208.70/185.90,
272.20/221.90 and 325.10/281.70 ms. Their fixed order and limited first-use
observations still do not establish a cold-start improvement.

Separate processes then ran the existing collected/progressive joining
comparison, with preparation caching disabled throughout. Their progressive
uncached medians were:

| Source / standalone report | Uncached progressive wall median |
| --- | ---: |
| [PandaSet](../benchmarks/cpu/c2c/real-source/windows-pandaset019-uncached-control.json) | 170.20 ms |
| [Autoware](../benchmarks/cpu/c2c/real-source/windows-autoware-planning-uncached-control.json) | 232.65 ms |
| [Tokyo](../benchmarks/cpu/c2c/real-source/windows-tokyo-takanawadai-uncached-control.json) | 273.85 ms |

These medians are close to the final cache experiment's uncached controls,
without a resident prepared pair. They support the repeated-calculation
benefit, while separate runs do not isolate memory/locality or host effects.
A [current-backend smoke comparison](../benchmarks/cpu/c2c/real-source/windows-webgpu-pandaset019-cached-backend-smoke.json)
also passed exact parity on real PandaSet inputs: its three timed CPU calls
used five cached workers, and its GPU calls used serial exact verification
with a reused physical GPU reference/device. This is a control/parity check,
not a comparison against the four-worker hybrid GPU mode.

The six repeated/final cache reports, three standalone controls and backend
smoke identify:

- Main worker: `01bb588d6291cab6cff57e6842af53488ba3e0d22a83d94ba3c7e5249801174c`.
- Pool worker: `73124711329934dcf01fc659df1c490cba009c17e1cd1b8515883e4d69bb798c`.
- WASM: `b49e945097dd022c7a633af57118298b0c559fbcd29999101264e26b4012b479`.
- Measurement script: `d768be8a0879f9e6e3c44b97f5094cfb3d105127070602127abdfae509e054a8`.

The initial three reports retain earlier main-worker and script hashes.
Pool-worker/WASM kernels, source files and preparation manifests remained
unchanged throughout this evaluation. Retained input bytes were also
unchanged between rounds.

Builds/tests finish before sequential timing runs. CPU clocks, power and
other applications are uncontrolled; these are development measurements on
one machine and do not establish portable speed guarantees.

## Verification

The production build passed 44 native checks (19 distance-filtered core unit
tests, three related integration tests and 22 WASM-crate tests) and 102
browser cases: 30 CPU distance/cache/summary/M3C2/Map quality cases and 36
GPU cases on each of software and physical adapters. Native/WASM tests
check prepared arithmetic against uncached/scalar/SIMD/brute-force paths,
original f64 bits, owned input lifetimes, repeated calls, tiny/odd/empty
queries, malformed inputs, duplicates and nonfinite/extreme magnitudes.

Browser coverage checks default reuse, serial thresholds, collected and
progressive joining, immutable concurrent jobs, pair/partition changes,
query/reference transforms and removal, first/mid-job worker loss,
shortened and held replies, whole-job draining/retry, release-time loss
and idle-worker replacement. GPU regressions verify reference reimport
without serial fallback and retain existing staged-query/device-loss checks.
The source-removal test's synchronization now recognizes cached preparation
and verification requests as well as ordinary distance requests.

Rust formatting, core/native/WASM Clippy with warnings denied, TypeScript
checking and release WASM/Vite builds passed. All 90 cache timed pairs,
45 cache warm-up pairs and nine recorded first-use pairs passed exact
displayed-distance and reported-statistic checks. The standalone controls
passed 30 timed and 15 warm-up pairs; the backend smoke passed three timed
and five warm-up pairs without fallback. Final artifact/script hashes match
the reports. Ten invalid CLI combinations and report overwriting were
rejected. Public content and filename scans found no omitted GPU model name.
