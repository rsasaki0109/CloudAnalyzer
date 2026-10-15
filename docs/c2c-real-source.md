# C2C comparisons on measured source geometry

The C2C benchmark accepts actual point-cloud files and can compare parallel
CPU with the hybrid WebGPU path. These runs use three existing local datasets,
with their measured coordinates preserved. Input-file, preparation, benchmark,
main-worker, pool-worker and WASM hashes are recorded for reproduction.

## Inputs and interpretation

| Source | Locally decoded points | Reference | Compared |
| --- | ---: | ---: | ---: |
| PandaSet scene 019 | 1,063,816 | 531,908 | 531,908 |
| Autoware sample-map-planning | 1,757,841 | 878,921 | 878,920 |
| Tokyo Takanawadai, prepared geometry | 1,883,866 | 941,933 | 941,933 |

[PandaSet](https://github.com/scaleapi/pandaset-devkit), by Hesai and Scale AI,
is licensed CC BY 4.0. The existing local PCD contains scene 019's Pandar64
frames 0, 8, ..., 72 in world coordinates, as prepared by
[`prepare_vector_map_pandaset.py`](../scripts/prepare_vector_map_pandaset.py).
Its stored XYZ values are f32; exporting them as f64 preserves those values
and does not restore precision lost before this evaluation.

The official Autoware [planning sample](https://docs.autoware.org/main/demos/planning-sim/)
is a surveyed map in MGRS 54SVE. Sample map: Copyright 2020 TIER IV, Inc.
The sample archive has no separate license text; the software repository's
license is not asserted to cover the survey. Original map bytes remain in
the local dataset directory; this change checks in metrics and provenance.
See the existing [archive provenance](vector-map-media.md#intersection-surveyed-context-and-new-connection-drafts).

[Hard Intersection Multimodal Samples](https://huggingface.co/datasets/dynamic-maps/hard-intersection-multimodal-sample),
Dynamic Map Platform Co., Ltd. (2026), is licensed CC BY 4.0. The local source
is pinned to revision `e8e8d2a5d49a8b9cb63b5b5ecef9b260ff48f39c`. Its original
LAS is a multi-run aggregated survey in EPSG:6677, with orthometric heights.
This benchmark reuses the existing 0.1 m preparation: the first original
point per voxel, retaining 1,883,866 of 35,668,990 points without averaging.
The preparation JSON is embedded in the benchmark provenance. See the
[source hashes and preparation](vector-map-hard-intersection.md).

`prepare-c2c-source.mjs` decodes each local source with the production WASM
before LOD indexing. Reference contains decoded indices 0, 2, 4, ...;
compared contains 1, 3, 5, .... Only XYZ is exported as binary f64 PLY.
Every selected coordinate is decoded again and checked bit for bit against
its source value. No offset, noise, registration, averaging or invented
coordinates are added. Original attributes and files remain in the source.
The worker subsequently loads every exported point, with `keepEvery: 1`.

These are disjoint source-index subsets of one measured cloud per dataset.
Repeated geometric positions can still exist. This exercises real geometry,
density and acquisition ordering; it is not a comparison of independent
surveys, a SLAM accuracy measurement or a map-change ground truth. The
runner also accepts independently acquired reference/compared files when
they already share a coordinate frame.

## Reproduction

Use new output directories and report names. The scripts refuse to overwrite
existing preparations or reports. Run from `web` with the production WASM
and app already built.

```powershell
node scripts/prepare-c2c-source.mjs --source ../demo_data/pandaset-vector-map/map.pcd --output ../notes/NEW-c2c-pandaset --source-url https://github.com/scaleapi/pandaset-devkit --attribution 'PandaSet, Hesai and Scale AI; locally prepared scene 019, Pandar64 frames 0,8,...,72 in world coordinates' --license CC-BY-4.0

$pair = @('--reference', '../notes/NEW-c2c-pandaset/reference.ply', '--compared', '../notes/NEW-c2c-pandaset/compared.ply', '--input-manifest', '../notes/NEW-c2c-pandaset/inputs.json', '--repeats', '10')
node scripts/benchmark-c2c.mjs --compare-summary @pair --output ../notes/NEW-pandaset-summary-cpu.json
node scripts/benchmark-c2c.mjs --compare-summary --backend webgpu --hardware @pair --output ../notes/NEW-pandaset-summary-gpu.json
node scripts/benchmark-c2c.mjs --compare-backend --hardware @pair --output ../notes/NEW-pandaset-cpu-vs-gpu.json
```

The corresponding existing sources are
`demo_data/autoware/sample-map-planning/pointcloud_map.pcd` and
`notes/hard-intersection-prepared-v2/geometry.las`. For Tokyo, pass
`--preparation ../notes/hard-intersection-prepared-v2/preparation.json` and
the pinned provider URL/CC BY attribution to the preparation script.

Actual file pairs can be supplied directly with `--reference` and
`--compared`; `--input-manifest` is optional. File inputs cannot be combined
with synthetic `--fixture`/`--sizes`. `--max-points` sets a per-cloud load
budget in 1..2,000,000, defaulting to 2,000,000; the report records announced
and loaded counts, thinning factors and bounds. Supplied manifests must
match file hashes, byte sizes and announced point counts. Files are
rechecked before they are served to the browser.

The benchmark runs the production worker on a blank page. It awaits CPU
pool preparation, warms both controls for five pairs and alternates their
order for ten timed pairs. File decoding, LOD preparation and rendering are
excluded. Worker request time includes planning, snapshots, transfers,
distance calculation, joining, summaries and coloring. GPU trials retain
default prefetch, bounds, SIMD verification, 64-invocation workgroups and
batch caps, and must reuse their reference/device. Timed GPU fallback fails
the run rather than being labelled a GPU result.

`--compare-summary` holds the backend and worker count fixed while changing
full sorting to adaptive exact medians. `--compare-backend` uses adaptive
summaries on both paths: production scalar CPU leaves with progressive
joining versus GPU candidates with exact f64 WASM verification. Worker
counts are recorded separately. GPU phase intervals exclude summaries and
can overlap; adding them does not reconstruct request wall time.

## Measured results

The runs used the same i7-9750H/approximately 32 GiB/Chromium 153 machine
and physical NVIDIA GPU as the earlier synthetic comparisons. Each table
entry comes from ten alternating timed pairs after five warm-up pairs.
Times are separate medians in ms. Paired change is the median of
`(optimized/control - 1) * 100` for each pair, rather than a ratio of the
two timing medians. Negative means shorter.

Full sorting versus adaptive exact summaries, with each backend held fixed:

| Source | Backend | WASM workers | Full sort | Adaptive | Median paired change |
| --- | --- | ---: | ---: | ---: | ---: |
| PandaSet 019 | CPU | 5 | 193.40 | 166.75 | −14.35% |
| Autoware planning | CPU | 8 | 292.40 | 245.75 | −14.96% |
| Tokyo Takanawadai | CPU | 8 | 342.65 | 297.65 | −14.81% |
| PandaSet 019 | WebGPU | 1 | 194.65 | 172.75 | −14.99% |
| Autoware planning | WebGPU | 1 | 309.85 | 267.45 | −13.26% |
| Tokyo Takanawadai | WebGPU | 1 | 306.55 | 253.45 | −16.83% |

These are complete C2C request times, not isolated summary-stage times.
Reported median distances are 0.074728 m, 0.156808 m and 0.069304 m
respectively; the comparisons are not an identical-cloud zero-distance case.

Parallel CPU versus hybrid GPU, with adaptive summaries on both backends:

| Source | Run | CPU workers | CPU | WebGPU | Median GPU/CPU paired change |
| --- | --- | ---: | ---: | ---: | ---: |
| PandaSet 019 | Initial | 5 | 166.80 | 165.90 | −1.34% |
| PandaSet 019 | Rerun | 5 | 252.90 | 279.80 | +9.19% |
| Autoware planning | Initial | 8 | 264.35 | 295.90 | +12.97% |
| Autoware planning | Rerun | 8 | 281.35 | 287.15 | −1.74% |
| Tokyo Takanawadai | Initial | 8 | 272.80 | 254.10 | −5.26% |
| Tokyo Takanawadai | Rerun | 8 | 295.05 | 250.00 | −16.01% |

GPU requests used one main WASM worker. These runs support retaining both
backends: PandaSet and Autoware paired differences changed direction between
runs, while Tokyo's GPU times were shorter in both runs with different
improvement sizes. They do not establish a universally faster backend.
The summary and backend tables come from separate runs; differences between
their absolute timings should not be attributed to a code change.

Reports, including every trial and embedded source/split provenance:

| Source | CPU summary | GPU summary | CPU versus GPU |
| --- | --- | --- | --- |
| PandaSet 019 | [report](../benchmarks/cpu/c2c/real-source/windows-i7-9750h-pandaset019-summary.json) | [report](../benchmarks/gpu/c2c/real-source/windows-webgpu-pandaset019-summary.json) | [initial](../benchmarks/gpu/c2c/real-source/windows-webgpu-pandaset019-cpu-vs-gpu.json), [rerun](../benchmarks/gpu/c2c/real-source/windows-webgpu-pandaset019-cpu-vs-gpu-v2.json) |
| Autoware planning | [report](../benchmarks/cpu/c2c/real-source/windows-i7-9750h-autoware-planning-summary.json) | [report](../benchmarks/gpu/c2c/real-source/windows-webgpu-autoware-planning-summary.json) | [initial](../benchmarks/gpu/c2c/real-source/windows-webgpu-autoware-planning-cpu-vs-gpu.json), [rerun](../benchmarks/gpu/c2c/real-source/windows-webgpu-autoware-planning-cpu-vs-gpu-v2.json) |
| Tokyo Takanawadai | [report](../benchmarks/cpu/c2c/real-source/windows-i7-9750h-tokyo-takanawadai-summary.json) | [report](../benchmarks/gpu/c2c/real-source/windows-webgpu-tokyo-takanawadai-summary.json) | [initial](../benchmarks/gpu/c2c/real-source/windows-webgpu-tokyo-takanawadai-cpu-vs-gpu.json), [rerun](../benchmarks/gpu/c2c/real-source/windows-webgpu-tokyo-takanawadai-cpu-vs-gpu-v2.json) |

The initial backend reports inherited a generic note suggesting shared
partitions/query order/tree construction between CPU and GPU. That wording
was incorrect: the backends use different planning, trees and worker counts.
The reruns correct the note; measured kernels, inputs and request options are
unchanged. Each report retains its actual measurement script hash.

All 120 recorded timed pairs matched every f32 distance bit and exact
reported f64 statistic. Source files, prepared files, manifests and
preparation scripts were rehashed against the reports. Benchmark script
hashes identify the initial and corrected-note versions separately. The
main-worker, pool-worker and WASM hashes also match the production assets
used in the previous [distance-summary validation](distance-summary.md).
This step adds preparation and comparison tooling using those existing
distance kernels. Three source preparations verified all 4,705,523 decoded
points' selected XYZ bits through production-WASM PLY round trips. Five
additional smoke runs covered actual-file CPU summaries/backend switching
and the existing synthetic SIMD, joining and GPU-summary CLI modes. Four
invalid-control checks rejected incomplete file pairs, a synthetic `files`
fixture and conflicting comparison/backend flags without creating reports.
Overwrite attempts also rejected existing preparations/reports and left
their hashes unchanged. Script syntax and local documentation links passed.

Benchmarks ran sequentially, with this project's tests/builds finished
before timing. Clocks, power, garbage collection and other applications
remain uncontrolled. For example, the Autoware backend run's CPU trials
ranged from 241.1 to 425.3 ms, with GPU trials also varying. These are three
real-source development workloads on one machine, not independent survey
accuracy measurements or cross-machine performance guarantees.

The backend reruns report median f64-verification intervals of 237.45 ms
(PandaSet), 226.75 ms (Autoware) and 202.65 ms (Tokyo), and leaf-certified
query proportions of 44.31%, 47.91% and 51.17%. These intervals overlap GPU
candidate processing and must not be added to it. Parallelizing or reducing
the remaining exact CPU verification is a useful next target to evaluate
against these measured-source pairs, retaining exact result checks.

The next [parallel f64 verification evaluation](gpu-c2c-verification.md)
reuses these exact files and provenance. It measures first-use worker tree
preparation separately from warmed serial/parallel pairs, with 2/4/8-lane
controls and unchanged exact output checks.

The later [CPU preparation cache](cpu-c2c-preparation.md) measures repeated
parallel CPU calculations on the same files. The backend reports here
retain the earlier uncached CPU path; current backend comparisons use the
new default preparation reuse and record it explicitly.

[Bidirectional Map quality preparation](cpu-map-quality-preparation.md)
also measures these unchanged full inputs. The two directions retain
their exact partitions/worker trees together, with fresh nearest searches,
voxel scores and inlier metrics on every call.
