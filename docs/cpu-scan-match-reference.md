# Scoped reference reuse in scan-match diagnostics

`ca scan-match-debug` prepares one immutable reference tree for its two
before/after distance searches when the optional Rust core provides
`PreparedReference`. The measured diagnostic/statistics interval improves by
20.55–24.45% across six real-source comparisons, with 60/60 faster timed pairs.
Preparation and destruction are included. Input loading, actual registration,
coloring and PLY writing are outside this timing scope.

A later [multi-frame SLAM change](cpu-slam-map-reuse.md) shares map loading and
an uncropped reference across selected attempts within one analysis. The
records below pin the preceding per-attempt implementation; standalone
scan-match diagnostics retain the lifetime described here.

This applies automatically to ICP/GICP, including calls from `ca slam-debug`,
with artifacts enabled or disabled. The working map is a private copy cropped/
downsampled before preparation. Registration adds normals and keeps its
coordinates fixed. Both query clouds share that exact reference tree, with
independent query order/guesses/results. Distance precision, registration,
summaries, thresholds, public options and report/PLY formats stay unchanged.

`prepare_nn_reference` checks the extension capability. Missing/disabled Rust
cores, older extensions without `PreparedReference` and empty targets retain
stateless dispatch. Actual constructor/query failures propagate. Installing a
supporting core enables the path; `CA_DISABLE_RUST_CORE=1` keeps Open3D dispatch.

## Lifetime and memory

Each invocation prepares a new tree from its working map. No reference is
shared across files/jobs. Input edits between invocations therefore use newly
loaded/preprocessed coordinates. The tree stays alive through registration,
serves the aligned search, and is released immediately after it, before
coloring/saving any artifact. Failure unwinds the invocation and creates no
artifact directory before successful searches.

The preceding [distance-array reuse](cpu-scan-match-distance-reuse.md) remains:
without artifacts, arrays are released after their metric blocks; with
artifacts, each stays until its colored cloud is saved. The tree is additional
live payload during registration:

| Source | Reference points | Native tree vector capacity, bytes |
| --- | ---: | ---: |
| PandaSet 019 | 531,908 | 18,084,896 |
| Autoware planning | 878,921 | 29,883,356 |
| Tokyo Takanawadai | 941,933 | 32,025,740 |

These are coordinate/index/node vector capacities from the
[owned reference API](cpu-c2c-python-prepared-reference.md), not process RSS or
a memory cap. Clouds, normals, distance/results, registration/search scratch
and allocator overhead are separate. Holding the tree increases registration-
stage live payload; avoiding the second build also avoids its temporary work.
Net peak RSS is not measured.

## Controlled measured interval

[Real-source preparation](c2c-real-source.md) supplies full original coordinates
and source-index subsets. No extra thinning, generated points, artificial
displacement or noise is added. Tokyo keeps its preceding original-point
selection. These are subsets of one measured source, not independent surveys
or accuracy ground truth. Generated fixtures serve only correctness tests.

The benchmark reuses the actual full-source ICP result/normals archived by the
preceding diagnostic comparison. Both workflows receive the same original
inputs and aligned cloud; recorded matrices are reapplied to verify every
point/normal bit. Loading and ICP solving are outside clocks. Loader and
registration entry points are replaced only inside the benchmark process to
supply these fixed inputs/results. Production keeps its usual solver.

The control is the immediately preceding two-search workflow, already sharing
distance arrays for PLY colors. Both workflows use the same current frozen
extension/kernel. The old workflow builds/drops a tree in each distance call;
the new workflow constructs one per invocation and drops it after the second
search. Timings include preprocessing/cloud copies, fixed normal/result
assignment, complete preparation, both changed-query searches, summaries,
0.5 m thresholds and cleanup. Artifacts are disabled in timed calls. These
results do not measure full CLI/loading/registration or artifact-writing speed.

Windows, 12 logical CPUs, Python 3.12.10, NumPy 2.3.4, Open3D 0.19.0 and
Matplotlib 3.10.8 use the preceding frozen release extension. Its 31 source/
configuration fingerprints are unchanged; ten current Python fingerprints
and an archived two-search control are recorded separately. Rayon/OpenMP
thread counts are 12 before imports. Each source/pass starts a new process.
A full artifact/f64 correctness pair and excluded first-use pair precede five
warm-up pairs and ten alternating timed pairs. Passes run PandaSet → Autoware
→ Tokyo, then reverse the source order. No builds/tests/scans/edits run during
performance comparisons.

| Source/pass | Two-build median, ms | Shared-tree median, ms | Median paired shortening | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| PandaSet initial | 352.55 | 266.92 | +24.45% | 10/10 |
| PandaSet final | 359.93 | 286.51 | +20.55% | 10/10 |
| Autoware initial | 582.17 | 449.78 | +21.19% | 10/10 |
| Autoware final | 576.28 | 460.51 | +21.91% | 10/10 |
| Tokyo initial | 605.83 | 471.74 | +22.75% | 10/10 |
| Tokyo final | 640.36 | 473.68 | +24.34% | 10/10 |

Shortening is `median(100 * (1 - shared_ms / two_build_ms))`, not the ratio of
marginal medians. All observations remain. Scheduling, clocks, power, other
applications, allocator/page retention and caches are uncontrolled. These
results include per-invocation preparation and differ from the standalone
API's constructor-excluded repeated-query measurement. Separate acceleration
percentages and operation scopes are not interchangeable or additive. This
step does not measure/change browser/WASM/GPU dispatch.

## Validation and reproduction

All 192 first/warm-up/timed diagnostic reports match in both sets of five
statistics, thresholds, transforms, preprocessing and improvement values.
Twenty-four traced full f64 arrays agree across six artifact correctness pairs
and match the preceding independent before/aligned distance hashes. All 36
complete PLY outputs in those pairs match every coordinate, normal and RGB byte.
PLY correctness checks run outside timing.

Fifty-five related Python tests pass with the supporting release extension,
including the CLI invocation. Tests cover ICP/GICP, Rust/Open3D, artifacts
on/off, thresholds, owned snapshots, one construction/two queries, reference
release before coloring, distance-array lifetime and failure without publication.
With the preceding extension, 52 pass and three feature-specific tests skip;
stateless dispatch remains functional. Rust binaries/kernels are unchanged.
Syntax/whitespace checks and an independent audit of fingerprints, complete
artifacts, retained capacities and every timing summary pass.

The [benchmark script](../cloudanalyzer/scripts/benchmark_scan_match_reference.py)
requires original `--reference`, `--compared`, `--input-manifest`, archived
two-search `--baseline`/`--baseline-sha256`, a preceding `--registration-record`,
`--source-root` and new `--work-dir`/`--output` paths. Defaults are
`--threads 12 --repeats 10` without artifact timing. `--with-artifacts` can
include production coloring/writes in a new comparison; these six reports omit
it. Snapshot matrices/points/normals and complete results/artifacts are checked.
Twelve negative controls cover options/counts, wrong data/baseline/registration
records and existing paths without replacing prior outputs.

- PandaSet: [initial](../benchmarks/cpu/scan-match/real-source/windows-pandaset019-reference-reuse-initial.json), [final](../benchmarks/cpu/scan-match/real-source/windows-pandaset019-reference-reuse-final.json)
- Autoware: [initial](../benchmarks/cpu/scan-match/real-source/windows-autoware-planning-reference-reuse-initial.json), [final](../benchmarks/cpu/scan-match/real-source/windows-autoware-planning-reference-reuse-final.json)
- Tokyo: [initial](../benchmarks/cpu/scan-match/real-source/windows-tokyo-takanawadai-reference-reuse-initial.json), [final](../benchmarks/cpu/scan-match/real-source/windows-tokyo-takanawadai-reference-reuse-final.json)
