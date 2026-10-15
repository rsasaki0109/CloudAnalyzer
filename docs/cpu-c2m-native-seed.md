# Native C2M seed-triangle deduplication experiment

Skipping the successful warm-start triangle's duplicate leaf test preserves
tested output bits, but does not produce consistent whole-call improvement.
`USE_PAR_MESH_SKIP_GUESS = false` remains the default. PandaSet changes by
−1.20% / +2.18% in the two measured passes; Autoware changes by +2.40% /
−0.31%; Tokyo improves by +2.52% / +1.90%. Only 33/60 timed pairs improve.
The experimental specialization and regression tests are retained, while
the preceding cached-bound traversal stays active.

The adopted [parallel BVH builder](cpu-c2m-native-builder.md),
[bottom-up bounds](cpu-c2m-bound-merge.md),
[indexed preparation](cpu-c2m-native-preparation.md),
[stack reuse](cpu-c2m-native-stack.md) and
[cached traversal bounds](cpu-c2m-native-bounds.md) remain enabled. These
observations measure native parallel C2M; they establish no Python conversion,
browser, WASM or GPU speedup.

## Candidate and correctness

The preceding traversal first tests the previous query's triangle with the
new query point, establishing a finite initial hit when possible. A later
visit to that triangle's leaf tests it again using an identical copy of its
immutable f64 corners. Strict `distance_sq < best.distance_sq` insertion
cannot accept that repeated result: the best distance already equals or
improves upon the seed distance. Other faces, including reversed faces with
equal distance and different normals, must still be visited in their
original order.

The candidate captures the initial successful hit's triangle ID and skips
only that ID in visited leaves. It preserves the exact original seed test,
every node-bound comparison, child order and other triangle tests. If the
seed has no computable distance, its hit remains the original missing-hit
sentinel and all real triangles retain their tests. The ID is captured
before traversal rather than following subsequent best-hit replacements.
No distances, closest points, normals or sign arithmetic are approximated.

Only `cloud_to_mesh_par` selects the candidate specialization when its
internal flag is true. Sequential/prepared/WASM/hinted APIs keep the preceding
traversal, and the normal cached-bound method remains available for exact
comparisons. Query Morton order, original 4,096-point chunks, warm starts,
[missing-hit recovery](cpu-c2m-missing-hit.md) and scatter order are unchanged.
No geometry/query/result cache, new retained buffer or altered node/triangle
capacity is introduced. The seed ID adds temporary scalar state and every
visited leaf slot adds an ID comparison. Retained capacities do not bound
peak process memory or allocator pages.

The candidate can save a full triangle closest-point calculation when the
seed is encountered again, but also adds comparisons to other leaf slots.
Some searches prune the seeded leaf before visiting it. The measured
results do not isolate which instruction, branch, scheduling or code-layout
effects account for the small changes.

## Measured results

[Measured-source preparation](c2c-real-source.md) supplies unchanged full
f64 XYZ and source-index subsets without generated coordinates,
displacement, noise or additional thinning. Tokyo retains its preceding
original-point preparation. These workloads are not independent surveys or
accuracy ground truth. Native PLY order differs from browser LOD order.
Full-input production 2.5D Delaunay TINs use automatic edge rejection and
recorded duplicate-XY handling. Complete geometry PLY bytes match across both
native builds/passes; equality with earlier browser hashes is not assumed.

| Source | Reference / queries | TIN vertices / triangles | Original query chunks |
| --- | ---: | ---: | ---: |
| PandaSet 019 | 531,908 / 531,908 | 501,097 / 844,324 | 130 |
| Autoware planning | 878,921 / 878,920 | 873,771 / 1,688,821 | 215 |
| Tokyo Takanawadai | 941,933 / 941,933 | 893,804 / 1,466,179 | 230 |

Windows/i7-9750H, 12 logical CPUs and 32 GiB RAM run archived native release
executables with fixed 12-thread pools, Rust 1.98.1/x86_64-pc-windows-msvc,
optimization level 3, LTO and one codegen unit. Twelve source/configuration
files match except for the seed-skip flag in `distance.rs`. Input/TIN and
idle pools remain resident in separate processes; only one computes at a time.

An independent archived baseline supplies full raw f64 distances, six exact
f64 summary values and scalar PLY oracles, then exits before timing. One
excluded first-use pair and five alternating warm-up pairs precede ten
alternating timed pairs per source/pass. Initial order is PandaSet → Autoware
→ Tokyo; the second pass reverses it with identical build/script fingerprints.
Whole API timing includes fresh BVH construction, Morton ordering, parallel
queries, result joining, BVH destruction and Rayon pool installation. Loading,
TIN/pool creation, first/warm-up observations, serialization, statistics/parity
and cleanup are excluded. No builds, tests, scans or edits run alongside timing.

| Source/pass | Baseline median, ms | Candidate median, ms | Median paired shortening | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| PandaSet initial | 294.37 | 293.68 | −1.20% | 4/10 |
| PandaSet final | 369.80 | 367.80 | +2.18% | 6/10 |
| Autoware initial | 599.78 | 591.08 | +2.40% | 7/10 |
| Autoware final | 680.71 | 691.60 | −0.31% | 3/10 |
| Tokyo initial | 1,776.77 | 1,800.66 | +2.52% | 6/10 |
| Tokyo final | 1,748.54 | 1,710.77 | +1.90% | 7/10 |

Shortening is `median(100 * (1 - unique_ms / baseline_ms))`, not a ratio of
marginal medians. PandaSet's initial candidate marginal median is lower
despite a negative paired median; Tokyo's initial candidate marginal median
is higher despite a positive paired median. All losses/spikes remain,
including final PandaSet observations near 333.3 / 428.2 ms and initial Tokyo
observations near 1,770.1 / 1,973.7 ms. Clocks, power, scheduling, other
applications, allocation, code layout and retained pages are uncontrolled.
PandaSet and Autoware do not reproduce their direction of change; Tokyo's
small gains are insufficient to justify a general default change against
that mixed evidence. No isolated traversal, retained-mesh query or
other-platform speedup is claimed. Percentages from different experiments
are not additive.

All 96 first-use/warm-up/timed pairs match complete f64 distance bytes,
six f64 summary values and full scalar PLY bytes: 192 outputs and six excluded
oracles. Geometry, threads and query counts remain identical. Archived source/
executable sets, script, inputs, manifests and original PCD are checked
before/after. A separate audit recomputes summaries and verifies pair order,
build fingerprints and provenance. No GPU model identifiers are collected.

## Validation and reproduction

With the candidate enabled, 404 native tests pass with the parallel feature,
together with native/WASM Clippy denying warnings, formatting and both native
release example builds. The new regression compares 22,272 hits across eight
seeds, six coordinate scales, 116 queries and four seed policies: none,
previous hit, changing arbitrary face and fixed face. It checks triangle ID,
closest-point f64 bits, squared-distance bits, signed-distance bits and
scratch capacities against the preceding specialization. Reversed/degenerate
faces, nonfinite geometry, signed zero, NaN payloads, infinities and finite
overflow queries are included. Existing missing-hit, original-chunk and
parallel-layout regressions also pass. Generated fixtures are for correctness
only. Benchmark syntax and twelve invalid CLI controls pass.

After measurement, the flag is restored to false and the native release
example is rebuilt. The original-chunk signed/missing-hit regression and
native Clippy/formatting are checked again for the restored default.
The restored twelve source/configuration files match the archived baseline.
The rebuilt executable has a different fingerprint, so a separate correctness
check also compares its complete PandaSet f64 distances, summary bits, scalar
PLY and geometry against that baseline oracle. All match; this extra check
supplies no timing comparison and does not alter the six measured reports.

Build `bench_c2m_source` in release mode with `--features parallel`, first
with `USE_PAR_MESH_SKIP_GUESS = false`, then true. Keep native stack reuse,
cached traversal bounds, indexed preparation, bottom-up bounds and parallel
tree construction enabled in both. Keep `USE_PAR_MESH_MORTON_PREPARATION = false`
before the later [Morton experiment](cpu-c2m-native-morton.md). Also keep
`USE_PAR_MESH_COMPACT_RESULTS = false` before
[compact temporary results](cpu-c2m-native-results.md). Archive each executable as
`bench_c2m_source.exe`, its twelve source/configuration files under repository
paths and `rustc -Vv` as `compiler.txt` in separate roots. Source guards cannot
identify binary mode; matching builds matter. Run from the repository root
with unchanged measured inputs and a new output:

```powershell
node web/scripts/benchmark-native-mesh-skip-guess.mjs --baseline-root notes/native-mesh-skip-guess-baseline-20261013 --candidate-root notes/native-mesh-skip-guess-measured-20261013 --reference notes/c2c-real-pandaset/reference.ply --compared notes/c2c-real-pandaset/compared.ply --input-manifest notes/c2c-real-pandaset/inputs.json --output notes/NEW-pandaset-native-skip-guess.json --threads 12
```

`--threads` accepts 1..64 (default logical CPU count); `--repeats` accepts
1..100 (default ten). Reports/run directories are not overwritten. Other
checkouts need equivalent archived builds and locally prepared inputs.
Native output files remain under ignored `notes/`.

- PandaSet: [initial](../benchmarks/cpu/c2m/real-source/windows-pandaset019-native-skip-guess-initial.json), [final](../benchmarks/cpu/c2m/real-source/windows-pandaset019-native-skip-guess-final.json)
- Autoware: [initial](../benchmarks/cpu/c2m/real-source/windows-autoware-planning-native-skip-guess-initial.json), [final](../benchmarks/cpu/c2m/real-source/windows-autoware-planning-native-skip-guess-final.json)
- Tokyo: [initial](../benchmarks/cpu/c2m/real-source/windows-tokyo-takanawadai-native-skip-guess-initial.json), [final](../benchmarks/cpu/c2m/real-source/windows-tokyo-takanawadai-native-skip-guess-final.json)
- [Benchmark controller](../web/scripts/benchmark-native-mesh-skip-guess.mjs)
- [Candidate and hit-bit regression](../rust/crates/ca-core/src/mesh.rs)
- [Parallel caller and default flag](../rust/crates/ca-core/src/distance.rs)
