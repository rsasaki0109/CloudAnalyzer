# Paired voxel cross-matrix eigensolver experiment

Map quality's retained reference path can fit two independent cross matrices
with the existing WASM `f64x2` Jacobi solver. The candidate preserves tested
outputs but does **not** show a consistent whole-call benefit: only **54/120
timed pairs** are faster. It remains **off by default**, with
`USE_VOXEL_PAIR_EIGEN = false`. The adopted
[paired normal fits](cpu-normal-pair.md) remain enabled.

## Computation and controls

[Reference voxel reuse](cpu-voxel-reference.md) already retains Gaussian
statistics and lazily computed reference covariance roots. Every new compared
cloud still needs a cross-matrix root for each matched voxel:
`sqrt(root_reference * covariance_compared * root_reference)`.
The experiment places two such matrices in independent SIMD lanes.

Each lane keeps the original rotation order, skip decisions, independent
convergence and maximum 50 sweeps. Matrix multiplication, root reconstruction,
mean-distance sums, trace arithmetic and final square root retain their scalar
operation order. The paired helper rejects nonfinite entries or absolute
entries above `1e150`; both members then use the original scalar calculation.
An odd tail also uses it. The existing unordered traversal and distance
insertion order are retained, without a full job vector or sorting.

Two copied fixed-size jobs and paired working matrices are transient. No new
reference/result cache, request field or retained payload is introduced.
Reference-root counters, capacities, invalidation and the 128 MiB idle payload
allowance retain their policy. This is not a peak process-memory bound.
Native execution, ordinary uncached voxel scores and initial automatic calls
keep their scalar solver. The candidate acts only inside owned reference
computation on SIMD-enabled WASM, including both moment-SIMD settings.

Changing compared identities is important: an unchanged identity can hit the
three-scalar final score cache, bypassing this calculation entirely. This
experiment accelerates CPU/WASM work and introduces no GPU dispatch.

## Measured comparison

Inputs are the full unchanged reference/compared files from
[measured-source preparation](c2c-real-source.md): PandaSet 019
(531,908/531,908 points), Autoware planning (878,921/878,920) and Tokyo
Takanawadai (941,933/941,933). All load with `keepEvery:1`. No generated
performance input, displacement, added noise or further thinning is used;
Tokyo retains its earlier original-point preparation. These source-index
subsets are not independent surveys or accuracy ground truth.

Windows/i7-9750H, 12 logical CPUs, 32 GiB RAM and Chromium 153.0.8010.12 run
the archived scalar and candidate production builds in separate workers.
**Both retain the same reference cache**, explicitly enabled, with final
score reuse disabled. A new compared identity is loaded from the original file
before every call. Both warm builds therefore compute zero reference roots
and reuse exactly one root per matched voxel. Reference payloads match at
11,472,896 / 45,879,296 / 11,472,896 bytes, with 15,650 / 56,008 / 20,122
matched voxels.

One excluded first-use pair prepares both references. Five excluded alternating
warm-up pairs precede ten alternating timed pairs for each method/source/pass.
An independent archived ordinary worker supplies an excluded oracle, then
terminates before timing. The initial source order is PandaSet → Autoware →
Tokyo; the second pass reverses it. Both passes use identical source, script
and build fingerprints, with `voxel:1`, `minPoints:5`, `voxelSimd:false` and
combined evaluation threshold 0.5.

Standalone RPC includes compared Gaussian construction, cross roots,
Wasserstein distances, neighbourhood SCS and returned scores. Combined RPC
also includes fresh exact bidirectional C2C preparation/search, joins,
summaries and distance arrays. Changing compared identities prevents whole
nearest-pair reuse. Loading, pool startup, exports, validation and removals
are excluded. No builds, tests or repository scans run alongside timing.

| Source/pass | Method | Scalar median, ms | Paired median, ms | Median paired shortening | Paired wins |
| --- | --- | ---: | ---: | ---: | ---: |
| PandaSet initial | Voxel scores | 97.30 | 90.55 | +5.34% | 6/10 |
| PandaSet final | Voxel scores | 98.65 | 105.60 | −13.20% | 2/10 |
| Autoware initial | Voxel scores | 252.80 | 249.95 | −1.64% | 3/10 |
| Autoware final | Voxel scores | 233.50 | 224.60 | +2.18% | 6/10 |
| Tokyo initial | Voxel scores | 117.10 | 115.70 | −0.86% | 5/10 |
| Tokyo final | Voxel scores | 110.15 | 105.80 | −1.98% | 5/10 |
| PandaSet initial | Combined Map | 362.85 | 360.75 | −0.42% | 5/10 |
| PandaSet final | Combined Map | 402.55 | 392.70 | +3.72% | 7/10 |
| Autoware initial | Combined Map | 582.80 | 590.00 | −1.01% | 4/10 |
| Autoware final | Combined Map | 621.45 | 644.75 | −5.07% | 4/10 |
| Tokyo initial | Combined Map | 535.40 | 569.85 | −3.28% | 5/10 |
| Tokyo final | Combined Map | 517.80 | 546.65 | −3.87% | 2/10 |

Shortening is `median(100 * (1 - paired_ms / scalar_ms))`, not a ratio of
marginal medians. Both methods have 27/60 wins. All spikes and losses remain
in the reports, including a 517.6/1,046.5 ms combined Tokyo pair. Clocks,
power, scheduling, other applications, GC, HashMap seeds and WASM pages are
uncontrolled. Pair staging, copies, differing lane convergence, compiler
layout and scheduling can also contribute; these results do not isolate an
instruction-level cause. The mixed/reversed standalone results and combined
regressions do not support enabling this path on the tested workloads.

All 192 first-use/warm-up/timed pairs preserve the original score allowance:
384 outputs plus 12 excluded oracles. AWD/SCS allow the existing `1e-12`
relative/absolute tolerance for randomized HashMap reductions. Combined
results preserve every distance f32 bit, all f64 statistics, non-voxel metrics
and every exported scalar PLY byte. Inputs, manifests, original measured
PCDs, thirteen implementation files, script and both production artifact sets
are hash-checked before/after each run and independently audited against the
saved measured source/build. No GPU adapter/model identifiers are collected.

## Validation and reproduction

The candidate passes 386 native tests with the parallel feature, native/WASM
Clippy with warnings denied, formatting and release WASM/TypeScript/Vite builds.
New native regressions compare 2,304 individual Wasserstein f64 bit patterns
with paired results across zero/tiny/large/indefinite/nonfinite cases and
scalar fallbacks. They also check odd voxel counts, empty maps, changed
compared geometry, root reuse and capacity. The existing shared solver tests
cover independent convergence and all eigenvalue/eigenvector bits.

Twenty-six candidate browser tests pass, including 720 archived-WASM
reference computations across scales, odd/even voxel counts, occupancy,
moment-SIMD controls and source ownership after the reference Cloud is freed.
Both readers also reject empty input; empty Gaussian sets are tested using
occupancy and native empty inputs. Generated fixtures are correctness tests
only. Existing tests cover final-score/reference policy, edits/removal,
budgets, combined metrics/distances, pool failures and recovery. Script syntax
and nine invalid-option/repeat/overwrite/manifest controls pass.

After restoring the default-off flag, all 386 native tests and 26 browser
regressions pass again, with native/WASM Clippy, formatting and release builds.
The final browser set covers the existing Map policies and 504 archived
normal-fit cases, confirming the adopted normal path remains intact. The
final WASM, analysis worker and lane worker artifacts are SHA-256/byte-identical
to the preceding production build. The benchmark also rejects the disabled
source flag without producing a report.

The measured candidate is archived separately from the final default-off
build. Reproduction requires archiving a preceding scalar production build,
setting `USE_VOXEL_PAIR_EIGEN = true`, and rebuilding matching release WASM/app.
The source-flag guard cannot identify binary mode. Reports cannot be overwritten.
Run from the repository root with full measured inputs and a new output:

```powershell
node web/scripts/benchmark-voxel-pair.mjs --baseline-dist notes/voxel-pair-baseline-20261013/dist --reference notes/c2c-real-pandaset/reference.ply --compared notes/c2c-real-pandaset/compared.ply --input-manifest notes/c2c-real-pandaset/inputs.json --output notes/NEW-pandaset-paired-voxel.json
```

`--repeats` accepts 1..100, default ten. Another checkout needs equivalent
archived artifacts and locally prepared measured inputs. The raw browser
oracle needs optional `VOXEL_PAIR_BASELINE` as an absolute snapshot root
containing its `dist` and matching `web/src/wasm/ca_wasm.js`; it skips without it.
Run that test against an enabled matching candidate build to exercise pairing.

- PandaSet: [initial](../benchmarks/cpu/map-quality/real-source/windows-pandaset019-paired-eigen-initial.json), [final](../benchmarks/cpu/map-quality/real-source/windows-pandaset019-paired-eigen-final.json)
- Autoware: [initial](../benchmarks/cpu/map-quality/real-source/windows-autoware-planning-paired-eigen-initial.json), [final](../benchmarks/cpu/map-quality/real-source/windows-autoware-planning-paired-eigen-final.json)
- Tokyo: [initial](../benchmarks/cpu/map-quality/real-source/windows-tokyo-takanawadai-paired-eigen-initial.json), [final](../benchmarks/cpu/map-quality/real-source/windows-tokyo-takanawadai-paired-eigen-final.json)
- [Benchmark script](../web/scripts/benchmark-voxel-pair.mjs)
- [Archived-WASM regression](../web/e2e/voxel-pair.spec.ts)
- [Implementation](../rust/crates/ca-core/src/map_quality.rs)
