# ICP prior-iteration correspondence radii

An optional ICP experiment recomputes a safe nearest-search radius from each
sample's previous correspondence. Exact results match the tested controls,
but comparisons against the preceding production build do not establish a
consistent whole-call speedup. `icpTemporal` therefore defaults to **false**
in the browser, WASM exports and ordinary core APIs. Existing reference/tree/
lazy-normal reuse remains enabled under its 128 MiB payload-capacity limit.

## Exact search and memory

Each fit owns a temporary vector of one `u32` tree slot per original sample.
It is filled before overlap partitioning, including samples discarded from
the closest-pair subset. The following iteration transforms the same sample
and recomputes its distance to the saved reference point in original f64
coordinates. The previous distance is never reused.

The saved point only supplies an upper radius. It never replaces the original
preceding-sample seed or enters the best-hit state. The root traversal, near
branch order and strict improvement comparisons remain unchanged. A far
branch must satisfy the original split-plane comparison and have a squared
split-plane distance at most the hint radius. Equality visits at this added
bound preserve tied winners and the original seed's tie priority.

An immutable tree's finiteness is checked once, lazily, in its prepared
reference. Invalid slots, nonfinite trees/queries and overflowing candidate
distances fall back to ordinary search. Candidates that do not tighten the
original seed also use the ordinary traversal. The first iteration has no
previous candidates. Reference replacement, edits and removal retain the
existing cache invalidation rules.

The vector is call-local, with no cross-call correspondence reuse. For the
browser's existing 50,000-point sampling limit it uses at most 200,000 bytes
of payload and is released before returning. It is separate from the
unchanged retained reference vector capacities; allocator overhead, temporary
queries and small object metadata are not included in those capacity counts.
Disabling the experiment allocates no slot vector, initializes no finiteness
record and takes the original query loop, with no per-query temporal dispatch
or slot bookkeeping. A single loop selection remains per iteration.

The core comparison is `IcpPrepared::compute_with_correspondence`, with
independent SIMD and temporal booleans. WASM `IcpPrepared.compute` and
`registerIcp` accept the optional temporal boolean after the SIMD argument.
The worker forwards `icpTemporal` through both cached and uncached fits.
`correspondenceTemporal` records the enabled mode, `temporalQueries` counts
queries with a tighter finite radius, and `temporalScratchBytes` records the
last call's temporary slot-vector capacity. A tighter radius does **not**
count branches avoided or establish an overall time saving. No GPU kernel
or new browser pool is introduced by this experiment.

The subsequent [shared radius-state experiment](cpu-icp-radius-context.md)
tested moving the invariant radius beside the incumbent. It was slower in
the measured whole calls. `USE_SHARED_NEAREST_STATE` remains false, retaining
the original radius argument for this optional temporal implementation.

## Measurements and decision

Both rounds use full unchanged f64 XYZ inputs from the existing
[real measured-source preparation](c2c-real-source.md). Original decoded
even/odd indices supply reference/moving subsets, without generated points,
added displacement/noise or further input thinning. ICP retains its existing
stride sampling limit of 50,000 points, overlap 0.8, maximum ten iterations,
point-to-plane metric and disabled initial centroid matching. Scalar leaves
are selected explicitly. Normal KNN, the full point-to-plane matrix,
fitting arithmetic, RMS reduction and convergence remain unchanged.

Each control has an independent worker with a fixed retained reference and
lazy normals. Every call loads a fresh unchanged moving cloud before timing.
One first-use pair and five alternating warm-up pairs are excluded, followed
by ten alternating timed pairs. Timings include the complete ICP RPC:
correspondences, overlap selection, normal lookups, fit, transform application
and output indexing. Load/pool startup, export checks and removal are
excluded. No builds, tests or repository scans run alongside timing.

On an i7-9750H with 12 logical CPUs and 32 GiB RAM, Chromium 153.0.8010.12
produced the following observations. Paired shortening is the median of
`100 * (1 - bounded_ms / control_ms)` for corresponding pairs. It is not
the ratio of the two marginal medians.

| Source | Reference / moving points | Iterations | Sampled points | Temporary slot bytes | Tighter-radius queries per bounded call |
| --- | ---: | ---: | ---: | ---: | ---: |
| PandaSet 019 | 531,908 / 531,908 | 9 | 48,356 | 193,424 | 434,807 |
| Autoware planning | 878,921 / 878,920 | 10 | 48,829 | 195,316 | 487,930 |
| Tokyo Takanawadai | 941,933 / 941,933 | 1 | 49,576 | 198,304 | 49,542 |

The first round compared disabled/enabled temporal radii in the same
prototype build. An excluded archived-build result/PLY oracle also matched.

| Source | Disabled median, ms | Bounded median, ms | Median paired shortening | Bounded wins |
| --- | ---: | ---: | ---: | ---: |
| PandaSet 019 | 429.95 | 424.05 | +2.72% | 7/10 |
| Autoware planning | 624.80 | 514.90 | +15.58% | 10/10 |
| Tokyo Takanawadai | 156.35 | 178.90 | −1.07% | 4/10 |

The second round compared the unchanged preceding production build against
the enabled prototype, including warm reference/normal reuse in both.
This checks whether apparent within-build improvements survive comparison
with the actual preceding implementation.

| Source | Archived median, ms | Bounded prototype median, ms | Median paired shortening | Prototype wins |
| --- | ---: | ---: | ---: | ---: |
| PandaSet 019 | 290.00 | 318.45 | −2.47% | 2/10 |
| Autoware planning | 456.55 | 465.30 | +0.66% | 5/10 |
| Tokyo Takanawadai | 127.05 | 127.45 | −0.20% | 5/10 |

Only 12 of 30 archived-build pairs improve. Autoware's small positive paired
median accompanies a slower prototype marginal median. The large first-round
Autoware gain is therefore insufficient evidence for standard adoption.
The rounds also have different timing ranges; CPU clocks, power, GC and other
applications are uncontrolled. The measurements do not isolate a unique
cause for the within-build/archived-build difference. Tokyo's one-iteration
fit has a different cost mix and does not establish better scaling.

The final source restores default-off controls and the original disabled
hot loop. Its formatting, native/WASM checks, production build and browser
tests are repeated after that change. **These performance tables measure the
saved enabled prototype, not the final default-off build.** No separate
final-build performance benefit is claimed. Both experiments use one timing
round per comparison, not independent surveys or registration ground truth.

## Validation and reproduction

Across the two rounds, all 96 pairs preserve every transform-matrix value,
initial/final RMS, iteration count, convergence flag and transformed PLY
byte. That includes 60 timed pairs and their excluded first-use/warm-up
observations: 192 paired outputs plus six excluded archived oracles. Full
input counts, expected mode/cache flags and equal bounded retained reference
capacities are checked. Those capacities remain 47,871,744 bytes for
PandaSet, 79,102,932 for Autoware and 84,773,988 for Tokyo.

Inputs, manifests, original measured sources, seven implementation sources,
scripts and current/archived worker/pool/WASM builds are hashed before and
after timing. The measured prototype's source, scripts and build were saved
before restoring the final defaults, and all six finished reports were
audited against those saved versions. Historical SIMD, scratch/reference,
triangular-matrix and octree benchmarks now pin `icpTemporal: false` to keep
their comparisons independent of this optional kernel.

Two new native tests cover exact hit indices/slots/distance bits, tied and
stale candidates, invalid/nonfinite/overflow fallbacks, sample-order records
after overlap selection, changing moving geometry, both metrics, sampling,
centroid matching and retained-reference mode changes. The native suite has
380 tests. Actual WASM tests compare 192 parameter/geometry cases, seven fits
each, including zero iterations, local/georeferenced coordinates, duplicate
ties, tiny/overflow scales, coincident/nonfinite geometry and SIMD controls.
Production-worker tests exercise both metrics, changing moving geometry,
cached/uncached references, explicit controls/default-off behavior and exact
PLY outputs. Eight browser tests cover the new and existing ICP/scratch/cache
paths. Native/WASM Clippy with warnings denied, formatting, TypeScript/Vite
build, script syntax and seven invalid/overwrite controls per new script are
also checked.

- PandaSet: [same-build report](../benchmarks/cpu/icp/real-source/windows-pandaset019-temporal-radius.json), [archived-build report](../benchmarks/cpu/icp/real-source/windows-pandaset019-temporal-radius-archived.json)
- Autoware: [same-build report](../benchmarks/cpu/icp/real-source/windows-autoware-planning-temporal-radius.json), [archived-build report](../benchmarks/cpu/icp/real-source/windows-autoware-planning-temporal-radius-archived.json)
- Tokyo: [same-build report](../benchmarks/cpu/icp/real-source/windows-tokyo-takanawadai-temporal-radius.json), [archived-build report](../benchmarks/cpu/icp/real-source/windows-tokyo-takanawadai-temporal-radius-archived.json)
- [Current-control benchmark](../web/scripts/benchmark-icp-temporal.mjs)
- [Archived-build benchmark](../web/scripts/benchmark-icp-temporal-archived.mjs)

Archive the preceding production `web/dist` before rebuilding. Run from the
repository root with the existing measured inputs and a new output path:

```powershell
node web/scripts/benchmark-icp-temporal.mjs --baseline-dist notes/icp-temporal-baseline-20261013/dist --reference notes/c2c-real-pandaset/reference.ply --compared notes/c2c-real-pandaset/compared.ply --input-manifest notes/c2c-real-pandaset/inputs.json --output notes/NEW-pandaset-temporal.json
```

Use `benchmark-icp-temporal-archived.mjs` for the archived warm comparison.
The final scripts explicitly select `icpTemporal: true` for the candidate and
false for the control; the saved prototype-era scripts instead used its
then-enabled omitted candidate option. An archived scalar build, full
measured inputs and manifest are mandatory; the output parent must exist,
reports cannot be overwritten, and `--repeats` accepts 1..100 (default ten).
The enabled option remains available without changing production defaults.
