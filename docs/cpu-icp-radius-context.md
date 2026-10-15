# ICP shared radius-query state

The shared-state experiment puts a bounded query's incumbent and invariant
radius in one call-local value. Recursive nodes receive that value by
reference instead of another f64 radius argument. Tested results remain
exact, but whole-call timings are worse across the three measured sources.
The preceding radius-argument implementation is retained:
`USE_SHARED_NEAREST_STATE = false`. The separate `icpTemporal` option also
remains off by default after its [earlier evaluation](cpu-icp-temporal.md).

## Implementation and validation

`BoundedNearest` contains the same `Nearest` incumbent and f64 upper radius
used by the previous implementation. The candidate radius is still recomputed
from original coordinates; the original preceding-sample seed and root/near/
far traversal order choose all hits and ties. Leaf arithmetic, strict
improvement comparisons and equality visits at the added radius are unchanged.
Invalid/nonfinite/overflow fallbacks also retain their original behavior.

The shared value is call-local and adds no heap allocation or retained
buffers. The existing per-fit u32 slot vector and reference/tree/lazy-normal
cache keep their capacities and lifetime rules. No GPU kernel, pool changes
or new product controls are introduced. Fewer source-level parameters are an
implementation hypothesis, not evidence of reduced CPU time.

The final implementation selects between both layouts with a compile-time
boolean. Its false default retains the old f64-argument search; ordinary
`icpTemporal: false` jobs retain the existing unbounded loop. The native
nearest-radius test explicitly compares both compile-time layouts, including
indices, tree slots, squared-distance bits and tighter-radius flags against
ordinary search. Existing complete-fit tests cover overlap selection,
changing geometry, both metrics, sampling, centroid matching and caches.

The measured shared-state build passes the 380-test native suite, native/WASM
Clippy with warnings denied, formatting, release WASM and TypeScript/Vite
builds. Eight browser tests pass, including 192 WASM parameter/geometry cases
with seven fits each, zero iterations, nonfinite inputs, SIMD combinations,
production-worker controls, reference invalidation and exact PLY outputs.
Those checks are repeated on the final default-off layout selection. Script
syntax, nine invalid-option/overwrite cases and a disabled-source guard are
checked too.

## Real-source comparisons

Both comparisons use full unchanged f64 inputs from the existing
[measured-source preparation](c2c-real-source.md). They add no generated
performance inputs, displacement, noise or further input thinning. ICP keeps
its existing 50,000-point stride sampling limit, overlap 0.8, maximum ten
iterations, point-to-plane metric and disabled initial centroid matching.
Leaves are explicitly scalar. The three fits execute nine, ten and one
iterations respectively for PandaSet, Autoware and Tokyo.

The machine has an i7-9750H, 12 logical CPUs and 32 GiB RAM; Chromium
153.0.8010.12 runs production WASM. Independent workers retain an unchanged
reference/tree/lazy-normal cache under the existing 128 MiB payload-capacity
limit. Every call loads a fresh moving cloud from the same measured file
before timing; correspondence slots are reused only within that call.

One excluded first-use pair and five excluded alternating warm-up pairs
precede ten alternating timed pairs. Whole ICP RPC timings include matching,
overlap selection, normal lookups, fit, transform application and indexing.
Loading/pool startup, export checks, validation and removal are excluded.
No builds, tests or repository scans run alongside timing.

`--control radius` compares the preceding enabled radius-argument prototype
against the enabled shared-state build. Every pair additionally requires
identical tighter-radius query counts and temporary slot-vector bytes.

| Source | Previous radius median, ms | Shared-state median, ms | Median paired shortening | Shared-state wins |
| --- | ---: | ---: | ---: | ---: |
| PandaSet 019 | 366.75 | 356.55 | −0.34% | 5/10 |
| Autoware planning | 448.40 | 530.15 | −19.75% | 1/10 |
| Tokyo Takanawadai | 158.10 | 180.15 | −16.41% | 3/10 |

`--control original` compares the archived pre-temporal production build
against the same enabled shared-state build. Both retain their reference
and lazy normals. This checks the practical effect against ordinary ICP.

| Source | Ordinary ICP median, ms | Shared-state median, ms | Median paired shortening | Shared-state wins |
| --- | ---: | ---: | ---: | ---: |
| PandaSet 019 | 392.05 | 379.85 | −3.17% | 4/10 |
| Autoware planning | 491.95 | 573.25 | −17.72% | 0/10 |
| Tokyo Takanawadai | 184.35 | 194.25 | −4.10% | 3/10 |

Paired shortening is the median of `100 * (1 - shared_ms / control_ms)` for
corresponding pairs, not the ratio of marginal medians. PandaSet's lower
shared-state marginal medians accompany nonpositive paired medians and
only four/five wins. Across all sources the shared state wins 9/30 against
the previous radius prototype and 7/30 against ordinary ICP. It is therefore
not selected for either the normal or optional temporal path.

These are whole-operation comparisons; they do not isolate parameter passing
from stack/memory access, compiler decisions, code layout or other costs.
CPU clocks, power, GC and other applications are uncontrolled. The rounds
have different timing ranges and provide no universal speed prediction.
Tokyo's one-iteration fit also has a different cost mix. The measured-source
subsets are not independent surveys or registration ground truth.

All 96 pairs preserve every transform-matrix value, initial/final RMS,
iteration count, convergence flag and transformed PLY byte. This covers
60 timed pairs and their first-use/warm-up observations: 192 paired outputs
plus six excluded archived scalar oracles. Full point counts, mode/cache
flags and equal retained capacities also match. Tighter-radius counts stay
434,807 / 487,930 / 49,542 per shared call; they do not count avoided branches
or establish a time saving. Temporary slot payloads stay
193,424 / 195,316 / 198,304 bytes and are released before return.

Input, manifest, original measured source, seven implementation sources,
script and worker/pool/WASM builds are hashed before and after timing.
The measured shared-state source/script/build are saved before selecting
the original layout again; all six reports are audited against those saved
versions. **These tables measure the shared-state prototype, not the final
default-off build.** No final-build performance benefit is claimed.

## Reports and reproduction

- PandaSet: [previous radius](../benchmarks/cpu/icp/real-source/windows-pandaset019-radius-context-radius.json), [ordinary ICP](../benchmarks/cpu/icp/real-source/windows-pandaset019-radius-context-original.json)
- Autoware: [previous radius](../benchmarks/cpu/icp/real-source/windows-autoware-planning-radius-context-radius.json), [ordinary ICP](../benchmarks/cpu/icp/real-source/windows-autoware-planning-radius-context-original.json)
- Tokyo: [previous radius](../benchmarks/cpu/icp/real-source/windows-tokyo-takanawadai-radius-context-radius.json), [ordinary ICP](../benchmarks/cpu/icp/real-source/windows-tokyo-takanawadai-radius-context-original.json)
- [Benchmark script](../web/scripts/benchmark-icp-radius-context.mjs)

Set `USE_SHARED_NEAREST_STATE = true` in `kdtree.rs` and rebuild matching
production WASM/app before comparison. The script rejects the false source
setting to avoid silently timing the preceding layout. Archive the appropriate
control build first, and run from the repository root:

```powershell
node web/scripts/benchmark-icp-radius-context.mjs --control radius --baseline-dist notes/icp-temporal-prototype-20261013/dist --reference notes/c2c-real-pandaset/reference.ply --compared notes/c2c-real-pandaset/compared.ply --input-manifest notes/c2c-real-pandaset/inputs.json --output notes/NEW-pandaset-radius-context.json
```

For the ordinary comparison use `--control original` (the default) and its
archived pre-temporal production build. The current candidate is explicitly
requested with `icpTemporal: true`; the archived scalar oracle always requests
false. Full measured inputs and a manifest are mandatory; the output parent
must exist, reports cannot be overwritten, and `--repeats` accepts 1..100
(default ten). Restore the constant to false and rebuild after comparison.
The saved measured prototype always used the shared state before introducing
the final layout constant; its exact script and source hashes remain in the
reports.
