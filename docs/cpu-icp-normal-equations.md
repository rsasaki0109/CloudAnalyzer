# Triangular point-to-plane ICP normal equations

The point-to-plane ICP experiment computes one half of its symmetric 6×6
normal matrix and copies the other half before solving. It preserves the
tested results, but measured whole-RPC speedups were inconsistent. The original
full matrix therefore remains the standard path:
`USE_TRIANGULAR_PLANE_SYSTEM = false` in `ca-core/src/icp.rs`.
The prototype is retained for developer comparisons and its regression test.

The final fixed-row candidate was faster in **27/60 timed pairs** across two
passes on three measured clouds. Its second pass was slower by paired medians
of **1.99% for PandaSet, 1.65% for Autoware and 5.74% for Tokyo**. This does not
justify enabling it. Existing ICP reference/tree/lazy-normal reuse remains
enabled; its earlier measurements are in [compute scratch](cpu-compute-scratch.md).
Point-to-point fitting, correspondence guesses, overlap selection and the
reference cache policy are unchanged.

## Implementation and precision

Each valid point pair supplies the same six row coefficients and residual as
before. The experimental accumulator updates the upper triangle's **21**
products instead of all **36**, while updating the same six right-hand-side
terms. Every cell receives its products in the original pair order, with
separate f64 multiplication and addition. Finite operand multiplication is
commutative, so the corresponding lower cell has the same history. Before
damping and Gaussian elimination, 15 assignments fill the lower triangle.

If a row contains any nonfinite coefficient, the accumulator first restores
the lower half accumulated so far and switches permanently to the full
historical update for that solve. Both halves then keep their original operand
order, including NaN propagation. Nonfinite residuals alone do not change this
choice. No extra retained arrays, allocations, approximation, thread scheduling
or public request/response fields are added.

The first prototype selected each triangular loop's first column at runtime.
The second uses six fixed row indices via `accumulate_plane_upper<const I>`
so the compiler can simplify the short loops. Both variants use the same
arithmetic and nonfinite fallback. The regression invokes the experimental
accumulator directly even when the production default is false.

These are source-level product counts, not a claim that generated machine
instructions or whole calculation time fall by 42%. Correspondence search,
normal fitting, overlap selection, transformation and output indexing still run.

## Measured-source protocol

[benchmark-icp-normal-equations.mjs](../web/scripts/benchmark-icp-normal-equations.mjs)
compares the preserved production build from before this experiment with a
triangular production build. Each independent worker owns one immutable
reference/tree/lazy-normal cache; both explicitly request `icpCache:true`.
Every fit loads a fresh, unchanged moving cloud outside timing. All pairs
require equal retained capacities, expected cold/warm reuse flags, identical
transform entries, RMS values, iteration counts and convergence flags, and
byte-identical transformed PLY exports.

The full measured files in [real sources](c2c-real-source.md) are used unchanged.
Reference/moving counts are 531,908/531,908 for PandaSet 019, 878,921/878,920 for
Autoware planning and 941,933/941,933 for Tokyo Takanawadai. Loading requires
`keepEvery=1` and full manifest counts/hashes. Original licenses and attribution
are embedded in each report. Even/odd measured-source splits are not independent
surveys or registration accuracy ground truth.

Settings are point-to-plane, maximum ten iterations, overlap 0.8 and no initial
centroid matching. ICP retains its normal maximum sample count of 50,000;
algorithmic sampling is unchanged. The observed fits take nine, ten and one
iterations respectively. Reference vector capacities are 47,871,744, 79,102,932
and 84,773,988 bytes respectively, within the existing 128 MiB idle budget.
This is not a total process-memory cap.

Each report has one excluded first-use pair, five excluded alternating warm-up
pairs and ten alternating timed pairs. Whole RPC includes correspondence
search, overlap selection, lazy normals, accumulation/solve, transform
application, output indexing and messages. Loading, pool startup, exports,
validation and removal are outside timing. First-use observations are single
samples, not cold medians. No builds, tests or scans ran during timing; CPU
clocks, power, garbage collection and other applications were uncontrolled.
All spikes and losses are retained. No GPU adapter/model metadata is collected.

## Whole-RPC results

Times are separate medians in milliseconds. Reduction is the median of each
pair's `(full − triangular) / full`; positive means faster. It is not the
ratio of separate medians, which can have a different sign.

The runtime-loop prototype ran in PandaSet → Autoware → Tokyo order:

| Source/pass | Full / triangular median ms | Paired reduction | Faster pairs |
| --- | ---: | ---: | ---: |
| PandaSet initial | 338.30 / 339.25 | -1.32% | 5/10 |
| Autoware initial | 416.80 / 415.25 | +2.20% | 6/10 |
| Tokyo initial | 137.05 / 132.55 | +2.33% | 7/10 |

The fixed-row prototype first reversed that order (`fixed`), then repeated
the original order (`final`). These six reports share the same candidate:

| Source/pass | Full / triangular median ms | Paired reduction | Faster pairs |
| --- | ---: | ---: | ---: |
| PandaSet fixed | 282.00 / 278.70 | +1.24% | 7/10 |
| PandaSet final | 274.20 / 280.85 | -1.99% | 1/10 |
| Autoware fixed | 414.75 / 397.55 | +2.19% | 7/10 |
| Autoware final | 360.10 / 358.50 | -1.65% | 3/10 |
| Tokyo fixed | 157.05 / 155.10 | +3.40% | 6/10 |
| Tokyo final | 143.50 / 146.00 | -5.74% | 3/10 |

The final pass does not reproduce the earlier small gains. No isolated kernel
gain or registration accuracy improvement is inferred from these measurements.

## Validation

- The new regression compares every matrix and right-hand-side f64 bit after
  **12,288 row updates** against the full historical accumulator. It covers
  finite values, underflow/subnormal products, finite overflow, signed zero,
  infinity and distinct NaN payloads, including switching after many finite
  rows and nonfinite residuals.
- Both triangular candidates passed **368 native tests**, Rust formatting,
  native Clippy and wasm32 SIMD Clippy with warnings denied, and SIMD WASM /
  TypeScript / Vite production builds.
- Both candidates passed the same **11 distinct Playwright tests**: exact
  scalar/SIMD kernel controls, ICP reference invalidation/reuse, existing
  voxel/M3C2 reuse, raw-WASM kNN bounds, two ICP pose-graph UI workflows, and
  four hardware-opt-in GPU regressions.
- Nine reports validate **144 pairs**: nine excluded first-use, 45 warm-up
  and 90 timed pairs. All fit results and complete transformed PLY bytes match.
  The fixed-row candidate accounts for 96 of those pairs.
- Nine original CLI guards passed: missing/unknown/repeated options, missing
  values, invalid repeats, manifest mismatch, identical builds and attempted
  output overwrite. Existing report bytes were preserved. After disabling the
  standard, a new guard also rejects a comparison with disabled candidate source.
- Both inputs, manifest, script, seven implementation files and both worker/WASM
  builds are hashed before and after each run. Only the ICP source changed
  between each measured candidate and the archived implementation.

After restoring the standard to false, all **368 native tests** and the same
**11 Playwright tests** passed again on the final production build. Formatting,
native/wasm32 SIMD Clippy, SIMD WASM and TypeScript/Vite builds also passed.

## Reproduction and fingerprints

The normal production build uses the full matrix. For a developer comparison,
enable `USE_TRIANGULAR_PLANE_SYSTEM` in the candidate source, rebuild the SIMD
WASM and production web app, then run from the repository root:

```powershell
node web/scripts/benchmark-icp-normal-equations.mjs --baseline-dist notes/icp-normal-equations-baseline-20261012/dist --reference notes/c2c-real-pandaset/reference.ply --compared notes/c2c-real-pandaset/compared.ply --input-manifest notes/c2c-real-pandaset/inputs.json --output NEWREPORT.json
```

Return the constant to false and rebuild after comparison. The script defaults
to ten timed pairs, refuses output overwrite and rejects disabled candidate
source. Source inspection does not replace rebuilding; the candidate artifacts
must match the enabled source. Inputs and archived builds are ignored, so another
checkout needs equivalent preserved artifacts and original measured files.

The initial and fixed-row builds, their seven implementation files and measured
script were preserved locally under `notes/icp-normal-equations-candidate-20261012`
and `notes/icp-normal-equations-fixed-20261012`. Their source hashes are
`f41ccb60ef5275de611d38971c9b5ae472c1021af17e59d4e499532b47bf288a` and
`524e8f4e735aee3367b7871ff6756a87319da6f18d8c133630b2341e70c6347e`; archived ICP source is
`c02eb7433cfda08446317e7752e6d99110d2b038553e45091c108c1647a71ecb`.
Reports retain their historical script/source/build fingerprints. Their
`optimized` label means candidate, not an adopted speedup. The disabled-source
guard was added after measurement, so the current script has a different hash.
Final default-off performance was not measured as a separate cross-build run.

| Artifact | SHA-256 |
| --- | --- |
| baseline `c2c-worker-DZRdZqhd.js` | `6bdc184592148f9fec7209b14eff935502465b0104646762dd1a20f3a0590f3c` |
| baseline `ca_wasm_bg-Cqb-Gons.wasm` | `67033ddc33d7540cbb50ef2aabd57d39672b65ca94f90aea8490a176c3d5b1b1` |
| baseline `worker-DVUZlhH3.js` | `4ffe1691efecd15db17f9fe8b8435d8aab81c17653b56b058a837d2a763ffe91` |
| initial `c2c-worker-BXTNdsb5.js` | `7f1b8adf88f17eefc5e6e4e4ebdc351080c999e25e5a92141a0c11c0c7349ee7` |
| initial `ca_wasm_bg-CQhKQQz3.wasm` | `1cc45df09821d7ec0103e4c19470b06609cd6172b06a223a0e6c654c791aae1b` |
| initial `worker-CitS-HDe.js` | `81668aac05ece9ff61e6294bdb73e283e32ff31f68ae9fae18f0d2fc778ec143` |
| fixed `c2c-worker-B28LR-Z9.js` | `20390c81abc93b8a4176c74953c3cea0d934fdf02d51fc5a787d1245ed3f6143` |
| fixed `ca_wasm_bg-BoT_dKfA.wasm` | `e0a5be7733e60b38e867dd7609ff9ff56453bd36e05473a4f2d9460fc958a902` |
| fixed `worker-DxO1h5l5.js` | `1d0f69c65ff6fd8e1bc040b5ac5734d60b7270978637373c135517928cc3cdd6` |
| Measured benchmark script | `e2bfba6cbfd962b18f0280fc7dc0746bbdbd791521c3510bc89c95db138dbdbb` |

| Final default-off artifact/source | SHA-256 |
| --- | --- |
| `c2c-worker-nC3pJ-DH.js` | `cc4e470f6998b80cabcf0785936ebef1414bb5fe76180239a8346cab8f94cb19` |
| `ca_wasm_bg-BrHnY6FE.wasm` | `2bb05665390ee2ef67028307ee571f16b11904a91e8f6b9cdf13cceee810d970` |
| `worker-BSJuKkLq.js` | `f308346bc77eba5733318bf008733363825f255f548059ced5969d963aaeaaba` |
| `rust/crates/ca-core/src/icp.rs` | `7d3dc9dfdd71a551542babbf25dbcc18cf1d31eda5388609df0b80667905af48` |
| `web/scripts/benchmark-icp-normal-equations.mjs` | `b3498f655dbd55f34127f0da00ab14e175548153a766352163f29d08a9105001` |

Raw reports and checksums:

| Report | SHA-256 |
| --- | --- |
| [windows-autoware-planning-triangular-final.json](../benchmarks/cpu/icp-normal-equations/real-source/windows-autoware-planning-triangular-final.json) | `c377e695124d710f1e8b9b0191c1d4c7a44e31f82a02c05943323f7fa19b260a` |
| [windows-autoware-planning-triangular-fixed.json](../benchmarks/cpu/icp-normal-equations/real-source/windows-autoware-planning-triangular-fixed.json) | `804978f87635f77f50ebf621655fb1e779f551c8ff62003171a2594a255c362c` |
| [windows-autoware-planning-triangular-initial.json](../benchmarks/cpu/icp-normal-equations/real-source/windows-autoware-planning-triangular-initial.json) | `1013774b5801706db9d29672d4f98376e48f93a3103296d7fadd39e6a5f929fd` |
| [windows-pandaset019-triangular-final.json](../benchmarks/cpu/icp-normal-equations/real-source/windows-pandaset019-triangular-final.json) | `c92680e153a4a3fca1e150e94421d3f67da494cbdcd9a06696b07570c41e16ef` |
| [windows-pandaset019-triangular-fixed.json](../benchmarks/cpu/icp-normal-equations/real-source/windows-pandaset019-triangular-fixed.json) | `55498f387ca8104a0e2fc680897f6655faf6ee40e3b37e0a1b0f03a428813ba9` |
| [windows-pandaset019-triangular-initial.json](../benchmarks/cpu/icp-normal-equations/real-source/windows-pandaset019-triangular-initial.json) | `673a2d7db77abc7404d4c7f82bc9f9d767b01993eba7e31bfe930cce19fcd326` |
| [windows-tokyo-takanawadai-triangular-final.json](../benchmarks/cpu/icp-normal-equations/real-source/windows-tokyo-takanawadai-triangular-final.json) | `0c019e32bcdf7666efc41a1f737ef5fcda769d7decc63af18ce1495a9d6e531c` |
| [windows-tokyo-takanawadai-triangular-fixed.json](../benchmarks/cpu/icp-normal-equations/real-source/windows-tokyo-takanawadai-triangular-fixed.json) | `59e55b07bbb8eedd53e41fde033474831ca074273f59691112f1e049d09de0ec` |
| [windows-tokyo-takanawadai-triangular-initial.json](../benchmarks/cpu/icp-normal-equations/real-source/windows-tokyo-takanawadai-triangular-initial.json) | `69e72f897c2a69e88c6f3265b0e8ed1ed4661a5e25abb71ec3a9cc28d647e38a` |
