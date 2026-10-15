# ICP full-matrix column SIMD

The point-to-plane ICP experiment updates two independent f64 matrix columns
together with WASM `f64x2` operations. Tested transforms and exported points
remain exact, but whole-call gains vary across measured sources. The standard
path therefore keeps scalar accumulation: `USE_SIMD_PLANE_SYSTEM = false`.
This is CPU/WASM SIMD; no GPU work is introduced.

## Implementation and validation

The candidate retains the full 6×6 normal matrix. It loads the six Jacobian
values in three pairs, broadcasts each row factor and performs a separate
multiply and add for each column pair. Each matrix cell receives the same
operands in the same correspondence order as before. RHS accumulation,
normal fitting/cache, damping and the solver retain their scalar arithmetic.
The earlier [triangular matrix experiment](cpu-icp-normal-equations.md) stays
disabled; this comparison changes neither triangle selection nor reduction
order.

Pairing requires all six row values to be finite and at most `1e150` in
absolute value. Their products are finite, so adding them to a matrix that
starts at zero cannot create NaNs, even after accumulated overflow. An unsafe
row switches the remainder of that system to the original scalar path,
preserving historical NaN/Inf propagation. The RHS stays scalar even for a
nonfinite residual. Native and non-SIMD WASM builds use the original path.
No heap buffers, retained capacities or public request fields are added.

Inspection of the raw linked LLVM WASM, before `wasm-opt`, finds 18
`f64x2.mul` and 18 `f64x2.add` instructions in the ICP compute function.
The preceding function has none. This confirms compiler emission, but does
not measure isolated matrix time or establish a whole-operation saving.

The measured candidate passes 381 native tests with the parallel feature,
native/WASM Clippy with warnings denied, formatting and release WASM/app
builds. Nine targeted browser tests pass. A new archived-build comparison
checks 192 geometry/parameter cases with four fits each: preceding scalar,
current standalone, fresh prepared and repeated prepared. It checks every
transform-matrix and transformed-coordinate bit, RMS, iteration count,
convergence, every PLY byte and retained capacity. Cases include both metrics,
zero/eight iterations, overlap/centroid controls, duplicate/coincident points,
local/georeferenced coordinates, tiny/huge values and nonfinite coordinates.
This comparison requires the optional `ICP_PLANE_BASELINE` snapshot; it is
skipped when no independent baseline is supplied. Existing tests additionally
cover correspondence options and cache invalidation after edits/removal.
The final default-off build also passes the nine targeted browser tests.

## Real-source comparison

Inputs are the full unchanged f64 clouds from the existing
[measured-source preparation](c2c-real-source.md). No generated performance
data, displacement, noise or further input thinning is added. ICP retains its
existing 50,000-point stride sampling limit, 0.8 overlap, maximum ten
point-to-plane iterations and disabled initial centroid matching. Both builds
explicitly set `icpSimd: false` and `icpTemporal: false`, isolating this matrix
experiment from the earlier correspondence prototypes.

Measurements use the existing Windows/i7-9750H machine, 12 logical CPUs,
32 GiB RAM and Chromium 153.0.8010.12. Independent production workers retain
the same reference/tree/lazy-normal preparation under its 128 MiB payload
capacity limit. Every call loads a fresh unchanged moving cloud before timing.
One excluded first-use pair and five excluded alternating warm-up pairs
precede ten alternating timed pairs for each source.

Whole ICP RPC includes correspondence search, overlap selection, normal
lookups, matrix assembly/solve, transform application and output indexing.
Loading/pool startup, exports, validation and removal are excluded. No build,
test or repository scan runs alongside timing. First use is not a cold median.

| Source | Scalar median, ms | Paired median, ms | Median paired shortening | Paired wins |
| --- | ---: | ---: | ---: | ---: |
| PandaSet 019 | 362.20 | 384.20 | −7.25% | 3/10 |
| Autoware planning | 557.10 | 534.35 | +1.35% | 6/10 |
| Tokyo Takanawadai | 193.75 | 182.35 | +6.62% | 8/10 |

Paired shortening is `median(100 * (1 - paired_ms / scalar_ms))` for the
corresponding pairs, not the ratio of marginal medians. Pairing wins 17/30
timed pairs and makes PandaSet slower. It is therefore retained only for
comparison, with the compile-time constant false in the final build.
The fits execute nine, ten and one iterations respectively; Tokyo has a
different cost mix. These whole-call timings do not isolate arithmetic,
guard checks, register/memory access or compiler/code-layout effects.
CPU clocks, power, scheduling, GC and other applications remain uncontrolled.
The measured-source subsets are not independent surveys or registration
accuracy ground truth, and these results do not predict every workload.

All 48 pairs preserve every transform value, initial/final RMS, iteration
count, convergence flag and transformed PLY byte: 96 paired outputs plus
three excluded archived scalar oracles. Point counts, cache flags and retained
payload capacities match. Reference payloads remain 47,871,744 / 79,102,932 /
84,773,988 bytes, with zero temporal queries and zero temporal scratch bytes.
Inputs, manifest, original measured source, seven implementation sources,
script and both production worker/pool/WASM builds are hashed before/after.
The measured SIMD-on source/script/build are saved and hash-audited before
restoring scalar accumulation. **The table measures that candidate, not the
final default-off build.** No final-build performance improvement is claimed.

## Reports and reproduction

- [PandaSet report](../benchmarks/cpu/icp/real-source/windows-pandaset019-plane-system-simd.json)
- [Autoware report](../benchmarks/cpu/icp/real-source/windows-autoware-planning-plane-system-simd.json)
- [Tokyo report](../benchmarks/cpu/icp/real-source/windows-tokyo-takanawadai-plane-system-simd.json)
- [Benchmark script](../web/scripts/benchmark-icp-plane-simd.mjs)
- [Archived-build browser test](../web/e2e/icp-plane-simd.spec.ts)

Archive the scalar production build first. Set `USE_SIMD_PLANE_SYSTEM = true`
in `icp.rs`, leave triangular accumulation false and rebuild matching release
WASM/app. The script rejects the disabled source setting; that guard cannot
identify the binary's mode, so a matching build remains required. Run from
the repository root with full measured inputs and their manifest:

```powershell
node web/scripts/benchmark-icp-plane-simd.mjs --baseline-dist notes/icp-plane-simd-baseline-20261013/dist --reference notes/c2c-real-pandaset/reference.ply --compared notes/c2c-real-pandaset/compared.ply --input-manifest notes/c2c-real-pandaset/inputs.json --output notes/NEW-pandaset-plane-simd.json
```

The output parent must exist and existing reports cannot be overwritten.
`--repeats` accepts 1..100, default ten. Eight invalid-option/overwrite cases
and the disabled-source guard are checked. For the browser oracle, supply
`ICP_PLANE_BASELINE` as an absolute snapshot root containing `dist` and the
corresponding `web/src/wasm/ca_wasm.js`. Restore the constant to false and
rebuild after comparison.
