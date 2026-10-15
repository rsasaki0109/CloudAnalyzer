# CPU C2C leaf SIMD

An optional ordinary browser C2C path can evaluate two reference points at
once inside a k-d-tree leaf using fixed-width WASM `f64x2` arithmetic. This
applies to the main worker's serial distance kernel and spatially partitioned
pool slices. Map quality shares these kernels and omits the option, retaining
scalar leaves in both directions. Worker selection,
reference pruning, Morton query order, original coordinates and exact
summaries keep their existing behavior.

Each pair occupies six consecutive f64 coordinates. Three unaligned vector
loads and shuffles assemble x/y/z lanes; separate multiply/add instructions
compute `(dx*dx + dy*dy) + dz*dz` in the original order. Strict vector `<`
comparisons and bit selection retain each lane's minimum and original tree
slot. At the end of the leaf, the closest strict improvement wins; tied
improvements choose the earliest tree slot, and a tied initial guess survives.
A scalar odd tail follows. NaN comparisons and overflowing squared distances
retain scalar semantics.
The loads stay within the complete pair and never read beyond a leaf.

The GPU candidate verifier retains its separate leaf strategy: it reduces
pairs against a usually good hint and rescans only when a closer point is
needed. Ordinary C2C instead reduces each leaf's two lanes and chooses the
winning slot once, avoiding that second scan when its initial guess needs
improvement. General k-d-tree `nearest`, k-neighbour and radius queries
retain their existing scalar paths; native
and non-SIMD WASM targets also use scalar arithmetic.

ICP can optionally use this general tree's paired nearest-point kernel for
correspondences. Its scalar default remains after inconsistent whole-call
measurements; see [ICP correspondence SIMD](cpu-icp-simd.md).

The internal C2C request `cpuNearestSimd: true` opts into paired leaves;
`false` or omission selects the scalar control through both serial and pool
kernels. `C2cOutput.cpuSimd` records whether
paired leaves are enabled in ordinary CPU C2C and supported by the WASM
build. The experimental control defaults off because measured ordinary C2C
wall time did not consistently improve. GPU verification continues to use
`compute.verifySimd` instead.

```sh
cd web
npm run wasm
npm run build
node scripts/benchmark-c2c.mjs \
  --sizes 10000,100000,1000000,2000000 --repeats 10 \
  --output ../notes/c2c-simd-grid.json
# Add --fixture scatter for the second fixture; use a new output path.
```

The benchmark runs scalar and paired leaves on the same production worker
and pool. It first awaits the automatic pool warm-up already started by the
main worker, recording that preparation separately. Five unmeasured pairs
warm both paths before timed pairs alternate
their order. Timings include reference construction, query planning,
coordinate copies/transfers, nearest search, joining and summaries. File
load and rendering are excluded. Every timed pair must match displayed f32
distance bits and exact f64 statistics, with identical worker counts.
Reports preserve main-worker, pool-worker and WASM hashes.

Native tests compare hit indices, tree slots and squared-distance bits with
scalar guesses, duplicates, tiny leaves, odd tails and nonfinite inputs.
Browser tests execute the actual SIMD WASM: exact f64 distances are compared
against the scalar control and brute force across large coordinates,
subnormal scales, overflow, duplicate ties and partial leaves. Separate
production-worker tests cover switching controls across serial and parallel
jobs. Map quality's regression baseline explicitly selects scalar C2C in
both directions before comparing its combined evaluation.

## Measured decision

On an i7-9750H (12 logical CPUs, 32 GiB RAM), Chromium 153 runs one worker
for 10k/100k points and eight pool workers for 1M/2M points. The lane-minimum
experiment awaited pool preparation, warmed both controls five times and
alternated the order of ten timed pairs. Both controls collected pool
results before joining, isolating the leaf arithmetic. Medians in ms:

| Fixture | Points per cloud | Scalar | Paired leaves | Median paired change |
| --- | ---: | ---: | ---: | ---: |
| Grid | 10,000 | 3.65 | 3.70 | 0.00% |
| Grid | 100,000 | 39.90 | 42.40 | +3.83% |
| Grid | 1,000,000 | 245.60 | 250.65 | +1.38% |
| Grid | 2,000,000 | 508.90 | 510.05 | −0.11% |
| Scatter | 10,000 | 5.00 | 4.70 | −0.03% |
| Scatter | 100,000 | 59.95 | 57.80 | −4.08% |
| Scatter | 1,000,000 | 388.25 | 388.55 | +1.29% |
| Scatter | 2,000,000 | 684.30 | 692.50 | +1.12% |

Paired change is the median of `(paired/scalar − 1) * 100` per trial;
negative means shorter. This differs from a ratio of the two timing medians.
All 80 timed pairs matched every displayed f32 distance bit and all f64
statistics. The grid is a 0.1 m UTM wavy surface; scatter uses deterministic
points in a 50×50×10 m box. Compared clouds are lifted by 0.05 m.

Reports: [grid](../benchmarks/cpu/c2c/windows-i7-9750h-simd-reduction-grid.json)
and [scatter](../benchmarks/cpu/c2c/windows-i7-9750h-simd-reduction-scatter.json).
These preserve the experimental build hashes from before SIMD was switched
off by default. The final implementation retains the same optional lane
kernel and validates the off-by-default controls separately.

The earlier [direct-update grid](../benchmarks/cpu/c2c/windows-i7-9750h-simd-grid.json)
and [scatter](../benchmarks/cpu/c2c/windows-i7-9750h-simd-scatter.json) reports
used a prototype that returned each pair to scalar comparisons. They did
not explicitly await the automatic pool warm-up and are historical records,
not a controlled comparison with the reduction build.

Neither strategy consistently shortened total C2C time. CPU clocks, power,
other applications and allocation/collection effects remain uncontrolled;
cross-build wall times cannot establish a kernel speedup. SIMD stays an
internal experiment, with scalar leaves as the production default. See
[CPU C2C joining](cpu-c2c.md) for the separate result-assembly change.
