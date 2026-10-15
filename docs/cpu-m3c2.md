# CPU parallel M3C2 in the browser

M3C2 automatically uses the existing WASM worker pool for sufficiently large
core-point sets. This works without WebGPU, SharedArrayBuffer or COOP/COEP
headers. C2C, C2M, SOR, normals, octree construction and LAS/COPC decoding
already use this pool; M3C2 now joins them. The UI reports how many workers
were used.

The current automatic path uses [exact tree transfer and reuse](cpu-m3c2-preparation.md)
by default. It preserves the core scheduling and numerical order described
below, while retaining only trees on each lane and computing fresh results.
The historical comparisons here describe the original copying path,
available through the internal `preparedTrees: false` control.

The earlier [resident analysis lane](cpu-m3c2-main-worker.md) can admit one
additional CPU lane within the same budget by borrowing the already loaded
source clouds. That document records measured-source comparisons; the
historical synthetic measurements below retain their original build hashes.

## Exact computation

Only core points are divided between workers. Each worker receives both
complete input clouds, in the same original f64 point order, and builds the
same two k-d trees as the serial implementation. A core near a part boundary
therefore sees every neighbour used by the serial computation, including
points outside its part. Tree traversal and floating-point accumulation order
within each core remain the same.

Core points are assigned in cyclic blocks of 1,024, rather than one long
contiguous range per worker. A spatial region with many neighbours is thus
spread across lanes, while each block retains local access. Each lane still
builds its two trees once and receives the same complete input snapshots.
The main worker scatters returned blocks back into original core order;
attributes, significance and result indexing use the existing finish step.
The internal `interleaved: false` request control retains contiguous ranges
for parity checks and timing comparisons. Serial computation and worker
selection use the same kernels and limits in both modes.
The last incomplete cycle is split evenly between lanes, so their core
counts differ by at most one and preserve the 10,000-core minimum.

Workers return interleaved f64 distance and LoD95 pairs. The main worker joins
them in core order, then uses the same Rust function as the serial path to
derive significance, store the result attributes and build the result octree.
Core selection, colors and existing attributes are preserved; previous M3C2
attributes are replaced. NaN distances and LoD95 remain undefined, with zero
significance, as before.

The core cloud and both input coordinate arrays are owned snapshots before
the job first yields. Transforming or removing either source during a
parallel job does not change its result or invalidate its memory. The result
name is also captured before yielding. An error does not publish a partial
result, and the core snapshot is freed. Worker inputs use borrowed WASM slices
to avoid an additional owned copy inside the Rust kernel.

## Worker and memory limits

- At least 20,000 selected core points are required; the intended minimum
  is 10,000 cores per worker. Voxel core spacing can make a large input's
  selected core set small enough to stay serial.
- Use at most four workers, never more than the existing pool makes available
  (`hardwareConcurrency - 1`, with the pool's own limit of eight).
- For each worker, estimate 128 bytes per input point across both full clouds
  for input copies, bindings and trees. Reduce worker count so that this
  estimate fits a 256 MiB job budget. Fewer than two eligible workers means
  the original serial path.

These are the original pool-only limits. The resident analysis lane uses
an 80-byte-per-input-point tree-construction allowance and participates only
when that admits a higher lane count under the same budget and core limits.
Two pool lanes can exceed the budget while one resident and one pool lane
still fit. Large pairs that cannot fit that combination remain serial.

The budget is a conservative scheduling estimate, not a total heap limit.
It excludes the source clouds, core attributes, main-worker snapshots, and
previously allocated WASM pages retained by the pool. Concurrent jobs can
also overlap their memory usage. Large input pairs can therefore run serial
even when their core sets are large. This avoids multiplying both full input
clouds across all CPU threads.

## Validation

Native Rust tests compare every f64 distance/LoD95 bit against the serial
kernel, including UTM offsets, sloped surfaces, irregular final slices,
undefined isolated cores and voxel core selection. Result PLY files are
byte-identical, including source colors and intensity. Snapshot ownership is
tested after editing source positions and attributes.
The cyclic-block test varies two to four lanes, block sizes from one to
1,024 cores, partial tails and voxel-selected cores. It verifies exactly one
visit per selected core, identical f64 bits and byte-identical indexed PLY
exports after restoring core order.

Six browser tests cover full and voxel-selected cores, byte-identical PLY
exports against both serial and contiguous controls, serial selection for
small jobs, CPU/copy-budget limits, error recovery, and changing/removing
both source clouds during delayed dispatch.
Threshold jobs with 20,001 and 30,001 cores exercise balanced tails on two
and three workers; the larger jobs exercise four workers.
The existing M3C2 UI test checks the known result of a plane lifted by 0.3 m.

```powershell
cd rust
cargo test -p ca-wasm --lib
cargo clippy -p ca-wasm --all-targets -- -D warnings
cargo clippy -p ca-wasm --target wasm32-unknown-unknown --lib -- -D warnings
cd ../web
npm run wasm
npm run build
npx playwright test e2e/m3c2-parallel.spec.ts --workers=1
npx playwright test e2e/app.spec.ts --grep "M3C2: a flat grid" --workers=1
```

## Reproducible benchmark

```powershell
cd web
node scripts/benchmark-m3c2.mjs --sizes 10000,40000,100000,200000 --repeats 4 --output ../benchmarks/cpu/m3c2/windows-i7-9750h.json
```

The benchmark launches only the production analysis worker, with no viewer or
extra UI worker pool. Both modes are warmed once, then serial and automatic
modes alternate their execution order. Modes run sequentially. Every trial
checks distance, LoD95 and significance bits; warm-up also checks full PLY
bytes. Reports include main/pool-worker and WASM hashes, browser version,
machine details and parameters, and refuse to overwrite an existing report.

Timings include core selection, input snapshots/copies, per-worker tree
construction, computation, result assembly and result indexing. File loading,
exports and rendering are excluded. Data are synthetic UTM wavy grids at
0.1 m spacing with a 0.07 m vertical shift, using a 0.4 m normal radius,
0.25 m projection radius and 0.5 m depth. Other applications and CPU
power/clocks are uncontrolled. Results depend on point density, search radii,
core spacing and the machine; this is not a real-survey throughput claim.

Measured on an i7-9750H (12 logical CPUs), 32 GB RAM, Windows 11 and Chromium
153.0.8010.12. The [saved report](../benchmarks/cpu/m3c2/windows-i7-9750h.json)
contains four trials per size for the initial contiguous-slice implementation,
with artifact hashes identifying that measured build. Values below are
medians, with the full trial range in parentheses:

| Points/cores per cloud | Workers | Serial, ms | Automatic, ms | Median speedup |
| ---: | ---: | ---: | ---: | ---: |
| 40,000 | 4 | 627 (546–919) | 231 (220–269) | 2.71× |
| 100,000 | 4 | 1,595 (1,446–1,765) | 731 (698–754) | 2.18× |
| 200,000 | 4 | 1,559 (1,399–2,975) | 665 (637–765) | 2.35× |

The 10,000-core case used the same serial path in both modes. Its median
was 141 ms versus 129 ms; that difference is timing variation, not a
parallel speedup. Serial timings also varied substantially in the larger
cases, so this run does not establish point-count scaling or a guaranteed
speedup. All result attributes matched bit-for-bit in every trial, and
the warm-up PLY exports were byte-identical for every size.

## Reproducing the load-distribution comparison

```powershell
cd web
node scripts/benchmark-m3c2.mjs --compare-layouts --fixture grid --sizes 10000,40000,100000,200000 --repeats 6 --output ../benchmarks/cpu/m3c2/windows-i7-9750h-interleaved-balanced-grid.json
node scripts/benchmark-m3c2.mjs --compare-layouts --fixture mixed-density --sizes 10000,40000,100000,200000 --repeats 6 --output ../benchmarks/cpu/m3c2/windows-i7-9750h-interleaved-balanced-mixed-density.json
```

Use new output filenames when repeating these runs. The uniform fixture is
the original 0.1 m wavy grid. The mixed-density fixture puts one quarter of
the points on a 0.035 m grid and the rest on a 0.14 m grid, separated by 100 m
in x. Dense cores have more radius-search neighbours and projection work.
Both fixtures retain the 0.07 m change and the same M3C2 parameters.

All three modes warm up once, then serial, contiguous and cyclic layouts
rotate their order over six sequential trials. Contiguous and cyclic modes
use the same lane count, complete inputs, per-worker tree construction and
per-core arithmetic. Timings include packing and scattering core blocks,
result attributes and octree construction. Every trial checks distance,
LoD95 and significance bits; warm-up compares PLY bytes across all modes.
CPU power, clocks and other applications remain uncontrolled. These
measurements are separate from the initial implementation's report above.

On the i7-9750H and Chromium 153, balanced-core-build medians of six trials were:

| Fixture | Points/cores per cloud | Workers | Serial | Contiguous control | Cyclic blocks | Reduction vs contiguous |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Uniform grid | 40,000 | 4 | 255.60 ms | 96.55 ms | 102.70 ms | -6.4% |
| Uniform grid | 100,000 | 4 | 621.60 ms | 265.80 ms | 251.95 ms | 5.2% |
| Uniform grid | 200,000 | 4 | 1,220.80 ms | 513.95 ms | 532.65 ms | -3.6% |
| Mixed density | 40,000 | 4 | 427.25 ms | 324.85 ms | 170.65 ms | 47.5% |
| Mixed density | 100,000 | 4 | 993.95 ms | 776.95 ms | 401.75 ms | 48.3% |
| Mixed density | 200,000 | 4 | 2,042.05 ms | 1,604.45 ms | 794.60 ms | 50.5% |

The mixed-density fixture demonstrates the benefit of distributing an
expensive region across all four lanes. In the 200k case, cyclic trials
ranged from 765 to 863 ms and contiguous trials from 1,469 to 1,858 ms. The
uniform grid has smaller differences in both directions, including measured
slowdowns at 40k and 200k; this is a load-distribution tradeoff, not a speedup
for every geometry. The 10k cases remain native serial for every request
mode, with substantial timing variation and no parallelization change.

All 48 final trials match distance, LoD95 and significance bits against both
serial and contiguous controls. Warm-up PLY exports match across all modes.
The [uniform report](../benchmarks/cpu/m3c2/windows-i7-9750h-interleaved-balanced-grid.json)
and [mixed-density report](../benchmarks/cpu/m3c2/windows-i7-9750h-interleaved-balanced-mixed-density.json)
record main-worker, pool-worker and WASM hashes matching the tested production
build at the balanced-core stage, before GPU workgroup tuning. The earlier
[cyclic experiment](../benchmarks/cpu/m3c2/windows-i7-9750h-interleaved-mixed-density.json)
retains the initial incomplete-cycle allocation; the final version balances
that tail, so lane counts differ by at most one core.

At the balanced-core stage, the build passed all 18 native WASM-crate tests, six M3C2 browser
tests, nine Map quality browser tests, 14 SwiftShader GPU cases, 14 physical
NVIDIA GPU cases and both related UI tests (45 browser cases in total).
Rust formatting, native and wasm32 Clippy checks, TypeScript checking and
the production build passed. Browser suites ran sequentially after the
benchmarks, so their shared Playwright trace directories did not compete
and no test/build workload overlapped the timing trials.
