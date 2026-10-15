# Resident analysis lane for M3C2

The current default uses [exact tree transfer and reuse](cpu-m3c2-preparation.md).
This document records the earlier resident implementation and its historical
builds. Set `preparedTrees: false` to isolate that copying policy; the
resident comparison script now supplies this control explicitly. Its saved
script/build hashes describe the measurement revision, rather than later builds.

M3C2 can use the analysis worker alongside the CPU pool when doing so admits
one more core lane within the existing input-copy budget. The analysis lane
borrows its already loaded `Cloud` inputs and constructs its two trees
directly, avoiding a complete input pair copied through another worker's
JS/WASM bindings. Core partitioning and the exact f64 M3C2 kernel remain
the same. The copying control uses this option by default.

## Scheduling and ownership

The pool-only policy estimates 128 bytes per input point per independent
worker, across both input clouds. The resident lane estimates 80 bytes per
input point: tree partition items, copied tree points/indices and growing
node capacity, without the complete input binding copies. These figures
retain the same 256 MiB scheduling budget, minimum 10,000 cores per lane,
and total worker cap `min(4, poolSize())`. An analysis lane replaces one
potential pool lane; it does not exceed the existing CPU concurrency limit.

With `n = reference.length + compared.length`, the maximum eligible lane
counts are:

```text
pool only: min(limit, floor(budget / (128 * n)))
resident:  min(limit, 1 + floor((budget - 80 * n) / (128 * n)))
```

The driver also caps the count by selected cores. It involves the analysis
worker only when that adds a lane. Jobs already eligible for the selected
number of pool workers retain their pool-only execution. Small core sets,
CPU availability, or input pairs too large even for one resident plus one
pool lane retain serial computation. Neither sources nor core points are
thinned automatically by this scheduling change.

For PandaSet's full 1,063,816-point input pair, two pool lanes would estimate
272,336,896 bytes and exceed the budget. One resident plus one pool lane
estimates 221,273,728 bytes and fits. The existing Autoware and Tokyo full
pairs are too large for that configuration and stay serial.

This is a scheduling estimate, not a total memory cap. Source clouds, the
owned core plan and attributes, main-worker coordinate snapshots, core
bindings, result buffers, allocator overhead, retained WASM pages and
concurrent jobs remain outside it. The 80-byte resident allowance includes
temporary tree construction, not just the final tree vectors.

Both full input snapshots and the owned core plan are captured before the
driver's first yield. It dispatches all pool lanes immediately, then calls
`m3c2CloudDistances` synchronously for the final lane. That call borrows the
resident source clouds only until it returns. Source transforms/removal
cannot run on the analysis worker during that synchronous computation;
after it yields, all later work uses the snapshots and completed owned
distance/LoD95 pairs. The native input point order, tree construction,
neighbour iteration and accumulation are identical to the pool kernel.

Pool messages are awaited with `Promise.allSettled`. Main or pool failure
drains every dispatched lane before the plan is released and rejects
without creating a result cloud. Indexed attributes and LOD construction
still use the existing consuming `M3c2Plan.finish` function after cyclic
blocks are restored to core order.

`mainWorker: false` is the internal request control for the previous
pool-only policy. `M3c2Output.mainWorker` reports whether a resident lane
participated alongside the pool. A serial request reports false; its
existing single-worker computation still runs on the analysis worker.
The experimental [radius pruning](cpu-m3c2-radius.md) remains disabled by
default and is explicitly disabled in the scheduling comparisons.

## Measured results

Measured on Windows with an i7-9750H (12 logical CPUs), 32 GB RAM and
Chromium 153. All cases use a 0.4 m normal radius, 0.25 m projection radius,
and 0.5 m half-depth. These are scheduling comparisons, with radius pruning
disabled in both modes. Each PandaSet row has five warm-up pairs and ten
timed pairs. Medians include the complete computation and RPC as described
below; the worker's own elapsed time is also recorded separately.

| PandaSet input / run | Loaded points per cloud | Core spacing, m | Selected cores | Previous / new lanes | Previous median, ms | Resident median, ms | Median paired reduction | Faster pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [Full inputs](../benchmarks/cpu/m3c2/real-source/windows-pandaset019-resident-main.json) | 531,908 | 0.25 | 212,541 | 1 / 2 | 1,091.85 | 779.30 | 31.17% | 10/10 |
| [Full inputs, all cores](../benchmarks/cpu/m3c2/real-source/windows-pandaset019-resident-main-all-cores.json) | 531,908 | 0 | 531,908 | 1 / 2 | 3,733.05 | 2,283.70 | 37.80% | 10/10 |
| [Full inputs, all cores, repeat](../benchmarks/cpu/m3c2/real-source/windows-pandaset019-resident-main-all-cores-repeat.json) | 531,908 | 0 | 531,908 | 1 / 2 | 4,740.10 | 2,624.15 | 44.62% | 10/10 |
| [Measured stride-2 subsets](../benchmarks/cpu/m3c2/real-source/windows-pandaset019-resident-main-three-to-four.json) | 265,954 | 0.25 | 140,710 | 3 / 4 | 323.80 | 303.60 | 6.91% | 8/10 |

Paired reduction is the median of `100 * (1 - resident_i / previous_i)`;
it differs from a ratio of the two marginal medians. The all-core condition
was measured twice on the same build: all 20 timed pairs improved, with
37.80% and 44.62% paired median reductions. Absolute timings varied between
runs, so these results do not establish a guaranteed speedup. Across all
four PandaSet conditions, the resident path is faster in 38 of 40 timed
pairs. The stride-2 condition uses an explicit 300,000-point budget and
retains every second original input point; it is a smaller measured subset,
not a full-source throughput claim. Its four-lane input estimate is
246,805,312 bytes and remains within the budget.

The full [Autoware planning](../benchmarks/cpu/m3c2/real-source/windows-autoware-planning-resident-main.json)
pair (878,921 / 878,920 points, 696,739 selected cores) and full
[Tokyo Takanawadai](../benchmarks/cpu/m3c2/real-source/windows-tokyo-takanawadai-resident-main.json)
pair (941,933 points each, 299,755 selected cores) each ran five warm-up and
five timed pairs. Both stay on exactly the same one-worker path in the
control and resident-enabled request. Their timing differences are variation
between identical computations, not improvements from added CPU concurrency.

All 50 timed and 30 warm-up scheduling pairs preserve distance, LoD95 and
significance bits and complete indexed PLY bytes. This includes both
full-source serial fallbacks and the measured stride-2 case. The default
resident policy is adopted for eligible inputs; it changes neither the
input selection nor the lane count when the same number of pool workers
already fits the budget.

The six scheduling reports share the final production artifacts and script:

| Artifact | SHA-256 |
| --- | --- |
| `worker-CwqWnkwZ.js` | `97882eba32b3849f9e5fda22dcb68152cf5a57cef8da5be510802417cd8c360c` |
| `c2c-worker-D-Qfto05.js` | `91143bf818cbf35a265f1aa567c86ed225382fd3f242cec5411485b23989ccab` |
| `ca_wasm_bg-DfoBjP_C.wasm` | `25ba5977e2382f5533b80211e1f92d3b4c81ae3931a2ba94edc06d9f6d733f2c` |
| `benchmark-m3c2-workers.mjs` | `02ffdc622fea226eb521128d55ad2fce760286a704392d4c79d93ef6b6a3005e` |

## Validation

The final build passes TypeScript/Vite, native and wasm32 Clippy with warnings
denied, and Rust formatting checks. Forty-five native tests pass: 16 k-d tree,
six M3C2 (including Rayon parity), and all 23 WASM-crate tests. Resident
Cloud lanes and copied worker inputs produce the same raw f64 distance/LoD95
bits, including undefined cores, and outputs survive source destruction.

Twenty-nine browser checks pass on the final production artifacts:

- Ten M3C2 scheduling tests plus the raw f64 radius-control test, including
  two and three resident-lane jobs, irregular tails, contiguous comparisons,
  source colors/intensity, invalid-radius recovery and source edits/removal.
  Instrumented-driver tests verify CPU/budget limits and wait for every
  dispatched lane after either a resident or pool failure.
- Four CPU nearest/voxel SIMD regressions and two M3C2/Map quality UI cases.
- Six selected WebGPU regressions on software and six on the physical adapter,
  including CPU-identical exports, serial/parallel device-loss recovery,
  cached verification lanes and brute-force candidate/leaf parity.

```powershell
cd rust
cargo test -p ca-core kdtree::tests --features parallel
cargo test -p ca-core m3c2::tests --features parallel
cargo test -p ca-wasm --lib
cargo clippy -p ca-core -p ca-wasm --all-targets --features ca-core/parallel -- -D warnings
cargo clippy -p ca-core -p ca-wasm --target wasm32-unknown-unknown --lib -- -D warnings
cargo fmt --all -- --check
cd ../web
npm run wasm
npm run build
$env:PW_PORT='4193'
npx playwright test e2e/m3c2-parallel.spec.ts e2e/m3c2-radius.spec.ts --workers=1
npx playwright test e2e/c2c-simd.spec.ts e2e/map-quality-simd.spec.ts --workers=1
npx playwright test e2e/app.spec.ts --grep 'M3C2: a flat grid|map quality:' --workers=1
npx playwright test e2e/gpu-c2c.spec.ts --grep 'WASM SIMD leaf verification|WebGPU C2C export matches CPU exactly|parallel exact verification preserves partial batches|device loss after the first batch|GPU candidates match brute force' --workers=1
$env:CA_GPU_HARDWARE='1'
npx playwright test e2e/gpu-c2c.spec.ts --grep 'WASM SIMD leaf verification|WebGPU C2C export matches CPU exactly|parallel exact verification preserves partial batches|device loss after the first batch|GPU candidates match brute force' --workers=1
```

Run browser suites sequentially because they share test-result paths, and
finish builds/tests before timing. The saved scheduling reports contain
50 timed and 30 warm-up pairs; the separate radius experiment contains
40 timed and 20 warm-up pairs. All nine reports were checked against input
and manifest hashes, result/export parity and point-selection controls;
the six final scheduling reports also matched the production and script
hashes at that measurement revision. Public tracked and non-ignored file names/content contain
no device model metadata prohibited by the project owner.

## Reproduction

`web/scripts/benchmark-m3c2-workers.mjs` requires two input files and has
no synthetic fixture mode. It loads the full supplied measured inputs by
default, warms the pool, then runs five warm-up pairs and ten alternating
sequential comparisons. The existing source manifests identify source
data, attribution, unchanged coordinates, original point counts and hashes.
The even/odd subsets are not independent surveys or change-detection ground
truth. Core spacing selects original measured points, without averaging.

Each call rebuilds its plan and trees, computes fresh normals and results,
joins core blocks, creates result attributes and indexes the output. All
of this and RPC delivery is timed. Loading, pool startup, parity validation
and export are excluded. Every pair requires identical distance, LoD95 and
significance bits and complete PLY output bytes. No results or trees are
cached. Source/manifest and production worker/WASM/script hashes are checked
before and after timing; existing reports cannot be overwritten. Other
applications, CPU clocks, power and garbage collection are uncontrolled.

The optional `--max-points` budget selects every `keepEvery`-th original
file point and records both original and loaded counts. It must be
distinguished from measurements on the full source files.

```powershell
cd web
node scripts/benchmark-m3c2-workers.mjs --reference ../notes/c2c-real-pandaset/reference.ply --compared ../notes/c2c-real-pandaset/compared.ply --input-manifest ../notes/c2c-real-pandaset/inputs.json --repeats 10 --output ../benchmarks/cpu/m3c2/real-source/windows-pandaset019-resident-main-repeat.json
```
