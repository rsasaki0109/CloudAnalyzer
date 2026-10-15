# Resident analysis lane for GPU C2C verification

The worker protocol accepts experimental `gpuMainWorker: true`. It replaces
the last pool lane with the analysis worker's existing immutable GPU C2C job.
The total verification lane count, GPU hints, exact kernel, batch size and
summary arithmetic stay the same. **The default remains false:** repeated
real-source comparisons improved first-use work but regressed warm processing
on two inputs. The UI retains its existing pool-only parallel verification.

This differs from the adopted [M3C2 resident lane](cpu-m3c2-main-worker.md).
GPU C2C already overlaps candidate dispatch with exact verification; adding
synchronous work to the analysis worker did not consistently improve that
pipeline. No dataset-specific activation heuristic is used.

## Implementation and ownership

For four requested lanes, pool lanes 0–2 receive immutable reference trees.
The analysis worker verifies lane 3 from its already owned tree and original
f64 queries. Thus a consistently resident reference needs three additional
pool reference payloads rather than four, a 25% reduction in that replication.
This is not a 25% reduction in total memory: source clouds, the main tree,
GPU buffers, results and retained WASM heap pages are also resident. Explicitly
switching to a larger pool-only configuration can retain the extra cached lane
until that reference retires.

Pool messages are submitted before the synchronous resident calculation.
Next-batch GPU prefetch still precedes both. Each batch splits into the same
contiguous ranges as pool-only verification; a final batch with fewer queries
uses fewer lanes, and its last lane is resident. No query coordinates or tree
are copied for the resident range. In optional staged-query mode,
`workerQueries(..., mainWorker)` excludes that last range from each batch,
including tiny tails. Existing callers omitting the new optional argument
keep their original partitioning. Query staging itself remains off by default.

`GpuC2cJob.verify(first, candidates)` returns owned f64 distances and its
certificate count without mutating the job's serial counter. Both this method
and serial `complete` use the same exact kernel. All submitted lanes settle
before any failure is recovered. If a pool or resident lane fails, its results
and certificate count are discarded, and the complete batch is recomputed
from the immutable main snapshot. This prevents resident certificates from
being counted again during serial recovery. Edits, removal and concurrent jobs
retain the existing snapshot leases.

Pool verification and staged-query preparation now use the prepared worker's
generation. A missing or replaced instance rejects the follow-up without
starting a replacement that lacks its reference/query state. A later job can
prepare a new instance normally. This protection also applies to the ordinary
pool-only GPU path.

`compute.verifyWorkers` remains the total lane cap. New metadata includes
`verifyMainWorker` and `verifyPoolWorkers`. Serial verification and recovered
jobs report false and zero respectively. `verifyReferenceBytes` is still the
reference payload size per pool lane. `verifyQueryBytes` counts only staged
pool coordinates when the resident option is enabled.

## Real-source measurements

All inputs are the unchanged measured-source pairs described in
[real-source C2C evaluation](c2c-real-source.md). PandaSet has 531,908 points
per side, Autoware 878,921/878,920, and Tokyo 941,933 per side. The full inputs
use the 2,000,000-point budget with `keepEvery: 1`; no coordinates, offsets or
noise are generated. These original-index splits are not independent surveys
or change-detection ground truth.

Each sequence uses five warm-up pairs and ten timed pairs, alternating the
mode order. Both modes request four total lanes. Each mode loads identical
reference bytes under a fresh identity, then performs an excluded serial GPU
call to prepare only the main tree/device. The cold call copies/imports exactly
the required worker references; its immediately following warm call reuses
them. Removing each reference prevents the other mode's extra cache lane from
affecting memory/preparation comparisons.

The table gives median complete RPC times in milliseconds and the median of
the ten paired reductions, `100 * (1 - resident / poolOnly)`. Positive paired
values mean faster resident processing. Separate marginal medians need not
give the same reduction. First-use rows include worker reference setup, but
exclude loading and main GPU preparation.

| Input / sequence | Cold pool → resident ms | Cold paired reduction | Faster cold pairs | Warm pool → resident ms | Warm paired reduction | Faster warm pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PandaSet / initial | 127.35 → 117.75 | +10.67% | 7/10 | 93.00 → 98.30 | +3.49% | 6/10 |
| PandaSet / repeat (`final`) | 136.40 → 127.55 | +10.54% | 6/10 | 98.90 → 102.70 | −4.70% | 3/10 |
| Autoware / initial | 225.70 → 197.75 | +10.18% | 8/10 | 146.50 → 142.20 | +2.46% | 7/10 |
| Autoware / repeat (`final`) | 213.00 → 194.95 | +9.27% | 6/10 | 144.35 → 145.65 | +0.40% | 5/10 |
| Tokyo / initial | 228.05 → 207.85 | +14.09% | 9/10 | 137.85 → 139.05 | −0.85% | 5/10 |
| Tokyo / repeat (`final`) | 223.70 → 206.40 | +5.57% | 6/10 | 133.00 → 137.80 | −3.54% | 1/10 |

There are 60 timed trials, each containing one cold and one warm paired
comparison (120 paired comparisons), plus 30 warm-up trials (60 paired
comparisons). All distance f32 bits, reported
f64 statistics and certificate counts match both modes, the serial control
and the warm run. Native and direct WASM tests additionally compare individual
f64 distance bits. CPU/GPU clocks, power, GC and other applications are
uncontrolled. Host GPU wait intervals include scheduling, and phase intervals
overlap; do not sum them or interpret them as shader execution times.

For these references, the one avoided payload is 16,832,040 bytes (PandaSet),
27,755,508 (Autoware) or 29,519,844 (Tokyo), plus the corresponding worker tree
and query-copy costs. The timing regressions still prevent default activation.

## Reports and reproduction

From `web/`, after building the SIMD-enabled WASM and production app:

```powershell
node scripts/benchmark-gpu-resident.mjs --reference ../notes/c2c-real-pandaset/reference.ply --compared ../notes/c2c-real-pandaset/compared.ply --input-manifest ../notes/c2c-real-pandaset/inputs.json --output ../notes/gpu-resident-new.json
```

Use the corresponding `c2c-real-autoware` or `c2c-real-tokyo` pair for the
other inputs. The harness requires files and a physical WebGPU adapter,
rejects unknown/repeated options and existing reports, validates full/strided
point counts, and checks source, manifest, script and production artifact
hashes before/after timing. Both compared flags are explicit, so their behavior
does not depend on the function's default. Reports contain coarse adapter
vendor/architecture/fallback information only.

| Report | SHA-256 |
| --- | --- |
| [PandaSet initial](../benchmarks/gpu/c2c/real-source/windows-webgpu-pandaset019-resident-main-initial.json) | `a91aeb9ae5887f81e17ab722e0ee5f636c7c159b5737ffdc336260e9648debc2` |
| [PandaSet repeat](../benchmarks/gpu/c2c/real-source/windows-webgpu-pandaset019-resident-main-final.json) | `a52e4926d141317364623e14b04d6cc688fe916cd5b0f4eb1993f7f853dc5639` |
| [Autoware initial](../benchmarks/gpu/c2c/real-source/windows-webgpu-autoware-planning-resident-main-initial.json) | `a9ac9185ef6c87ba64f488353fd6200aa4cff0bbc4a0af2871051de31d6056a9` |
| [Autoware repeat](../benchmarks/gpu/c2c/real-source/windows-webgpu-autoware-planning-resident-main-final.json) | `82a826fc20127a4228e0e120555b7abbab7981fa6aca4bb0eb14d3bcb59853f7` |
| [Tokyo initial](../benchmarks/gpu/c2c/real-source/windows-webgpu-tokyo-takanawadai-resident-main-initial.json) | `0cf7168f7786ef105c8c14ecb1fb5fb4892c43b4526a434a6f2a9fd644ec932f` |
| [Tokyo repeat](../benchmarks/gpu/c2c/real-source/windows-webgpu-tokyo-takanawadai-resident-main-final.json) | `254b8da597351d59a0e9e23ae39ba687bcef58abb6b0bfd348d7197c625142bb` |

The benchmark script SHA-256 is
`f0acb95e121fda124bf1e697f9ad4800a58e3d82c0a8499237a9dac34d8bd881`.
All six reports use the same WASM and pool artifact:

- `ca_wasm_bg-CWqnGpGE.wasm`: `85f7b01f7bad88dd338fc1e6846644c1502607a167f40fbb61348957dfebf17e`
- `c2c-worker-CygevjPy.js`: `52b287233d638951e6247414b8a0e18498cd59a78b06dd5495a79adfeaee6842`

Initial reports and the retained production default use
`worker-Yy6kZ-za.js`, SHA-256
`2f9586094ac5ddd9f80f3c150612720b5b6c7a1a1241b6f4cfea9926990cc157`.
Repeat reports named `final` were measured during provisional default
activation using `worker-C1WJHwVc.js`, SHA-256
`c420859f099fc2b180c87791b23a91f15d89d2ed0bb6748e489bfad519716bce`.
The comparisons explicitly set false/true in both builds; the provisional
default was reverted after those warm regressions. `final` identifies the
second measurement sequence, not the selected default. Reports record the
dirty checkout and artifact hashes rather than claiming an unmodified commit.

## Validation

Native `ca-wasm` library tests cover resident/copy f64 parity, both bounds/SIMD
controls, unchanged serial counters, owned outputs and staged partitions from
one-query batches through uneven tails and eight lanes. Browser checks cover
the real WASM SIMD kernel, pool cache growth, default policy, staged coordinates,
both hint modes, scalar/bounds controls, pool/main/generation failures and
draining, concurrent source edits/removal, and exact UI PLY exports. The complete
GPU suite also exercises device loss and all existing candidate/buffer cases.
The retained production artifacts above pass 24 native library tests,
47 software-GPU checks and 47 physical-GPU checks, plus six CPU/M3C2 browser
regression checks (100 browser checks in total). TypeScript/Vite production
build, native/WASM Clippy with warnings denied, Rust formatting, report
provenance/parity checks, local documentation links and seven invalid CLI /
overwrite guards also pass. Public file names and contents contain no GPU
model identifiers.

```powershell
# From rust/:
cargo test -p ca-wasm --lib
cargo clippy -p ca-wasm --all-targets -- -D warnings
$env:RUSTFLAGS = '-C target-feature=+simd128'
cargo clippy -p ca-wasm --target wasm32-unknown-unknown -- -D warnings
# From web/, against the production build:
$env:PW_PORT = '4193'
npx playwright test e2e/gpu-c2c.spec.ts --workers=1
$env:CA_GPU_HARDWARE = '1'
npx playwright test e2e/gpu-c2c.spec.ts --workers=1
# CPU/M3C2 regression selection:
npx playwright test e2e/c2c-simd.spec.ts e2e/map-quality-simd.spec.ts e2e/m3c2-radius.spec.ts --workers=1
npx playwright test e2e/m3c2-parallel.spec.ts --grep 'resident analysis lane admits large inputs' --workers=1
```
