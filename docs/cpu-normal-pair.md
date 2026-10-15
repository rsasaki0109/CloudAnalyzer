# Paired normal eigensolvers on WASM

Ordinary WASM normal estimation now fits two adjacent Morton-order points
with independent `f64x2` Jacobi eigensolver lanes. Each point keeps its own
rotation decisions and convergence state. The paired path is enabled by
default with `USE_NORMAL_PAIR_EIGEN = true`.

Across two passes on three unchanged measured clouds, whole-normal-estimation
paired median reductions are **5.69–11.42%**, with **55/60 timed pairs** faster.
All normal f32 bits and complete PLY exports with normal attributes match the
preceding individual fits. This accelerates CPU/WASM work in the existing pool.

## Computation and exactness

The preceding compiled WASM already unrolls all three single-matrix Jacobi
rotations and automatically pairs x/y centroid arithmetic. Those operations
do not need an additional single-point SIMD wrapper. The new kernel instead
places one complete 3×3 covariance matrix in each SIMD lane.

Neighbour queries remain in the same Morton order, with the same exact tree,
insertions, ties and original-index normalization. Two result buffers retain
the two neighbourhoods while their moments are computed independently with
the historical divide/add and covariance order. Both searches run before the
two fits; this scheduling change can also affect cache behavior.

The eigensolver preserves separate multiply/add/divide operations, row/column
rotation order (0,1), (0,2), (1,2), the `1e-300` rotation skip, per-point
`1e-30` convergence criterion and maximum 50 sweeps. Converged lanes retain
their previous matrix/eigenvector bits while the other lane continues. A
rotation mask also preserves a lane when only its partner needs that rotation.
When both lanes rotate, the full-mask branch avoids those preservation selects.

Only finite covariance entries with absolute value at most `1e150` enter the
paired solver. Too few hits or an unsafe matrix uses the unchanged individual
normal kernel for that pair. An odd tail does too. Eigenvalue selection,
collinearity checks, orientation and final scalar normalization retain their
historical arithmetic. No f32 approximation is introduced before normal output.

Pairing selects ordinary retained-scratch WASM queries (`reuse:true`,
`simd:false`, `bounds:false`). Explicit neighbour-SIMD/bounds controls,
allocating controls, native normal estimation and GPU-hinted paths keep their
individual kernels. Native tests run a portable two-lane implementation of
the same rotation/mask schedule, checking all eigenvalue/eigenvector f64 bits.
M3C2, Map quality and ICP's own eigensolver callers are unchanged.

One additional nearest-k buffer and fixed-size paired working matrices belong
to each normal calculation; they are reused across points and freed on return.
At k=24 the additional live hit payload is 384 bytes before allocator slack.
Prepared capacities and their 128 MiB idle payload policy are unchanged;
this is not a total process-memory bound. Normal outputs are recomputed on
every request. No additional result cache or public request field is added.
The existing pool retains its part-local neighbourhood definition at borders;
this comparison preserves that partitioned operation, not a new global fit.

## Measurements

Inputs are the full unchanged `compared.ply` clouds from
[measured-source preparation](c2c-real-source.md): PandaSet 019 (531,908 points),
Autoware planning (878,920) and Tokyo Takanawadai (941,933). No generated
performance inputs, displacement, noise or further thinning is added. Original
preparation, attribution, licence and hashes are embedded in each report.
These subsets are not independent surveys or normal-accuracy ground truth.

Measurements use Windows/i7-9750H, 12 logical CPUs, 32 GiB RAM and Chromium
153.0.8010.12. Two independent production workers compare the preceding
individual build with the paired build, with k=24, upward orientation,
`knnCache:true` and `knnBounds:false`. Part counts and preparation capacities
match, cache jobs stay stable and reuse counters increment exactly. A third
archived worker supplies one excluded normal/PLY oracle and terminates before
timing. Source order is PandaSet → Autoware → Tokyo initially, then reversed.

One excluded first-use pair and five excluded alternating warm-up pairs
precede ten alternating timed pairs per source/pass. Whole RPC includes
nearest-k search, moments, eigen fits, orientation, normalization, messages,
joins and storing the normals. Startup/loading, cache diagnostics, exports
and validation are excluded. First use is not a cold median. No builds,
tests or repository scans run alongside timing.

| Source/pass | Individual median, ms | Paired median, ms | Median paired shortening | Paired wins |
| --- | ---: | ---: | ---: | ---: |
| PandaSet initial | 368.05 | 331.70 | +9.26% | 9/10 |
| PandaSet final | 373.35 | 339.10 | +9.83% | 10/10 |
| Autoware initial | 521.25 | 463.85 | +10.01% | 9/10 |
| Autoware final | 586.05 | 525.55 | +11.42% | 10/10 |
| Tokyo initial | 485.10 | 447.75 | +7.93% | 9/10 |
| Tokyo final | 611.35 | 545.05 | +5.69% | 8/10 |

Paired shortening is `median(100 * (1 - paired_ms / individual_ms))`, not the
ratio of marginal medians. All spikes and losses are retained. Tokyo's second
pass has large timing variation, including a 1,017/1,042 ms timed pair. CPU
clocks, power, scheduling, GC and other applications are uncontrolled. These
whole-call comparisons do not isolate SIMD instruction savings from grouping,
compiler layout, cache behavior or other costs, and do not predict every
workload or other k. Six positive paired medians and 55/60 wins support the
default paired path in the tested workloads.

All 96 first-use/warm-up/timed pairs preserve every normal f32 bit and every
PLY byte: 192 paired outputs plus six excluded archived oracles. Point counts
remain unchanged. Cache payloads remain 35,138,912 / 58,041,728 / 62,200,604
bytes across five/eight/eight parts. All six reports use the same source,
script and build fingerprints, independently hash-audited against the saved
adopted source/build. Input, manifest, original measured source, thirteen
implementation files, script and both worker/pool/WASM builds are checked
before and after each run. No GPU adapter/model metadata is collected.

## Validation and reproduction

The candidate passes 384 native tests with the parallel feature, native/WASM
Clippy with warnings denied, formatting and release WASM/TypeScript/Vite builds.
The new solver regression compares 3,072 individual decompositions with all
eigenvalue/eigenvector f64 bits, covering different convergence/skip decisions,
zero/tiny/large scales and unsafe-input rejection. The paired normal pipeline
additionally checks odd tails, short neighbourhoods, both orientations,
duplicates, georeferenced coordinates and nonfinite fallback against allocating
individual fits. Generated fixtures test correctness only.

Fifteen targeted browser tests pass. A new archived-WASM oracle checks 504
count/geometry/k/bounds cases, including both orientations, odd/even counts,
fallbacks, retained capacities and every normal bit. It requires optional
`NORMAL_PAIR_BASELINE` and skips when no independent snapshot is supplied.
Existing regressions cover exact SOR fields/exports, normal PLY, budgets,
source edits/removal, worker loss/recovery, joins and prior computations.
Raw linked LLVM WASM inspection before `wasm-opt` finds the paired arithmetic
inlined into the prepared normal kernel, including vector sqrt and preservation
select instructions. This confirms emission, not isolated execution time.
Script syntax and nine invalid-option/overwrite/manifest controls pass.

- PandaSet: [initial](../benchmarks/cpu/normals/real-source/windows-pandaset019-paired-eigen-initial.json), [final](../benchmarks/cpu/normals/real-source/windows-pandaset019-paired-eigen-final.json)
- Autoware: [initial](../benchmarks/cpu/normals/real-source/windows-autoware-planning-paired-eigen-initial.json), [final](../benchmarks/cpu/normals/real-source/windows-autoware-planning-paired-eigen-final.json)
- Tokyo: [initial](../benchmarks/cpu/normals/real-source/windows-tokyo-takanawadai-paired-eigen-initial.json), [final](../benchmarks/cpu/normals/real-source/windows-tokyo-takanawadai-paired-eigen-final.json)
- [Benchmark script](../web/scripts/benchmark-normal-pair.mjs)
- [Archived-WASM test](../web/e2e/normal-pair.spec.ts)
- [Paired solver](../rust/crates/ca-core/src/normal_pair.rs)

Archive the individual production build first, then build matching paired
release WASM/app with `USE_NORMAL_PAIR_EIGEN = true`. Run from the repository
root with full measured inputs and a new output in an existing parent:

```powershell
node web/scripts/benchmark-normal-pair.mjs --baseline-dist notes/normal-pair-baseline-20261013/dist --input notes/c2c-real-pandaset/compared.ply --input-manifest notes/c2c-real-pandaset/inputs.json --output notes/NEW-pandaset-paired-normal.json
```

The source-flag guard cannot identify binary mode, so a matching rebuild is
required. Reports cannot be overwritten. `--repeats` accepts 1..100, default
ten. Another checkout needs equivalent archived artifacts and local measured
inputs. For the browser oracle, supply `NORMAL_PAIR_BASELINE` as an absolute
snapshot root containing `dist` and its matching `web/src/wasm/ca_wasm.js`.
