# Shared GPU C2C verification with fresh query jobs

The shared verifier did not improve GPU operation time when each request owned new query data. Across four real-file reports, its median paired GPU-pipeline duration was **4.87–7.04% longer than the existing streaming verifier pool**, and 0.34–1.36% longer than the staged pool. Keep the existing application defaults. This is a standalone measurement increment, with no application cache integration or service-worker registration change.

The [previous prepared-job measurements](wasm-shared-gpu-c2c.md) excluded per-job query preparation and reused one immutable job. Their small GPU-pipeline advantage did not establish a speedup for repeated user operations. This follow-up measures a different boundary; do not pool the two studies.

## Timing boundary and controls

The reference k-d tree, GPU buffers and existing pool references are prepared once. The query cloud is already parsed, as it is when an operation uses an existing loaded cloud. Every measured request then:

1. Creates a new production `GpuC2cJob`, owning an exact coordinate snapshot.
2. Stages that request's queries when the chosen verifier requires it.
3. Runs fixed captured hints or the physical WebGPU dispatch/readback pipeline, with one-batch prefetch.
4. Computes the same ordered f64 statistics and exported f32 distances.
5. Releases the query state, waits for worker release acknowledgements and transfers the result back.

File parsing, reference import, worker/device initialization and diagnostic f64 exports are excluded. This measures a computation on loaded clouds; it does not establish cold application startup time.

The four controls are the production pool with streaming queries, the same pool with queries staged per request, serial verification in the threaded shared binary, and shared parallel verification. Shared serial and parallel use the same binary, cached reference and query layout. All modes create a fresh owner job. The shared setter consumes its incoming WASM allocation without another Rust coordinate copy; JS/WASM copies and worker transfers remain timed.

The experiment awaits one release acknowledgement per pool lane or the shared worker, rather than allowing releases to overlap a later request. The regular application schedules pool query releases asynchronously. The measured boundary deliberately completes cleanup before the next request, so these times are not a direct benchmark of the application's exported function.

Increasing owner job sequences and shared query generations must be recorded on every fresh request. The shared reference generation must stay at one, pool references must remain cached, and a verification after query release must fail before the first fresh request. Invalid query replacements must preserve the previous valid job.

## Real inputs and protocol

The inputs contain unchanged measured coordinates split into even and odd source indices, using the same byte-identical f64 PLY pairs as the prior study:

- PandaSet: 531,908 reference and 531,908 query points. Credit: PandaSet, Hesai and Scale AI; CC-BY-4.0.
- Autoware sample map: 878,921 reference and 878,920 query points. Credit: TIER IV's sample map; the input specification retains its source URL and attribution.

These are subsets of one cloud per dataset, not independent surveys or SLAM-accuracy ground truth. No generated coordinates are used for timing. Each complete report uses four actual shared lanes, five warmups, twelve measured trials and batches of 131,072 points. Twelve trials form three complete Williams blocks, balancing positions and directed predecessor pairs within trials. Report order was PandaSet, Autoware, Autoware, PandaSet. The predeclared four-report timing budget was used exactly; no trials or long waits were removed. No heavy builds ran during measurement. Other host activity, GC, clocks and power remain uncontrolled.

The optimized production WASM binary is reused byte-for-byte from the prior study, with all production sources checked unchanged. Experimental stable and fixed-nightly builds include SIMD; the threaded build also includes atomics and the shared-memory linker configuration. Compiler and package fingerprints are in every raw report. No adapter identity is collected.

## Full GPU pipeline

Values are per-report medians in milliseconds. Positive paired percentages mean the shared verifier took longer. Paired percentages are medians of trial-by-trial ratios, not ratios of independently computed medians, so their direction can differ from the adjacent median columns.

| Report | Streaming pool | Staged pool | Shared parallel | Shared / streaming paired | Shared / staged paired |
|---|---:|---:|---:|---:|---:|
| pandaset-4-01 | 112.86 | 115.21 | 118.14 | +7.04% | +1.36% |
| autoware-4-01 | 150.50 | 159.47 | 160.74 | +5.24% | +0.39% |
| autoware-4-02 | 153.76 | 162.03 | 162.42 | +5.55% | +0.34% |
| pandaset-4-02 | 111.58 | 117.21 | 115.88 | +4.87% | +1.31% |

Before dispatch, the median query staging interval was:

| Report | Staged pool (ms) | Shared parallel (ms) |
|---|---:|---:|
| pandaset-4-01 | 5.06 | 7.45 |
| autoware-4-01 | 8.78 | 11.84 |
| autoware-4-02 | 8.20 | 11.81 |
| pandaset-4-02 | 5.19 | 6.94 |

The streaming pool has no full-job staging interval: its coordinate packing and transfers happen within batch verification. Zero upfront staging therefore does not mean zero query-copy cost. Do not add medians from separate phase distributions to reconstruct a wall-time median.

Long requests remain visible in the raw reports: shared parallel GPU requests reached 619–678 ms in three reports; the streaming pool reached 662 ms in another, and the staged pool reached 648 ms. Host `gpuWaitMillis` reached roughly 518–556 ms in these long requests. This measures time awaiting a JavaScript promise, not GPU shader execution time, and does not identify the cause. GPU prefetch can overlap CPU verification, so their intervals must not be summed as disjoint work.

## Fixed captured hints

This control excludes repeated GPU dispatch/readback but includes fresh query preparation and release. Same-binary serial/parallel ratios show that the exact verifier itself can run in parallel; they do not measure an application improvement over an already parallel pool.

| Report | Streaming pool (ms) | Staged pool (ms) | Shared serial (ms) | Shared parallel (ms) | Paired serial / parallel |
|---|---:|---:|---:|---:|---:|
| pandaset-4-01 | 73.18 | 81.03 | 186.88 | 77.51 | 2.44x |
| autoware-4-01 | 92.86 | 94.85 | 236.65 | 93.82 | 2.62x |
| autoware-4-02 | 96.03 | 98.19 | 248.50 | 93.87 | 2.63x |
| pandaset-4-02 | 73.40 | 68.70 | 176.80 | 71.47 | 2.47x |

Every report passed untimed full f64-bit comparisons against complete CPU search, stable-serial parity, and displayed f32 distances, ordered summary fields and certificate counts on every timed request. Candidate hashes, distance hashes and certificate counts agree across the two repetitions of each dataset and the prior prepared-job data. WASM committed-page counters are high-water values; all alternatives coexist in these reports, so no isolated application RSS reduction is established.

## Evidence and reproduction

The [packet manifest](../benchmarks/wasm-shared-gpu-fresh/manifest.json) includes all four raw reports, input specifications, 110 normalized source fingerprints and compiler/binary provenance. It also maps the four changed sources from the previous packet to preserved copies; the previous reports and manifest are unchanged. The packet integrity test validates both studies against their executed sources and checks the fresh-job boundaries.

Build the packages as described in the [experiment README](../web/experiments/shared-gpu-c2c/README.md). From `web/`, with a new output path and the verified real input pair:

```sh
node scripts/benchmark-shared-gpu-c2c.mjs --output NEW.json --reference reference.ply --compared compared.ply --input-manifest inputs.json --threads 4 --warmups 5 --repeats 12 --gpu physical --jobs fresh
node --test tests/shared-gpu-c2c-order.test.mjs tests/shared-gpu-packets.test.mjs
```

`--jobs prepared` remains the default for compatibility. CI runs both lifecycles with `--gpu disabled` on bundled measured inputs, checking the experiment-scoped static-host isolation and serial initialization fallback. Those smoke reports are not GPU performance evidence. Production integration, recovery, cancellation and automatic selection remain separate work.
