# First GPU batch before fresh query staging

Beginning the first GPU batch before query staging reduced the matched staged-pool duration by **4.58–8.69%** and the matched shared-verifier duration by **6.82–8.35%** across four real-file reports. Against the existing streaming pool, shared overlap ranged from **2.17% shorter to 0.70% longer**, and staged-pool overlap ranged from 1.69% shorter to 0.60% longer. This does not establish a consistent improvement over the current application strategy. Keep the production defaults.

The [previous fresh-query study](wasm-shared-gpu-fresh.md) awaited full query staging before beginning GPU work. This follow-up changes the scheduling order in the standalone driver. The shader, exact f64 verifier, ordered statistics, compiled production WASM and experimental shared WASM are unchanged. No application integration or isolation registration is added.

## Operation boundary and controls

Each request creates a fresh owned `GpuC2cJob` from an already parsed cloud, stages the current queries when needed, runs the chosen pipeline, computes ordered f64 statistics and exported f32 distances, waits for query release acknowledgements and transfers its result. Immutable reference trees and GPU buffers remain cached. File parsing, reference imports, worker/device initialization, correctness oracles and diagnostic f64 exports are outside timing. As before, acknowledged cleanup differs from the regular application, which schedules pool releases asynchronously. These are standalone loaded-cloud operations, not direct timings of the exported application function.

The six controls are:

- `pool`: existing Worker pool, streaming exact query coordinates per batch.
- `pool-staged`: existing pool, full-job query staging before the first GPU batch.
- `pool-staged-overlap`: the same staged pool, with the first GPU batch started before staging.
- `shared-serial`: serial exact verification in the threaded shared binary, with queries staged first.
- `shared`: shared parallel verification, with queries staged first.
- `shared-overlap`: the same shared parallel verifier, with the first GPU batch started before staging.

Only the GPU pipeline changes dispatch order. Fixed-hint controls reuse captured physical-GPU candidates and issue no GPU dispatch; their overlap-labelled variants execute the same CPU pipeline as their matched controls. One-batch GPU prefetch continues during later CPU verification. GPU promises handle rejection immediately, and the driver retains the existing pending-batch drain before query cleanup. Failure injection and application cancellation recovery were not evaluated here.

Every GPU request checks its batch count and the first-dispatch offset against the query-stage boundaries. The offset records the host API call before rounded query packing; it is not a hardware timestamp. These fields establish scheduling order, not the amount of simultaneous GPU execution. `gpuWaitMillis` measures host promise waiting, not shader duration. Do not add phase medians to reconstruct a wall-time median.

## Real inputs and bounded protocol

The original maps and all coordinates are unchanged. The even/odd f64 PLY pairs reproduce the earlier studies byte-for-byte: PandaSet has 531,908 reference and 531,908 query points; the Autoware sample has 878,921 reference and 878,920 query points. These are measured subsets of one cloud per dataset, not independent surveys or SLAM-accuracy ground truth. Credit: PandaSet, Hesai and Scale AI (CC-BY-4.0), and the TIER IV Autoware planning sample; input specifications preserve source URLs and attribution. No generated timing coordinates or adapter identity are used.

Each report uses four actual shared lanes, batches of 131,072 points, six warmups and twelve measured trials per pipeline. Six Williams rows balance positions and directed predecessor pairs within trials; twelve trials form two complete blocks. The predeclared report order was PandaSet, Autoware, Autoware, PandaSet. The four-report budget was exhausted exactly, with every trial and long wait retained. No heavy builds ran during timing; host activity, GC, clocks and power remain uncontrolled.

## Complete GPU pipeline

Values are per-report medians in milliseconds. The serial shared control is also timed and retained in each raw report.

| Report | Streaming pool | Staged pool | Staged pool, early GPU | Shared parallel | Shared, early GPU |
|---|---:|---:|---:|---:|---:|
| pandaset-4-01 | 114.15 | 122.45 | 116.05 | 123.70 | 113.43 |
| autoware-4-01 | 156.92 | 164.78 | 157.57 | 169.79 | 158.28 |
| autoware-4-02 | 163.47 | 175.58 | 160.32 | 173.87 | 160.33 |
| pandaset-4-02 | 118.45 | 130.59 | 117.98 | 126.39 | 115.86 |

Paired percentages are medians of within-trial ratios. Negative means the early variant was shorter. They need not equal ratios of the separately calculated medians above.

| Report | Pool early / staged | Shared early / shared | Pool early / streaming | Shared early / streaming |
|---|---:|---:|---:|---:|
| pandaset-4-01 | -4.58% | -6.82% | -1.69% | -0.93% |
| autoware-4-01 | -5.07% | -7.73% | +0.60% | +0.70% |
| autoware-4-02 | -8.69% | -8.35% | -1.69% | -2.17% |
| pandaset-4-02 | -8.01% | -8.14% | +0.20% | -0.79% |

Early dispatch recovers much of the sequential staging cost. The comparison against streaming remains small and changes direction on Autoware. It does not justify an application default change. Long requests persist: staged-pool GPU operations reached 608 and 625 ms in the PandaSet reports, shared overlap reached 606 ms, and the serial shared control reached 738 ms. Host promise waits reached about 502–521 ms in these tails. No root cause or shader-time reduction is established.

## Fixed-hint control and correctness

The fixed pipeline performs no new GPU dispatch, so an overlap label does not implement an optimization there. Its per-report medians were:

| Report | Streaming pool | Staged pool | Pool overlap label | Shared serial | Shared parallel | Shared overlap label |
|---|---:|---:|---:|---:|---:|---:|
| pandaset-4-01 | 76.26 | 79.99 | 79.91 | 201.10 | 76.96 | 80.27 |
| autoware-4-01 | 139.04 | 137.94 | 131.66 | 368.26 | 137.53 | 145.92 |
| autoware-4-02 | 122.68 | 127.24 | 127.04 | 327.20 | 125.96 | 132.83 |
| pandaset-4-02 | 74.87 | 84.84 | 77.19 | 191.34 | 74.09 | 80.53 |

Despite identical fixed-hint work, paired differences between matching labels reached −5.71% for the pool and +5.18% for shared verification. Order, state and host effects are not fully controlled by the schedule. This negative control reinforces the limit on interpreting the small differences against streaming; there is no fixed-hint overlap speedup claim.

Every report checks full f64-bit parity against complete CPU search and stable-serial verification outside warm trials. Every measured request checks displayed f32 bits, ordered statistics, certificate counts, cached references, increasing owner/query generations and acknowledged cleanup. Distance hashes, candidate hashes and certificate counts agree across both repetitions and all prior GPU packets. All six alternatives coexist; committed WASM pages are high-water counters, not isolated application RSS.

## Evidence and reproduction

The [manifest](../benchmarks/wasm-shared-gpu-overlap/manifest.json) contains four raw reports, two LF input specifications, 111 normalized source fingerprints, compiler and artifact provenance, the predeclared protocol and reused-build records. It maps the two modified PR337 sources to preserved copies. Prior packets and their raw reports remain unchanged. The integrity test validates all sixteen GPU timing reports, preserved source versions, exact input specifications and scheduling/lifecycle boundaries.

Build using the [experiment instructions](../web/experiments/shared-gpu-c2c/README.md), then run from `web/` with new output paths:

```sh
node scripts/benchmark-shared-gpu-c2c.mjs --output NEW.json --reference reference.ply --compared compared.ply --input-manifest inputs.json --threads 4 --warmups 6 --repeats 12 --batch-size 131072 --gpu physical --jobs fresh --overlap compare
node --test tests/shared-gpu-c2c-order.test.mjs tests/shared-gpu-packets.test.mjs
```

`--overlap off` preserves the four-mode schedule; `--jobs prepared` remains the default lifecycle. Overlap comparison requires fresh jobs. CI runs the existing prepared and fresh smokes plus a six-mode fresh smoke with GPU disabled. Those checks cover packaging, static-host isolation, serial fallback and CPU results; physical-GPU scheduling and timings come from the separate real-file reports. The remaining work is to reduce query copies or test application integration against streaming with a consistent whole-operation benefit.
