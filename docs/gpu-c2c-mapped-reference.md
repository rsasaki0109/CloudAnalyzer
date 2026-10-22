# Immutable mapped reference uploads

The [mapped reference packet](../benchmarks/wasm-gpu-mapped-reference/manifest.json) compares queued writes with `mappedAtCreation` initialization for immutable reference points and ordinary flat nodes through the production distance Worker. The candidate stays opt-in: two of four reports passed the frozen joint gate, so queued upload remains the default.

The preceding [preparation profile](gpu-c2c-reference-preparation.md) found long error-scope completion waits after host upload calls. It did not isolate physical transfer time or predict this candidate's benefit. This study compares complete RPCs without diagnostic instrumentation or GPU timestamp negotiation.

Four independent browser sessions use the frozen PandaSet, Autoware, Autoware, PandaSet order, four exact CPU workers, 131072-query batches and ordinary 32-slot traversal. Each session has six warm-up pairs, twelve balanced primary warm pairs, then four balanced cold pairs. Every cold operation reloads the reference outside the RPC and includes complete reference/tree/export/upload/verifier preparation; its immediate cached follow-up is also retained. The ordinary device, pipeline and CPU pool are prepared beforehand. The inputs reproduce the same real measured even/odd subsets byte for byte, with no generated coordinates or noise.

Every report had to achieve median paired cold RPC reduction of at least 5% with at least three of four cold wins, median paired primary warm ratio at most 1.02, and median paired immediate cached-follow-up ratio at most 1.05. All four reports had to pass. Positive cold reduction means faster; positive warm change means slower.

| Session | Cold paired reduction | Cold wins | Primary warm change | Joint gate |
|---|---:|---:|---:|---|
| pandaset 1 | 16.94% | 4/4 | -3.92% | pass |
| autoware 1 | 4.84% | 4/4 | +2.39% | fail |
| autoware 2 | 46.69% | 3/4 | +0.74% | pass |
| pandaset 2 | -13.89% | 1/4 | +2.71% | fail |

Twelve of sixteen cold pairs were faster, but the repeated cold effect ranges from -13.89% to +46.69%, and primary warm regressions reach 2.71%. All 32 cold operations, 32 immediate cached follow-ups and 96 primary warm operations retain exact indexed-CPU f32 distance bits, ordered statistics and certificate counts. No tail is discarded or additional timing session added. These measurements do not establish a stable acceleration result or explain browser scheduling, driver or GC causes.

Mapped mode copies exactly the source typed array's byte offset and length into a newly mapped immutable point or node buffer, then unmaps before use. Initial and lazily selected flat/packed formats use the same helper, including cleanup if allocation, mapping, byte copy, unmap or scope completion fails before the caller receives the buffer. Node formats share points and batch slots within one search. All WGSL shader bytes and authoritative f64 verification remain unchanged. Query/readback and optional box/parent uploads keep their existing paths.

The request control `gpuMappedReference` defaults to false; compute metadata reports `mappedReference`. Search caches distinguish upload mode and diagnostic setting, while the ordinary GPU device and exact verifier pool remain shared. At most two upload modes by two diagnostic settings exist per immutable reference. The warm comparison retains both modes' separate point/node/batch caches. Selected/retained node bytes describe each search and do not represent total device memory or measured traffic. Source edits, retired leases and failed/device-lost references retain their normal cleanup and fallback behavior.

Validation covers both node formats, nonzero source offsets and independent upload bytes, all workgroup sizes and traversal entry points, lazy shared node preparation/retries, buffered mutations and concurrent maps, injected mapping/unmap/allocation/scope failures and complete CPU fallback after device loss. The direct real 261-query fixture compares 7830 raw hint slots and checks 32 full f64 results against brute force per run. Flat and packed software/physical Worker smokes are compatibility checks excluded from adoption timings. The historical 154 packet files remain byte-identical, and executed runtime/harness versions are preserved.

An ordinary production build record from `build-gpu-frontend.mjs` or `build-query-export.mjs` is required. A small compatibility run uses a new output path:

```sh
node web/scripts/benchmark-mapped-reference.mjs --output mapped-smoke.json --build-info normal-build.json --reference web/public/samples/lidar_reference.pcd --compared web/public/samples/lidar_candidate.pcd --threads 2 --warmups 0 --repeats 1 --cold-pairs 1 --nodes flat --gpu software
```

`--nodes packed` checks the separate packed-node combination. The frozen adoption study uses flat nodes. Rust sources and all four WASM/glue artifacts were unchanged and checked before rebuilding the frontend; CI also compiles a fresh production WASM package. No adapter identity is collected. Original data is reused, and generated subsets are removed after preserving the small reports.
