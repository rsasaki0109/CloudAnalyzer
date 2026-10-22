# Fresh-reference GPU C2C preparation profile

The [reference preparation packet](../benchmarks/wasm-reference-preparation/manifest.json) localizes the long fresh-reference readiness intervals seen in the packed-node comparison. This is an explanatory study. It changes no production backend or default and establishes no acceleration result.

An isolated Vite build inserts bounded host phase events into the ordinary distance Worker at three identified source files. The original production sources, all WGSL shader strings, exact f64 verification, and nine normal build artifacts remain unchanged. The build records original, transformed and bundled hashes and reuses four byte-identical WASM/glue artifacts only after checking the complete Rust source set. Probe source and executed helper versions are preserved in the packet.

The frozen four-report order is PandaSet, Autoware, Autoware, PandaSet. Four exact CPU workers and 131072-query batches process the same hash-identified real even/odd point subsets. Each report runs two balanced cycles of flat/packed nodes with logging off/on. Every arm gets a newly loaded reference, one complete cold C2C RPC, and an immediate cached follow-up. The CPU pool, ordinary GPU device and both node pipelines are prepared beforehand; parse/index time is outside the RPC. All 32 cold operations and 32 cached follow-ups retain exact indexed-CPU f32 distance bits, ordered statistics and stable per-mode certificates. No timestamp-query feature is requested.

Nine of the sixteen instrumented cold operations had node readiness of at least 50 ms. In all nine, error-scope pop completion covered 89.99–99.52% of that readiness interval. The intervals occurred in both node formats. The serialized scope-entry queue was at most 0.10 ms in these samples; the long interval followed the host upload calls.

The synchronous reference-tree construction is another material cold cost. The following medians pool eight instrumented cold observations per dataset across both node formats. Units are milliseconds. These overlapping spans must not be added into a total.

| Real input | f64 tree | Query snapshot | Verifier snapshot export | Candidate export | Scope pop median | Scope pop maximum |
|---|---:|---:|---:|---:|---:|---:|
| pandaset | 95.35 | 1.20 | 8.80 | 4.45 | 4.15 | 396.70 |
| autoware | 161.55 | 2.00 | 13.25 | 8.35 | 263.00 | 412.50 |

Observed host point-write calls were at most 2.60 ms and node-write calls at most 3.10 ms. The eight packed-node validation/encoding spans ranged from 3.60 to 27.50 ms. These are host wall-clock call spans, with browser scheduling and GC potentially included. Closed error scopes do not identify isolated physical transfer time or prove that uploads caused the delay.

Logging can perturb an operation, and cold/warm tails are large. For example, Autoware includes an 868.8 ms logging-off cold operation and a 674.6 ms logging-on cached follow-up; both remain in the raw reports. The cached tail is outside fresh-reference setup and cannot be explained by the initial-upload interval alone. On/off arms run the same instrumented binary, with phase clock reads skipped when off; this does not quantify guard overhead against a byte-identical uninstrumented binary. No cold speedup or adoption decision is made from these observations.

Phase records use one Worker `performance.now()` clock, include scope identifiers, and are capped at 256 events. Tests verify every shader string, zero additional phase clock reads when disabled, independent bounded snapshots, fail-closed source matching, serialized scopes and cleanup after work/validation failure. Software and physical sample smokes also check exact results, selected/retained node bytes, one reused GPU device, every initial phase, cached preparation reuse and complete scope closure. The ordinary application does not load the probe.

A sample compatibility run uses new output paths:

```sh
node web/scripts/build-reference-preparation.mjs --output prep-build.json --out-dir web/prep-profile-dist --wasm-build-info normal-build.json
node web/scripts/benchmark-reference-preparation.mjs --output prep-smoke.json --build-info prep-build.json --reference web/public/samples/lidar_reference.pcd --compared web/public/samples/lidar_candidate.pcd --threads 2 --cycles 1 --row 0 --gpu software
```

`normal-build.json` must identify the current generated WASM package and unchanged Rust sources, for example from `build-query-export.mjs`. The builder refuses existing output paths. The sample smoke checks compatibility and is excluded from the frozen real-data budget.

A next candidate is immutable reference upload through `mappedAtCreation`, followed by a new bounded complete-RPC comparison if correctness holds. The current profile does not predict that candidate's benefit. CPU tree construction remains a separate target. No physical occupancy, spill, total device-memory or adapter-identity data is collected.
