# Shared-memory WASM C2C experiment

This standalone module evaluates a shared reference tree, query snapshot and
cached Morton order on Rayon Web Workers. The normal application continues to
use its existing independent WASM workers and WebGPU backends. Neither its
service-worker registration nor its Pages deployment changes.

[Measured results and limitations](../../../docs/wasm-shared-c2c.md) compare it
with the existing worker pool; the shared path remains experimental.

The module uses `ca-core::distance::PreparedC2c`. Its opt-in `distances_par`
method splits the retained order into Morton chunks, gives each chunk its own
nearest-hit guess, and restores every result to original query order. All lanes
read the same immutable tree and coordinates. Each call computes fresh distances
and ordered f64 summaries; returned display distances are f32.

The static-host isolation mechanism is informed by
[visloc-rs PR 311](https://github.com/rsasaki0109/visloc-rs/pull/311).
The scoped `coi-sw.js` adds COOP/COEP to navigations and worker responses in
the experiment. The benchmark runs below `/CloudAnalyzer/app/shared-c2c/` on a
plain local HTTP server, with one initial reload. It also checks that an
unisolated worker rejects the threaded build and can initialize the serial
build. This is a local static-host test, not a deployed Pages/browser matrix.

## Build

Install stable Rust with `wasm32-unknown-unknown`, the fixed nightly with
`rust-src`, and `wasm-bindgen-cli` matching the locked version:

```sh
rustup toolchain install nightly-2026-10-20 --profile minimal --component rust-src --target wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.129 --locked
cd web
node scripts/build-shared-c2c.mjs serial
node scripts/build-shared-c2c.mjs threads
```

The script uses stable for `pkg-serial` and nightly-2026-10-20 for `pkg-threads`.
Both enable fixed-width SIMD. The latter rebuilds `std` with atomics, imports a
shared memory capped at 1 GiB and uses `wasm-bindgen-rayon`'s no-bundler helpers.
Generated packages and Cargo targets are ignored. Existing output directories
are refused; move an old experiment package aside before rebuilding.
`CA_CARGO`, `CA_WASM_BINDGEN` and `CARGO_TARGET_DIR` can select existing tools
and an isolated build directory. Build metadata records compiler versions,
flags, raw and normalized source hashes, and generated asset hashes.

## Compare measured files

Build the normal application first (`npm run wasm && npm run build`). Prepare
unchanged measured coordinate subsets as described in
[the real-source C2C protocol](../../../docs/c2c-real-source.md), then run from
`web/`:

```sh
node scripts/benchmark-shared-c2c.mjs --output ../notes/NEW-shared-c2c.json --reference ../notes/PAIR/reference.ply --compared ../notes/PAIR/compared.ply --input-manifest ../notes/PAIR/inputs.json --threads 4 --warmups 5 --repeats 10
```

The four rotated controls are stable serial, serial in the threaded binary,
the production worker pool, and shared parallel. Both standalone controls use
the identical indexed f64 coordinates exported by the production worker.
Untimed diagnostics compare every authoritative f64 distance bit between the
two builds and both thread modes. Every timed request must preserve all f32
distance bits and f64 summary fields against the production result. Tests also
cover failed replacement, release and serial-build rejection of parallel work.

Cold pool startup, module/thread initialization, snapshot preparation and first
requests are recorded separately. Warm request times include search, exact
summaries, output copies and RPC. Production keeps spatial partitions; the
shared alternative keeps one full tree, so their comparison includes both
partition and scheduling differences. A served-only `hardwareConcurrency`
override caps the production pool; the application files remain untouched.
The query-count threshold can select fewer production workers than requested.

Retained vector capacity and committed WASM memory are separate measurements.
Committed pages are a high-water allocation, excluding JavaScript, native
worker stacks and browser overhead. Both alternatives coexist during paired
timing; these values do not establish total application memory savings.
Five warmups and ten trials constitute a timing comparison. Smaller runs are
labelled compatibility smoke tests. Measured subsets of one source are not
independent surveys or SLAM accuracy ground truth.

CI builds both variants and exercises the static-host path on the existing
RELLIS-derived demo files. It checks compatibility and output parity; its
single trial is not performance evidence.

The separate [CPU tree collection screen](../../../docs/cpu-kdtree-collection.md)
builds two ordinary stable/SIMD production WASM variants in one isolated source
directory. It changes only the final point/index collection and requests no
shared memory or GPU. Its real-data constructor timings did not pass the frozen
gate, so no production implementation or backend selection changes. This CI
workflow also builds those variants and checks exact real-sample compatibility.
