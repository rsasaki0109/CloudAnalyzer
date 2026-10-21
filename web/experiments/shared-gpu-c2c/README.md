# Shared exact GPU C2C verification experiment

This separate WASM package checks production WebGPU nearest-neighbour hints against one immutable f64 k-d tree using Rayon. The tree's validated snapshot preserves candidate slots. Invalid slots and failed certificates use the same complete exact search as production. Queries stay in WASM for a job; only hints and verified distances cross the worker boundary per batch.

The regular application and its existing verifier pool remain unchanged. This experiment needs cross-origin isolation; its service worker is scoped to the experiment. The serial build supports initialization without isolation.

From `web/`, after installing the normal dependencies and building production WASM (`npm run wasm`):

```sh
rustup toolchain install nightly-2026-10-20 --profile minimal --component rust-src --target wasm32-unknown-unknown
# Install a matching wasm-bindgen-cli 0.2.129, or set CA_WASM_BINDGEN to its path.
node scripts/build-shared-gpu-c2c.mjs serial
node scripts/build-shared-gpu-c2c.mjs threads
node node_modules/typescript/bin/tsc --noEmit -p experiments/shared-gpu-c2c/tsconfig.json
node scripts/build-shared-gpu-driver.mjs
node scripts/benchmark-shared-gpu-c2c.mjs --output NEW.json --reference reference.ply --compared compared.ply --threads 4 --warmups 5 --repeats 10
```

Builders refuse existing output folders. `CARGO_TARGET_DIR`, `CA_CARGO` and `CA_WASM_BINDGEN` can point to scoped caches/tools. The shared memory has a 1 GiB maximum. Both builds enable SIMD. Native checks use `cargo +stable fmt`, `clippy --all-targets --features threads -- -D warnings`, and `test` with and without `--features threads`, passing this crate's manifest path.

The measurement script requires a physical WebGPU adapter by default. `--gpu disabled` runs a correctness/packaging smoke with full-search sentinel hints and is never labelled a GPU timing comparison. CI checks both query lifecycles on small bundled inputs. No adapter identity is collected.

By default (`--jobs prepared`), four controls share one owner job: the existing pool with streaming coordinates, the existing pool with staged coordinates, serial verification in the threaded binary, and shared parallel verification. Fixed-hint trials hold every hint constant; prepared-job GPU trials rerun dispatch/readback with one-batch prefetch. Imports and query preparation are outside timed requests; the immutable job is reused. These are not complete application-operation measurements. Four Williams rows balance positions and immediate predecessors within each full block. Stable serial is checked outside timing. Every trial must preserve f32 distances, ordered f64 statistics and certificate counts; untimed checks also compare every f64 bit against full search.

Use `--jobs fresh --repeats 12` to measure new query jobs against the cached immutable reference. Each request creates an owned production query snapshot from the already parsed cloud, stages queries when needed, computes the selected pipeline and summaries, and releases its queries before returning. Shared jobs use `setQueries` and `releaseQueries` on the existing tree. The report checks increasing job/query generations, a fixed reference generation, and acknowledged release. Pool release acknowledgements are awaited for measurement; the application normally schedules them asynchronously. Parsing and initial reference preparation stay outside timing. Twelve measured trials give three complete Williams blocks. See [fresh-query measurements](../../../docs/wasm-shared-gpu-fresh.md).

This prototype does not implement application cache integration, cancellation recovery, or automatic selection. The driver serializes requests and drains GPU work before release; terminating its worker cancels the experiment. Do not select a production default from a verification-only timing or a WASM committed-page counter. See [measurement notes](../../../docs/wasm-shared-gpu-c2c.md).
