# Compact occupancy for octree indexing

A compact bit table preserves the tested point order, geometry, octree nodes
and attributes, while reducing the default occupancy table from **8 MiB to
256 KiB**. Whole browser timings were inconsistent, so **generation stamps
remain standard**: `COMPACT_CELL_MARKS = false`.
The byte bitmap remains available through the native comparison API
`Octree::build_for_cloud_with_occupancy(..., true)`.

Across two passes on three measured clouds, the byte candidate was faster in
**30/60 load pairs**, **38/60 transform pairs** and **30/60 ICP pairs**.
The worker's load-index phase was faster in **33/60 pairs**. Some transform
comparisons improved, but load and ICP gains did not reproduce consistently.
These observations do not justify enabling the bitmap for all index builds.
Existing CPU/GPU acceleration and ICP reference reuse remain enabled; the
earlier [ICP matrix experiment](cpu-icp-normal-equations.md) remains disabled.

## Implementation and precision

The existing Morton quantization, cube arithmetic, node traversal, first-point
choice, swaps and child partitioning are unchanged. Generation stamps use one
u32 per lattice cell to distinguish each node. The experiment instead starts
each node with a cleared bitmap and marks one bit per occupied cell. Clearing
before a child is safe because its parent has finished selecting its own
points and never consults that occupancy again.

The initial implementation used u64 words and stored the OR result for every
point, including duplicates. The refined byte version uses u8 entries and
writes only when the bit was previously clear. The smaller operand avoids
64-bit shift work in this WASM path; this is an implementation choice, not an
isolated machine-instruction speed claim.

Bitmap allocation starts at the first non-leaf node. At deep nodes the live
length is truncated to the required number of cells, cleared, then extended
with zeros if necessary. Capacity is reused across siblings and released with
the builder. With the standard grid=128, the maximum table contains
128³ bits, or 262,144 bytes. A leaf-only build allocates no bitmap. The
historical control retains its generation-stamp allocation and update policy.

This table is transient builder scratch, not retained octree or cache storage.
The 32-fold reduction refers only to occupancy storage, not total process
memory or elapsed time. Points, Morton codes, order arrays, attributes, nodes,
worker heaps and concurrent builds still need their existing storage.

Both the serial cloud builder and bucket subtree builder use the standard
selection. Explicit native controls can still compare either occupancy scheme.
Public WASM methods and browser requests are unchanged. No new GPU shader,
coordinate approximation or parallel scheduling is introduced.

## Measured-source protocol

[benchmark-octree-occupancy.mjs](../web/scripts/benchmark-octree-occupancy.mjs)
compares immutable production builds from before the change and with the
candidate enabled. Each method gets two independent analysis workers and pools.
Full unchanged reference/compared files and their manifests are required;
every load checks full counts and `keepEvery=1`.

Inputs are the coordinate-only measured-source files from
[real sources](c2c-real-source.md): PandaSet 019 (531,908 compared points),
Autoware planning (878,920) and Tokyo Takanawadai (941,933). Source attribution,
licenses and input hashes are embedded in every report. They are even/odd
subsets of measured scans, not independently surveyed registration truth.
The measured load inputs are below the pool's one-million-point indexing
threshold; no real-data parallel-index speedup is claimed.

- **Load** times the complete load/parse/index/description RPC. The existing
  worker parse/index/prepare timings are also recorded; the index field
  includes the worker's awaited index call, not a separately isolated kernel.
- **Transform** reloads the unchanged compared cloud outside timing, then
  applies the actual ICP-fitted matrix and times transformation, octree
  reconstruction, description and RPC. The fitted matrix is derived once per
  report outside timing in both builds; fit results and complete transformed
  PLY must agree. No arbitrary displacement or generated coordinates are used.
- **ICP** reloads the unchanged moving cloud outside timing and times the full
  fit, transformation, octree reconstruction, description and RPC. Both builds
  explicitly retain the same reference/tree/lazy-normal cache within its
  existing 128 MiB idle capacity budget. Capacity and cold/warm reuse flags
  must match. Full matrix accumulation remains standard. Parameters are ten
  maximum point-to-plane iterations, overlap=0.8, no centroid matching and the
  unchanged maximum sample count of 50,000.

Every pair compares all render/attribute/LOD bytes, full coordinate PLY bytes,
bounds, shifts and counts. ICP additionally compares all matrix entries, RMS,
iteration counts and convergence flags. Export/verification/removal are outside
timing. Each method has one excluded first-use pair, five excluded alternating
warm-up pairs and ten alternating timed pairs. First-use observations are
single samples, not cold medians.

The initial word candidate ran PandaSet → Autoware → Tokyo. The byte candidate
ran Tokyo → Autoware → PandaSet, then the opposite order. No builds, tests or
file scans ran during timing. CPU clocks, power, garbage collection and other
applications were uncontrolled. All spikes and losses are retained. No GPU
adapter/model metadata is collected.

## Whole-RPC and index results

Times are separate medians, stamps / bits, in milliseconds. Reduction is the
median of each pair's `(stamps − bits) / stamps`, not the ratio of the separate
medians. Positive means faster; separate medians can have a different sign.

The initial u64 candidate did not establish a repeatable whole-call gain:

| Source/pass | Load RPC: median ms; paired reduction | Worker index: median ms; paired reduction | Transform RPC: median ms; paired reduction | ICP RPC: median ms; paired reduction |
| --- | ---: | ---: | ---: | ---: |
| PandaSet initial | 60.50 / 61.65; -1.17% | 39.00 / 39.70; -3.11% | 60.95 / 59.45; -4.72% | 279.95 / 280.25; -0.14% |
| Autoware initial | 99.85 / 101.20; -1.97% | 60.35 / 60.05; -0.68% | 122.65 / 139.15; -12.03% | 472.40 / 499.75; -9.37% |
| Tokyo initial | 121.45 / 115.60; +4.47% | 68.40 / 67.05; +1.71% | 113.00 / 117.45; -9.29% | 131.55 / 143.80; -8.48% |

The two byte passes share the same candidate and improve some transform
comparisons, while load and ICP remain mixed:

| Source/pass | Load RPC: median ms; paired reduction | Worker index: median ms; paired reduction | Transform RPC: median ms; paired reduction | ICP RPC: median ms; paired reduction |
| --- | ---: | ---: | ---: | ---: |
| PandaSet byte | 89.35 / 86.90; +5.53% | 56.55 / 55.35; +8.76% | 68.75 / 67.55; +2.16% | 365.35 / 366.75; +0.32% |
| PandaSet final | 77.20 / 85.65; -2.15% | 51.85 / 55.80; -4.69% | 69.40 / 64.65; +1.44% | 352.65 / 370.15; -4.82% |
| Autoware byte | 111.60 / 116.90; -5.29% | 67.25 / 73.30; -11.64% | 93.90 / 99.00; -0.30% | 437.65 / 440.20; -3.92% |
| Autoware final | 155.00 / 137.20; +8.45% | 97.00 / 86.70; +9.58% | 141.50 / 126.15; +13.39% | 462.40 / 438.60; +5.65% |
| Tokyo byte | 108.90 / 117.10; +3.80% | 66.45 / 67.35; +5.16% | 101.70 / 99.30; +1.46% | 165.70 / 166.30; +2.66% |
| Tokyo final | 123.25 / 117.45; +5.59% | 69.80 / 69.30; -0.08% | 111.35 / 102.30; +11.58% | 174.50 / 169.75; +1.94% |

The byte candidate's faster-pair counts are:

| Source/pass | Load | Worker index | Transform | ICP |
| --- | ---: | ---: | ---: | ---: |
| PandaSet byte | 6/10 | 6/10 | 6/10 | 5/10 |
| PandaSet final | 3/10 | 4/10 | 6/10 | 4/10 |
| Autoware byte | 2/10 | 4/10 | 5/10 | 2/10 |
| Autoware final | 7/10 | 8/10 | 7/10 | 7/10 |
| Tokyo byte | 6/10 | 6/10 | 6/10 | 6/10 |
| Tokyo final | 6/10 | 5/10 | 8/10 | 6/10 |

## Validation

- A new native regression compares the historical stamp and compact paths for
  **48 source/parameter combinations**, including six empty-input rejections.
  It checks original-index order, every node field, all coordinate f64 bits,
  RGB, u8 attributes and f32 attributes with distinct NaN payloads. Coordinates
  include signed zero. It covers tiny/large clouds, projected coordinates, duplicates,
  nonfinite coordinates, varying grids, depth limits and leaf thresholds.
- Both candidates passed **369 native tests**, including existing serial and
  bucketed octree structure/attribute checks. Rust formatting, native Clippy
  and wasm32 SIMD Clippy passed with warnings denied.
- Both candidates passed the same **22 distinct Playwright tests**: query
  precision and cache/lifetime controls, joins and exact exports, SOR/normal UI
  workflows, 1.2M-point pool indexing/SOR, two ICP pose-graph workflows and
  four hardware-opt-in GPU regressions. SIMD WASM and TypeScript/Vite builds passed.
- Nine reports validate **432 comparison pairs**: 27 excluded first-use,
  135 warm-up and 270 timed pairs. Nine further untimed fitted-transform pairs
  also validate fit/geometry/LOD/PLY parity. No real-input tolerance is used.
- Ten CLI guards passed: missing/unknown/repeated options, missing values,
  invalid repeats/methods, manifest mismatch, identical builds and attempted
  output overwrite. Existing report bytes were preserved. The disabled-source
  guard also rejects comparisons after returning the standard to false.
- Inputs, manifest, script, six implementation sources and both builds'
  worker/WASM artifacts are hashed before and after each report. Only the
  octree source changed between the measured implementations.

After restoring the standard to false, all **369 native tests** and the same
**22 Playwright tests** passed again on the final production build. Formatting,
native/wasm32 SIMD Clippy, SIMD WASM and TypeScript/Vite builds also passed.

## Reproduction and fingerprints

For a developer comparison, enable `COMPACT_CELL_MARKS` in the candidate
source, rebuild SIMD WASM and the production web app, then run from the root:

```powershell
node web/scripts/benchmark-octree-occupancy.mjs --baseline-dist notes/octree-occupancy-baseline-20261012/dist --reference notes/c2c-real-pandaset/reference.ply --compared notes/c2c-real-pandaset/compared.ply --input-manifest notes/c2c-real-pandaset/inputs.json --output NEWREPORT.json
```

Default is ten timed pairs and all three methods. `--methods load,transform,icp`
selects methods explicitly. Output must not exist. The source guard requires
an enabled candidate; it does not replace rebuilding matching artifacts.
Return the constant to false and rebuild after comparison.

Inputs and archived builds are ignored, so another checkout needs equivalent
preserved artifacts and original measured inputs. The initial and byte builds,
all six hashed implementation files and the measured script are saved locally
under `notes/octree-occupancy-candidate-20261012` and
`notes/octree-occupancy-byte-20261012`. Archived, initial and byte octree source
hashes are respectively:

- `d86ab3088396deeb84f6f75fb01e760e74335e28a09154fa4529cf0d7f12ed43`
- `8f4bd1525b59992a90de7ded3cbb916f2403427c7089b7edae4619b00975c67f`
- `32f3af147de559982bef7ee5be8536e7a82e9107685492768a4b89c4e82ca4ff`

Reports retain their historical fingerprints; their `optimized` label means
candidate, not an adopted speedup. Final default-off performance was not
measured as a separate cross-build comparison.

| Artifact | SHA-256 |
| --- | --- |
| baseline `c2c-worker-nC3pJ-DH.js` | `cc4e470f6998b80cabcf0785936ebef1414bb5fe76180239a8346cab8f94cb19` |
| baseline `ca_wasm_bg-BrHnY6FE.wasm` | `2bb05665390ee2ef67028307ee571f16b11904a91e8f6b9cdf13cceee810d970` |
| baseline `worker-BSJuKkLq.js` | `f308346bc77eba5733318bf008733363825f255f548059ced5969d963aaeaaba` |
| initial `c2c-worker-FW3z7Lw3.js` | `b18708641f0b78b25535534f235df311710bdaa7fd55471a7e25350e6fb84910` |
| initial `ca_wasm_bg-BQw-lYQb.wasm` | `6301b2b65d4751cb287a7fb3495cda288b78398d5517e0c9eee2346573b39aa4` |
| initial `worker-DP_3xQD9.js` | `6075212093775527164cadb78c957b0218d5d33a0a5c52ca0a5922fe866213b0` |
| byte `c2c-worker-C-QBXkqx.js` | `3386f262aa4eb40038d0df09bee060bfeeb5eb5110b71bab37aaa1e2683d2dc1` |
| byte `ca_wasm_bg-BcnHyEyV.wasm` | `487183c84ca62e57e74aebdfa4ee8040a73da80e734eca460a9748f13df5bd36` |
| byte `worker-C6Yu2kkq.js` | `4baf6271bc8d341bd8909f1cf4f8e44437561be788e4352c140e025fc9086f8f` |
| Measured/current script | `31804679d6f3ab7555e16d3817df34e2ac841918f9cae66e031e95319458c03e` |

| Final default-off artifact/source | SHA-256 |
| --- | --- |
| `c2c-worker-DBQN-YoI.js` | `4a621d02afd772d40879371c17fb15ce4ff3f924f5cc3227584d213066ed7d33` |
| `ca_wasm_bg-BzK_u8lC.wasm` | `7a61155d6771857b62c387d6b0c6fdd21fb1aebb0b5806389bf56b2ae4bed577` |
| `worker-Bw6KVgy3.js` | `e31bca88019784f6af69bd86558e693d16d7fc3c0e90f1020d9c04bc3a17b66f` |
| `rust/crates/ca-core/src/octree.rs` | `ee2ccac9c55eb0533bfd2123a876bdba9df90c7a38d7e83c8044df188ab06e60` |
| `web/scripts/benchmark-octree-occupancy.mjs` | `31804679d6f3ab7555e16d3817df34e2ac841918f9cae66e031e95319458c03e` |

Raw reports and checksums:

| Report | SHA-256 |
| --- | --- |
| [windows-autoware-planning-compact-byte.json](../benchmarks/cpu/octree-occupancy/real-source/windows-autoware-planning-compact-byte.json) | `875d84932a9ba2f07fd70c8bbdd960f5667abc13ed861556940e06381fb9de7c` |
| [windows-autoware-planning-compact-final.json](../benchmarks/cpu/octree-occupancy/real-source/windows-autoware-planning-compact-final.json) | `f5724bcc1beb1dd2855de79b79d23744ac2d39b2ebcf580fd0153b3cd0c7abb3` |
| [windows-autoware-planning-compact-initial.json](../benchmarks/cpu/octree-occupancy/real-source/windows-autoware-planning-compact-initial.json) | `5967d724b7d8d6fbb2fba0ffa6d01f36a5ad5c5ac8cb7c599b36002679cc63b5` |
| [windows-pandaset019-compact-byte.json](../benchmarks/cpu/octree-occupancy/real-source/windows-pandaset019-compact-byte.json) | `be8701995c2a01c3e577372ed3045aacfe931776124ba183435d1a000339265c` |
| [windows-pandaset019-compact-final.json](../benchmarks/cpu/octree-occupancy/real-source/windows-pandaset019-compact-final.json) | `db10fb8f5e97e4c8fc7fac5a03a8b4739d7f95fad84cdb24fbe8dc1a82e511be` |
| [windows-pandaset019-compact-initial.json](../benchmarks/cpu/octree-occupancy/real-source/windows-pandaset019-compact-initial.json) | `ee295cddd229db769ced33566df2a06c5476016320536acfb2bb2d192257b2b1` |
| [windows-tokyo-takanawadai-compact-byte.json](../benchmarks/cpu/octree-occupancy/real-source/windows-tokyo-takanawadai-compact-byte.json) | `80bc89f8e1f3d9758a433911c86ba1e267561a0aad97931c345df0bddc55457c` |
| [windows-tokyo-takanawadai-compact-final.json](../benchmarks/cpu/octree-occupancy/real-source/windows-tokyo-takanawadai-compact-final.json) | `fff402408843e83eb10ee57a9d7432e27e9d3de5c367aadab55a8cc7b57f8d2f` |
| [windows-tokyo-takanawadai-compact-initial.json](../benchmarks/cpu/octree-occupancy/real-source/windows-tokyo-takanawadai-compact-initial.json) | `f18de0c13abe8b3863dfd38143c93ffc5b64eb17591603fedd836d40e0057f6e` |
