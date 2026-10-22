# CPU reference-tree collection screening

The [collection packet](../benchmarks/wasm-kdtree-collection/manifest.json) tests
replacing the two final point/index collection scans in `KdTree::new` with
`items.into_iter().unzip()`. The production Rust implementation stays unchanged:
only two of four reports passed the frozen screening gate.

The preceding [GPU preparation profile](gpu-c2c-reference-preparation.md) found
CPU reference-tree construction to be a material cold cost. This experiment
builds two complete ordinary stable/SIMD WASM packages in the same isolated
source directory. Their only Rust source difference is the final collection
block. The baseline WASM is byte-identical to the existing ordinary production
package; the candidate WASM differs. The generated JavaScript and declarations
are identical between variants. No new toolchain or dependency download is used.

Four browser sessions use the frozen PandaSet, Autoware, Autoware, PandaSet
order. Each Dedicated Worker loads both packages with separate ordinary WASM
memories and parses the same byte-identified real even/odd point subsets. Six
warm-up pairs precede twelve timed pairs. Pair order alternates, and each repeated
dataset reverses its initial order. All pairs and tails are retained.

The clock surrounds the complete exported `GpuC2cReference` constructor,
including partitioning, collection and derived bounds/certificates. Parsing,
reference destruction, correctness checks and bookkeeping are outside the
clock. No GPU adapter is requested. This does not measure a complete C2C RPC,
verification pool, application load or GPU execution.

Every report had to achieve at least 3% median paired reduction, at least eight
of twelve faster pairs, and committed WASM memory ratio at most 1.02. All four
had to pass before using the separately bounded complete-RPC comparison.
Positive reduction means the candidate was faster. Medians below are in ms;
paired reduction is the median of pair ratios, not the ratio of the two medians.

| Session | Baseline median | Unzip median | Paired reduction | Faster pairs | Gate |
|---|---:|---:|---:|---:|---|
| PandaSet 1 | 88.65 | 81.10 | 3.56% | 8/12 | pass |
| Autoware 1 | 158.40 | 150.85 | 4.67% | 8/12 | pass |
| Autoware 2 | 173.85 | 167.35 | 1.32% | 9/12 | fail |
| PandaSet 2 | 103.35 | 102.25 | 1.08% | 7/12 | fail |

The candidate wins 32/48 pairs, but the repeated effect is too small and variable
to pass the screen. The four-report screen budget is closed. All four unused
conditional RPC reports are discarded, and no application speedup is claimed.
The Autoware baseline includes a 324.9 ms tail and the candidate includes a
292.9 ms tail; neither is removed or attributed to a particular browser cause.

Each variant commits 86,376,448 bytes of WASM linear memory on PandaSet and
141,754,368 bytes on Autoware. Both memories coexist. These values describe
committed pages after construction, not live/peak heap, allocation traffic or
total browser memory, and do not establish memory savings.

Before and after timing, both variants have identical full reference snapshots,
GPU point/node export bits, exact full query exports, complete f64 distances and
displayed f32 distance bytes. Validation also checks twelve hint/bounds/SIMD
combinations, query-copy independence, range rejection, reference-wrapper
release before verification and complete object cleanup. The 167 historical
packet files and all nine normal build artifacts remain byte-identical. The
source datasets retain their recorded PandaSet/TIER IV attribution and original
license terms; two subsets of one measured cloud are not independent surveys or
SLAM ground truth.

Use new workspace output paths for a small compatibility check:

```sh
node web/scripts/build-kdtree-collection.mjs --output kdtree-study/build-info.json --out-dir kdtree-study
node web/scripts/benchmark-kdtree-collection.mjs --output kdtree-smoke.json --build-info kdtree-study/build-info.json --reference web/public/samples/lidar_reference.pcd --compared web/public/samples/lidar_candidate.pcd --warmups 0 --repeats 1
```

The builder requires the existing stable WASM toolchain and cached dependencies,
uses offline Cargo, and refuses existing outputs or an unexpected collection
block. `CA_WASM_PACK_NO_INSTALL=1` also disables wasm-pack installation. It copies
the tracked Rust workspace, checks all 94 compiled Rust inputs and records each
variant's four WASM/glue hashes. Original production sources and generated
packages are untouched. CI compiles both variants afresh and runs the real demo
sample as a compatibility smoke; that single pair is not performance evidence.
