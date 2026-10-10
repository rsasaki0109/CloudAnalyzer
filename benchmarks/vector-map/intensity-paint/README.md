# Explicit retained-intensity paint checks

This records guarded capability and source-only regressions, not a new accuracy
evaluation. See [controls, normalization and the actual-source screenshot](../../../docs/vector-map-intensity-paint.md).

- `tokyo/`: six existing development courses, RGB before / explicit intensity
  after. Twelve maps retain byte-identical IR and OSM between channels. Each
  `*-audit.json` reports the exact held reason; **no new paint fit applies**.
  Both modes use physical anchors, curb alignment, corridor/divider fitting and
  source-surface fitting. Lane-edge inference is disabled in both modes.
- `tokyo-ui/`: an isolated arterial-middle build for the browser's six-lane
  native baseline, without the preceding course's existing map. Its two modes
  also retain identical geometry. Manual inputs are three lanes each way and
  3.5 m width.
- `rosbag-rejected/`: another cached XYZ-only source and an operator-placed trace.
  Both attempted builds reject all road intervals for insufficient source support.
  `verification.json` preserves both failures; **no map was generated**.
- `native-verification.json`: an independently rebuilt/installed native module
  reproduces all twelve Tokyo maps, OSM exports, extraction reports and source
  audits exactly.
- `rgb-regression.json`: all 83 outputs from 18 existing source-only generation
  runs match the previous lane-edge stage byte for byte. Hashes are retained
  instead of copying those outputs again.
- `verification.json`: runtime/harness/native/production-WASM hashes, actual-source
  browser checks, held reports and this package's artifact hashes. It also binds
  the previous evidence files that remain unchanged.

Generation used no reference map; `generation-freeze.json` records the inputs and
outputs before any evaluation. No reference geometry was opened for this iteration.
These are reused development inputs and a negative check at another location,
not held-out accuracy results. The source clouds are cached externally and are
not included. Old real-map errors and inferred outside widths remain applicable.

The known synthetic Rust/native/Web fixtures establish 3.0 m measured spacing
and expected heading with uniform RGB and affine-rescaled intensity. They also
check missing/flat/invalid/wide-band holds, inferred gaps and preserved outside
divider geometry. They do not prove performance on the Tokyo intersection.

## Reproduce

Use runtime commit `38d24a0ef2c4681e1278dca1fbc5cd0c48198525` and the cached prepared
Tokyo XYZRGBI LAS whose SHA256 is
`e84b7e8e81e133fa67372badf2e468b1e561b6393d5782e851cd8a1666790a15`.
The [previous source record](../paint-corridor/README.md) describes preparation
and attribution. From the repository root, with new output directories:

```sh
cargo run --manifest-path rust/Cargo.toml -p ca-wasm \
  --example vector_map_anchor_compare -- \
  PATH-TO-CACHED-TOKYO.las benchmarks/vector-map/intensity-paint/tokyo/inputs.json NEW-tokyo-proof
cargo run --manifest-path rust/Cargo.toml -p ca-wasm \
  --example vector_map_anchor_compare -- \
  PATH-TO-CACHED-TOKYO.las benchmarks/vector-map/intensity-paint/tokyo-ui/inputs.json NEW-ui-proof
```

The example's `intensity_paint` comparison changes only the paint channel between
modes; existing-map accumulation within each Tokyo batch is recorded separately.
Source paths/configs are not reference-map arguments. The XYZ-only ROS-bag source
SHA256 is `63cfc18d32c9a46db0b7546b75be522e307abef6af08e207f792f6c2a09d126a`.
Running the comparator with `rosbag-rejected/inputs.json` is expected to fail
before publishing a map, as recorded in its verification.

For the optional production-browser proof, build WASM/Web and run in `web` with
these environment values (absolute source/proof paths):

```text
VECTOR_MAP_ANCHOR_SOURCE=PATH-TO-CACHED-TOKYO.las
VECTOR_MAP_ANCHOR_PROOF=PATH-TO-NEW-ui-proof
VECTOR_MAP_PAINT_CHANNEL=intensity
VECTOR_MAP_SOURCE_POINTS=1883866
VECTOR_MAP_EACH_DIRECTION_LANES=3
VECTOR_MAP_CURB_ALIGNMENT=1
VECTOR_MAP_PAINT_CORRIDOR=1
VECTOR_MAP_PAINT_DIVIDER=1
VECTOR_MAP_EVIDENCE_OVERLAY=1
```

```sh
npx playwright test --config playwright.media.config.ts vector-map-physical-anchors.spec.ts
```

The harness commit is recorded separately from the generation runtime. It loads
all source points, matches 49 native boundary vertices to exported nodes, checks
full-source support, verifies OSM byte invariance after the display toggle,
inspects the source list and confirms one-step Undo with no page errors. It does
not check whole-map topology equality or claim zero survey error. The six
intensity source vertices existed in ordinary cross-section extraction; they
are not six newly fitted paint observations. The fresh RGB source-browser run
also retains its previous four-lane / 66-vertex baseline and canvas inspection.

Tokyo source: Dynamic Map Platform Co., Ltd. (2026),
[pinned Hard Intersection sample](https://huggingface.co/datasets/dynamic-maps/hard-intersection-multimodal-sample/tree/e8e8d2a5d49a8b9cb63b5b5ecef9b260ff48f39c),
CC BY 4.0. Screenshot adapted from its previously prepared point cloud. Official
Autoware sample sources are cached separately; see the
[physical-anchor source record](../physical-anchors/README.md) for the planning
source used by the unchanged RGB regression.
