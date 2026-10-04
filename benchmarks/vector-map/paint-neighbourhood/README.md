# Bounded paint-neighbourhood proof

This is source-only indexing/diagnostic verification, not a new real-map accuracy
evaluation. [Behaviour and remaining limits](../../../docs/vector-map-paint-search.md)
describe the adaptive coarse/fine index and the actual source result.

`generation.json` freezes artifact hashes for 30 batch builds and four isolated
browser baseline builds before comparison with earlier **generated** maps.
No reference geometry was opened in this iteration. IR/OSM and source-profile
geometry stay unchanged. Full map batches are not copied again; their existing
versions are in [the intensity proof](../intensity-paint/) and
[the lane-edge proof](../lane-edge-inference/).

The six Tokyo intensity fits still hold. Arterial-middle now completes scanning
(7,147 bright candidates → 24 contrasted points → no narrow components); two
other courses hit later neighbourhood caps. `initial-query-source-probe.json`
and `remaining-query-cross-check.json` are independent Python source checks, not
native point-visit telemetry. The latter retains the west case's conservative
boundary-cell overhead and the east case's genuine inside-circle excess.

`geometry-regression.json` checks all previous geometries and source positions.
Default RGB artifacts comprise 79 byte-exact files plus four documents changed
only by optional `budget_stage` / `budget_query` fields. The source-profile
documents embed extraction reports; a report change is not a source-position change.
`native-verification.json` binds the rebuilt module and reproduces sixteen
native maps, exports, extraction reports and full-source audits exactly.
`xyz-rejections.json` retains two unsupported XYZ-only build failures, not maps.

`verification.json` records the runtime, separate browser-test commit, source
manifests, production WASM, fresh browser checks and 536 preserved old artifact
hashes (533 benchmark files plus the main GIF and two evidence images). No source
cloud or new screenshot is included. The historical intensity screenshot is
unchanged and remains labelled as a draft whose paint corrections were held.

Known-shape tests compare circular queries to brute force, preserve completed
query order, retain a 3 m fit in the presence of 5,000 unrelated elevated returns,
and hold genuinely dense nearby queries. The new browser fixture checks the
first ground-query limit and one-step Undo. The initial assertion expected a
contrast-query limit; it was corrected to the actual earlier ground-query stage
without changing runtime or evidence thresholds.

## Reproduce

Runtime commit: `2242539` (full SHA in `verification.json`). Source files and
preparation remain external cached inputs documented by the previous proofs.
Generate into new directories from the repository root:

```sh
cargo run --manifest-path rust/Cargo.toml -p ca-wasm \
  --example vector_map_anchor_compare -- PATH-TO-CACHED-TOKYO.las \
  benchmarks/vector-map/intensity-paint/tokyo/inputs.json NEW-tokyo-proof
cargo run --manifest-path rust/Cargo.toml -p ca-wasm \
  --example vector_map_anchor_compare -- PATH-TO-CACHED-TOKYO.las \
  benchmarks/vector-map/intensity-paint/tokyo-ui/inputs.json NEW-intensity-ui-proof
cargo run --manifest-path rust/Cargo.toml -p ca-wasm \
  --example vector_map_anchor_compare -- PATH-TO-CACHED-PLANNING.pcd \
  benchmarks/vector-map/paint-neighbourhood/planning-ui-inputs.json NEW-rgb-ui-proof
```

The default RGB batch uses the existing `lane-edge-inference/{planning,tokyo}/inputs.json`.
Compare each new artifact against its `generation.json` hash. The example reads
source-only configurations, not a survey/reference map. The negative XYZ source
and its operator trace remain in [the earlier rejected-build inputs](../intensity-paint/rosbag-rejected/).

For the production-browser harness, follow
[the intensity environment settings](../intensity-paint/README.md#reproduce),
using the newly generated isolated native baseline and the current WASM/Web build.
For RGB, use the cached planning source / `NEW-rgb-ui-proof`, channel `rgb`,
1,757,841 source points, one lane per direction and enable `VECTOR_MAP_LANE_EDGES=1`;
retain alignment, corridor, divider and evidence flags. Run in `web`:

```sh
npx playwright test --config playwright.media.config.ts vector-map-physical-anchors.spec.ts
```

Both source runs check native boundary vertices, full-source support, invariant
OSM after display toggles and one-step Undo. The intensity run checks the source
list; RGB also checks canvas inspection. Vertex agreement is an implementation
check, not complete topology equality or zero survey error. Lane counts, directions,
width assumptions and semantic marking roles still need review.

Source licensing and pinned dataset attribution are retained in the linked
previous proofs; these are reused development inputs, not held-out data.
