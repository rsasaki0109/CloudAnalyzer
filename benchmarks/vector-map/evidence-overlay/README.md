# Build evidence display verification

This is a read-only diagnostic addition, not a new geometry or accuracy experiment.
See [controls and source screenshot](../../../docs/vector-map-evidence.md).
`verification.json` records the runtime source commit, production WASM, independent
historical native geometry baseline and cached-source/browser regression evidence.

The source-only comparison example reproduced all 83 generated outputs from the
[previous lane-edge stage](../lane-edge-inference/) byte for byte: three planning
courses and six Tokyo courses, each in two modes. No reference map was used to
generate them. The existing 455 proof artifacts across five stages stay unchanged.
The evidence collector and ordinary build also produce identical IR and reports
in the split/reversed-lane Rust lifecycle test.

The production browser loaded all 1,757,841 cached planning points, generated four
lane fragments, checked full source support and matched all 66 boundary vertices
to the frozen native map (zero maximum distance to exported nodes). This is a
vertex check, not a whole-topology equality assertion. Turning both display
switches on leaves the exported Lanelet2 bytes identical. A canvas click and the
profile/vertex inspector work; one Undo removes the generated map.

The optional source browser harness is `web/media/vector-map-physical-anchors.spec.ts`.
The snapshot and dash limits declare a partial display rather than changing geometry.
Snapshot metadata is not saved in vectormap IR or Lanelet2; reopening a saved map
has no source history. Source-cloud changes do not re-audit this build snapshot.
