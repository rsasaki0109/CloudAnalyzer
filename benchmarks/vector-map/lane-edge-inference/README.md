# Source-only lane-edge inference inside distant curb candidates

Actual before/after editable JSON, Lanelet2 OSM, operator poses, measured profiles,
full-source audits and fixed-target diagnostics for three planning and six Tokyo
cases. `infer_lane_edges` is default-off and places a **configured-width prior**
relative to guarded source paint; it does not detect a shoulder or outer paint.
Original road-edge candidates remain in the extraction report before trimming.

[Explanation, actual maps and remaining error](../../../docs/vector-map-lane-edges.md).

Both modes retain previous physical anchors, curb alignment, complete paint and
interior-divider options. Only AFTER enables lane-edge inference. Case sources
and options remain fixed before reference evaluation; configurations use exact
LF bytes. Both complete scene generations were frozen before either survey opens.
A separately generated isolated planning-1 case supplies the actual-source UI
proof, which has no existing input map. The public three-case chain is cumulative.

Runtime declaration and measured binary hashes are in [verification.json](verification.json);
these are not embedded build attestations. Generation references are empty.
Source and reference SHA256, generation freezes and every output are recorded.
References enter only `evaluation.json` and the plotted diagnostic after freeze.

Planning path 1: SAME 38.141 m / 312 ordered samples, three-slot mean
0.542 → 0.327 m, right-only maximum 1.531 → 0.631 m. The right edge still has
zero samples within 0.5 m. The independent before-nearest diagnostic has
0.579 → 0.369 m mean, 1.530 → 0.743 m max. Left/interior are exact; both modes
retain 38.141 m / defer 8 m. Nine RGB, twelve curb, forty-five inferred final
vertices distinguish observation from the configured 3.5 m width prior.

Planning 0/2 hold this stage; all six Tokyo cases hold and retain byte-identical
JSON/OSM, 88.078 m generated / 34.259 m deferred. Tokyo ordered reference
comparison stays held. These are known development scenes, not held-out accuracy
or whole-intersection topology. No source cloud is copied into this directory.
Earlier four frozen stages and both existing GIFs remain unchanged.

To regenerate source only (before evaluating):

```powershell
cargo build --manifest-path rust/Cargo.toml -p ca-wasm --example vector_map_anchor_compare
rust/target/debug/examples/vector_map_anchor_compare.exe SOURCE inputs.json NEW_OUTPUT
```

Use `scripts/vector_map_anchor_evaluate.py --evaluate-frozen` after freezing BOTH
scene outputs. It verifies source/config/executable/artifact hashes before opening
the reference map. Plot with `scripts/plot_vector_map_lane_edges.py`.

Planning sample: Copyright 2020 TIER IV, Inc., original cached Autoware source.
Tokyo: DMP 2026 CC BY 4.0 sample, revision
`e8e8d2a5d49a8b9cb63b5b5ecef9b260ff48f39c`; source provenance is linked in the
[earlier paint proof](../../../docs/vector-map-paint-corridor.md).
