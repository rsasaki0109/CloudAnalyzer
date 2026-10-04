# Bright-candidate check outcomes

This source-only record adds exclusive check counts to the existing optional
paint fits. It does not apply a new ground mask or establish a map-accuracy gain.
[Check meanings and actual central-course outcomes](../../../docs/vector-map-paint-candidates.md)
explain why bright returns do not yet form a supported lane-paint corridor.

- `generation.json`: frozen source/config/runtime/output hashes for 30 batch maps
  and four isolated browser baselines; compares against previous generated outputs.
  Every IR/OSM and source-profile position stays unchanged. Changed JSON matches
  the previous full document after only `candidate_diagnostics` is omitted.
- `native-verification.json`: a freshly rebuilt native module independently
  reproduces sixteen maps, exports, extraction reports and full-source audits.
- `xyz-rejections.json`: two cached XYZ-only negative attempts publish no map.
- `verification.json`: exact core/Web runtime identities, native/WASM hashes,
  actual source browser checks and 545 preserved historical artifact hashes.

The browser checks compare all six displayed rejection counts plus the accepted
and bright totals against independent native reports. They also retain the
49 / 66 exported-vertex checks, complete-source support, display-toggle export
invariance and one-step Undo. These are implementation checks, not survey errors.
Fresh screenshots remain local; the published historical images and GIF stay
unchanged. No duplicate source cloud or map batch is added here.

Inputs, source preparation and attribution remain in the [explicit-intensity
record](../intensity-paint/README.md) and [earlier lane-edge record](../lane-edge-inference/README.md).
[Bounded-search verification](../paint-neighbourhood/README.md) remains frozen
in its original runtime context. All development courses are reused; no new
reference geometry was opened or source thresholds changed in this iteration.
