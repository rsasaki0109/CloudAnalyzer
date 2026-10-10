# Optional physical width-anchor comparison

See the [protocol, measured gains and regressions](../../../docs/vector-map-physical-anchors.md).
This road-only experiment keeps default behavior unchanged and excludes outside
coverage-edge candidates from inferred width offsets when explicitly requested.

- `planning/inputs.json` and `tokyo/inputs.json` embed the prior small operator CSV
  traces and explicit lane priors, with empty reference inputs. No private path is
  required to reproduce those inputs.
- Each scene contains every `before/after-N.json/.osm` cumulative generated map,
  its full-source `*-audit.json`, extracted `*-profiles.json` and trajectory CSVs.
  Profiles are independently extracted source geometry and verified to be present
  in the actual built maps, including reversed shared boundaries.
- `generation-freeze.json` hashes all source-only generation outputs before any
  reference read. `evaluation.json` reports full-map proximity, extent, exclusions
  and primary common-source-interval distances against fixed before-selected
  survey samples. `comparison.json` summarizes each build.
- `verification.json` records exact runtime/input/public-artifact hashes, native
  byte-exact reproduction and actual source-only production-browser checks.

The same 36.606 m of planning path 0 improves from 1.997 to 1.650 m mean distance,
but another 4.5 m is deferred. Tokyo east south slightly worsens from 1.260 to
1.286 m. Surveyed lane identity is not established; known development scenes and
source-gated coverage do not establish independent accuracy. These outputs are
editable geometric drafts, not reviewed traffic/legal maps. Later source-footprint
fitting and directly selected coverage limits remain separate review concerns.
The original source clouds and surveyed maps are not copied here.

Planning sample: Copyright 2020 TIER IV, Inc.; [official instructions](https://docs.autoware.org/main/demos/planning-sim/).
Tokyo data: Dynamic Map Platform Co., Ltd. (2026), [pinned source](https://huggingface.co/datasets/dynamic-maps/hard-intersection-multimodal-sample/tree/e8e8d2a5d49a8b9cb63b5b5ecef9b260ff48f39c),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). CloudAnalyzer adds derived
geometry and reports; the code's MIT license does not replace these data terms.
