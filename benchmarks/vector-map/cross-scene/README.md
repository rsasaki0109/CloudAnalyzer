# Second-scene development audit

See [protocol, results and limitations](../../../docs/vector-map-cross-scene.md).
`planning-inputs.json` contains three prior source-reviewed operator traces and
explicit lane priors. Generation does not consume the reference map.

- `before/after-evaluation.json` and `before/after-relation-proposals.json` record
  separate unmasked and per-query target-masked component checks. Four changed
  movement candidates are held after the fix; three matching pairs remain.
  Masked mode recovers zero expected targets; missing marking context is retained.
- `before/after-road-2.json/.osm` are the actual source-only default/fitted final
  road drafts; `after-projector-info.yaml` preserves Local metre coordinates.
  Twelve fragments defer 17.5 m and are not twelve inferred traffic lanes.
- `discovery.json` and `junction-preview.json` contain unconfirmed geometric
  evidence, including caps, ambiguities and unsupported windows.
- `generation-freeze.json` hashes all original intermediate reproduction files
  before reference reads. This directory distributes final maps and reports;
  intermediate maps/CSVs can be reproduced into a new proof directory.
- `verification.json` records input/runtime/artifact hashes, the actual browser
  checks and the unchanged original Tokyo source-scene adoption regression.

The generation fix leaves roads, detector output and source quality unchanged;
it prevents stop proposals from replacing/dropping reviewed vehicle lanes.
Nearest-survey-boundary distance still averages 1.42 m despite zero source flags.
Nearby feature pairs are not semantic precision/recall, and this project development
scene is not a held-out accuracy benchmark. The source and surveyed map are not
copied into this directory.

Sample map: Copyright 2020 TIER IV, Inc.; see the
[official instructions](https://docs.autoware.org/main/demos/planning-sim/).
CloudAnalyzer adds derived road geometry and development evaluation reports.
