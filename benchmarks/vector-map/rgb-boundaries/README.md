# Frozen RGB boundary experiment

See [the protocol, figure and unchanged real-scene results](../../../docs/vector-map-rgb-boundaries.md).
No real-scene accuracy improvement is demonstrated; this remains draft work.

`planning-inputs.json` and `tokyo-inputs.json` embed only small CSV trajectories
and explicit lane priors. No point clouds, semantic labels or surveyed shapes are
generation inputs. Tokyo CSVs derive from the pinned Dynamic Map Platform sample
(2026), CC BY 4.0, with the same source/operator tracing provenance as the linked
preparation protocol; no complete drive is redistributed.

Each scene directory contains `prior-N.json/osm`, `rgb-N.json/osm`, source audits,
CSV traces, `generation-freeze.json` and `evaluation.json`. Both modes enable
source-footprint drafting. Full maps are cumulative across the fixed traces.
All corresponding prior/RGB maps are byte-identical. Generation was frozen before
references were opened; hashes refer to canonical LF bytes.

Survey comparison distances are unpaired nearest sampled driving-boundary XY
distances. Report generated/deferred extent, fragments and source-support totals
alongside them. These are known development scenes, not held-out accuracy.
No junction choices or equipment associations are adopted in these experiments.
Root `verification.json` records artifact/runtime hashes and actual browser checks.
