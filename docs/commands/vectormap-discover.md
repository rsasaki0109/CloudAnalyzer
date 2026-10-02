# Find road equipment without feature boxes

`ca vectormap-discover` searches retained point-cloud geometry and brightness for
repeated paint bands, bright bars and compact elevated panels. It supplies measured
proposals for crosswalks, stop lines and signal housings. A person must identify the
object and confirm its lane IDs. The detector does not infer traffic priority,
lamp colours, stop signs, or signal/stop-line relationships.

Start with roads generated from a cloud and a trajectory in the same metre frame:

```sh
ca vectormap-build survey.pcd drive.csv --out roads
ca vectormap-discover survey.pcd --map roads/vector_map.json --out proposals
```

No manually located feature box or surveyed feature geometry is required. Preview
writes an unchanged map, projector metadata, editable IR and `report.json` into a
**new** directory. `report.discovery` contains the original-coordinate bounds,
measured geometry, point support, nearby lane suggestions, review flags and keys.
Nearby lanes describe distance, not which lanes a signal controls or a crossing
intersects. The search does not classify the objects automatically.

To search a scene without roads or a recorded drive:

```sh
ca vectormap-discover survey.pcd --scope ground_surface --out scene-proposals
```

This uses lower supported surfaces as scan anchors. Roofs and other levels can
also be searched. It does not generate a road network or infer a heading for paint
bars. In Web, paths can be traced over the original points with **Fit drawn paths
to the point cloud** enabled. The traced path, lane counts and nominal widths are
operator inputs; points supply ground heights and supported boundary candidates.

Review proposals over the source points. Put the chosen zero-based `id`, its full
`key`, a confirmed `classification` and distinct existing `lanes` into a JSON array:

```json
[
  {
    "candidate": 0,
    "key": "COPY THE EXACT KEY FROM THE PREVIEW",
    "classification": "stop_line",
    "lanes": [4]
  }
]
```

The ID and lane above are illustrative. Valid classifications are `crosswalk` for
repeated paint, `stop_line` for a bright bar, and `vehicle_signal` or
`pedestrian_signal` for an elevated panel. Run against the original source/map
with the **same search settings**:

```sh
ca vectormap-discover survey.pcd --map roads/vector_map.json --out reviewed \
  --confirmations confirmations.json
```

The core recomputes the evidence and rejects stale keys, incompatible types and
missing lanes before applying the complete batch. It stores measured coordinates,
the user's classification and lane assignments with review-required provenance.
Identical confirmations reuse matching additions after Lanelet2 reload, including
their subsequent manual geometry edits. This is duplicate prevention, not automatic
replacement of changed measurements or imported features. IDs and keys are local
to the search settings and retained source; they are not persistent object IDs.

In Web, **Find road equipment automatically** is available before a map is opened.
Road builds also search by default. Select a candidate, focus or inspect its
original points, discard false candidates, then explicitly choose the type and
lane IDs. Inspecting copies only that candidate's original points and hides the
source; cloud Undo restores it. Adding keeps the original cloud as the measurement
input and creates one complete-map Undo step. Geometry editing supports crossings,
stop lines and signal faces. Hiding unconfirmed geometry affects display only;
proposals never enter exports before review and addition.

Search radius defaults to 12 m (allowed 4–18 m). `--brightness-fraction` defaults to
0.65 (allowed 0.4–0.9). Intensity is preferred, with RGB minimum-channel fallback
when intensity is absent or constant. The repeated-band fitter uses ground P10/P95;
bright-bar search uses P10/P99.9 so narrow markings are not swallowed by the large
automatic window. Both require actual darker returns around/between paint: missing
returns are not evidence of dark paint. Shape and contrast tests remain heuristics,
not calibrated probabilities or survey accuracy guarantees.

Elevated-panel search groups supported horizontal slices before fitting compact
faces, excluding narrow pole slices. Signs, vegetation and background can still
pass. The measured face is exported without invented lamps, pole geometry or a
stop line. Stop-line additions use a marking rule, never an inferred stop sign.
Observed crossing bands remain provenance/display metadata; standard Lanelet2
crosswalk geometry is the measured outline.

This is a local, attribute-preserving **whole-file compatibility reader**. It loads
the source before searching: point caps do not bound the reader's peak memory.
Road-corridor search retains at most 2 million points and 500 windows; whole-ground
search accepts at most 2 million source points and 2,000 supported tiles. Windows
with fewer than 100 or more than 200,000 points are reported as unsupported. Each
preview keeps at most 64 highest-support proposals per evidence type and explicitly
reports truncation. Sparse loaded/display LOD is not full-density input. Export an
attribute-bearing scene for larger sources; this is not a 10-billion-point search.

Development checks reuse PandaSet 019 (1,063,816 points and 80 recorded poses) and
the Autoware planning sample (1,757,841 points, no recorded drive). The former
produced a repeated-paint proposal and numerous panel proposals; no bright bar
passed the final dark-flank checks. Whole planning search detected 806 proposals,
retaining 64 bars, 64 panels and one repeated-paint proposal, with 71 unsupported
windows out of 906. Inspection found false paint/panel proposals and missed heads
in road-corridor mode. Two planning panel centres were within 0.284 m and 0.890 m
of surveyed heads in a **post-hoc** comparison; surveyed geometry was never a
generator input, and incomplete annotations prevent precision/recall claims.
These are development observations, not held-out accuracy results. Positive
crosswalk/stop-line/head fixtures, missing-return negative fixtures, atomic review,
editing, exact Undo and Lanelet2 replay are also tested.

The Python function and MCP tool are both named `discover_vector_map_features`.
Reports distinguish point evidence, inferred road width, human classification and
lane assignments. Review traffic rules and geometry independently before use.
