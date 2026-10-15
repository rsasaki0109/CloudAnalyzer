# Inspect a vector map over its point cloud

Open the point cloud and its Lanelet2 `.osm` or vectormap IR `.json` in the Web
app. Both inputs must use the same metre frame. The Vector map panel's **Map
display** controls provide a plan view, a 3D overview, layer switches and point
cloud brightness. **Frame map** keeps the current camera direction. These controls
change the display, not geometry, point colors, visibility, traffic rules or Undo.

- Blue surfaces show road lanes; teal surfaces distinguish turning lanes.
- Chevrons follow each stored lane centerline. Select a lane to see its ID, turn
  and speed limit, its incoming lanes in purple, and its outgoing lanes in green.
  Yellow marks the selected lane. These connections come from map topology;
  they do not establish permitted turns or signal priority.
- Stop lines are pink. White crosswalk bands are clipped to the map's outline,
  preserving interpolated heights, including concave outlines. The bands use a
  regular display pattern; the map stores the crossing outline, not individual
  measured paint stripes.
- Amber signal faces use the saved bottom polyline and known positive height.
  A missing height leaves only the stored line. Faces and labels show geometry,
  not red/amber/green lamp states, poles or automatic object classification.
- Object labels follow the camera and avoid overlapping each other. Virtual
  boundaries are initially hidden to make complex junctions easier to read;
  enable them for geometry inspection. Vertex editing still shows every handle.

Turn surfaces are drawn from the actual boundaries, without visual curve fitting
that would hide kinks in the saved map. Use **Edit vertices** to correct a shared
boundary, **Undo** to revert it, and **Save for Autoware** to download
`lanelet2_map.osm` and `map_projector_info.yaml`.

Dragging keeps Z. Select a yellow handle to open **Edit boundary vertex height**,
or choose a boundary and vertex in that panel. It shows original coordinates and
affected lanes. **Apply height** changes only the selected vertex's Z; shared lanes
update together. **Focus vertex** frames the yellow cross. Inspect source points
and [recheck source coverage](vector-map-quality.md) after editing.

## Real intersection check

The display was checked on the official Autoware `sample-map-planning` point
cloud (1,757,841 points, MGRS 54SVE) and the surveyed approach lanes used by the
[junction ablation](vector-map-validation.md). Its 72 imported lanes, 34 signal
geometries, 95 stop lines and four crosswalks are surveyed context. They were not
automatically extracted from the point cloud by this demonstration.

The Web app proposed the same 83 ground-supported connections as the native
implementation. Twelve connections through the four-way intersection were selected
and added as one batch. All twelve retained their review-required tags after MGRS
OSM export and reimport. Layer switches, camera changes and point-cloud dimming
left the exported OSM byte-identical. Undo restored the original OSM exactly.
UI tests also cover drawing, shared-boundary editing, signal measurement,
crosswalk creation and export. Geometry tests cover clipped crossing bands,
survey slopes, large coordinates and absent signal heights.

These checks establish display and editing behavior, not automatic survey accuracy
or traffic-rule correctness. See [draft validation](vector-map-validation.md),
[point-cloud road drafting](commands/vectormap-build.md),
[junction proposals](commands/vectormap-connect.md) and
[measuring a signal housing](commands/vectormap-signal.md) for the distinct inputs,
review requirements and limitations of each workflow.

[Feature geometry editing](vector-map-feature-editing.md) supports XY dragging,
numeric XYZ changes and signal housing height adjustments. Observed paint and
stored lamps retain their source coordinates; edits are marked for review.

## Build evidence inspection

After generating roads in this session, enable the optional [build evidence display](vector-map-evidence.md)
to inspect selected paint/curb-source dots, inferred connectors and saved pre-trim
road-edge drafts. Imported maps have no build history. These switches do not change
map geometry or exports; source history is not persisted in the map.
