# Measure crosswalk paint from points

`ca vectormap-crosswalk` proposes repeated bright bands on a fitted road surface.
It measures their direction and visible footprint from retained intensity or RGB.
The user identifies a crosswalk and confirms its crossing lanes. Other repeated
road markings can pass the pattern checks; this is assisted creation, not automatic
object classification or an inference of legal priority.

`--brightness-fraction` defaults to `0.75`, between ground-brightness P10 and
P95. It accepts `0.4`–`0.9`; a lower threshold can retain dim paint but also admits
more background. Web exposes the same setting as a percentage. Keep the box,
threshold and lane inputs identical between preview and confirmed addition.

The cloud and editable map must already share the same metre frame. Select an
original-coordinate road-paint box, excluding kerbs and buildings. Preview keeps
the map unchanged and writes the usual map, projector, editable IR and report
artifacts to a **new** directory:

```sh
ca vectormap-crosswalk survey.pcd vector_map.json --out crossing-preview \
  --box "3882,73743,18.5,3899,73764,20.5"
```

Review `report.json`: each candidate includes its four outline vertices, measured
band rectangles, band count, dimensions, direction and an uncalibrated ranking
statistic. Inspect the actual paint, including missing ends. Then supply confirmed
lane IDs and the chosen **zero-based** candidate index, using the original map:

```sh
ca vectormap-crosswalk survey.pcd vector_map.json --out crossing-added \
  --box "3882,73743,18.5,3899,73764,20.5" \
  --lane 85 --candidate 0 --add
```

Coordinates and lane 85 are illustrative, not a verified lane assignment. Repeat
`--lane` for all confirmed crossing lanes. Adding recomputes the measurements,
creates the measured outline and crossing rule, and retains existing geometry,
IDs and rules. It never invents a stop line. Geometry and rules carry source and
review-required tags through IR and Lanelet2 export/import. Identical requests
reuse the added feature without a new Undo step in Web. Replay assumes the same
source cloud; it is duplicate prevention, not an update operation. Different boxes,
indices or lane assignments can create duplicates, including over imported walks.
Review existing features first; imported geometry is never silently replaced.

In Web, open **Measure crosswalk paint from points**, choose a cloud and enter the
box (or use the bounds of an isolated working cloud). Enter confirmed lane IDs,
then **Find paint candidates**. Select a candidate to frame its measured cyan
outline and individual bands. **Add reviewed crossing** requires both a selection
and lane IDs. Changes to inputs, cloud or map invalidate the preview. One map Undo
removes the new crossing and lane association. Preview/export/reimport/replay are
tested. Measured band rectangles are retained in the optional custom tag
`cloudanalyzer_paint_bands` and displayed clipped to the crossing outline. Their
orientation and observed gaps remain visible after addition and OSM reimport.
This is provenance/display metadata: standard Lanelet2 crosswalk geometry is the
outline. Imported/manual crossings without band metadata use decorative regular
stripes; these do not claim observed paint.

**Inspect box points** copies the previewed ROI with its original attributes and
hides the source for inspection. Re-run the preview on this isolated cloud before
adding. Cloud Undo restores the source. If sparse points appear black, disable
EDL or increase the point size; RGB remains useful for inspecting paint contrast.
Use a 3D view when inspecting a signal head, so its measured height is visible.

Only the selected loaded points are measured. Sparse LOD is not full-density data.
For COPC in Web, first create an attribute-bearing **COPC full-density box** working
cloud. Native Python/CLI/MCP currently use the local whole-file compatibility
reader, retaining RGB/intensity; HTTP and XYZ-only inputs are unsupported. For a
large native source, export an attribute-preserving ROI first. The selected-point
cap does not bound the memory of a whole-file reader.

Python and MCP expose `measure_vector_map_crosswalk(cloud, vector_map, out_dir,
bounds=..., lanes=None, candidate=0, brightness_fraction=0.75, preview_only=True)`. Preview may omit lanes;
adding requires explicitly confirmed distinct existing IDs. Failed requests and
existing output directories never publish partial artifacts.

The deterministic heuristic uses 0.25 m low-quantile surface cells and a robust
local plane, selecting ground returns within 0.08 m. Usable intensity is preferred,
with minimum RGB channel brightness as fallback. Directions are searched every
2 degrees using 0.1 m profile bins. At least three 0.3–0.9 m bands, separated by
0.2–1.2 m gaps, must have at least 1.5 m of transverse extent, aligned centres,
and 80% transverse occupancy at 0.25 m spacing. The footprint is limited to 8 m
along the band sequence and 35 m across it. Boxes are limited to 40 m in XY, 5 m
in Z and 200,000 finite selected points; at least 100 ground returns and 25
supported surface cells are needed. These constants are development heuristics.
Worn paint, shadows, occlusion, nonplanar roads and sparse returns can yield no
candidate or a shortened footprint. No geometry is substituted when support fails.

Synthetic rotated/sloped paint, blank ground, a single line, above-ground patterns,
attribute parity, invalid inputs, caps and atomic state changes are tested. The
existing Autoware sample provides a real RGB-backed proposal check, not a held-out
detection or survey-accuracy benchmark. Reference-map geometry is not passed to the
paint detector; existing lanes are used only for explicit association validation.
