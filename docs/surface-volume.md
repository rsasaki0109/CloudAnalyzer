# Compare two ground surveys: cut, fill and height changes

In **Map coverage & quality**, select the later map as **Source** and the earlier
survey as **Reference / another pass**. **Prepare cut/fill volume comparison**
opens **Volume (cut / fill)** with reference as Before, source as After, and
interpolation disabled. It does not run the calculation until you choose
**Compute volume**.

The existing volume engine rasterizes both surfaces onto one XY grid. Choose
cell size and Mean, Min or Max cell height. The lowest data XY corner is the
center of the first cell; each cell represents its full square area. For each
cell with both surface heights, `difference = after − before`:

- Positive difference contributes to **Fill (added)** and is colored red.
- Negative difference contributes to **Cut (removed)** and is colored blue.
- **Net** is fill minus cut.
- Undefined cells are excluded and counted as **Uncompared cells**. They do not
  mean zero change. The total grid also includes space in the union bounding box
  where neither survey has data.

Use **Hide input surfaces** to see the colored difference cells without the
overlapping input points. It preserves the input data and volume result; show
inputs again with their cloud-list visibility checkboxes.

**Fill empty cells** explicitly enables interpolation; results then include
estimated heights. The summary, export and QA report record this setting.
Choose the coordinate unit for area and volume labels (default metres); changing
it does not convert coordinates or numbers.

This is a **2.5D surface-height calculation**, not a closed-solid volume
measurement. Use aligned, comparable ground surfaces for earthwork; mixed
vegetation, roofs and multiple height layers can change the representative
height. A different cell size or height mode can change the estimated volume.
Meshes and a constant reference height remain available in the standalone panel.

Under **Review area**, enable **Limit volume to a world XY rectangle** and enter
original-coordinate min/max X/Y. Only grid cells whose centers lie inside the
closed rectangle contribute to totals, compared/uncompared counts and the
derived difference cloud. Outside cells are counted separately. Missing heights
inside remain uncompared, not zero change. An area containing no grid-cell
centers has zero selected cells and no new difference cloud.
Counts cover the original union grid only. Extending the rectangle beyond the
loaded surfaces does not create extra missing cells outside that grid; its
compared-cell ratio is not a surveyed-coverage percentage of the whole rectangle.

**Use distance review area** copies the distance rectangle into volume;
**Use this area for distance review** copies it back. Copying does not run either
analysis. Settings remain independent afterward; changing one does not silently
change the other. Each report preserves its own run's rectangle.

The grid origin, cell heights and optional interpolation still use the **full
selected loaded surfaces**. Included cells contribute their full square areas;
boundary squares are not clipped and may use points outside the rectangle.
Interpolation may use neighbors outside it. This is cell-center selection, not
exact polygon/footprint integration. Choose a suitable cell size and inspect
boundaries. Area selection does not reduce the full-grid allocation budget;
crop inputs first to reduce processing. Preview/thinning scope and loaded/file
counts remain in JSON. Grids above 1,000,000 cells are rejected before allocation.

**Save volume summary JSON** records the exact runtime input identities,
Before/After roles, settings, unit, added/removed/net volume, areas and compared/
uncompared counts, selected bounds and outside-cell count. **Save compared cells CSV** exports every compared grid-cell
center at its After height and its signed height difference, up to 100,000 cells.
Larger CSV requests fail without a partial download; increase cell size or crop.
CSV uses the displayed f32 cell records shifted to original coordinates; totals
use the existing f64 engine, so summing rounded CSV values can differ slightly.
Keep JSON with CSV for input identities, settings, units and scope.

**Mesh compared After cells** prepares [terrain meshing](terrain-mesh.md) from
the compared cell centers at After height. The summary JSON includes this
derived cloud's identity for provenance. This meshes the later surface in
compared cells, not signed height differences or the complete later survey.

Changing settings, editing/removing/replacing an input, or changing the derived
cell geometry invalidates the current result and its QA report. Each run holds
the chosen input snapshots and settings, so changes during computation cannot
publish a stale result. Cancellation discards a result after the active WASM
calculation finishes and releases any unpublished derived cloud.

For general 3D discrepancy candidates and an explicit review rectangle, use
[distance discrepancy regions](distance-regions.md). M3C2 remains available in
the distance tool when a normal-direction change and its level of detection are
needed; neither C2C nor raw height difference alone establishes survey accuracy.

The [recorded browser smoke check](../benchmarks/map-coverage/surface-analysis.json)
uses six synthetic Before cells and five After cells: fill 2, cut 1, net 1,
with one uncompared cell. It also verifies signed CSV, input roles, stale-result
invalidation and cancellation. This is known-cell arithmetic and workflow
validation, not an independent survey-accuracy result.
