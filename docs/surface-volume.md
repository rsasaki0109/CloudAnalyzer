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

## Named areas and independent area summaries

In **Review area**, enter an **Area name** and save the current enabled rectangle
with **Save / update area by name**. Names are unique after trimming, with at
most 80 characters and 32 definitions. Saving an existing name updates its bounds.
Select a saved area and choose **Apply selected area** to restore its name and
bounds for a single-area run or copying to distance review. Applying runs no
analysis. A named single-area run includes the name in its JSON and QA title.

Definitions are saved with projects, workspace snapshots and browser project
recovery. **Save area definitions JSON** also exports the bounded list without
point data. **Load area definitions** replaces the list after validation; bad
files retain the existing list. Check coordinate frame and coordinate units
before reuse. Bounds do not perform registration or unit conversion.

**Compute all saved areas** uses the same selected Before/After, cell size,
height mode, interpolation and coordinate-unit label for each saved rectangle.
Runs are sequential and cancellable between active engine calls. No compared
clouds or point exports are created for these summary-only runs. All rows are
published together; cancellation or changed inputs/settings discards staged rows.
The full-grid allocation limit still applies to every run. To inspect a row's
colored cells or mesh them, apply that area and compute the ordinary volume.

Rows are independent. Overlapping rectangles, including shared closed edges,
may count the same cells, so there is no combined total. **Save area summaries
JSON** records exact bounds, input runtime identities and loaded/file scope,
settings, cut/fill/net, areas, compared/uncompared/outside counts and an explicit
comparison-available flag. **Save area summaries CSV** exports one row per area;
quantities are blank when no cells were compared. JSON retains arithmetic zero
totals with `comparisonAvailable: false`; those are not measured zero change.
CSV protects formula-leading labels for spreadsheet use; JSON preserves names.

The QA report also records the current batch with area-specific metrics. Empty
comparisons have counts but no volume gate metrics. Editing inputs, settings or
definitions clears the batch and QA section; recompute for the new configuration.
Calculated rows are exported separately and recomputed after opening a project.

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

## Repeated visits on saved areas

Load one aligned baseline and up to eight visit surfaces. Under **Repeated
visits by area**, select the visit surfaces (Ctrl / Command for multiple),
choose **Before** as the fixed baseline, save the named rectangles, and click
**Compare visits with baseline**. **After** is unused for this operation. A
constant Before height is supported, but it defines a plane across the full
grid, including locations beyond an observed baseline's footprint.

All comparisons use one union grid over the baseline and all selected visits,
the same cell size, height mode, unit label and interpolation option. Grid
placement follows the existing nearest-center convention, anchored at the
lowest XY corner of that union. Adding/removing a selected visit can move the
grid and change results; compare a fixed selection. A remote visit can enlarge
the grid: the existing 1,000,000-cell budget applies to the entire union, not
just the saved rectangles. At most 8 × 32 serial summaries are run. Each result
compares one visit to the baseline; no derived scene clouds are created.

The table and JSON/CSV exports retain each visit/area's fill, cut, net, compared,
uncompared and outside cell counts. No-support rows show unavailable values
and blank CSV quantities. JSON retains baseline/visit runtime descriptors,
loaded/preview/decimation information, settings, shared grid origin/dimensions
and area bounds. Visit indices follow loaded order, not inferred acquisition
dates; use dated filenames. Duplicate filenames remain distinguishable by
visit index and runtime descriptors. Save JSON alongside CSV for full provenance.
Selections and computed rows are session-only; projects retain area definitions,
and visits must be selected and computed again after reload.

A common grid does **not** establish a common observed footprint: even equal
compared-cell counts can represent different cells. Do not subtract these
totals to infer between-visit earthwork. Missing heights are excluded rather
than treated as zero change; interpolation can create estimates in unobserved
cells. Area overlaps remain independent and must not be summed. The selected
inputs are not registered or checked for matching frames/units. Results apply
to loaded data, including any preview/decimation, and remain 2.5D estimates.

Cancellation finishes the in-flight computation then discards every staged
row. Changes to settings, areas or an input invalidate results; removing an
input does too. An unrelated newly loaded cloud does not enter an existing
comparison or change its selected visits. Synthetic checks cover shifted visit
extents, exact known heights, missing/empty support, constant baselines,
budget rejection, cancellation and stale-result invalidation. These checks do
not establish accuracy on independent real surveys.
