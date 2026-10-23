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
just the saved rectangles. A baseline-only run makes at most 8 × 32 serial
summaries; enabling direct intervals adds up to 7 × 32 requests. Each baseline row
compares one visit to the baseline; no derived scene clouds are created.

The table and JSON/CSV exports retain each visit/area's fill, cut, net, compared,
uncompared and outside cell counts. No-support rows show unavailable values
and blank CSV quantities. JSON retains baseline/visit runtime descriptors,
loaded/preview/decimation information, settings, shared grid origin/dimensions
and area bounds. Visit indices follow the explicit numbered list (initially loaded order), not
inferred acquisition dates; review the order and use dated filenames. Duplicate filenames remain distinguishable by
visit index and runtime descriptors. Save JSON alongside CSV for full provenance.
Selections and computed rows are session-only; projects retain area definitions,
and visits must be selected and computed again after reload.

In the default each-pair mode, a common grid alone does **not** establish a
common observed footprint: even equal
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

## Save a standalone survey-volume report

After a completed repeated-visit comparison, **Save volume report HTML** exports
a standalone, read-only document that opens without CloudAnalyzer, network
access or the source clouds. It includes the fixed-baseline table, optional
direct adjacent-visit table, named bounds, fill/cut/net and affected areas,
compared/uncompared/outside counts, input runtime descriptors and
loaded/file/preview/thinning information, explicit visit order, settings and
shared-grid metadata. Export time is UTC and is not an acquisition date.
No combined area or interval total is calculated. No-support quantities display
**Unavailable**. Independent support and interpolated estimates are labeled;
a constant baseline is identified as a plane rather than an observed survey.

The expandable **Complete comparison JSON snapshot** contains the same fields
as **Save visit summaries JSON**, with full numeric values. Names and bounds
are rendered as text, including filenames or area labels containing HTML.
The report has no scripts or external resources; browser print/PDF is available.
The printed summary omits the expandable JSON; retain the HTML/JSON for it.

In common-observed mode, the report records the mask filename, encoding, byte
length and SHA-256, but does not embed the binary. Save **common cells mask**
separately and retain it with the JSON to identify the exact full-grid cells;
apply each row's bounds separately. The mask hash identifies a cell mask, not
the source point contents. Runtime IDs and names are not content fingerprints.
The report contains no point coordinates and does not register surveys,
calibrate uncertainty, certify field accuracy or approve quantities for payment.

The download is available only for the current valid completed comparison.
Editing settings/order/areas or changing/removing inputs invalidates it with the
other visit exports. A saved HTML remains an independent historical snapshot;
it does not update when the live analysis changes or apply results to clouds.

## Restrict repeated visits to common observed cells

Under **Repeated visits by area**, set **Comparison cells** to **Observed in
baseline and all visits**. This mode requires point clouds for Before and every
selected visit, and **Fill empty cells** off. Constant planes and meshes are
rejected because their heights do not establish raw point observations. The
existing each-pair mode remains the default and retains constant/mesh and
optional interpolation support.

A cell is common only when the selected height mode (mean/min/max) produces a
finite raw, unfilled height from finite XYZ points in the baseline and **every** selected visit, on the
same full union grid. The intersection is prepared once, processing one input
at a time. Each area then selects full square cells by the same closed center
bounds and applies that identical intersection to every visit. Compared counts
are identical across visits within an area. Uncompared counts include cells
missing in any selected input, even if the current baseline/visit pair has both
heights there. Counts outside the area remain separate. An empty intersection
shows unavailable quantities, not evidence of zero change.

JSON records `support.mode`, the whole-grid common count/area, and the mask's
encoding/size/SHA-256. **Save common cells mask** exports `common-observed-cells.bin`:
one uint8 byte per union-grid cell, 0 excluded or 1 common, row-major `j*nx+i`.
Use the JSON shared-grid boundary origin, dimensions and cell size to place it,
and apply each area's bounds separately. The mask includes the whole grid,
not only saved areas. Retain JSON with the binary; verify SHA-256 to bind the
files. CSV appends support mode and mask SHA-256 so common-support results are
explicitly distinguished from independent-pair results. No compared-cell scene
clouds are created. Mask preparation and row computations remain cancellable
between worker calls; an active call finishes before all staged results are
discarded. Changed input/settings invalidate both the summaries and mask.

The full-grid 1,000,000-cell budget and 8-visit/32-area limits still apply. Mask
preparation retains one mask and rasterizes inputs serially; row requests copy
the bounded mask and reuse it without rerasterizing all other visits. No full
point data is copied to the UI for this operation. Adding/removing visits can
change the union-grid placement and common footprint; compare a fixed selection.

This supports baseline-relative volumes on identical observed cells. Differences
of **net** baseline volumes have a consistent cell footprint (subject to floating
point rounding); differences of **gross fill or cut** are not between-visit
fill/cut. Those require directly comparing the two visits on the common mask.
Loaded/decimated points define this support, not full-density survey coverage.
A finite cell height is not proof of a common physical ground surface, adequate
sampling, correct registration or acquisition order. Results remain 2.5D, with
whole boundary cells and no fractional clipping, unit conversion or field
accuracy guarantee. Synthetic equal-count/different-footprint tests validate
the mask and arithmetic, not actual map accuracy.

## Direct intervals between ordered visits

Select at least two visits, choose **Observed in baseline and all visits**, and
check **Also compute adjacent visit intervals**. Review the numbered visit list;
use **Earlier / Later** to set the desired acquisition order. Initial order is
loaded order, not a date inferred from filenames. Deselecting visits removes
them from the order, new selections append, and unrelated newly loaded clouds
do not change it. Order, selections, option and results remain session-only.
Switching to independent-pair support or fewer than two visits clears/disables
the interval option.

The operation still computes baseline-relative rows, then directly compares
each consecutive pair in the explicit list using after minus before heights.
An interval is visit 1 → visit 2, then visit 2 → visit 3, etc.; the baseline is
not inserted into that interval sequence. Every comparison uses the same mask
intersecting the **baseline and all selected visits**, even if two interval
inputs alone cover more cells. Grid, height mode, units and named area bounds
are identical across baseline and interval rows. Interpolation remains off and
inputs must be point clouds. No additional scene clouds are created.

For example, a zero baseline with two cells and visits A = [5, 1], B = [3, 4]
has baseline fill 6 and 7. Subtracting those fill totals gives 1, but directly
comparing A → B gives fill **3**, cut **2**, net **1**. Reversing the order swaps
fill/cut and negates net. Changing order alone retains the common mask; changing
the input set may change both grid and observed support.

The **Direct intervals (visit to visit)** table is separate from baseline rows.
**Save visit summaries JSON** includes `ordering` runtime cloud IDs and optional
`intervals` with a direct-comparison scope and per-row from/to indices, IDs,
names, bounds, quantities and counts. **Save direct intervals CSV** saves only
interval rows, including support mode and common-mask SHA-256. Retain the JSON
and binary mask for full provenance. A row with no compared cells is unavailable,
with blank CSV quantities. Counts outside the area remain distinct from cells
uncompared because they are missing in any selected input.

Baseline and interval rows publish together only when every request succeeds.
Cancellation or changed settings/order/inputs during an interval discards the
staged baseline rows, interval rows and mask together. Reordering or removing an
input after completion invalidates both tables and their exports. At most
(8 baseline comparisons + 7 intervals) × 32 areas = 480 serial summary requests
are performed on the existing bounded grid. Existing loaded inputs and the
development environment are reused.

These remain loaded-data **2.5D** height-surface differences, with no automatic
registration, acquisition-date inference, physical-ground correspondence or
field-accuracy guarantee. Whole boundary cells are counted without fractional
clipping. Overlapping areas and gross volumes across intervals must not be
summed as unique earthwork: the same location may rise and fall repeatedly.
Synthetic checks validate arithmetic, order, persistence within the session,
export identity and invalidation; independent surveys are still needed for
accuracy and construction-quantity validation.
