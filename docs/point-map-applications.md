# Applications built on repeated point-cloud maps

The application work covers three uses: construction/cut-fill progress,
map-update inspection, and terrain-change monitoring. Reuse the existing map
comparison, surface-volume, mesh and export tools; their implementation is not
a new milestone.

## Shared review area

[Distance review](distance-regions.md) and [surface volume](surface-volume.md)
can use the same explicitly copied world-XY rectangle. Each result retains its
own input identities, settings, loaded-data scope and selected bounds. Copying
bounds runs neither analysis, and later edits remain independent.

Distance review selects query points inside the rectangle and searches the full
loaded reference. Volume selects full grid cells whose centers lie inside it;
boundary squares are not fractionally clipped. Matching rectangles therefore
give common geographic context, not identical support or interchangeable counts.
Volume grid placement and optional interpolation still use full loaded inputs.
Missing cells and cells outside the rectangle remain separately accounted for.
Counts cover the original loaded union grid; an area extending beyond that grid
does not add missing cells outside it or establish the area's true coverage.

These operations require aligned coordinates and suitable ground surfaces.
They do not register surveys, infer their true shared coverage, or convert units.
Loaded previews are not full-density processing. Volume is a 2.5D height-surface
estimate, not a closed-solid measurement.

## Application paths

| Use | Available workflow | Next application work |
| --- | --- | --- |
| Construction / earthwork | Before/After heights, named rectangles, independent per-area cut/fill/net and missing-cell summaries, project/JSON definition reuse, JSON/CSV results, fixed-baseline repeated-visit comparisons on one union grid, optional intersection of raw observed cells with exportable mask, direct adjacent-visit cut/fill in an explicit order | Independent field validation and real-survey quantity workflows |
| Map updates | Bidirectional distance regions, density/gap candidates, sections, scoped evidence; independent human reviewed/deferred records with comments and JSON/CSV/QA exports; read-only archives and HTML reports without clouds; A/B status/comment comparisons for identical archived diagnoses | Independent real-map validation and review across different diagnoses |
| Terrain monitoring | Height changes, user-threshold screening for single pairs, saved areas and repeated/direct visits with rise/fall/within-band counts, affected areas/volumes and bounded cell inspection with camera focus, sections, M3C2 with detection limits, terrain meshes, repeated-visit and direct interval summaries with common-observed-cell support | Independent field validation and uncertainty-aware monitoring |

Screened-cell inspection also opens a Before/After point section for that row's
actual pair. Direction, length and width are adjustable; JSON/CSV preserve
selected band points and provenance. Bands include loaded context outside parent
bounds/masks and use display-coordinate precision, so they do not reproduce the
original f64 raster or establish full-density survey accuracy. See the
[section workflow](surface-volume.md#compare-the-loaded-beforeafter-points-around-a-screened-cell).

Construction and monitoring comparisons can export a standalone **volume report
HTML** with baseline and optional direct interval tables, loaded-data scope,
settings, bounds, support/mask metadata and an expandable comparison JSON
snapshot. It opens offline without point clouds. Keep the separate mask and
JSON for exact common-cell provenance; the report supplies no combined total,
source authentication or field-accuracy certification.

The next-work column is planned. Named area summaries reuse the current input
pair and full-grid settings; repeated visits use one fixed baseline and a shared
union grid. Independent-pair mode can have different observed support; the optional raw
common-cell mode intersects every selected input without interpolation. Optional direct interval comparisons use the explicit
visit order; gross fill/cut is computed from the pair, not baseline-total subtraction. Overlapping rows are not combined into a total.
Candidate distances and height changes do not establish causes such as erosion,
landslides, missing map objects or construction quantities accepted for payment.
User-defined thresholds are not automatically calibrated survey uncertainty.

## Validation

Synthetic known-height cells check arithmetic, closed area edges, grid-origin
stability, absent coverage, full-grid interpolation and exact selected CSV rows.
Browser checks cover settings invalidation, cancellation and bounds copying.
These are implementation checks; independent repeated surveys with references
are still required to establish each application's field accuracy.
