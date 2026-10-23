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
| Construction / earthwork | Before/After heights, cut/fill/net, compared area, missing cells, JSON/CSV | Named work areas, per-area summaries and comparison across visits |
| Map updates | Bidirectional distance regions, density/gap candidates, sections, scoped evidence | Review decisions for change candidates and a report of confirmed/deferred updates |
| Terrain monitoring | Height changes, sections, M3C2 with detection limits, terrain meshes | Consistent repeated-visit inputs and reports retaining registration/coverage limitations |

The next-work column is planned, not implemented by the shared-area change.
Candidate distances and height changes do not establish causes such as erosion,
landslides, missing map objects or construction quantities accepted for payment.
User-defined thresholds are not automatically calibrated survey uncertainty.

## Validation

Synthetic known-height cells check arithmetic, closed area edges, grid-origin
stability, absent coverage, full-grid interpolation and exact selected CSV rows.
Browser checks cover settings invalidation, cancellation and bounds copying.
These are implementation checks; independent repeated surveys with references
are still required to establish each application's field accuracy.
