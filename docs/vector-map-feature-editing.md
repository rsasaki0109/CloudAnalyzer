# Edit crossing, stop-line and signal geometry

In Web, open **Edit crossing, stop line or signal geometry** in the Lanelet2 panel and
choose an existing feature. **Focus feature** frames its geometry. Use **Drag
feature vertices** to move the yellow handles in XY while keeping Z; Escape
cancels the current drag. Alternatively choose a vertex, enter its original
metre coordinates and click **Apply geometry edit**. Signals also expose their
stored housing height. Empty height keeps the existing value.

Each accepted edit is one map Undo step, including geometry and provenance.
An unchanged edit adds no Undo step. Invalid coordinates, collapsed sides,
self-intersecting/touching crossing outlines, reversed crossing vertex order,
excessive extents or vertex movement, and invalid housing heights leave the
map intact. Existing vertex counts and sequence are retained. This editor
supports crossings represented by two edges; crossings with a separate
polygon need a dedicated polygon/edge editor and cannot be changed here.

Editing preserves feature IDs, road geometry, connectivity, controlled/crossing
lane assignments, other features and stored lamps. Measured paint bands remain at
their observed coordinates and are clipped to the edited crossing outline for
display; widening an outline does not invent missing paint. Moving a signal
housing does not move its surveyed lamps. Review these observations and lane
associations separately, including the signal face direction and legal priority.

The feature's geometry source gains `_user_edited`, with
`cloudanalyzer_user_edited=yes` and `cloudanalyzer_review_required=yes`.
Associated existing rules are marked for review. These extension tags survive
Lanelet2 export/reimport, together with measured paint metadata. Identical
measurement requests reuse the existing feature, retaining the explicit edit;
this assumes the same source cloud and does not reclassify the object.

Crossings and stop lines are limited to 256 vertices and 100 m extent/movement; signal edges
to 256 vertices and 20 m extent/movement, with a positive housing height at
most 20 m. These are input sanity limits, not survey-quality acceptance criteria.
All geometry still requires operator review before use in Autoware.

## Measured paint envelopes

New paint proposals retain opposing polylines through the observed band ends,
instead of forcing every crossing into a rectangle. The envelope is simplified
with a 0.2 m displacement tolerance in XY; this is a display/geometry tolerance,
not a survey accuracy claim. The measured band rectangles remain unchanged.
Both envelope edges survive confirmation, Lanelet2 export and reimport, geometry
editing and Undo. The `cloudanalyzer_outline_source` extension tag records this
construction separately from operator-confirmed object classification.

Source-supported bright cells seed individual bands. Nearby bands are chained
only when the original ground returns support their extents and contrasting gaps.
Missing returns cannot supply dark-gap evidence. Each measurement reports its
64-band and 64-profile seed limits. Discovery remeasures window-clipped paint over
a larger source box, up to 32 additional windows, and reports unsupported or
omitted refinements. A longer source-supported chain suppresses the fragments
whose observed bands it covers; separate or differently oriented patterns remain
available for review. No reference-map outlines or semantic labels supply these
geometry steps.
