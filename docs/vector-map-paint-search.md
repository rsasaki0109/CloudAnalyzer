# Bounded paint-neighbourhood search

Paint fitting now subdivides dense spatial cells before holding an over-budget
neighbourhood. The query radius, ground/contrast rules and 4,096-point examination
cap remain unchanged. No points are sampled away and no partial scan selects a
paint corridor. This applies to both explicit paint channels and to the shared
ground queries used by divider/lane-edge inference.

Previously, a 0.8 m circular query counted every point in its surrounding 0.5 m
square-grid cells. Unrelated cells outside the circle could exhaust the cap.
Ordinary queries still use the original traversal. A potential overflow instead
prunes cells outside the circle and uses 0.125 m subcells for dense cells with
more than 128 points. Cell lengths are checked before point visits, and completed
results retain the previous source order for component sums and nearest-source
ties. Raw XYZ/RGB/intensity are unchanged.

This remains conservative: intersecting cells can contain points outside the
circle, and their potential points consume the cap. It is not an unlimited or
out-of-core scan. Ground checks still use the original height quantiles; elevated
returns inside the search radius have not been silently discarded. ROI,
brightness and contrasted-paint limits remain 250,000 / 50,000 / 10,000.

When a fit is held by a limit, its report now includes optional `budget_stage`.
Neighbour-query limits also include `budget_query` with local trace coordinates,
radius, potential candidate count and cap. The Web report describes the failing
operation, for example **ground near the trace**, and its counts. These are
search diagnostics, not observed marking coordinates or confidence values.
The divider propagates scan diagnostics and reports its own output-ground limit.

## Actual source result

The same six cached Tokyo development courses were generated without a reference
map, using the previous explicit RGB/intensity inputs. Their twelve IR/OSM maps
and source-profile geometry remain identical to the previous stage. All new
paint corrections still hold; **no geometric accuracy improvement is established**.

| Intensity course | Previous corridor result | Current corridor result |
| --- | --- | --- |
| arterial-south | Curved trace | Curved trace; unchanged |
| arterial-middle | Neighbour query limit | Scan completes: 7,147 bright candidates, 24 contrasted points, no narrow longitudinal components |
| arterial-north | ROI limit | ROI limit; diagnostic identifies `roi_points` |
| west-south | Neighbour query limit | A later contrast query reaches 4,115 potential points; correction held |
| east-south | Neighbour query limit | A later contrast query reaches 4,105 potential points; correction held |
| west-north | Curved trace | Curved trace; unchanged |

An independent source-only cross-check of the remaining circular queries finds
3,321 actual points for west-south and 4,576 for east-south. The west case still
illustrates conservative boundary-cell overhead; the east case is genuinely
above 4,096 inside the circle. These exact-circle counts are a separate Python
cross-check, not native point-visit telemetry. The current query reports count
potential cells up to the first rejected cell, rather than completing a scan.
Neither count is a lane-detection or survey-accuracy measurement.

Counts, directions, initial widths and marking roles remain manual. The two
cached XYZ-only negative builds still reject all unsupported road intervals.
No reference geometry was opened in this iteration, and these reused development
inputs are not held-out evaluation data.

Known-shape tests confirm that 5,000 elevated returns outside a paint query no
longer prevent the same 3 m fit, while genuinely dense nearby returns hold the
entire optional correction. Brute-force comparisons cover negative coordinates,
cell/radius boundaries and the preserved traversal order. Existing affine-intensity,
curb, gap and ambiguity checks continue to pass.

The default RGB batch retains all 36 IR/OSM files and all 18 source-profile
geometries exactly. Of its 83 generated artifacts, 79 remain byte-identical;
four audit/profile documents add only the optional ROI-limit diagnosis. They
match the previous documents when those two diagnostic keys are omitted.
The previous verification artifacts and published GIF stay unchanged.

[Source-only hash manifests, reports and native/browser verification](../benchmarks/vector-map/paint-neighbourhood/README.md)
record the exact runtime and separately reproduce sixteen native builds
(twelve accumulated-course maps and four isolated browser baselines). No source
cloud or duplicate generated-map batch is added to this proof package. The
[previous intensity screenshot](vector-map-intensity-paint.md) remains a
historical source-only draft with held corrections.

Source preparation, licensing, hashes and frozen operator inputs are retained in
the [explicit-intensity record](../benchmarks/vector-map/intensity-paint/README.md)
and [previous lane-edge record](../benchmarks/vector-map/lane-edge-inference/README.md).
