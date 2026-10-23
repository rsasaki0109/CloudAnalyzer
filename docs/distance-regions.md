# Locate distance discrepancy regions

In **Map coverage & quality**, select a source map and an aligned reference,
then expand **Locate distance discrepancies**. Choose a distance threshold,
grouping XY cell size and minimum exceeding points per region, and click
**Compute & locate regions**.

Use **Compare both directions** to compute source → reference and reference →
source sequentially with the same settings. The summary shows exceeding and
measured counts, their ratio, and unmeasured/invalid counts separately for each
direction. Ratios exclude unmeasured/invalid rows; without measured points the
ratio is unavailable. **Show direction** switches the heatmap, region list and
cell inspection to the corresponding source cloud without rerunning distances.
Switching direction clears the previous cell inspection. **Save both directions
JSON** retains both source identities, full counts, settings and listed regions;
the existing region JSON/CSV exports apply to the currently shown direction.
QA reports retain both-direction counts as well as the shown direction's detail.

Each direction caches its own distances for the exact input snapshots. Starting
with a one-direction diagnosis reuses that direction and computes only the
missing reverse distances. Threshold/grouping edits can reuse both snapshots.
The both-direction result is published only after both directions finish and
remain valid; cancellation or input/settings changes publish neither partial
diagnosis. Running **Compute & locate regions** returns to one-direction output.
Both-direction mode requires XYZ plus distances to fit within 64 MiB for each
cloud, rather than the reference-XYZ-only limit of one-direction mode.

The tool computes existing unsigned CPU C2C nearest-neighbor distances from the
loaded source to the loaded reference. Points strictly above the threshold are
grouped into eight-neighbor-connected XY cells, ranked by maximum distance,
then exceeding-point count. Click a row to frame its source-coordinate bounds.
The mean is over exceeding points in that region, not all points in its box.
Cell size controls grouping; connected chains can form a broad region spanning
much of a map. This is not object segmentation, and different heights can join.

For a broad region, set **Inspection XY cell** (positive and no larger than the
grouping cell), then click its **Inspect cells** button. This reuses the same
distance snapshot and regroups only exceeding points belonging to that exact
connected component. A different component inside its bounding box is excluded.
Fine cells are independent bins, so adjacent cells do not merge into another
broad chain. The XY origin is the minimum finite source XY, shared with coarse
grouping; grid indices are zero-based, with half-open cells.

The inspection ranks up to 50 occupied cells by maximum distance, then point
count, then grid X/Y. It has no minimum-support filter within a region: a lone
outlier can rank first. Click a cell row to frame its exceeding-point bounds
with surrounding context (padding at least two inspection cells or 1.1 times
its maximum distance) so an isolated source point does not fill the view alone.
The summary explicitly counts points in unlisted cells. Separate cell JSON/CSV
exports retain the parent region and settings (JSON), grid indices, counts,
maximum/mean distances and original-coordinate point bounds. Cell CSV contains
only listed rows; JSON records the grid origin and omitted-point count. These
are point envelopes, not the full grid-cell footprint. QA reports include the
current cell inspection. Editing the inspection size clears that inspection;
changing the comparison/grouping settings or either cloud clears both levels.
Cancellation stops the inspection worker and preserves the parent diagnosis.

JSON includes the pair's runtime identities, loaded/file counts and sampling
scope, settings, measured/unknown counts, total exceedances and listed regions.
CSV contains only the listed ranked regions and their original-coordinate
bounds. At most 50 eligible regions are listed/exported; counts for unlisted
eligible points and below-minimum regions remain explicit. QA reports include
the current diagnosis; changing settings or invalidating either input clears it.

Distances are reused when only grouping settings change and both exact loaded
cloud snapshots remain valid. Editing/removing/replacing either input invalidates
that cache, including when another cloud has the same name. Changing the shown
source's distance result also hides the old diagnosis. Cancelling grouping stops
its worker immediately. An active C2C call can finish before cancellation takes
effect; its diagnosis is discarded without publication.

Units are the source clouds' coordinate units; no registration or CRS conversion
is performed. The effective threshold is rounded to the precision of the stored
f32 distances and recorded in JSON. Input coordinates plus distances are limited
to 64 MiB, reference XYZ to 64 MiB, and exceeding occupied cells to 50,000. Limit
failures request a crop/coarser cell rather than silently truncating inputs.
Fine inspection has the same 50,000 occupied-cell limit and rejects an
unrepresentable or over-budget grid rather than truncating it.

**Large distances are inspection candidates, not confirmed registration defects.**
Scene changes, occlusion, sparse sampling and missing reference coverage can all
produce them. This direction does not locate points missing from the source;
use **Compare both directions** or swap the pair to inspect the reverse direction.
Reference → source can expose reference areas absent from the source, but this
is not proof of missing map data without an expected common survey footprint.
It does not use signed M3C2
distances or their significance tests. Use an expected overlap/survey footprint
and source inspection before attributing a cause.

![Actual directed-distance review of existing Autoware survey subsets](images/web/distance-regions.png)

The [recorded smoke check](../benchmarks/map-coverage/distance-regions.json)
uses unchanged alternating-index subsets of one Autoware planning survey:
878,920 source and 878,921 reference points. With threshold 0.3, XY cell 5 and
minimum support 5, there are 20,785 exceeding points, 40 components and 14 eligible
regions; 53 points belong to below-minimum components. The largest region links
20,421 points across much of the map. These results are a data-handling/review
check, not evidence of 14 registration failures or an independent survey.

![Fine-cell review of the same measured survey partition, with surrounding context](images/web/distance-region-cells.png)

The [cell inspection receipt](../benchmarks/map-coverage/distance-region-cells.json)
records the same largest component inspected with a 1-unit XY cell: 20,421
exceeding points in 10,237 occupied cells. The top 50 cells contain 63 points;
20,358 points remain explicitly unlisted. The highest-ranked cell is a single
point with distance 16.375; its focused view includes nearby context. This shows
why maximum-distance ranking requires source inspection and is not a defect
classifier. The earlier region receipt remains bound to its original commit.
