# Locate distance discrepancy regions

In **Map coverage & quality**, select a source map and an aligned reference,
then expand **Locate distance discrepancies**. Choose a distance threshold,
grouping XY cell size and minimum exceeding points per region, and click
**Compute & locate regions**.

The tool computes existing unsigned CPU C2C nearest-neighbor distances from the
loaded source to the loaded reference. Points strictly above the threshold are
grouped into eight-neighbor-connected XY cells, ranked by maximum distance,
then exceeding-point count. Click a row to frame its source-coordinate bounds.
The mean is over exceeding points in that region, not all points in its box.
Cell size controls grouping; connected chains can form a broad region spanning
much of a map. This is not object segmentation, and different heights can join.

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

**Large distances are inspection candidates, not confirmed registration defects.**
Scene changes, occlusion, sparse sampling and missing reference coverage can all
produce them. This direction does not locate points missing from the source;
swap the pair to inspect the reverse direction. It does not use signed M3C2
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
