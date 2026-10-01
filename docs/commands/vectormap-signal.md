# Measure a traffic signal from a point cloud

`ca vectormap-signal` measures the housing of a **user-identified** traffic signal
head from a tight 3D box. It uses surveyed XYZ points for the position, width,
height and horizontal orientation. Object classification and controlled lane IDs
are supplied by the user. This is assisted map creation, not automatic signal detection.

The cloud and map must already use the same metre frame. Prefer editable
`vector_map.json` to preserve all rules and metadata; OSM import issues are reported.
Exclude the pole, nearby lamps and background from the box. Bounds are original
coordinates in `xmin,ymin,zmin,xmax,ymax,zmax` order, with increasing limits.

```sh
ca vectormap-signal survey.pcd vector_map.json --out signal-preview \
  --box "3822.5,73783.1,24.65,3824.1,73784.5,25.5" --lane 85
```

Preview is the default: the map remains unchanged and the four usual artifacts
include `report.json` with measured geometry and support. Inspect it over the cloud
with `ca web-view survey.pcd signal-preview`. Use a bright solid point color when
dark RGB returns are hard to see. Confirm the object, head extent, face direction
and controlled lanes before adding to a **different new directory**:

```sh
ca vectormap-signal survey.pcd vector_map.json --out signal-added \
  --box "3822.5,73783.1,24.65,3824.1,73784.5,25.5" --lane 85 --add
```

Repeat `--lane` for multiple explicitly confirmed lanes. `--kind pedestrian`
selects a pedestrian signal; vehicle is the default. Example coordinates and lane
85 are illustrative inputs, not a verified control relationship.

The fit uses horizontal PCA around a local f64 origin and a vertical housing
plane. The bottom edge and height use the 2nd–98th percentiles of observed returns;
they estimate the visible footprint, not the full occluded housing. Left-to-right
ordering follows the first supplied lane's end heading. Check this orientation
for curved approaches. Boxes are limited to 10 m per axis and 200,000 selected
points. At least 12 finite points, width 0.15–3.5 m, height 0.15–2.5 m, thickness
at most 0.5 m and plane RMS at most 0.2 m are required. These checks reject sparse
or poorly isolated shapes; signs and other compact planar objects can still pass.

Adding recomputes the fit, preserves existing lane geometry and rules, and creates
a traffic-light rule only for the supplied lanes. No stop line, lamp color, arrow,
or light state is synthesized. Both signal and rule carry review-required tags.
Replaying the identical box/kind/lane request on its output IR reuses the added
signal without changing the map, including after OSM import. Different boxes, kinds
or lane assignments can produce duplicates;
review existing heads and use the identical request for replay.

In the Web panel, open **Measure a signal from points**. Segment an identified
head into a cloud and use **Use cloud bounds**, or enter the six bounds directly.
Supply lane IDs or select a lane and click **Use selected lane**. **Measure and
preview** frames the measured cyan rectangle; **Add reviewed signal** adds it as
one Undo step. Changing inputs or the cloud/map invalidates the preview.
If background makes the points hard to distinguish, **Inspect box points** copies
the validated box into a separate cloud, gives it a solid color and hides its
source. Measure that isolated cloud again before adding. Cloud Undo restores the
source; map Undo removes the signal. Original coordinates and attributes are retained.
Only loaded points are used: a sparse LOD display is not a full-density measurement.
For COPC, first use **COPC full-density box** to read all overlapping density
levels into a capped working cloud, then select that cloud for measurement.

Python and MCP expose the same workflow:

```python
from ca.vector_map import measure_vector_map_signal

report = measure_vector_map_signal(
    "survey.pcd", "vector_map.json", "signal-preview",
    bounds=[3822.5, 73783.1, 24.65, 3824.1, 73784.5, 25.5],
    lanes=[85], preview_only=True,
)
```

MCP tool `measure_vector_map_signal` defaults to `preview_only=true`. Python,
CLI and MCP use the same high-level reader: local LAS/LAZ and CSV are scanned in
10,000-point chunks, filtering the inclusive box and retaining at most 200,000
selected XYZ64 points. Local `.copc.laz` and HTTP(S) COPC `.laz` URLs use bounded
full-density spatial traversal. A small box does not avoid a sequential scan of
ordinary LAS/LAZ; COPC can skip non-overlapping nodes. Oversized boxes or selected
outputs fail without publishing artifacts. The current native core is required
for spatial signal inputs; no display-level thinning substitutes for selection.

Other formats, including PCD/PLY/XYZ, retain the complete-file compatibility
reader. Its selection cap does not limit source memory. The low-level native
`cloudanalyzer_core.measure_vector_map_signal` file method also reads whole files;
`measure_vector_map_signal_points` instead accepts at most 200,000 finite `(N,3)`
float64 points supplied by a caller. These limits exclude libraries, metadata and
decoder buffers; see [large-cloud limits](../large-point-clouds.md).
Reports identify the processing strategy. Remote source URLs in published reports
omit query strings, fragments and user credentials. This is not a physical
10-billion-point benchmark. See [real-data validation](../vector-map-validation.md).
