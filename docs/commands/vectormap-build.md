# ca vectormap-build

Draft Autoware Lanelet2 roads from a surveyed point cloud and a recorded trajectory:

```sh
ca vectormap-build map.pcd trajectory.tum --out draft-map
```

Requires the updated Rust core (`pip install "cloudanalyzer[fast]"`, or build
`rust/crates/ca-py` with `python -m maturin build --release` and install its wheel).
The output directory must be new. All four artifacts are published together:
`lanelet2_map.osm`, `map_projector_info.yaml`, `vector_map.json` and `report.json`.
The command prints that report as JSON, including candidate counts, inferred boundary
counts, supported length, Autoware issues and geometry validation results.

Inputs must already share a coordinate frame in metres. TUM, KITTI and timestamped XYZ
CSV trajectories are accepted. The trajectory must follow the outside forward lane.
Ground heights come from the cloud, not sensor poses. Intensity peaks, curb steps and
the end of ground coverage provide boundary candidates. Missing features use a width prior;
missing ground splits the road. Coverage edges can be scan gaps. Review the draft geometry,
lane counts, directions, repeated passes and junctions before using it.

Inferred lines are anchored to detected outer edges when available, while keeping their
inferred label. This reduces dependence on the trajectory being exactly at the lane centre.
`--no-anchor-width-prior` disables that adjustment; the MCP option is `anchor_width_prior`.
Missing observations are excluded from the five-section offset median. An anchor can
extend to two neighbouring sampled sections with decreasing strength; farther missing
slices retain the trajectory-centred width prior. This does not bridge missing ground.

Opt in with `--physical-anchors-only`, Python/MCP `physical_anchors_only=True`,
or Web's **Anchor inferred lane widths to paint and curbs only** under **Build from a trajectory**.
This excludes outside point-coverage candidates from the offset median for inferred
width lines. Curb and intensity observations can still supply offsets. Direct
coverage-edge geometry remains selectable; the option does not certify road edges.
Later source-footprint fitting remains independent and can still infer a footprint
from low-surface coverage.
`coverage_edge_anchor_candidates_ignored` counts excluded outside candidates before
tracking and surface deferral, including candidates that are later dropped. It is
not a false-detection or removed-geometry count. The option defaults to off and has
no effect with `--no-anchor-width-prior`. See the [same-interval gains, losses and
coverage changes](../vector-map-physical-anchors.md).

Opt in with `--align-trace-to-curbs`, Python/MCP `align_trace_to_curbs=True`,
or Web's **Align straight traces using paired curbs**. This independently translates
a straight trace when a stable pair of source curbs encloses the configured road
width. It requires a majority of sections, three consecutive pairs, consistent
offsets and no ambiguous sections; curved or unconfirmed traces stay unchanged.
Lane counts and widths remain explicit priors. `trace_alignment` records the
shift and held reason. This translation precedes the separate 0.5 m local curve
fit, so its reported movement may exceed that cap. The option is off by default.
See the [actual maps and correspondence-aware evaluation](../vector-map-curb-alignment.md).

Opt in with `--fit-paint-corridor`, Python/MCP `fit_paint_corridor=True`, or Web's
**Fit straight lanes using paint candidates**. This independently fits
straight parallel boundary heading and spacing from thin source RGB paint with
dark ground on both sides. Counts, directions and marking roles remain manual.
`paint_corridor` reports measured widths and each line's source observation
intervals, interpolation and extrapolation before source-footprint trimming.
Sparse outer lines can be extended; unobserved vertices remain inferred.
Missing/ambiguous bundles and scan limits hold the fit. It is off by default,
and its separate relocation is not bounded by the later local curve-fit cap.
See [actual maps, retained extent and both fixed-target diagnostics](../vector-map-paint-corridor.md).

Opt in with `--fit-paint-divider`, Python/MCP `fit_paint_divider=True`, or Web's
**Correct the interior line using paint and paired curbs**. This corrects only
the interior boundary of an explicitly configured two-lane road when one strong
RGB paint track lies inside confirmed physical curb pairs in a majority of
sections. Outside candidates stay intact; a complete paint corridor takes
precedence. `paint_divider` reports the guard counts, movement and observed,
interpolated and extended source intervals. Missing or ambiguous evidence holds
the correction. Counts, boundary roles and directions remain manual; nearby
observed paint alone receives RGB evidence. The option defaults to off and its
separate relocation can exceed the 0.5 m local curve-fit cap.
See [actual maps and per-boundary errors](../vector-map-paint-divider.md).

Choose the paint-fit input explicitly with `--paint-channel intensity`, Python/MCP
`paint_channel="intensity"`, or Web's **Paint-fit source → Retained intensity**.
RGB remains the default; channel selection alone does not enable a fit or provide
an automatic fallback. Retained intensity is temporarily normalized within the
ROI while raw points stay unchanged. Observed fitted paint uses intensity evidence;
gaps and extensions stay inferred. Missing, ambiguous or over-budget evidence holds
the correction. See [normalization, source-only checks and real-data holds](../vector-map-intensity-paint.md).

Over-budget paint neighbourhoods use a finer index in dense cells before holding
the fit. Reports identify `budget_stage` and, for query limits, `budget_query`
with potential point count, radius and cap. Web displays the failing operation.
These diagnostics do not certify markings or relax the examination limits.
See [bounded search and remaining source failures](../vector-map-paint-search.md).

Curb candidates must have nearby road-side support and two raised outside bins no taller
than `curb_height + 0.3 m` above the candidate. This rejects wall/vehicle steps and isolated
low returns, but can also reject genuine curbs in sparse or cluttered scans. The report's
`rejected_curb_candidates` counts rejected cross-section height-step candidates, not
selected vertices or known false detections. Disable with `--no-verify-curb-profiles`,
Python/MCP `verify_curb_profiles=false`, or Web's **Check curb profiles** checkbox.
Remaining candidates and inferred lines still need review; filtering does not guarantee
better geometry. See the [fixed-input measurements](../vector-map-validation.md#curb-profile-checks-and-missing-observation-anchors).

Boundary candidates are tracked across supported slices to reject isolated peaks. Missing
evidence can become a labelled width prior. Trajectory-relative lateral curve fitting then reduces jitter,
with at most 0.5 m XY movement per vertex and unchanged ground heights. This is geometry
fitting, not detection of additional lane markings. Source counts and observed fractions
refer to selected positions **before** fitting. The report also includes `tracked_vertices`,
`fitted_vertices` and `maximum_fit_displacement`.
`--no-track-boundaries` and `--no-fit-boundaries` disable these stages independently;
Python/MCP use `track_boundaries=false` and `fit_boundaries=false`. Web has corresponding
checkboxes. Disable both to reproduce the earlier independently selected boundary geometry.

The fitter smooths lateral deviations from the sampled trajectory instead of independently
fitting XY coordinates. It uses metre spacing, gives observed candidates more weight than
width assumptions, and preserves sharp corners with straight observed support on both sides.
Those weights are heuristics, not calibrated confidence probabilities. Heights, source
positions and evidence labels stay fixed; fitted positions are inferred geometry. See the
[fixed-input shape comparison](../vector-map-validation.md#trajectory-relative-boundary-stability).

Traffic keeps left by default. Options include `--right-hand`, `--forward-lanes 1`,
`--backward-lanes 1`, `--lane-width 3.5` (metres), `--speed-limit 40` (km/h) and
`--segment-length 50` (metres; zero keeps each stretch whole).

Opt in to source-footprint drafting with `--fit-source-surface`, Python/MCP
`fit_source_surface=True`, or Web's **Fit road footprint to source points** under
**Build from a trajectory**. Source-backed incoming candidates are preserved and
clipped first; weak candidates use a coherent low surface to infer widths and
heights. Missing intervals and short fragments are deferred, not filled. Lane
counts remain your input, and existing maps are not refitted automatically.
`extraction.surface_fit` reports retained width range and deferred/evaluated length.
The mode defaults to off. See the [same-input maps and coverage losses](../vector-map-source-footprint.md).

```sh
ca vectormap-build cloud.las trajectory.csv --out footprint-draft --fit-source-surface
```

For an existing map frame, copy only its coordinate metadata:

```sh
ca vectormap-build pointcloud_map.pcd trajectory.csv --out draft-map --reference-map original/lanelet2_map.osm
```

Its geometry is not used for extraction or copied into the result. Alternatively use
`--projection mgrs --origin-lat LAT --origin-lon LON`: the representative origin selects
a single UTM MGRS 100 km tile. For `utm` or `transverse_mercator` it defines the local
origin. These options describe the input frame; they do not transform either input.
Without coordinate metadata the map uses Autoware Local. Multi-tile MGRS and polar UPS
frames are unsupported.

Append another recorded pass to an editable map:

```sh
ca vectormap-build map.pcd next-drive.csv --existing-map draft-map/vector_map.json --out combined-map
```

Lanelet2 OSM is also accepted. `--existing-map` keeps the existing geometry, IDs, traffic
rules and coordinate metadata; omit `--reference-map` and explicit projection options.
Use the editable JSON to retain all IR editing information.
OSM imports report unsupported members and types in `import_issues`; preservation applies
to the imported IR, so review those issues before appending an external OSM map.
Matching is enabled by default, including in the Web build panel and MCP tool
(`existing_map`, `merge_repeated_passes`).
`--no-merge-repeated-passes` explicitly adds the entire pass instead.

Reuse requires the lane centre and both corresponding boundaries to agree within 0.5 m,
both boundary directions within 15 degrees and ground heights within 0.3 m. Uniquely
connected sections are compared as continuous edges; comparison centres use normalized
arc length at a fixed 0.5 m resolution so inserted split vertices do not change the test.
Explicit existing centreline geometry is respected. Nearby parallel lanes, opposite
travel, different ground levels and ambiguous disconnected duplicates are not fused.
Only uncovered trajectory intervals are added; existing geometry and rules stay fixed.
Coincident unambiguous endpoints can be connected, but gaps are not snapped.

`report.json` records incoming supported length, `reused_length`, `added_length`,
`reused_intervals` and `joined_connections`. Evidence counts describe the incoming pass;
they do not establish better survey accuracy. Partial overlap can retain duplicate or
disconnected fragments when extraction differs, so inspect validation and geometry.
Inputs must already be aligned; this operation does not correct drift between surveys.
An exact replay adds no geometry and creates no additional Web Undo entry.

To inspect and edit:

```sh
ca web-view map.pcd draft-map
claude mcp add vectormap -- vectormap mcp draft-map/lanelet2_map.osm
```

Register CloudAnalyzer with `claude mcp add cloudanalyzer -- ca mcp`. Its
`build_vector_map` tool takes `cloud`, `trajectory`, `out_dir`, lane and projection
options, and returns the same report. Call `view_link([cloud, out_dir])` to view the
cloud and map together. Later map editing uses the separate `vectormap mcp` server.
See [real-data validation](../vector-map-validation.md) for measured accuracy and limitations.

### Lane edges inside distant curb candidates

Opt in with `--infer-lane-edges`, Python/MCP `infer_lane_edges=True`, or Web’s
“Infer lane edges inside distant curbs (width assumption)”. Requires an applied
`--fit-paint-divider` correction and verified curb profiles. Review the configured
width: the result is an inferred lane edge, not observed outer paint or a certified
shoulder. Selected far-curb candidates require majority/run guards; every new
vertex needs source ground and must move inward. Outside paint takes precedence.
Original road-edge candidates remain separately in the report before footprint
trimming; exported lane vertices use width-prior evidence.
See [actual maps and fixed-reference comparison](../vector-map-lane-edges.md).
