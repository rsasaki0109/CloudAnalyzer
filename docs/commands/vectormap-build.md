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

Traffic keeps left by default. Options include `--right-hand`, `--forward-lanes 1`,
`--backward-lanes 1`, `--lane-width 3.5` (metres), `--speed-limit 40` (km/h) and
`--segment-length 50` (metres; zero keeps each stretch whole).

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
Use the editable JSON to retain all IR editing information. Matching is enabled by default,
including in the Web build panel and MCP tool (`existing_map`, `merge_repeated_passes`).
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
