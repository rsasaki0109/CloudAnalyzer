# ca vectormap-connect

Preview junction connections between open road ends, using a point cloud in the map's
metre coordinate frame:

```sh
ca vectormap-connect map.pcd draft-map/vector_map.json --preview --out junction-preview
ca web-view map.pcd junction-preview
```

The preview leaves map geometry and topology unchanged. `report.json` contains all
supported candidates, with incoming/outgoing lane IDs, centre and boundary geometry,
gap, heading change, ground support and a flag for branches sharing road ends.
In the Web **Draft junction connections** panel, select a cloud and preview the cyan
curves; **Show** frames a candidate. Select the intended branches and add them together.
One Undo removes the batch. Editing the map or changing inputs invalidates the preview.

After reviewing the report, use the original input and another new output directory:

```sh
ca vectormap-connect map.pcd draft-map/vector_map.json --pair 7:8 --pair 7:9 --out connected-map
```

Omit `--pair` to add all supported proposals automatically as drafts. Each selected pair
is checked again against the current map and cloud; an unsupported pair rejects the whole
batch. The Python/MCP tool `connect_vector_map_junctions` uses `preview_only`, `lane_pairs`,
`max_gap` and `min_ground_support`; `lane_pairs=[]` adds nothing, and omitting it adds all.
All four artifacts (`lanelet2_map.osm`, `map_projector_info.yaml`, `vector_map.json`,
`report.json`) are published together. Existing output directories are never overwritten.
Requires the updated native Rust core, as [vectormap-build](vectormap-build.md) does.

Only driving lanes without turn labels and with open directed ends participate.
Existing connections remain authoritative. Default gates are a 0.5–30 m XY gap,
endpoint height difference at most 0.3 m, headings pointing into/out of the gap, and
turn magnitude at most 135 degrees. The exact generated connector centre is sampled
every 0.5 m. A sample requires at least three cloud points within 0.75 m XY; their 15th
percentile height must be within 0.3 m of the sample. At least 90% of samples must pass.
Change the distance with `--max-gap` (1–100 m) and the fraction with
`--min-ground-support` (0.5–1). Inputs are loaded into memory; this is not large-cloud
streaming or registration.

Connections use virtual boundaries and inherit the lower endpoint speed limit when
available. New lanes carry `cloudanalyzer_geometry_source=ground_supported_connection`
and `cloudanalyzer_review_required=yes`, including in OSM. Existing geometry, IDs,
rules and projector metadata remain fixed. Prefer editable IR JSON: unsupported OSM
members/types can be dropped on import and are reported in `import_issues`.

Ground beneath a centre curve does **not** establish a permitted turn, unobstructed lane
width, marking accuracy or right-of-way. No signal, stop or priority rules are inferred.
Branches remain available for review rather than being reduced to a single choice.
Occlusion, sparse scans, ramps, elevated crossings, U-turns and existing partial topology
can cause missed or incorrect candidates. Once any connection occupies a road end, that
end is excluded from later automatic passes; add further reviewed branches manually or
select the complete group together from the original input. Review the exported map and
traffic rules before use. See [real-data validation](../vector-map-validation.md).
