# cloudanalyzer-core

Python bindings for the Rust core of [CloudAnalyzer](https://github.com/rsasaki0109/CloudAnalyzer):
point cloud I/O (PLY, PCD, LAS, **LAZ**, XYZ), cloud-to-cloud and cloud-to-mesh
distances, ICP registration and filters. Points are `(N, 3)` float64 NumPy
arrays; heavy calls release the GIL and use all cores.

```python
import cloudanalyzer_core as cc
import numpy as np

scan = cc.read("scan.laz")
# COPC, local or remote: only the octree levels that fit max_points are read
tile = cc.read_copc("https://s3.amazonaws.com/hobu-lidar/autzen-classified.copc.laz", max_points=3_000_000)                 # {"positions", "colors"?, "intensity"?, "classification"?}
ref = cc.read("reference.pcd")["positions"]
d = cc.nearest_distances(scan["positions"], ref)   # C2C, one distance per point

vertices, triangles = cc.read_mesh("design.stl")
signed = cc.cloud_to_mesh(scan["positions"], vertices, triangles)  # C2M

result = cc.icp(scan["positions"], ref)    # {"transformation": 4x4, "rms_final", ...}
keep = cc.statistical_outliers(scan["positions"], k=8, ratio=1.0)  # multi-threaded
ground = cc.ground_csf(scan["positions"], cloth_resolution=1.0)  # bool per point

normals = cc.normals(scan["positions"], k=12)  # (N, 3) float32, facing +z

# Cross-section: points within 0.25 of a polyline, and their distance along it
idx, along = cc.profile(scan["positions"], np.array([[0.0, 0.0], [50.0, 20.0]]), half_width=0.25)

# M3C2 change from ref to scan at core points (multi-threaded; NaN = no data)
distance, lod95, significant, normals = cc.m3c2(scan["positions"][::10], ref, scan["positions"],
                                                normal_radius=1.0, projection_radius=0.5)

v = cc.volume(0.0, scan["positions"], cell=0.5)   # cut/fill vs a z = 0 plane
print(v["added"], v["removed"], v["net"])
```

The `cloudanalyzer` CLI uses this package automatically when it is installed
(`pip install "cloudanalyzer[fast]"`); set `CA_DISABLE_RUST_CORE=1` to force
the Open3D implementation.

Build from source with [maturin](https://www.maturin.rs/): `maturin develop --release`.

For full-density spatial batches install `cloudanalyzer-core[copc]` (CloudAnalyzer
already includes laspy). Use a context to close a local file on early exit:

```python
with cc.CopcStream("survey.copc.laz", bounds=(0, 0, -5, 100, 100, 20), chunk_size=10000) as stream:
    for batch in stream:
        xyz = batch.positions  # float64, in the source coordinate system
        classification = batch.records.classification
        # Consume and discard: retaining batches grows caller memory.
```

Local files and HTTP(S)/presigned URLs use the same bounded full-density walker.
`CopcLimits` sets per-page/node/metadata limits; oversized required items fail.
`CopcBatch.records` retains all LAS fields and `node_offset`/`ordinals` identify
source records. `cancel=event.is_set` raises `CopcCancelled` at range/node/batch
boundaries. [Large-cloud conditions and real measurements](../../../docs/large-point-clouds.md)
distinguish this iterator from the LOD display API and a resumable tile job.

`build_vector_map(cloud, trajectory, options="{}", reference_map=None, georeference=None)`
returns a JSON string containing a draft Lanelet2 map, projector metadata, editable map IR
and evidence/validation report. Inputs must already share a metre frame. For publication
of all artifacts together, use [`ca vectormap-build`](../../../docs/commands/vectormap-build.md)
or the `build_vector_map` tool from `ca mcp`.

`connect_vector_map_junctions(cloud, vector_map, options="{}", lane_pairs=None, preview_only=False)`
returns the same artifacts with ground-supported junction candidates and added lane IDs.
Options and optional `[from,to]` pairs are JSON strings. Preview is read-only; omitted
pairs add all supported branches. Existing IR geometry/rules/coordinates remain fixed;
new connections require review of permitted turns and clearance. Use
[`ca vectormap-connect`](../../../docs/commands/vectormap-connect.md) or its MCP tool for
atomic publication into a new directory.

`measure_vector_map_signal(cloud, vector_map, options, preview_only=True)` measures
a user-identified signal head. Options are JSON `min`, `max`, explicit `lanes`
and optional `kind`. It returns measured geometry and support plus the usual map
artifacts. Classification, controlled lanes, lamps and stop lines are not inferred.
Use [`ca vectormap-signal`](../../../docs/commands/vectormap-signal.md) or its MCP
tool to preview and publish the artifacts together.
