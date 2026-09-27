# cloudanalyzer-core

Python bindings for the Rust core of [CloudAnalyzer](https://github.com/rsasaki0109/CloudAnalyzer):
point cloud I/O (PLY, PCD, LAS, **LAZ**, XYZ), cloud-to-cloud and cloud-to-mesh
distances, ICP registration and filters. Points are `(N, 3)` float64 NumPy
arrays; heavy calls release the GIL and use all cores.

```python
import cloudanalyzer_core as cc

scan = cc.read("scan.laz")                 # {"positions", "colors"?, "intensity"?, "classification"?}
ref = cc.read("reference.pcd")["positions"]
d = cc.nearest_distances(scan["positions"], ref)   # C2C, one distance per point

vertices, triangles = cc.read_mesh("design.stl")
signed = cc.cloud_to_mesh(scan["positions"], vertices, triangles)  # C2M

result = cc.icp(scan["positions"], ref)    # {"transformation": 4x4, "rms_final", ...}
keep = cc.statistical_outliers(scan["positions"], k=8, ratio=1.0)
ground = cc.ground_csf(scan["positions"], cloth_resolution=1.0)  # bool per point

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
