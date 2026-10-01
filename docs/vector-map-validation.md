# Vector map draft validation

The builder takes a surveyed point cloud and a measured trajectory in the same metre frame.
It assumes the trajectory follows the outside forward lane, uses configured lane counts and
widths, and estimates road elevation from the cloud. It detects intensity peaks, curb steps
and ground coverage edges within a limited distance of each nominal boundary. Missing
features use the explicit width prior. Detected features are candidates, not proof of a correct
lane boundary. Repeated passes and junctions require manual review and connection.

## Recorded Autoware drive and reference map

Inputs are the official [Autoware demo artifacts](https://github.com/autowarefoundation/autoware/blob/main/ansible/roles/demo_artifacts/tasks/main.yaml):

- `maps/demos/sample-map-rosbag.zip`, SHA256 `07e2da0b0bf12e2324f7083c2ce5556fb8044c50cef1da6428ab9084c3903bc8`.
- `recordings/bags/demos/sample-rosbag.zip`, SHA256 `5f9d36353393b3d249212153c19049822b1298db56512aa045b4f7f6fc37cf88`.

The map has 326,867 surveyed points and no intensity attribute. The recording supplies 30
NavSatFix measurements over about 127 m; they are projected into MGRS tile 54SUE with
PROJ. The reference Lanelet2 geometry is used only for scoring, never for extraction.
The evaluation copies only its coordinate metadata into the generated map.

With the original defaults (outer-edge anchoring disabled), the builder produced two supported stretches, six lane sections and
122.13 m of reference line. One section lacked ground support. Of 192 boundary vertices,
52 were curb candidates, 24 coverage edge candidates and 116 width priors. The measured
fractions for the three boundaries were 50%, 0% and 68.75%.

Reference boundaries are those of the nearest lanes visited within 5 m of the recorded
fixes and their immediate neighbors, excluding virtual boundaries. The scoring samples
both reference and generated polylines at approximately 0.1 m and uses XY distance; their
elevation datums differ. Whole selected reference boundaries are scored, including portions
beyond the recorded drive. At a 0.3 m tolerance:

| Metric | Result |
|---|---:|
| Generated samples close to a reference boundary (precision) | 15.96% |
| Reference samples close to a generated boundary (recall) | 5.68% |
| F1 | 0.0838 |
| Detected candidate vertices close to reference boundaries | 26.32% |

These low scores establish that this GNSS-driven, intensity-free draft requires substantial
geometry review. Export format checks passing do not establish lane geometry accuracy.
No survey-grade accuracy is claimed. Inputs are not redistributed or used as README artwork.

## Outer-edge anchoring and reproducible diagnostics

The original width prior placed missing lane lines relative to the trajectory, assuming it
was at the outside forward lane's centre. The recorded drive is often off the reference
centreline, so this assumption displaces otherwise plausible lines. The builder now uses
detected outer-edge offsets to position inferred boundaries, with a five-section median
to reject isolated candidates. Detected vertices stay fixed and inferred vertices remain
labelled as width priors. This cannot recover unobserved paint or establish lane counts.

On the same files, lane counts and sampling method, with no reference geometry supplied
to extraction:

| Metric at 0.3 m | Original / anchoring disabled | Anchoring enabled |
|---|---:|---:|
| Boundary precision | 15.96% | 26.56% |
| Boundary recall | 5.68% | 9.55% |
| F1 | 0.0838 | 0.1404 |
| Detected candidate vertex precision | 26.32% | 26.32% |

The improvement is limited to inferred geometry. Road-edge detection remains imperfect,
and overall accuracy still requires manual review. Changing ground extraction or interpreting
coverage gaps as actual curbs is not justified by this experiment. The disabled mode
reproduces the original measurements; no scoring tolerances or reference selections changed.

The evaluation example also writes `comparison.json`, containing external reference
boundaries, a cloud preview, the recorded trajectory and draft vertices with evidence labels.
Plot an overlay, per-evidence error distribution and elevation comparison:

```sh
pip install numpy scipy matplotlib
python scripts/vector_map_diagnose.py results/comparison.json results/plots
```

It writes `comparison.png`, `comparison.svg` and `diagnostics.json`; keep these external
data artifacts outside version control. The original candidate errors were median 0.60 m
for curbs, 0.52 m for coverage edges and 0.74 m for width priors. These are distances to
the selected reference, not a measured GNSS sensor error. To reproduce the ablation,
pass a fifth argument to the Rust example: a JSON options file containing
`{"anchor_width_prior": false}`. CLI uses `--no-anchor-width-prior`; MCP accepts
`anchor_width_prior=false`; the Web build panel has the corresponding checkbox.

To reproduce after downloading and extracting the two official archives:

```sh
pip install rosbags pyproj
python scripts/vector_map_gnss.py sample-rosbag/sample-rosbag_0.db3 trajectory.csv --zone 54
cargo run --manifest-path rust/Cargo.toml -p ca-wasm --release --example vector_map_evaluate -- sample-map-rosbag/pointcloud_map.pcd trajectory.csv sample-map-rosbag/lanelet2_map.osm results
```

The example writes the map, projector metadata and `report.json` with extraction counts,
XY scores and export issues. Filenames are external inputs; the repository contains no
reference geometry or captured sensor data.

## Intensity path on PandaSet

[PandaSet](https://github.com/scaleapi/pandaset-devkit) scene 019 (CC BY 4.0), Pandar64
frames 0, 8, …, 72 in world coordinates, supplies approximately 1.07 million actual
intensity-bearing points. The original 80 sensor poses supply an 87.20 m trajectory.
With right-hand traffic and otherwise default options, the builder produced one stretch,
four lane sections, 29 intensity candidate vertices, 46 curb candidates and 60 width priors.
No sections lacked ground support. Detected fractions were 77.78%, 20% and 68.89%.

This checks that real intensity values reach and exercise the detector. No corresponding
Lanelet2 reference was available for this test, so these counts are not accuracy metrics.
Use `-` as the reference argument of the evaluation example to obtain counts without scores;
that mode uses right-hand traffic and a Local projector. The existing
`scripts/fetch_pandaset.py` documents the public archive and sensor/world transforms.
