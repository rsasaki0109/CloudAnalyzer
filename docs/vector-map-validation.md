# Vector map draft validation

The builder takes a surveyed point cloud and a measured trajectory in the same metre frame.
It assumes the trajectory follows the outside forward lane, uses configured lane counts and
widths, and estimates road elevation from the cloud. It detects intensity peaks, curb steps
and ground coverage edges within a limited distance of each nominal boundary. Missing
features use the explicit width prior. Detected features are candidates, not proof of a correct
lane boundary. Repeated passes must already be aligned and require review. Ground-supported
junction connections can be drafted automatically; geometry and traffic rules require review.

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
to reject isolated candidates. In this anchoring ablation, detected vertices stay fixed and inferred vertices remain
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
`{"anchor_width_prior": false, "track_boundaries": false, "fit_boundaries": false}`.
Keep tracking and fitting disabled for both sides of this historical anchoring ablation.
CLI uses `--no-anchor-width-prior`; MCP accepts
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
Before continuity tracking and fitting, with right-hand traffic and then-default options,
the builder produced one stretch,
four lane sections, 29 intensity candidate vertices, 46 curb candidates and 60 width priors.
No sections lacked ground support. Detected fractions were 77.78%, 20% and 68.89%.

This checks that real intensity values reach and exercise the detector. No corresponding
Lanelet2 reference was available for this test, so these counts are not accuracy metrics.
Use `-` as the reference argument of the evaluation example to obtain counts without scores;
that mode uses right-hand traffic and a Local projector. The existing
`scripts/fetch_pandaset.py` documents the public archive and sensor/world transforms.

## Boundary continuity and local curve fitting

Independently chosen cross-section peaks caused visible zigzags. The default builder now
selects a continuous candidate path, penalizing sudden lateral changes without consulting
reference geometry. Selected candidates can be replaced by labelled width priors when
evidence is inconsistent. Inferred candidates retain the original robust outer-edge anchor.
The path is fitted with weighted local quadratic curves over up to five vertices. XY
movement is capped at 0.5 m; ground heights stay fixed. Fits that cross adjacent boundaries
at a sampled cross-section are rejected. This does not guarantee topology between samples.

On the same recorded inputs, lane counts and 2 m cross-section sampling:

| Metric | Independently selected, anchored | Tracking and fitting |
|---|---:|---:|
| Autoware boundary precision at 0.3 m | 26.56% | 29.00% |
| Autoware boundary recall at 0.3 m | 9.55% | 10.16% |
| Autoware F1 | 0.1404 | 0.1505 |
| Autoware selected source vertex precision | 26.32% | 32.08% |
| Autoware adjacent-segment heading change, P95 | 46.66 degrees | 11.14 degrees |
| PandaSet adjacent-segment heading change, P95 | 35.92 degrees | 7.77 degrees |

Fitting moved 192 Autoware vertices by at most 0.194 m, and 135 PandaSet vertices by at
most 0.170 m. Tracking changed 103 and 41 selected sources respectively. Autoware source
counts changed from 52 curbs, 24 coverage edges and 116 width priors to 42, 11 and 139;
PandaSet counts changed from 29 intensity peaks, 46 curbs and 60 width priors to 27, 33
and 75. Thus continuity can reject measurements; it does not create new observed evidence.
Autoware scores still use the same nine reference lanes and 10,895 reference samples.
Generated supported lengths and unsupported sections did not change.

Heading variation measures zigzags, **not accuracy**. True corners can also be smoothed.
PandaSet still has no independent lane-boundary reference, and Autoware position accuracy
remains low. These two datasets informed development; there is no held-out generalization
claim. Inspect the cloud and source observations before accepting a draft. The evaluation
JSON retains `source_boundaries` before fitting; evidence scatter points and candidate
errors use these sources, while the red overlay lines and polyline accuracy scores use
fitted geometry.

To reproduce, run the Rust evaluation example once with default options and once with
`{"track_boundaries": false, "fit_boundaries": false}` in an options file. Include
`"left_hand_traffic": false` in **both** options files for PandaSet. Then compare:

```sh
python scripts/vector_map_fit_evaluate.py baseline/comparison.json fitted/comparison.json heading-change.json
python scripts/vector_map_diagnose.py fitted/comparison.json fitted/plots
```

The comparison checks identical input previews, references, trajectories, other options
and cross-section sampling, and refuses to overwrite its result. Disable stages independently
through Web, CLI or Python/MCP as described in the command documentation.

The generated PandaSet map also exercises Web vertex editing: a shared internal boundary
vertex was dragged with its cloud-relative height unchanged, both adjacent lanes updated,
and one Undo restored the complete map view exactly. This checks editing behavior, not
whether the new position matches a surveyed marking. A production E2E separately covers
large coordinates, reversed shared boundaries, cancellation, export and reload.

## Conservative repeated-pass integration

Appending uses `existing_map` (editable IR JSON or Lanelet2 OSM), distinct from the
metadata-only `reference_map`. Existing geometry, entity IDs, rules and projector metadata
stay fixed. Corresponding centres and both boundaries must agree within 0.5 m, edge
directions within 15 degrees and ground heights within 0.3 m. Uniquely connected sections
are compared as continuous edges, with comparison centres sampled by normalized arc length
at a fixed 0.5 m resolution. This avoids false mismatches caused by inserted split vertices,
unequal edge lengths or comparing a trajectory heading against a derived lane centre.
Explicit existing centreline geometry is respected. Ambiguous disconnected duplicates,
opposite travel and different ground levels are not fused. Uncovered intervals are added;
only coincident, unambiguous directed ends are connected automatically.
OSM preservation applies to the imported IR: unsupported members/types are reported
in `import_issues` and can be dropped on import. Prefer editable JSON for IR fidelity.

On the same public inputs above, exact replay retained the complete IR document unchanged:

| Recorded input | Incoming supported length | Reused length | Added length | Added lanes |
|---|---:|---:|---:|---:|
| PandaSet 019 | 87.199 m | 87.199 m | 0 m | 0 |
| Autoware sample rosbag, MGRS 54SUE | 122.134 m | 122.134 m | 0 m | 0 |

The PandaSet Web replay also exported byte-identical OSM, and one Undo removed the original
build: the no-op replay did not create an Undo entry. Native tests cover retained speed
rules, IR and OSM imports, rejected invalid imports, atomic failure, explicit centres,
curved edges, sharp height changes, section cuts, partial extensions and ambiguity.

For partial overlap, PandaSet poses 0–49 and 30–79 have 20.667 m of common recorded XY
trajectory. With continuity tracking and fitting, the first subset generated 54.705 m;
appending the second reused 20.000 m and added 33.161 m in one fragment (two lane sections).
Existing lane and boundary documents
remained unchanged. The conservative matcher did **not** remove all overlap: different
sampling and extraction geometry retained unmatched fragments, and no endpoints coincided
exactly enough to connect. Those duplicates and gaps require review. The same subset
procedure on Autoware retained existing geometry and reused 13.864 m, adding 102.093 m
in four fragments (eight lane sections);
its stationary GNSS prefix and missing ground also limit this experiment.

These are overlapping subsets of **one** drive, not independently recorded repeated
surveys. They check reuse and preservation, not improved boundary accuracy or correction
of frame drift. Evidence counts continue to describe the incoming pass. Reproduce with
an installed CloudAnalyzer package and an updated native Rust core:

```sh
python scripts/vector_map_integrate_evaluate.py pandaset/map.pcd pandaset/trajectory.csv results/pandaset-integration --right-hand
python scripts/vector_map_integrate_evaluate.py sample-map-rosbag/pointcloud_map.pcd trajectory.csv results/autoware-integration --reference-map sample-map-rosbag/lanelet2_map.osm
```

The input CSV must have `timestamp,x,y,z` columns. Results must be a new directory.
The script writes maps and reports for first/replay/overlap, checks complete replay equality
and retention of old lane/boundary documents, and records the limitations in `summary.json`.

## Branching junction connection drafts

A controlled ablation uses the official Autoware **sample-map-planning** cloud and map
(1,757,841 points, MGRS 54SVE). Removing 114 lanes with turn labels leaves 72 input lanes,
including 69 surveyed driving legs. Their existing geometry, directly connected ends,
rules and coordinate metadata are retained. Reference topology supplies 88 expected
leg-to-leg pairs through the removed connector chains; it is used only after generation
for scoring, never passed to the proposal or connection algorithm.

With default 30 m gap and 90% ground-support thresholds:

| Pair measure | Result |
|---|---:|
| Generated connection pairs | 83 |
| Pairs matching reference topology | 76 |
| Incorrect pairs | 7 |
| Missed reference pairs | 12 |
| Pair precision | 91.57% |
| Pair recall | 86.36% |

All existing lane/boundary/rule documents and metadata remained unchanged. Preview was
read-only; replay added no geometry. New lanes' predecessor/successor pairs and review
tags survived MGRS OSM export/import. Branches are retained, including multiple supported
choices at one incoming or outgoing road end. Every generated connection is marked for
review; no new traffic-rule permissions are inferred.

This measures reconstruction of topology from **surveyed input legs**, not end-to-end
lane extraction or boundary accuracy. This sample informed development, so it is not a
held-out accuracy estimate. Centre ground support does not check full lane width or
obstacle clearance and cannot decide legal manoeuvres. The seven incorrect pairs make
manual inspection necessary. The removed lanes' rules can become orphaned; import,
input/output validation and Autoware/export issues are recorded separately to distinguish
existing map issues from changes introduced by the experiment.
Input and output had zero validation errors. Isolated-lane warnings decreased from
42 to 3, and disconnected-topology warnings from 35 to 2; the other validation,
Autoware and export issue counts were unchanged. Those improvements measure connectivity,
not driving permission or lane-boundary accuracy.

Reproduce with external official sample-map-planning inputs and a new output directory:

```sh
cargo run --manifest-path rust/Cargo.toml -p ca-wasm --example vector_map_junction_evaluate -- sample-map-planning/pointcloud_map.pcd sample-map-planning/lanelet2_map.osm results/junction-ablation
```

An optional fourth argument supplies a JSON options file. The example writes ablated
`input.json`, generated IR/OSM/projector artifacts and a report containing expected,
generated, incorrect and missed pairs, preservation/replay checks and validation issues.
Use the [preview and selection workflow](commands/vectormap-connect.md) to review drafts
on your own map; neither reference maps nor ground support establish driving permission.

## Assisted signal-head measurement on the public planning survey

The same official planning cloud (1,757,841 original points, MGRS 54SVE) was used
to measure a selected signal head. The input IR retained the ablated road legs
but removed signal entities and traffic-light rules. Reference head 353 guided
selection of the box `[3832.1,73770.1,24.74]`–`[3833.9,73771.4,25.27]`; object
kind and lane 85 were explicit inputs. **Lane 85 exercises publication only: its
control relationship was not verified.** Housing geometry was fitted from XYZ
returns, not copied from the reference. Because the box was reference-assisted
and this sample informed development, this is a measurement workflow check,
not automatic-detection precision/recall or a held-out accuracy estimate.

The box supplied 61 points: measured width 1.42061 m, height 0.45651 m,
5th–95th-percentile thickness 0.44356 m and vertical-plane RMS 0.14842 m.
The broad box `[3822.5,73783.1,24.65]`–`[3824.1,73784.5,25.5]` at another
head was rejected (thickness 0.512 m); tightening its height still produced
0.503 m thickness and was rejected. Bounds/background matter, and a valid
signal can fail these conservative shape checks.

MGRS Lanelet2 export/reimport retained the measured head, height and source
tags. Existing unsupported-member export warnings remained visible. Unit tests
also check existing lane/boundary/stop-rule preservation, no fabricated lamps or
stop lines, atomic rejection, replay after IR/OSM import, and exact Undo.
The production Web app reproduced the same 61-point measurement. Inspection copied
the selected points into a separate cloud with unchanged fit statistics; preview
kept the map unchanged, adding changed the export, and Undo restored it byte for
byte. The isolated head and housing overlay were visually reviewed; no JS errors
were reported. This checks the assisted workflow, not the unverified lane assignment.

Reproduce a selected-box measurement with your input map and an options JSON
containing `min`, `max`, `lanes` and `kind`:

```sh
cargo run --manifest-path rust/Cargo.toml -p ca-wasm --example vector_map_signal_measure -- sample-map-planning/pointcloud_map.pcd input.json signal-box.json
```

The example prints fitted support, import/export issues and roundtrip geometry;
it checks the source tag and rejects roundtrip displacement above 5 mm.
See [the assisted workflow](commands/vectormap-signal.md) for Web/CLI/Python/MCP
use. This path currently loads the complete cloud: no 10-billion-point benchmark
or automatic object-classification claim is implied.
