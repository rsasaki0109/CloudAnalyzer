# Partial source-footprint road draft

Actual source-only before/after road generation from the same six frozen paths.
`after-road-5.json` is the cumulative editable draft; `after-road-5.osm` is its
Lanelet2 export. For external use, rename the matching
`after-projector-info.yaml` to `map_projector_info.yaml` and the OSM to
`lanelet2_map.osm`. The Local projector does not transform the source frame.
Numbered JSON files retain intermediate cumulative maps for
reproducing the branch comparison image; they are generated output, not inputs
to fitting. `comparison.json` records original point/trajectory hashes, declared
runtime source commit, measured native binary hash, support flags and deferred
extent. The original 49-lane equipment/junction map is not replaced here.

The new draft retains 88.078 m of 122.337 m road-stretch extent and defers 34.259 m
(28%). Its 38 lane fragments result from splitting missing ground; lane counts
and traffic directions are operator inputs. Zero source-review flags is not
independent accuracy. See [methods, limitations and reproduction](../../../../docs/vector-map-source-footprint.md).

Derived from Hard Intersection Multimodal Samples, Dynamic Map Platform Co., Ltd.
(2026), [pinned source](https://huggingface.co/datasets/dynamic-maps/hard-intersection-multimodal-sample/tree/e8e8d2a5d49a8b9cb63b5b5ecef9b260ff48f39c),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). CloudAnalyzer adds
generated curves, interval filtering and reports; no survey authorship or
endorsement is claimed. Raw points, teacher labels and reference maps are absent.
