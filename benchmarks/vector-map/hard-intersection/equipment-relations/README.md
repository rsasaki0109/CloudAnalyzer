# Reviewed equipment associations on the generated intersection

`reviewed.json`, `reviewed.osm` and Local projector metadata retain the physical
59-road generated map and add two operator-reviewed signal associations. Rule
171's legacy vehicle-lane assignment is cleared and stays unresolved. See
[the workflow and limitations](../../../../docs/vector-map-equipment-relations.md).
The frozen input is the source-supported generated map at
[media commit 3f93de0](https://github.com/rsasaki0109/CloudAnalyzer/blob/3f93de0f4100e2bb09de6d0e985aa411a7c6a63d/benchmarks/vector-map/hard-intersection/supported-intersection/reviewed.json)
(PR198 remains draft). These are geometric review drafts, not proven legal rules.

`operator-inputs.json` supplies the complete explicit target edits.
`native-verification.json` records geometry equality, no-op, JSON/Lanelet2 reload,
source hashes, unchanged full-source coverage and retained warnings. The report
separates 18 UI validation warnings from the Local export's missing-georeference
warning. No teacher/reference shapes or control labels enter these edits.

Derived from DynamicMapPlatform Co., Ltd. (2026), [hard-intersection multimodal
sample](https://huggingface.co/datasets/dynamic-maps/hard-intersection-multimodal-sample/tree/e8e8d2a5d49a8b9cb63b5b5ecef9b260ff48f39c),
CC BY 4.0. Source subsetting/attribute clearing and earlier manual proposals are
documented in the input workflow. This review changes only regulatory targets
and their provenance; no lamp states or geographic origin are invented.
