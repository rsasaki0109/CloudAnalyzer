# Source-generated intersection with reviewed geometric targets

Generated from the original retained source and six explicit paths: 38 source-footprint
road fragments plus 21 reviewed connections, seven measured paint crossings, two
transverse marking drafts and four housing drafts. `generated.json/.osm` retain the
unlinked generated map. `reviewed.json/.osm` add two explicitly reviewed target
suggestions (vehicle rule 169 to stop 164; pedestrian rule 173 to crossing 156) and
clear legacy pedestrian rule 171, leaving two signals unresolved. Open the final
`reviewed.osm` with Local `map_projector_info.yaml`.

**Partial draft:** 34.259 m (28%) of input-road extent and two requested turns remain
deferred. Physical geometry is unchanged by target review. Full-source quality
checks 6,752 samples with zero source-review flags; this shared development-scene
gate does not establish independent accuracy or legal semantics. Fifteen connectivity
warnings, one orphan pedestrian rule and two unresolved control/stop warnings remain
(18 UI warnings); Local export separately warns about missing geographic reference.
Equipment discovery is capped. Types, movement choices and target adoption require
operator review; unsigned housing normals do not prove front face or phases.

`quality.json` records final full-source coverage and structural validation.
`relation-proposals.json` retains supported and held target evidence, including the
closer but incompatible crossing 152. `manifest.json` separates generated/associated
validation and records source/runtime/native/output hashes, explicit choices and
no-op object replay after import. `web-verification.json` records actual production
generation, native geometry/targets, held alternatives, Undo and warning reload.
All artifact text uses canonical LF. Source clouds/CSVs are reused in place; the
historical 49-lane map and earlier evaluations remain unchanged.

See [protocol and reproduction](../../../../docs/vector-map-supported-intersection.md).
Data and derived geometry: Dynamic Map Platform Co., Ltd. (2026),
[Hard Intersection Multimodal Samples](https://huggingface.co/datasets/dynamic-maps/hard-intersection-multimodal-sample/tree/e8e8d2a5d49a8b9cb63b5b5ecef9b260ff48f39c),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). CloudAnalyzer adds derived
geometry and explicit review; no survey authorship or endorsement is claimed.
