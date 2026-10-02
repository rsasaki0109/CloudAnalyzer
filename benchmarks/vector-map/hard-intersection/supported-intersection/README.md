# Source-supported intersection draft

Generated from the original retained source and six explicit input paths, without a reference map: 38 source-footprint road fragments plus 21 reviewed connections, seven measured paint crossings, two transverse marking drafts and four housing drafts. Open `reviewed.osm` with `map_projector_info.yaml` (Local coordinates); `reviewed.json` is the native editable IR.

**Partial draft:** 34.259 m (28%) of input-road extent and two requested turns are deferred. There are zero structural errors, 15 connectivity warnings and zero source-review flags across 6,752 samples. Zero flags do not establish independent accuracy or legal semantics. The app's Autoware profile adds four warnings for unreviewed signal-stop links (19 total); these links are not invented. Equipment preview is capped; types, selected manoeuvres and controlled lanes are operator choices.

`quality.json` retains all lane fractions, endpoint checks and validation issues. `manifest.json` records source/runtime/native/output hashes, explicit selections and no-op confirmation replay after Lanelet2 import. These hashes use canonical LF bytes. The original source/trajectories are reused in place, not stored here. The old 49-lane map and equipment baselines remain unchanged.

See [protocol, limitations and reproduction](../../../../docs/vector-map-supported-intersection.md). Data and derived geometry: [Hard Intersection Multimodal Samples](https://huggingface.co/datasets/dynamic-maps/hard-intersection-multimodal-sample/tree/e8e8d2a5d49a8b9cb63b5b5ecef9b260ff48f39c), Dynamic Map Platform Co., Ltd. (2026), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). CloudAnalyzer adds derived geometry and review; no survey authorship or endorsement is claimed.
