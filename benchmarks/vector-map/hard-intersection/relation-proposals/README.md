# Geometric target proposals on the source-generated intersection

`native-proposals.json` records read-only geometric candidates, held alternatives
and two explicit adoptions: vehicle rule 169 to stop 164, and pedestrian rule 173
to crossing 156. The closer crossing 152 is held for its incompatible axis.
Rules 167/171 remain unresolved; clearing rule 171's legacy vehicle-lane target is
a separate explicit edit. `native-verification.json` and `web-verification.json`
record unchanged physical map/full-source quality, exact native/browser Lanelet2,
all-edit Undo and reload. The final map is exactly the previously published
[equipment review](../equipment-relations/reviewed.json); no duplicate cloud or
map copy is needed here.

Reproduce using `scripts/preview_vector_map_relations.py` and the optional Web
proof. See the [workflow, fixed gates and limits](../../../../docs/vector-map-relation-proposals.md).
The input is the frozen source-generated development map from media commit
`3f93de0f4100e2bb09de6d0e985aa411a7c6a63d`. No reference/teacher shapes or control
labels are input. The source's original points are reused in place.

Derived from DynamicMapPlatform Co., Ltd. (2026),
[hard-intersection sample](https://huggingface.co/datasets/dynamic-maps/hard-intersection-multimodal-sample/tree/e8e8d2a5d49a8b9cb63b5b5ecef9b260ff48f39c),
CC BY 4.0. Geometric targets require review and do not prove legal control, front
face or phases. Deferred road extent and unresolved warnings are retained.
