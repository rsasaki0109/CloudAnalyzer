# Suggest equipment targets

`ca vectormap-suggest map.json --rule ID` previews geometric targets for an existing
vehicle or pedestrian signal. JSON and Lanelet2 inputs are supported. No point
cloud, output directory or automatic adoption is needed for preview.

```sh
ca vectormap-suggest map.json --rule 173
ca vectormap-suggest map.json --rule 173 --candidate crosswalk:156 --snapshot CURRENT_SNAPSHOT --out reviewed
```

Adoption requires `--candidate`, `--snapshot` and a NEW `--out` directory together.
Use the key and opaque snapshot from the current preview after reviewing its
source context. Stale, rejected and incomplete candidates fail atomically. Reports
retain candidate distance, unsigned axis difference, road context, held reasons,
existing links and ambiguity. No legal control, front face or phases are inferred.

Python/MCP expose `propose_vector_map_relations`. See the
[Web workflow and actual-source proof](../vector-map-relation-proposals.md).
