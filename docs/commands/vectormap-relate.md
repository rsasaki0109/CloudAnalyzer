# Review equipment associations

`ca vectormap-relate` inspects existing signal/crosswalk/stop-marking rules or
explicitly replaces their targets in a new output directory. It changes no
physical map geometry or lamp observations. Targets are operator-reviewed drafts.

```sh
ca vectormap-relate map.json
ca vectormap-relate map.json --rule 169 --lane 76 --lane 77 --stop 164 --out reviewed
ca vectormap-relate reviewed/vector_map.json --rule 173 --crosswalk 156 --out pedestrian
```

Omit `--rule` and `--out` for read-only inspection. Edits require both. Repeat
`--lane`, `--crosswalk`, or `--stop` per target ID. Lists replace the whole existing
association, including empty lists. Pedestrian signals accept crosswalk targets;
vehicle signals accept lanes and at most one previously reviewed transverse stop.
Missing/duplicate IDs and incompatible participants fail before publishing files.
An existing output directory is never overwritten.

Outputs are `lanelet2_map.osm`, `map_projector_info.yaml`, `vector_map.json` and
`report.json`. The report includes current relationships, an edit/no-op result,
retained validation/export issues and review limitations. These lists do not
infer legal control or phases. Python/MCP expose `edit_vector_map_relations`.
See the [Web workflow and actual-source review](../vector-map-equipment-relations.md).
