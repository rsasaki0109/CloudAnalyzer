# Frozen interior paint correction comparison

Both modes enable previous physical anchors, curb alignment, complete paint
fitting and source-footprint checks. Only after enables `fit_paint_divider`.
Counts, directions, 3.5 m starting width and original traces stay fixed. BOTH
complete scene generations were frozen before reference geometry opened.
Actual cumulative editable maps, Lanelet2 exports, per-case profiles, source
audits and inline CSV inputs are included. Raw clouds/reference maps are not copied.

| Source | Cases | Retained before → after | Deferred before → after | Result |
|---|---:|---:|---:|---|
| Original planning XYZRGB, 1,757,841 points | 3 | 116.512 → 116.512 m | 12.000 → 12.000 m | Interior of path 1 corrected; 0/2 held |
| Prepared Tokyo XYZRGBI, 1,883,866 points | 6 | 88.078 → 88.078 m | 34.259 → 34.259 m | All held; IR/OSM byte-identical |

On planning path 1, 99 paint points and 13/25 guarding physical curb sections
support a −2.320° interior-line fit, with 0.061 m P90 source residual and 0.923 m
maximum movement. Before-footprint source intervals contain 18.114 m observed
components, 27.504 m interpolation and 0.523 m extension. Final path 1 has only
9 RGB vertices, plus 28 curb and 29 inferred vertices; missing paint stays inferred.
Outside candidate geometry/evidence and reference vertices are byte-exact.

Ordered BEFORE-selected lane-pair targets on **38.141 m / 312 samples** give
three-boundary mean **0.695 → 0.542 m**. Interior alone, on 104 samples, improves
mean **0.504 → 0.044 m**, maximum **0.969 → 0.046 m**. The **right outside maximum
remains 1.531 m**; source curbs do not certify driving-lane edges. The independent
fixed-before-nearest diagnostic improves mean 0.712 → 0.579 m, but its 1.530 m
maximum is unchanged. Before-only/after-only extent is zero; 8 m remains deferred.
Both diagnostics and all three per-boundary errors are retained in evaluation.

Full-planning unpaired mean changes 0.395 → 0.351 m; maximum stays 1.530 m.
These are known development scenes, not held-out accuracy or certified roles.
All Tokyo ordered-correspondence extent is held, not assigned zero error.
Path 2 retains its previous curb fix and 2 m reference-pair seam hold.

See [method, actual map figure, usage and browser proof](../../../docs/vector-map-paint-divider.md).
[verification.json](verification.json) binds runtime declarations, measured
binary/source hashes, eighteen independent native reproductions, the production
actual-source browser proof and SHA256 for every other file here. Generation
freezes bind source-only outputs; later evaluation hashes reference inputs
separately and never regenerates source maps. Source declarations are not
embedded binary attestations. Browser proof checks boundary vertices and lane
counts, not complete topology or byte-identical OSM. Divider-off maps reproduce
every previous paint-stage map/OSM exactly; older frozen artifacts/GIFs are unchanged.

Planning sample: Copyright 2020 TIER IV, Inc.; [official instructions](https://docs.autoware.org/main/demos/planning-sim/).
Tokyo source: Dynamic Map Platform Co., Ltd. (2026), [pinned Hard Intersection sample](https://huggingface.co/datasets/dynamic-maps/hard-intersection-multimodal-sample/tree/e8e8d2a5d49a8b9cb63b5b5ecef9b260ff48f39c),
CC BY 4.0. Prepared source classifications were cleared, RGB is uniform and
coordinates/heights unchanged. Reviewed traces are retained; this is not an untouched raw tile.
