# NCLT drive to an editable Lanelet2 draft

This checks whether a real drive can complete the browser workflow: load and
correct the survey, generate roads, edit a speed limit, inspect source coverage,
save OSM and projector YAML, then reopen the saved map.

## Reproduced blocker

On main `83c415b4678f7a235366d0982af8f7d508004195`, the April NCLT demo's
corrected trajectory failed with the default 50 m piece length:

```text
Could not build draft roads: invalid geometry: boundary:8 does not extend across the split position
```

The road builder projected each reference cut onto nearby lane/boundary geometry.
The returning trajectory and fitted boundaries made that projection ambiguous.
Cloud extraction already supplies corresponding cross-section vertices. The fix
cuts every line at the same edge/fraction, retaining geometry and interpolating
XYZ only at cuts. Identical consecutive vertices are removed. Pieces remain
connected in each lane's travel direction. Piece length uses 3D arc length;
zero disables splitting. Excessive piece counts fail within the atomic build.

Checking the entire round trip exposed a second problem: with two nominal lanes,
an inner boundary folds through itself at this sharp turn. A draft could pass
the native structural check yet reopen with a reversed boundary and a broken
connection. Segmented generation now checks Lanelet2's boundary-direction
inference before storing each piece, subdividing ambiguous returning sections
when possible. Incompatible geometry fails with a review message and preserves
the map. Lane counts, widths and surveyed geometry are never silently changed.

## Browser evidence

The test first checks that the contradictory two-lane configuration fails
atomically. It then explicitly selects one forward lane and no backward lane
for a controlled drafting exercise, retaining the 50 m piece length and other
road options. This does not establish the actual lane count or permitted travel.
Automatic equipment proposals are disabled to isolate road creation.
One lane's speed is changed to 25 km/h. The test requires structural checks to
pass both before saving and after reopening, with boundary coordinates, travel
directions, connections, neighbours and lane speeds retained.

Source coverage still requires review. The writer emits
`lanelet2.no_georeference` and `projector_type: Local`; the browser now displays
that warning. Structural checks, source coverage, and writer warnings are
separate results. This verifies an editable draft and a format round trip; it
does not certify surveyed geometry, traffic rules, or an Autoware deployment.
The test is automated workflow evidence, not a user usability study.

The [2026-10-08 run](verification.json) used 199 corrected poses and generated
5 lane pieces / 10 boundaries over 243.14 m. Four pieces needed source review.
The workflow took approximately 48 seconds in the cloud development environment
while Rust checks also ran; that timing is an observation, not a performance
target. Structural validation passed before and after reopening.

## Reproduce

Build the WASM package and production web app using the repository setup, then:

```sh
cd web
CLOUDANALYZER_REAL_DATA=1 npx playwright test map-workflow-real-data.spec.ts --workers=1
```

The opt-in test saves `workflow.json`, `saved-draft.png`, `corrected-drive.kitti`,
`lanelet2_map.osm`, `map_projector_info.yaml`, and before/after map IR in its
Playwright results. It asserts successful generation, native structural checks, editing, the visible
Local-coordinate warning, and coordinate/speed preservation on reopening.
The regular `map-export.spec.ts` regression also checks that map edits clear
stale export warnings and that a georeferenced export has no Local warning.
The small NCLT geometry fixture in `ca-core/tests/fixtures/vector-map` exercises
the folded inner boundary in ordinary Rust CI, without loading a browser/bag.

Data: University of Michigan NCLT, April 29, 2012. The bundled MCAP is a thinned
190-second excerpt; see [sample attribution](../../../web/public/samples/ATTRIBUTION.md)
for the ODbL / Database Contents License and preparation details.
