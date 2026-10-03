# Frozen source-curb trace correction comparison

Both modes use physical width anchors and source-footprint fitting. Before has
`align_trace_to_curbs=false`; after enables the optional straight-trace stage.
The lane counts, directions, 3.5 m width prior and all source traces stay fixed.
Source-only generation reads no survey. All generated JSON/OSM, source audits,
profiles and trajectories are frozen before surveyed geometry is evaluated.

Each scene has inline CSV inputs, cumulative maps, actual per-case profiles,
full-source audits, generation freezes, unpaired distances, fixed before-nearest
targets and fixed adjacent-lane-pair correspondence. Every profile polyline is
verified in the actual editable map. Operator coordinates must invert the
reported source translation; they are used only to match common intervals.
Reference targets are selected using BEFORE only and remain fixed after.

| Source | Cases | Retained before → after | Deferred before → after | Result |
|---|---:|---:|---:|---|
| Original planning XYZRGB, 1,757,841 points | 3 | 106.512 → 106.512 m | 22.000 → 22.000 m | Path 2 translated; other two held |
| Prepared Tokyo XYZRGBI, 1,883,866 points | 6 | 88.078 → 88.078 m | 34.259 → 34.259 m | All held, maps byte-identical |

Planning path 2 translates +3.550 m laterally from nine of seventeen source
sections. All 31.765 m remain generated. Ordered three-boundary correspondence
on 29.765 m gives mean 2.877 → 0.352 m, P90 3.411 → 0.566 m and maximum
3.569 → 0.581 m. **2.000 m of reference coverage are held**, not counted as zero
error. Original before-nearest targets on 31.765 m give mean **1.369 → 2.296 m**,
a regression under that diagnostic; nearby boundary roles differ. Both results
are retained. Planning path 0 still has a **5.674 m** maximum. Tokyo's ordered
lane-pair correspondence holds all 88.078 m, including unsupported lane counts;
it supplies no accuracy score for those held intervals.

These are known development scenes, not held-out or certified map accuracy.
Passing source support is also a generation gate. Lane identity, legal traffic
directions, counts, controls and equipment still require review. No raw point
clouds or surveyed maps are included. Existing physical-anchor artifacts and
GIFs keep their historical measurements and runtimes.

See [method, usage and actual map figure](../../../docs/vector-map-curb-alignment.md).
[verification.json](verification.json) records runtime hashes, eighteen independent
native reproductions, old baseline preservation, browser proof and SHA256 for
every other file in this folder. `generation-freeze.json` files bind each scene's
source-only generation; `evaluation.json` files record later reference inputs
and evaluation-script hashes separately. Source commit declarations are not
embedded binary attestations.

Planning sample: Copyright 2020 TIER IV, Inc.; [official instructions](https://docs.autoware.org/main/demos/planning-sim/).
Tokyo source: Dynamic Map Platform Co., Ltd. (2026), [pinned Hard Intersection sample](https://huggingface.co/datasets/dynamic-maps/hard-intersection-multimodal-sample/tree/e8e8d2a5d49a8b9cb63b5b5ecef9b260ff48f39c),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). CloudAnalyzer adds derived
drafts and measurements. The code's MIT license does not change data licenses.
