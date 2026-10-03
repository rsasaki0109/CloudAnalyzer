# Frozen straight RGB paint corridor comparison

Both modes enable physical width anchors, paired-curb alignment and source-footprint
checks. Only after enables `fit_paint_corridor`. Counts, directions, starting
3.5 m prior and original source traces stay fixed. BOTH complete scene generations
are frozen before any surveyed geometry is opened. Actual cumulative editable
maps, Lanelet2 exports, per-case profiles, full-source audits and inline CSV inputs
are included; original clouds and surveyed maps are not copied into the repository.

| Source | Cases | Retained before → after | Deferred before → after | Result |
|---|---:|---:|---:|---|
| Original planning XYZRGB, 1,757,841 points | 3 | 106.512 → 116.512 m | 22.000 → 12.000 m | Path 0 paint fit; 1/2 held |
| Prepared Tokyo XYZRGBI, 1,883,866 points | 6 | 88.078 → 88.078 m | 34.259 → 34.259 m | All held, maps byte-identical |

Path 0 uses source paint widths 2.900/2.894 m and heading correction −2.041°.
Ordered BEFORE-selected lane-pair slots on **34.606 m / 282 samples** give mean
3.312 → 0.037 m, P90 4.166 → 0.062 m and maximum 5.674 → 0.064 m.
**2 m before-only and 12 m after-only are excluded from paired accuracy**;
retained source-supported path increases 36.606 → 46.606 m with 4 m still deferred.
The separate fixed-before-nearest-point diagnostic **regresses mean 1.616 → 2.003 m**
and maximum 5.674 → 5.797 m; misplaced initial boundaries select different roles.
Both diagnostics keep the same before-selected targets; neither certifies lane identity.

The left paint track has **only 9 points over 1.796 m** and **48.811 m extrapolated
before trimming**. Other tracks have interpolation gaps. Only nearby observed
paint vertices receive `rgb_paint`; final path 0 has 24 such vertices and 54 inferred
vertices. Source component intervals are not a continuous survey of the output.
These are known development scenes, not held-out map accuracy. All Tokyo ordered
lane-pair intervals are held rather than given zero error. One Tokyo paint ROI
reaches its explicit budget; its unchanged map does not establish absent paint.

See [method, usage and actual map figure](../../../docs/vector-map-paint-corridor.md).
[verification.json](verification.json) binds runtime declarations and binary/source
hashes, eighteen independent native reproductions, actual-source browser proof
and SHA256 for every other file here. Generation freezes bind source-only outputs;
later evaluation records reference hashes separately. Declared source commits
are not embedded binary attestations. Browser proof is boundary-to-exported-node
and lane-count verification, not complete topology or OSM byte equality.
All previous curb-stage maps remain byte-identical with paint off; older frozen
benchmark files and existing GIFs retain their historical bytes and runtimes.

Planning sample: Copyright 2020 TIER IV, Inc.; [official instructions](https://docs.autoware.org/main/demos/planning-sim/).
Tokyo source: Dynamic Map Platform Co., Ltd. (2026), [pinned Hard Intersection sample](https://huggingface.co/datasets/dynamic-maps/hard-intersection-multimodal-sample/tree/e8e8d2a5d49a8b9cb63b5b5ecef9b260ff48f39c),
CC BY 4.0; prepared source classifications cleared, RGB uniform, coordinates/heights
unchanged, reviewed traces retained. This is not an untouched raw tile.
