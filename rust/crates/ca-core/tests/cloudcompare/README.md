# CloudCompare parity fixtures

Recorded with CloudCompare 2.13.1 by `scripts/make_cloudcompare_fixtures.py`; checked by `tests/cloudcompare.rs`.

- `before.xyz`, `after.xyz`: two synthetic 60 x 60 m surveys (the second adds a mound and a pit).
- `c2c.txt`: C2C distance of each `after` point to `before`.
- `sor_removed.txt`: `before` points removed by SOR (k = 8, 1.0 σ).
- `csf_ground.txt`: `before` points CSF classifies as ground (relief, resolution 1.0, threshold 0.3).
- `m3c2.txt`: M3C2 distance at each `before` point (normal Ø 2.0, projection Ø 1.0, depth 2.0), `nan` where not measured.
- `volume.txt`: added and removed volume, `before` as the ground, 1.0 m grid.
