# SLAM Leaderboard (generated)

Static snapshot produced by `scripts/build_leaderboard.py`.
Open [`index.html`](index.html) on GitHub Pages or regenerate locally:

```bash
pip install -e './cloudanalyzer[slam]'
python scripts/build_leaderboard.py --output docs/leaderboard
# add locally-prepared KITTI / Newer College rows when available:
python scripts/prepare_leaderboard_kitti.py --velodyne-dir ... --kitti-poses ...
python scripts/build_leaderboard.py --include-optional --output docs/leaderboard
```

Optional real-world datasets (`kitti-mini`, `newer-college-mini`) are
not bundled because of size and upstream licenses. Prepare them locally,
then rebuild with `--include-optional`.
