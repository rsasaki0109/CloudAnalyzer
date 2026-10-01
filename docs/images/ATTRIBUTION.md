# README Figure Attribution

The README figures in this repository are generated from documented public assets.

## Source Map

- Source repository: [`koide3/hdl_localization`](https://github.com/koide3/hdl_localization)
- Source file: [`data/map.pcd`](https://github.com/koide3/hdl_localization/blob/master/data/map.pcd)
- License: [BSD-2-Clause](https://github.com/koide3/hdl_localization/blob/master/LICENSE)
- Publisher: AISL, Toyohashi University of Technology
- Related public demo bag from the same README:
  [`hdl_400.bag.tar.gz`](http://www.aisl.cs.tut.ac.jp/databases/hdl_graph_slam/hdl_400.bag.tar.gz)

The repository README describes that bag as an example recorded in an outdoor environment,
and the repository ships `data/map.pcd` as the sample global map used by the localization demo.
This sample map and bag are published by AISL at Toyohashi University of Technology.

The README point-cloud triptych is derived from the deterministic RELLIS-3D perception demo
artifacts. See [perception-demo attribution](../demo/perception/ATTRIBUTION.md) for the
upstream dataset terms.

## Web App GIFs from PandaSet

`web/odometry.gif` and `web/dynamic.gif` show the web app on
[PandaSet](https://pandaset.org) scene 019 (San Francisco), by Scale AI and Hesai, licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The scans were converted to the app's input
with `scripts/fetch_pandaset.py` (the 360° Pandar64 only, moved into its own frame) and recorded with
`PANDASET_DIR=<dir> npm run media` in `web/`. The GIFs are adapted from PandaSet.

`web/vector-map.gif` is also adapted from PandaSet 019 under CC BY 4.0, using
Pandar64 frames 0, 8, …, 72 with original world coordinates and intensity and
all 80 recorded positions. It adds generated draft lane geometry, explanatory
captions and a cursor ring, and crops/scales the real Web app capture. Scale AI
and Hesai provided the source data. Preparation uses
[`scripts/prepare_vector_map_pandaset.py`](../../scripts/prepare_vector_map_pandaset.py);
recording uses `VECTOR_MAP_DEMO_DIR=<dir> npm run media:vectormap` in `web/`.
See [reproduction and accuracy limits](../vector-map-media.md).

## Web App Pictures from NCLT

`web/loop.gif`, `web/posegraph.jpg` and the pose graph part of `web/demo.gif` show the web app on session
2012-04-29, `web/seasons.gif` on sessions 2012-06-15 and 2012-12-01 (keyframes 200-2199) joined, and
`web/bags.jpg` on the two ROS 2 bag samples made from sessions 2012-04-29 and 2012-06-15 (see
`web/public/samples/ATTRIBUTION.md`), of the University of Michigan North Campus Long-Term Vision and Lidar Dataset (NCLT;
N. Carlevaris-Bianco, A. K. Ushani and R. M. Eustice, International Journal of Robotics Research, 2016,
http://robots.engin.umich.edu/nclt/). Contains information from NCLT, which is made available under the
[Open Database License](https://opendatacommons.org/licenses/odbl/1-0/); its contents are under the
[Database Contents License](https://opendatacommons.org/licenses/dbcl/1-0/). The scans were prepared with
`scripts/prepare_nclt.py` (a keyframe every metre, KISS-ICP odometry) and recorded with
`NCLT_DIR=<dir> NCLT_SEASONS=<summer dir>,<winter dir> npm run media` in `web/`.

## Social Preview

`social-preview.svg` is original CloudAnalyzer artwork and does not use external image assets;
`social-preview.png` is its 1280×640 upload-ready render. Upload the PNG from the repository's
**Settings → General → Social preview** panel when refreshing the GitHub repository card.

## Generated Files

- `density_hdl_localization_map.png`
- `f1_hdl_localization_v0_2.png`
- `f1_hdl_localization_v0_1.png`
- `f1_hdl_localization_v0_5.png`
- `readme-pointcloud-triptych.png`

## Regeneration Commands

```bash
git clone --depth 1 https://github.com/koide3/hdl_localization /tmp/hdl_localization

cd cloudanalyzer

python3 -m cloudanalyzer_cli.main density-map \
  /tmp/hdl_localization/data/map.pcd \
  -o ../docs/images/density_hdl_localization_map.png \
  -r 1.0 -a z

python3 -m cloudanalyzer_cli.main downsample \
  /tmp/hdl_localization/data/map.pcd \
  -o /tmp/map_v0.2.pcd \
  -v 0.2 \
  --evaluate \
  --plot ../docs/images/f1_hdl_localization_v0_2.png

python3 -m cloudanalyzer_cli.main downsample \
  /tmp/hdl_localization/data/map.pcd \
  -o /tmp/map_v0.1.pcd \
  -v 0.1 \
  --evaluate \
  --plot ../docs/images/f1_hdl_localization_v0_1.png

python3 -m cloudanalyzer_cli.main downsample \
  /tmp/hdl_localization/data/map.pcd \
  -o /tmp/map_v0.5.pcd \
  -v 0.5 \
  --evaluate \
  --plot ../docs/images/f1_hdl_localization_v0_5.png

python3 ../scripts/build_readme_pointcloud_figure.py
```

## Result Summary

These commands produced the metrics shown in the root README:

- `0.1m`: 67.5% kept, Chamfer `0.0147`, AUC `0.9984`
- `0.2m`: 31.2% kept, Chamfer `0.0460`, AUC `0.9770`
- `0.5m`: 7.2% kept, Chamfer `0.1266`, AUC `0.8775`
