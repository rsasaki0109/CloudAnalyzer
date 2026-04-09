# README Figure Attribution

The README figures in this repository are generated from documented public assets.

## Source Map

- Source repository: [`koide3/hdl_localization`](https://github.com/koide3/hdl_localization)
- Source file: [`data/map.pcd`](https://github.com/koide3/hdl_localization/blob/master/data/map.pcd)
- License: [BSD-2-Clause](https://github.com/koide3/hdl_localization/blob/master/LICENSE)
- Related public demo bag from the same README:
  [`hdl_400.bag.tar.gz`](http://www.aisl.cs.tut.ac.jp/databases/hdl_graph_slam/hdl_400.bag.tar.gz)

The repository README describes that bag as an example recorded in an outdoor environment,
and the repository ships `data/map.pcd` as the sample global map used by the localization demo.

## Generated Files

- `density_hdl_localization_map.png`
- `f1_hdl_localization_v0_2.png`
- `f1_hdl_localization_v0_1.png`
- `f1_hdl_localization_v0_5.png`

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
```

## Result Summary

These commands produced the metrics shown in the root README:

- `0.1m`: 67.5% kept, Chamfer `0.0147`, AUC `0.9984`
- `0.2m`: 31.2% kept, Chamfer `0.0460`, AUC `0.9770`
- `0.5m`: 7.2% kept, Chamfer `0.1266`, AUC `0.8775`
