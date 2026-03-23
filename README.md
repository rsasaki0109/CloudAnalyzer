# CloudAnalyzer

[![Test](https://github.com/rsasaki0109/CloudAnalyzer/actions/workflows/test.yml/badge.svg)](https://github.com/rsasaki0109/CloudAnalyzer/actions/workflows/test.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**1コマンドで点群の品質がわかる。**

```bash
$ ca evaluate map_v02.pcd reference.pcd --plot f1.png

Chamfer Distance:  0.0083
Hausdorff Distance: 0.1809
AUC (F1):          0.9852

F1 Scores:
  d=0.05  P=0.9471  R=0.8833  F1=0.9141
  d=0.10  P=0.9971  R=0.9899  F1=0.9935
  d=0.20  P=1.0000  R=1.0000  F1=1.0000
```

| Density Map (14.6M pts) | F1 Evaluation Curve |
|---|---|
| ![density](docs/images/density_utsukuba.png) | ![f1](docs/images/f1_curve.png) |

## Why CloudAnalyzer?

|  | 従来のワークフロー | CloudAnalyzer |
|---|---|---|
| 点群の中身を確認 | GUIツールを起動して目視 | `ca info map.pcd` |
| ダウンサンプリングの品質確認 | 「見た目で大丈夫そう」 | `ca evaluate down.pcd orig.pcd` → AUC=0.985 |
| 地図の密度分布 | スクリプトを書く | `ca density-map map.pcd -o density.png` |
| フィルタ→間引き→評価 | 3つのスクリプト | `ca pipeline noisy.pcd ref.pcd -o clean.pcd` |
| CIで品質チェック | なし | `quality-gate.yml` で AUC < 0.9 なら fail |
| ブラウザで共有 | Potreeセットアップ | `ca web map.pcd` |

## Install

```bash
cd cloudanalyzer
pip install -e .
```

## 30秒デモ

```bash
# 点群の基本情報
ca info cloud.pcd
# -> Points: 14656120, BBox, Centroid...

# ダウンサンプリングして品質評価
ca downsample cloud.pcd -o down.pcd -v 0.2
ca evaluate down.pcd cloud.pcd --plot quality.png
# -> AUC(F1): 0.9852

# ノイズ除去 → 間引き → 評価を1コマンドで
ca pipeline noisy.pcd reference.pcd -o clean.pcd -v 0.2
# -> Filter: removed 42418 → Downsample: 11.0% → AUC: 0.9819

# ブラウザで3D表示
ca web cloud.pcd
```

## 全23コマンド

### 評価・分析

| Command | What it does |
|---|---|
| `ca evaluate` | **F1 / Chamfer / Hausdorff / AUC** — 点群品質を定量評価 |
| `ca compare` | ICP/GICP レジストレーション付き比較 |
| `ca diff` | レジストレーションなしの距離統計 |
| `ca info` | 点数・BBox・重心 |
| `ca stats` | 密度・点間距離分布 |
| `ca batch` | ディレクトリ内の全ファイルを一括info |
| `ca pipeline` | filter → downsample → evaluate を1コマンドで |

### 加工

| Command | What it does |
|---|---|
| `ca downsample` | ボクセルダウンサンプリング |
| `ca filter` | 外れ値除去 (Statistical Outlier Removal) |
| `ca sample` | ランダムサンプリング |
| `ca split` | グリッドタイル分割 |
| `ca merge` | 複数点群を結合 |
| `ca align` | 連続レジストレーション + 結合 |
| `ca crop` | バウンディングボックスで切り出し |
| `ca normals` | 法線推定 |
| `ca convert` | フォーマット変換 (pcd/ply/las) |

### 可視化

| Command | What it does |
|---|---|
| `ca web` | **ブラウザで3D表示** (Three.js) |
| `ca view` | デスクトップ3Dビューア |
| `ca density-map` | 2D密度ヒートマップ |
| `ca heatmap3d` | 3D距離ヒートマップ画像 |

## ダウンサンプリング品質の定量評価

「0.2mボクセルでどれだけ品質が落ちるか？」を数値で答えられる:

| Voxel | Points | Chamfer | AUC (F1) | 判定 |
|---|---|---|---|---|
| 0.1m | 1.74M (97%) | 0.0011 | 0.998 | ほぼ劣化なし |
| 0.2m | 1.60M (90%) | 0.0083 | 0.985 | 実用上問題なし |
| 0.5m | 1.13M (63%) | 0.0544 | 0.886 | Recall低下あり |

| Voxel 0.1m (AUC=0.998) | Voxel 0.5m (AUC=0.886) |
|---|---|
| ![v01](docs/images/f1_voxel01.png) | ![v05](docs/images/f1_voxel05.png) |

## 自動化・CI対応

全コマンドが `--output-json` / `--format-json` に対応:

```bash
# jq でパイプ処理
ca evaluate a.pcd b.pcd --format-json | jq '.auc'

# CIで品質ゲート
AUC=$(ca evaluate new.pcd ref.pcd --format-json | jq -r '.auc')
[ $(echo "$AUC < 0.9" | bc -l) -eq 1 ] && exit 1
```

GitHub Actions の `quality-gate.yml` でAUC/Chamfer閾値チェックも可能。

## Python API

```python
from ca.evaluate import evaluate, plot_f1_curve
from ca.pipeline import run_pipeline

# 評価
result = evaluate("estimated.pcd", "reference.pcd")
print(f"AUC: {result['auc']:.4f}")
plot_f1_curve(result, "f1_curve.png")

# パイプライン
result = run_pipeline("noisy.pcd", "ref.pcd", "clean.pcd", voxel_size=0.2)
```

## 対応フォーマット

`.pcd` / `.ply` / `.las`

## License

MIT
