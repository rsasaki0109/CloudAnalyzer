# CloudAnalyzer

[![Test](https://github.com/rsasaki0109/CloudAnalyzer/actions/workflows/test.yml/badge.svg)](https://github.com/rsasaki0109/CloudAnalyzer/actions/workflows/test.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**点群を加工したら、品質がどう変わったか数値で出す。**

```bash
$ ca downsample map.pcd -o down.pcd -v 0.2 --evaluate

Original:     1784475 pts
Downsampled:  1597449 pts
Reduction:    10.5%
Saved:        down.pcd
  Chamfer=0.0083  AUC=0.9852
  Best F1=1.0000 @ d=0.20
```

`--evaluate` 1つ付けるだけで、加工前後の品質変化が即座にわかる。

| Density Map | F1 Evaluation Curve |
|---|---|
| ![density](docs/images/density_utsukuba.png) | ![f1](docs/images/f1_curve.png) |

## 他ツールとの違い

|  | CloudCompare | PCL | Open3D (Python) | **CloudAnalyzer** |
|---|---|---|---|---|
| 品質評価 (F1/AUC) | - | - | スクリプト必要 | **`--evaluate` で即時** |
| CLI | 限定的 | なし | なし | **23コマンド** |
| CI/自動化 | 不可 | C++で実装 | スクリプト必要 | **JSON出力 + 品質ゲート** |
| 加工 + 評価 | 別操作 | 別プログラム | 別スクリプト | **1コマンド** |
| ブラウザ表示 | 不可 | 不可 | 不可 | **`ca web`** |

## Install

```bash
cd cloudanalyzer && pip install -e .
```

## Core: 加工したら即評価

CloudAnalyzerの核心。**すべての加工コマンドに `--evaluate` を付けられる。**

```bash
# ダウンサンプリング → 品質を即確認
ca downsample map.pcd -o down.pcd -v 0.2 --evaluate --plot quality.png

# フィルタ → 品質を即確認
ca filter noisy.pcd -o clean.pcd --evaluate

# サンプリング → 品質を即確認
ca sample map.pcd -o sampled.pcd -n 100000 --evaluate

# パイプライン: フィルタ → 間引き → 評価 を1コマンドで
ca pipeline noisy.pcd reference.pcd -o production.pcd -v 0.2
```

## 評価指標

| 指標 | 意味 |
|---|---|
| **Precision** | 加工後の点が元データのどれだけ近くにあるか |
| **Recall** | 元データの点が加工後にどれだけカバーされているか |
| **F1 Score** | Precision と Recall の調和平均 |
| **Chamfer Distance** | 双方向の平均最近傍距離 |
| **Hausdorff Distance** | 最悪ケースの距離 |
| **AUC** | 複数閾値でのF1カーブの面積（総合スコア） |

### 品質判定の目安

| AUC (F1) | 判定 | 用途 |
|---|---|---|
| > 0.99 | 優秀 | 高精度ローカリゼーション用 |
| 0.95 - 0.99 | 良好 | ナビゲーション用 |
| 0.90 - 0.95 | 許容 | 粗い経路計画用 |
| < 0.90 | 要確認 | 品質劣化の可能性 |

### ボクセルサイズ別の品質比較

| Voxel | Points | Chamfer | AUC | 判定 |
|---|---|---|---|---|
| 0.1m | 97% | 0.0011 | 0.998 | 優秀 |
| 0.2m | 90% | 0.0083 | 0.985 | 良好 |
| 0.5m | 63% | 0.0544 | 0.886 | 要確認 |

| Voxel 0.1m (AUC=0.998) | Voxel 0.5m (AUC=0.886) |
|---|---|
| ![v01](docs/images/f1_voxel01.png) | ![v05](docs/images/f1_voxel05.png) |

## CI/自動化

```bash
# AUC を取得してスクリプトで判定
AUC=$(ca evaluate new.pcd ref.pcd --format-json | jq -r '.auc')
[ $(echo "$AUC < 0.9" | bc -l) -eq 1 ] && echo "FAIL" && exit 1

# GitHub Actions で品質ゲート
gh workflow run quality-gate.yml \
  -f source=new.pcd -f reference=ref.pcd -f auc_threshold=0.9
```

## 全コマンド一覧

### 評価

```bash
ca evaluate src.pcd ref.pcd --plot f1.png   # F1/Chamfer/Hausdorff/AUC
ca compare src.pcd tgt.pcd --register gicp  # レジストレーション付き比較
ca diff a.pcd b.pcd --threshold 0.1         # クイック距離統計
ca pipeline in.pcd ref.pcd -o out.pcd       # filter→downsample→evaluate
```

### 加工 (すべて `--evaluate` 対応)

```bash
ca downsample cloud.pcd -o d.pcd -v 0.2 -e  # ボクセルダウンサンプリング
ca filter cloud.pcd -o f.pcd -e              # 外れ値除去
ca sample cloud.pcd -o s.pcd -n 10000 -e     # ランダムサンプリング
ca merge a.pcd b.pcd -o m.pcd                # 結合
ca align s1.pcd s2.pcd -o a.pcd              # レジストレーション+結合
ca split map.pcd -o tiles/ -g 100            # グリッド分割
ca crop cloud.pcd -o c.pcd --x-min 0 ...     # BBox切り出し
ca convert in.pcd out.ply                     # フォーマット変換
ca normals cloud.pcd -o n.ply                 # 法線推定
```

### 分析

```bash
ca info cloud.pcd                   # 基本情報
ca stats cloud.pcd                  # 密度・点間距離統計
ca batch /path/to/dir/ -r           # ディレクトリ一括
```

### 可視化

```bash
ca web cloud.pcd                    # ブラウザ3D表示
ca view cloud.pcd                   # デスクトップ3D表示
ca density-map cloud.pcd -o d.png   # 密度ヒートマップ
ca heatmap3d src.pcd ref.pcd -o h.png  # 3D距離ヒートマップ
```

## Python API

```python
from ca.evaluate import evaluate, plot_f1_curve
from ca.pipeline import run_pipeline
from ca.plot import plot_multi_f1

# 評価
result = evaluate("down.pcd", "original.pcd")
print(f"AUC: {result['auc']:.4f}")  # -> 0.9852

# 複数条件の比較プロット
results = [evaluate(f"v{v}.pcd", "ref.pcd") for v in [0.1, 0.2, 0.5]]
plot_multi_f1(results, ["0.1m", "0.2m", "0.5m"], "comparison.png")
```

## Docs

- [Map Quality Gate Tutorial](docs/tutorial-map-quality-gate.md)
- [Command Reference](docs/commands/)
- [Architecture](docs/architecture.md)
- [CI / Quality Gate](docs/ci.md)

## License

MIT
