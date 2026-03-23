# Tutorial: Map Update Quality Gate

マップを更新したとき、品質が劣化していないことを自動で検証するワークフロー。

## シナリオ

1. SLAMで新しい地図を生成した
2. 既存のリファレンス地図と比較して、品質が落ちていないか確認したい
3. AUC(F1) < 0.9 なら更新を拒否する

## Step 1: リファレンス地図を用意

```bash
# 現行の地図を確認
ca info reference_map.pcd
# -> Points: 18685856, Extent: [15223, 12724, 283]
```

## Step 2: 新しい地図を評価

```bash
ca evaluate new_map.pcd reference_map.pcd \
  -t 0.05,0.1,0.2,0.5,1.0 \
  --plot quality_report.png \
  --output-json quality.json

# Output:
# Chamfer Distance:  0.0083
# AUC (F1):          0.9852
# -> PASS
```

## Step 3: 加工時に品質を同時確認

```bash
# ダウンサンプリング + 即時評価
ca downsample new_map.pcd -o down.pcd -v 0.2 --evaluate
# -> Reduction: 45.7%
# -> Chamfer=0.0083  AUC=0.9852  Best F1=1.0000 @ d=0.20

# フィルタ + 即時評価
ca filter noisy_map.pcd -o clean.pcd --evaluate
# -> Removed: 42418
# -> Chamfer=0.0002  AUC=0.9999
```

## Step 4: CIで自動チェック

### シェルスクリプト版

```bash
#!/bin/bash
AUC=$(ca evaluate new_map.pcd reference.pcd --format-json | jq -r '.auc')
echo "AUC: $AUC"

if (( $(echo "$AUC < 0.9" | bc -l) )); then
  echo "FAIL: Map quality regression detected (AUC=$AUC < 0.9)"
  exit 1
fi
echo "PASS: Map quality OK"
```

### GitHub Actions版

`.github/workflows/quality-gate.yml` を使用:

```bash
gh workflow run quality-gate.yml \
  -f source=new_map.pcd \
  -f reference=reference_map.pcd \
  -f auc_threshold=0.9 \
  -f chamfer_threshold=0.1
```

## Step 5: パイプラインで全自動

```bash
# 1コマンドで: フィルタ → ダウンサンプリング → 評価
ca pipeline new_map.pcd reference.pcd -o production.pcd \
  -v 0.2 -n 20 -s 2.0 \
  --output-json pipeline_result.json

# Output:
# Filter:     1784475 -> 1742057 pts (removed 42418)
# Downsample: 1742057 -> 1550910 pts (11.0%)
# Chamfer:    0.0557
# AUC (F1):   0.9819
```

## Step 6: 複数ボクセルサイズの比較

```bash
# 最適なボクセルサイズを見つける
for v in 0.05 0.1 0.2 0.5; do
  ca downsample map.pcd -o "down_v${v}.pcd" -v $v --evaluate
done

# 比較プロットを生成 (Python API)
python3 -c "
from ca.evaluate import evaluate
from ca.plot import plot_multi_f1

results, labels = [], []
for v in [0.05, 0.1, 0.2, 0.5]:
    r = evaluate(f'down_v{v}.pcd', 'map.pcd')
    results.append(r)
    labels.append(f'v={v}m (AUC={r[\"auc\"]:.3f})')

plot_multi_f1(results, labels, 'voxel_comparison.png')
"
```

## 判定基準の目安

| AUC (F1) | 判定 | ユースケース |
|---|---|---|
| > 0.99 | 優秀 | 高精度位置推定用マップ |
| 0.95 - 0.99 | 良好 | 一般的なナビゲーション用 |
| 0.90 - 0.95 | 許容 | 粗い経路計画用 |
| < 0.90 | 要確認 | 品質劣化の可能性 |
