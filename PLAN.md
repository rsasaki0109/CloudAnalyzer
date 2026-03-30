# CloudAnalyzer 開発プラン

## プロジェクト概要

点群（PCD/PLY/LAS）の加工・評価・可視化を行うCLIツール。
核心は **加工時に `--evaluate` で即品質評価できる** こと。

- リポジトリ: https://github.com/rsasaki0109/CloudAnalyzer
- ライセンス: MIT
- Python 3.10+, Open3D, Typer, Matplotlib

## 現在の状態 (2026-03-26)

### 規模
- **23コマンド**, **155テスト**, mypy通過, CI green
- コアモジュール: 26ファイル (`cloudanalyzer/ca/`)
- CLI: `cloudanalyzer/cli/main.py`

### コマンド一覧

**評価・分析:**
`evaluate` / `compare` / `diff` / `info` / `stats` / `batch` / `pipeline`

**加工 (downsample/filter/sample は `--evaluate` 対応):**
`downsample` / `filter` / `sample` / `merge` / `align` / `split` / `crop` / `convert` / `normals`

**可視化:**
`web` / `view` / `density-map` / `heatmap3d`

**その他:**
`version`

### アーキテクチャ
```
cloudanalyzer/
├── ca/                    # コアライブラリ (各モジュールが dict を返す)
│   ├── evaluate.py        # F1/Chamfer/Hausdorff/AUC + plot_f1_curve
│   ├── pipeline.py        # filter → downsample → evaluate
│   ├── metrics.py         # compute_nn_distance (Open3D vectorized)
│   ├── compare.py         # レジストレーション付き比較 (logging使用)
│   ├── web.py             # Three.js ブラウザビューア
│   ├── plot.py            # plot_multi_f1, heatmap3d
│   ├── split.py           # グリッドタイル分割
│   ├── log.py             # logging設定 (--verbose/--quiet)
│   └── ...                # io, registration, visualization, report, etc.
├── cli/
│   └── main.py            # Typer CLI (23コマンド)
│       ├── _run_evaluate() # --evaluate 共通ヘルパー
│       ├── _handle_error() # エラーヒント表示
│       └── @app.callback() # --verbose/--quiet
├── tests/                 # 155テスト (pytest)
├── pyproject.toml         # PEP 621, mypy, pytest設定
└── setup.py               # editable install shim
```

### 設計原則
1. **各モジュールは dict を返す** — JSON-serializable、テスト・合成しやすい
2. **CLIは薄い** — パース → コア関数呼び出し → フォーマット出力
3. **ログは stderr** — `--format-json` でstdout がクリーンなJSON
4. **ステートレス** — グローバル状態なし

### CI
- `.github/workflows/test.yml`: mypy + pytest (xvfb-run)
- `.github/workflows/quality-gate.yml`: 手動トリガーでAUC/Chamfer閾値チェック

### 実データ動作確認済み
- UTsukuba 2022 Map (14.6M pts) — tc-datasets
- Istanbul Leo Drive Route3 (18.7M pts) — Autoware datasets
- データは `demo_data/` (gitignore済み)

## dogfooding で発見・修正済みの問題

| 問題 | 修正内容 |
|---|---|
| stats Density: 0.00 | `:.4g` 表示に変更 |
| stats 1.7M点で10秒 | 50万点超でサンプリング |
| batch --format-json でログ混入 | format_json時にlogger抑制 |
| downsample -v 0 無言エラー | バリデーション追加 |
| sample -n 0 空出力 | バリデーション追加 |
| crop 0点時にOpen3D警告 | 空はファイル書き込みスキップ |
| np.trapz deprecation | np.trapezoid に置換 |

## 未着手・改善候補

### 高優先度
- [ ] `batch` の "Processing:" ログが多すぎる（115ファイルで溢れる）→ quiet時以外もサマリのみにすべき
- [ ] `stats` の spacing 計算がまだ遅い（KDTree k=2 ループ）→ サンプリングはしたが、Open3D vectorized にできるとさらに高速
- [ ] `ca web` の実データでの動作確認（大規模点群のダウンサンプリング含む）

### 中優先度
- [ ] `ca web` に距離ヒートマップモード追加（2ファイル指定時に距離で着色）
- [ ] `ca evaluate` に `--reference` オプションで3ファイル以上の比較をサポート
- [ ] `ca batch` に `--evaluate` 対応（ディレクトリ内の全ファイルをリファレンスと比較）
- [ ] PyPI への公開 (`python -m build` + `twine upload`)
- [ ] `ca web` でファイルドラッグ&ドロップ対応

### 低優先度
- [ ] `ca align` の精度改善（初期位置推定、NDT対応）
- [ ] `.las` フォーマットの読み書きテスト（実データでの確認）
- [ ] カバレッジ測定 (`pytest-cov`)
- [ ] `ca web` に測距ツール追加

## 開発ルール

### コミット
- `Co-Authored-By: Claude` を含めない
- PRの説明に「Generated with Claude Code」を書かない
- コミットはユーザーのみがコミットしたことにする

### コード規約
- mypy 通すこと (`python -m mypy ca/ cli/`)
- テスト全件通すこと (`python -m pytest tests/`)
- 作業ディレクトリ: `cd cloudanalyzer` してから実行
- ログは `ca.log.logger` を使う（print禁止）
- 新コマンド追加時はテストも追加

### テスト実行
```bash
cd cloudanalyzer
python -m pytest tests/ -v          # 全テスト
python -m mypy ca/ cli/             # 型チェック
```

### 実データでのテスト
```bash
# demo_data/ にダウンロード済み (gitignore)
TUNNEL=demo_data/istanbul/2024_08_16-route3/loam_feature_localization/local
ca info ${TUNNEL}/tunnel_local_corner.pcd
ca evaluate ${TUNNEL}/tunnel_local_corner.pcd ${TUNNEL}/tunnel_local_surface.pcd
```
