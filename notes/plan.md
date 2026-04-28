## 結論

CloudAnalyzer に **地図評価（MapEval 参考）** と **手動ループ閉じ QA** を段階的に積み上げる方針は成立している。現在の `feature/mapeval-reference` は PR としてレビュー可能なまとまりになっており、次は「運用で迷う点（exit code / 指標の違い）」の明文化と、軽量 fixture 戦略（合成 or 取得スクリプト）を固めるフェーズ。

## 確認済み事実

- **PR**: `https://github.com/rsasaki0109/CloudAnalyzer/pull/7`
- **ブランチ**: `feature/mapeval-reference`
- **実装済み（MapEval 参考）**
  - `cloudanalyzer/ca/experiments/map_evaluate/`
    - `nn_thresholds`: GT あり（accuracy/completeness@t + chamfer + fscore、任意で colored PLY artifacts）
    - `voxel_entropy`: GT なし proxy（近傍占有 entropy）
  - docs 生成へスライス追加済み（`process_docs.py` → `docs/{experiments,decisions,interfaces}.md`）
- **実装済み（手動ループ閉じ QA）**
  - `ca map-evaluate`
  - `ca posegraph-validate`（g2o/TUM/key_point_frame の軽量検証）
  - `ca loop-closure-report`
    - before/after/ref の地図比較（`ca.evaluate` ベース）
    - 任意で trajectory 評価（before/after/ref）
    - 任意で posegraph セッション検証（before/after）
    - `--before-session-root` / `--after-session-root` で path 自動探索（expected/exists を返す）
    - quality gate FAIL の場合に exit code 1
- **テスト**
  - 追加済み: `tests/test_loop_closure_report.py`, `tests/test_map_evaluate.py`, `tests/test_posegraph.py`（探索/欠頂点も）
  - 最新のローカル実行で `pytest tests/` / `mypy ca/ cloudanalyzer_cli/` は通過
- **方針**
  - GPL の `interactive_slam` / `Manual-Loop-Closure-Tools` は **コード統合しない**（CloudAnalyzer 側は独自/BSD 寄せ）

## 未確認/要確認項目

- **指標の混同防止**
  - `ca map-evaluate`（accuracy/completeness@t 等）と、`loop-closure-report` が使う `ca.evaluate`（AUC/F1 曲線）は **別の指標セット**。利用者が混乱しやすいので docs/PR 本文で明記する。
- **exit code ポリシー**
  - `loop-closure-report` は現状「quality gate FAIL でのみ exit 1」。
  - posegraph の `summary.ok == false` を “ゲートに含めたい” 場合は、追加 gate（例: `--require-posegraph-ok`）などの設計が必要。
- **大規模点群での実用性**
  - `nn_thresholds` は Open3D KD-tree を点ごとに呼ぶため、超大規模点群では重い可能性。ダウンサンプル運用か、将来的な高速化が必要。
- **fixture（データ同梱）戦略**
  - リポに小さな点群/セッションを同梱するか、外部データは URL + 取得/縮小スクリプトにするかを決める（ライセンス/CI負荷）。

## 次アクション

- **PR #7 を merge-ready にする**
  - PR 本文/ドキュメントに「指標の違い」「exit code ポリシー」を 2〜3 行で明記
  - 必要なら `docs/commands/` に `map-evaluate` / `posegraph-validate` / `loop-closure-report` のページ追加
- **dogfooding を継続できる形にする**
  - 合成の最小 fixture（before/after/ref + 最小 g2o）を `demo_assets/` か `tests/assets/` に置く案
  - HDL Graph SLAM 等の実データは **同梱せず**、取得→縮小→変換を行うスクリプト案（ライセンス安全）
- **運用ゲートの拡張（必要なら）**
  - posegraph `summary.ok` を gate に含めるオプション追加
  - map 評価の “期待改善量” をプロジェクトごとにテンプレ化（設定ファイル化）

