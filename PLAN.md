# PLAN / Claude Handoff

最終更新: 2026-04-05

このファイルは Claude への引き継ぎ用メモ。

## 0. 現在の状態

すべて main にマージ済み。dirty な worktree はない。

- branch: `main`
- 直近マージ: PR #1 (squash merge)
- pytest: 389 passed
- mypy: 87 source files, no issues
- CLI: 32 commands
- Experiment slices: 7

## 1. 入っている機能

### config-driven QA
- `ca check cloudanalyzer.yaml` — unified QA gate
- `ca init-check` — 雛形生成
- reusable GitHub Actions workflow

### check regression triage
- `ca check` fail 時に severity-first 順位付け
- core: `severity_weighted`
- experiments: `pareto_frontier`, `signature_cluster`

### baseline evolution
- `ca baseline-decision` — promote / keep / reject
- `ca baseline-save` — history dir に保存
- `ca baseline-list` — 一覧表示
- `--history-dir` で自動発見、`--keep` でローテーション
- core: `stability_window`
- experiments: `threshold_guard`, `pareto_promote`

### trajectory evaluation
- `ca traj-evaluate` — ATE, RPE, drift, coverage
- lateral / longitudinal error 分解
- `--max-lateral`, `--max-longitudinal` quality gate
- `ca check` の trajectory gate にも対応

### ground segmentation evaluation
- `ca ground-evaluate` — voxel-based precision/recall/F1/IoU
- core: `voxel_confusion`
- experiments: `nearest_neighbor`, `height_band`

### web inspection
- `ca web` — interactive browser inspection
- `ca web-export` — static HTML bundle
- experiment-driven: point cloud reduction, trajectory sampling, progressive loading

### public demo & benchmark
- GitHub Pages demo
- public benchmark pack
- reusable CI workflows

## 2. 次の探索候補

### CI integration
`ca check` → `baseline-save` → `baseline-decision` を GitHub Actions でつなぐ。

### perception public demo
ground-evaluate のサンプルデータ付き公開例。

### regression bundle / triage export
fail した regression を bundle にして共有。

### `ca check` に ground check kind 追加
ground-evaluate を config-driven QA に統合。

## 3. やらない方がよいこと

- いきなり大きい抽象 interface を設計し直すこと
- experiments を消して core だけ残すこと
- docs を最後にまとめて直そうとすること

今の repo は「具体の群を比較し、共通 contract を後から見つける」方針で進んでいる。
