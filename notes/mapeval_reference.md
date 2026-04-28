## 結論

`JokerJohn/Cloud_Map_Evaluation (MapEval)` は、CloudAnalyzer に「点群マップ評価」を入れるときの **評価軸の分離（GTあり/なし）** と **評価パラメータの再現性（設定の固定化）** の面で特に参考になる。CloudAnalyzer 側ではまず `ca/experiments/map_evaluate/` を新設し、GTありの距離系指標と、GTなしの自己整合性指標を並走比較できる形にする。

## 確認済み事実

- MapEval は SLAM 点群マップ評価フレームワークで、評価を以下の2観点に分離している
  - Global Geometric Accuracy（GTあり）
  - Local Structural Consistency（GTなしでも一部可）
- MapEval の設定例（`map_eval/config/config.yaml`）には以下が含まれる
  - 距離閾値リスト（`accuracy_level`）
  - 初期変換行列（`initial_matrix`）
  - ダウンサンプル（`downsample_size`）
  - “初期姿勢のまま評価する” モード（`evaluate_using_initial`）
  - GTなし時の評価として MME（`evaluate_mme`）を明示

## 未確認/要確認項目

- CloudAnalyzer で扱う「マップ」の定義（フレーム統合後点群なのか、占有/メッシュも含むのか）
- GTが無いケースでの採用指標（MME相当をそのまま導入するか、軽量 proxy を作るか）
- 既存の I/O 形式（`.pcd`/`.ply`/`.npz` など）と、CLI のサブコマンド設計

## 次アクション

- `cloudanalyzer/ca/experiments/map_evaluate/` に、まず2系統の実装を置く
  - GTあり: Chamfer/閾値付きNN（AC/COMの簡易版）
  - GTなし: 局所一貫性（MME proxy）
- 実験の勝者が固まったら、最小 Request/Result だけを `cloudanalyzer/ca/core/` に昇格

