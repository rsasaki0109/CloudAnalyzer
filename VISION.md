# Vision

## Mission

CloudAnalyzer の目的は、3D perception / mapping / reconstruction の出力物を
**測れる、比較できる、再現できる、CI に載せられる状態にすること**。

点群処理ライブラリ、3Dビューア、学習フレームワーク、SLAM/LIO 実装はすでに強い。
CloudAnalyzer はそれらと競合するのではなく、**それらの上で品質を判断する共通レイヤ**になる。

## 何の上位互換を目指すか

「上位互換」は、機能数の多さではなく、ユーザーがやりたい仕事をより高いレイヤで完結できることとして定義する。

### PCL / Open3D に対して

- 低レベルの点群処理 API 数では勝負しない
- 加工した結果をその場で評価できる
- 比較、差分可視化、レポート、回帰検知まで一気通貫で行える

### CloudCompare / Potree に対して

- 単発の GUI 操作ではなく、CLI と JSON を前提に自動化できる
- 目視確認だけでなく、定量評価と品質ゲートを回せる
- 共有しやすい HTML / 画像 / JSON レポートを出せる

### PyTorch 系に対して

- 学習基盤には踏み込まない
- 推論結果の幾何品質を、同じ指標と同じ可視化で比較できる
- データセット横断・手法横断でベンチマークできる

### Gaussian Splatting 系に対して

- 新しい表現を処理対象として取り込み、点群やメッシュと比較できる
- 表現ごとの美しさではなく、幾何的な整合性と差分を評価できる

### Continuous-time LIO 系に対して

- 推定器そのものは実装しない
- 軌跡、時系列、地図品質、ドリフトを比較・可視化できる
- 実験結果を再現可能な形で残せる

## プロダクト定義

CloudAnalyzer は、3Dデータ処理のための次の 5 層を一体で提供する。

1. Processing
2. Evaluation
3. Comparison
4. Visualization
5. Regression Testing / Reporting

短く言えば、CloudAnalyzer は **3D 処理結果の QA / Benchmark / Operations platform** を目指す。

## 設計原則への影響

このビジョンを実現するために、以下を優先する。

- `--evaluate` を核にして、加工直後に品質がわかること
- CLI / JSON / HTML を第一級に扱い、再現性を壊さないこと
- 各処理を dict ベースで返し、組み合わせやすくすること
- backend / adapter で外部ツールや新しい表現を取り込みやすくすること
- 可視化は viewer ではなく、判断を助ける diff / heatmap / overlay を重視すること

## 近い将来のロードマップ

### Phase 1: Point Cloud QA を完成させる

- `ca batch --evaluate` を追加し、複数ファイル比較を自動化する
- `ca web` に差分ヒートマップを入れる
- HTML / JSON / PNG レポートをまとめて出せるようにする
- 大規模点群での速度とメモリ使用量を改善する

### Phase 2: Trajectory / Temporal Evaluation を追加する

- 軌跡とタイムスタンプを扱える共通データモデルを追加する
- odometry / trajectory の誤差指標を追加する
- continuous-time LIO 系の結果を比較できるようにする

### Phase 3: Representation / Backend を拡張する

- point cloud 以外に mesh / splat を取り込めるようにする
- Open3D backend の外に adapter / plugin 方式を導入する
- PyTorch 系の推論結果や研究成果を評価に流し込みやすくする

## 今はやらないこと

以下は現時点では非目標とする。

- PCL や Open3D の全アルゴリズムを再実装すること
- CloudCompare の GUI を全面的に作り直すこと
- PyTorch の学習ループやトレーニング基盤を抱え込むこと
- LIO / SLAM 推定器そのものを内製すること

## 判断基準

今後の機能追加は、次の問いで判断する。

- 加工後に品質判断がしやすくなるか
- 複数手法の比較がしやすくなるか
- 実験の再現性と CI 連携が強くなるか
- 大規模実データでも現実的に回るか
- 他ツールの結果を取り込みやすくなるか

この答えが yes なら、CloudAnalyzer の中核に近い。
