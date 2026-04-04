# PLAN / Claude Handoff

最終更新: 2026-04-04

このファイルは Claude への引き継ぎ用メモ。
目的は「CloudAnalyzer を point cloud utility 集ではなく、mapping / localization / perception の出力を後処理・比較・回帰検知する QA platform として育てる」流れを止めずに再開できるようにすること。

## 0. 一番大事な要約

今のローカル worktree には、未 commit の大きな experiment slice が 2 本載っている。

1. `check_regression_triage`
`ca check` が fail したとき、どの check から inspection すべきかを severity-first で順位付けする仕組み。

2. `check_baseline_evolution`
`ca check --output-json` の summary JSON を入力にして、candidate baseline を `promote / keep / reject` する仕組み。

どちらも experiment-driven-package の方針で作ってあり、`core/` に最小 contract、`experiments/` に 3 実装、`docs/experiments.md` / `docs/decisions.md` / `docs/interfaces.md` の更新、process tests まで入っている。

この 2 本はローカルでは validation 済みだが、まだ commit / push していない。

## 1. 現在の Git 状態

- branch: `codex/web-pages-demo`
- HEAD: `caf94cd` (`Harden config-driven QA onboarding and benchmarks`)
- 現在の worktree: dirty

### `git status --short` の要点

追跡済み変更:

- `README.md`
- `cloudanalyzer/ca/core/__init__.py`
- `cloudanalyzer/ca/core/checks.py`
- `cloudanalyzer/ca/experiments/process_docs.py`
- `cloudanalyzer/cli/main.py`
- `cloudanalyzer/tests/test_check_suite.py`
- `cloudanalyzer/tests/test_cli.py`
- `cloudanalyzer/tests/test_process_docs.py`
- `docs/commands/analysis.md`
- `docs/decisions.md`
- `docs/experiments.md`
- `docs/interfaces.md`
- `docs/tutorial-run-quality-gate.md`

未追跡:

- `cloudanalyzer/ca/core/check_baseline_evolution.py`
- `cloudanalyzer/ca/core/check_triage.py`
- `cloudanalyzer/ca/experiments/check_baseline_evolution/`
- `cloudanalyzer/ca/experiments/check_triage/`
- `cloudanalyzer/tests/test_check_baseline_evolution_process.py`
- `cloudanalyzer/tests/test_check_triage_process.py`

重要:
`check_regression_triage` も `check_baseline_evolution` も、会話上は完了済みとして扱われていたが、実際にはまだ HEAD に入っていない。Claude はここを前提に作業を再開すること。

## 2. プロジェクトの今の立ち位置

CloudAnalyzer はもう「点群ツールの小さい CLI」ではなく、次の立場に寄せている。

- mapping の後処理 QA
- localization / SLAM trajectory QA
- perception / reconstruction / depth-derived 3D output QA
- config-driven contract による CI gate
- browser inspection
- public benchmark pack
- GitHub Pages demo

README もその方向に寄せてあり、今の CLI は `29 commands` として見せている。

現時点の主力 flow は以下。

1. `ca init-check` で `cloudanalyzer.yaml` の雛形を出す
2. `ca check cloudanalyzer.yaml` で unified QA を回す
3. fail 時は triage で inspection 順を出す
4. summary JSON を履歴比較して baseline 更新判断を出す
5. HTML / JSON / `ca web` で drill-down する

## 3. ここまでで HEAD に入っているもの

HEAD `caf94cd` までに入っている主なもの:

- `ca web` の interactive improvement
- web-only の experiment-driven slices
  - point cloud reduction
  - trajectory sampling
  - progressive loading
- public demo / Pages export
- public benchmark pack
- config-driven QA
  - `cloudanalyzer.yaml`
  - `ca check`
  - `ca init-check`
  - reusable workflow / benchmark CI

この時点では baseline evolution はまだ HEAD に入っていない。

## 4. ローカル未 commit slice その1: `check_regression_triage`

### 問題設定

`ca check` で複数の gated check が fail したとき、どの regression を先に見るべきかを決めたい。

問題文:

`When multiple mapping / localization / perception checks fail, rank the failures so the user can inspect the most informative regression first.`

### stable core

ファイル:

- `cloudanalyzer/ca/core/check_triage.py`

最小 contract:

- `CheckTriageItem`
- `CheckTriageRequest`
- `RankedCheckTriageItem`
- `CheckTriageResult`
- `CheckTriageStrategy`
- `rank_failed_checks(...)`
- `summarize_failed_checks(...)`

現在の stable strategy:

- `severity_weighted`

### experiments

ファイル:

- `cloudanalyzer/ca/experiments/check_triage/__init__.py`
- `cloudanalyzer/ca/experiments/check_triage/common.py`
- `cloudanalyzer/ca/experiments/check_triage/severity_weighted.py`
- `cloudanalyzer/ca/experiments/check_triage/pareto_frontier.py`
- `cloudanalyzer/ca/experiments/check_triage/signature_cluster.py`
- `cloudanalyzer/ca/experiments/check_triage/evaluate.py`

設計思想:

- `severity_weighted`: functional
- `pareto_frontier`: OOP
- `signature_cluster`: clustering / grouping oriented

### production integration

入っている場所:

- `cloudanalyzer/ca/core/checks.py`
  - `run_check_suite(...)` が `summary["triage"]` を返す
- `cloudanalyzer/cli/main.py`
  - `ca check` の human-readable output に top 3 triage を表示
- `cloudanalyzer/ca/core/__init__.py`
  - triage contract を export

### docs

更新対象:

- `docs/experiments.md`
- `docs/decisions.md`
- `docs/interfaces.md`
- `docs/commands/analysis.md`
- `docs/tutorial-run-quality-gate.md`
- `README.md`

### tests

追加 / 更新:

- `cloudanalyzer/tests/test_check_triage_process.py`
- `cloudanalyzer/tests/test_check_suite.py`
- `cloudanalyzer/tests/test_cli.py`
- `cloudanalyzer/tests/test_process_docs.py`

### 現在の判断

この slice はローカルでは閉じている。
次に Claude がやるべきことは「実装を進める」ことではなく、「この slice を baseline evolution とどう commit 分割するか決める」こと。

## 5. ローカル未 commit slice その2: `check_baseline_evolution`

### 問題設定

candidate の QA summary と、過去の summary 群を見て、baseline を更新してよいか判断したい。

問題文:

`Given current outputs, prior baselines, and QA results, decide whether to keep, reject, or promote a candidate baseline revision.`

### stable core

ファイル:

- `cloudanalyzer/ca/core/check_baseline_evolution.py`

最小 contract:

- `DecisionLabel = Literal["promote", "keep", "reject"]`
- `BaselineCheckSnapshot`
- `BaselineEvolutionSnapshot`
- `BaselineEvolutionRequest`
- `BaselineEvolutionResult`
- `BaselineEvolutionStrategy`
- `snapshot_from_check_result(...)`
- `build_baseline_evolution_request(...)`
- `decide_baseline_evolution(...)`
- `summarize_baseline_evolution(...)`

現在の stable strategy:

- `StabilityWindowBaselineEvolutionStrategy`

設計意図:

- failed candidate は即 `reject`
- history window が足りないときは `keep`
- 直近の pass window が安定しており、margin 改善が見えるときだけ `promote`

### experiments

ファイル:

- `cloudanalyzer/ca/experiments/check_baseline_evolution/__init__.py`
- `cloudanalyzer/ca/experiments/check_baseline_evolution/common.py`
- `cloudanalyzer/ca/experiments/check_baseline_evolution/threshold_guard.py`
- `cloudanalyzer/ca/experiments/check_baseline_evolution/pareto_promote.py`
- `cloudanalyzer/ca/experiments/check_baseline_evolution/stability_window.py`
- `cloudanalyzer/ca/experiments/check_baseline_evolution/evaluate.py`

設計思想:

- `threshold_guard`: functional
- `pareto_promote`: OOP / Pareto frontier
- `stability_window`: pipeline

共通評価:

- decision match
- critical match
- false promote
- false reject
- stability under perturbation
- readability
- extensibility
- runtime

### production integration

追加コマンド:

- `ca baseline-decision`

実装場所:

- `cloudanalyzer/cli/main.py`

使い方:

```bash
ca baseline-decision qa/current-summary.json \
  --history qa/baseline-2026-03-20.json \
  --history qa/baseline-2026-03-27.json
```

JSON モード:

```bash
ca baseline-decision qa/current-summary.json \
  --history qa/baseline-summary.json \
  --format-json --output-json qa/baseline-decision.json
```

重要な挙動:

- input は `ca check --output-json ...` が出した summary JSON
- `--format-json` 時は stdout を純 JSON に保つようにしてある
- decision が `reject` のときは exit code 1

### docs

更新対象:

- `docs/experiments.md`
- `docs/decisions.md`
- `docs/interfaces.md`
- `docs/commands/analysis.md`
- `docs/tutorial-run-quality-gate.md`
- `README.md`

### tests

追加 / 更新:

- `cloudanalyzer/tests/test_check_baseline_evolution_process.py`
- `cloudanalyzer/tests/test_cli.py`
- `cloudanalyzer/tests/test_process_docs.py`

### 現在の判断

この slice もローカルでは閉じている。
production への integration は「`ca check` に全部押し込む」のではなく、「`ca check` の result JSON を使う別 entrypoint」に切ってある。これは意図的で、QA gate と baseline 更新判断の責務を分離している。

## 6. process docs の状態

experiment-driven-package の流儀で、以下はすでに更新済み。

- `docs/experiments.md`
- `docs/decisions.md`
- `docs/interfaces.md`

`cloudanalyzer/ca/experiments/process_docs.py` は、今は以下の slice をまとめて再生成する。

1. `web_point_cloud_reduction`
2. `web_trajectory_sampling`
3. `web_progressive_loading`
4. `check_scaffolding`
5. `check_regression_triage`
6. `check_baseline_evolution`

つまり、Claude は docs を手で直す前に、まず process docs generator と evaluator を信じてよい。

## 7. 直近で実行して通っている検証

2026-04-04 時点で、ローカルの dirty worktree に対して以下を実行済み。

### full test

```bash
python3 -m pytest cloudanalyzer/tests/ -q
```

結果:

- `365 passed, 1 warning`

warning:

- Matplotlib の `Axes3D` import warning
- failure ではない

### full mypy

```bash
python3 -m mypy --config-file cloudanalyzer/pyproject.toml cloudanalyzer/ca/ cloudanalyzer/cli/
```

結果:

- `Success: no issues found in 78 source files`

### process docs regeneration

```bash
python3 -m ca.experiments.process_docs --write-docs --repetitions 1
```

結果:

- docs 再生成成功
- Axes3D warning のみ

### targeted tests

```bash
python3 -m pytest \
  cloudanalyzer/tests/test_check_baseline_evolution_process.py \
  cloudanalyzer/tests/test_process_docs.py \
  cloudanalyzer/tests/test_cli.py -q
```

結果:

- `71 passed, 1 warning`

## 8. Claude が最初にやるべきこと

Claude に最初にやってほしい順番はこれ。

1. `git status --short` を見て、triage と baseline evolution が両方 uncommitted なことを確認する
2. `git diff --stat` と `git diff` で、どこまでが triage でどこからが baseline evolution かを切り分ける
3. ユーザーが望むなら、2 commit に分ける
4. ユーザーが望まないなら、まとめて 1 commit にしてもよい
5. commit 前にもう一度 `pytest` と `mypy` を流す
6. push / PR 更新はユーザーの明示指示が出てから行う

## 9. commit 分割のおすすめ

理想は 2 commit。

### commit A: triage

内容:

- `cloudanalyzer/ca/core/check_triage.py`
- `cloudanalyzer/ca/experiments/check_triage/`
- `cloudanalyzer/ca/core/checks.py`
- triage 関連 test
- triage 関連 docs

理由:

- `ca check` の fail triage は独立価値が高い
- baseline evolution がまだ不要でも取り込める

### commit B: baseline evolution

内容:

- `cloudanalyzer/ca/core/check_baseline_evolution.py`
- `cloudanalyzer/ca/experiments/check_baseline_evolution/`
- `cloudanalyzer/cli/main.py` の `baseline-decision`
- baseline evolution 関連 test
- baseline evolution 関連 docs

理由:

- `ca check` の結果を後段でどう収束させるか、という別 slice だから

ただし、もしユーザーが「まとめて push」したいなら 1 commit でも構わない。

## 10. 次の探索候補

baseline evolution を commit / push したあとに、次に切るとよい problem slice は以下。

### 候補A: baseline history management

問題:

history JSON を手で `--history` に並べるのではなく、履歴の保存・選択・ローテーションを contract にしたい。

狙い:

- `qa/history/` の命名規約
- `latest promoted baseline` の発見
- retain policy

これが入ると `baseline-decision` が実運用しやすくなる。

### 候補B: CI integration for baseline decision

問題:

`ca check` の summary artifact と `baseline-decision` を GitHub Actions 上でつなぎたい。

狙い:

- current summary を artifact 化
- baseline summary をダウンロード / checkout
- `promote / keep / reject` を workflow summary に出す

これは platform 感をさらに強くする。

### 候補C: regression bundle / triage export

問題:

fail した regression 一式を bundle にして、`ca web-export` と合わせて共有したい。

狙い:

- triage top N の report / JSON / web bundle をまとめる
- reviewer が再現しやすくする

### 候補D: perception demo を public に増やす

今の公開 demo は map 寄り。
README の主張に合わせるなら perception 系 public example を 1 本増やす価値がある。

## 11. やらない方がよいこと

Claude には以下を避けてほしい。

- いきなり大きい抽象 interface を設計し直すこと
- `ca check` に baseline policy をべったり埋め込むこと
- experiments を消して core だけ残すこと
- docs を最後にまとめて直そうとすること
- triage と baseline evolution を混ぜた巨大 refactor にすること

今の repo は「設計を先に決める」のではなく、「具体の群を比較し、共通 contract を後から見つける」方針で進んでいる。Claude もその流れに乗るべき。

## 12. 現在の重要ファイル早見表

production core:

- `cloudanalyzer/ca/core/checks.py`
- `cloudanalyzer/ca/core/check_triage.py`
- `cloudanalyzer/ca/core/check_baseline_evolution.py`
- `cloudanalyzer/ca/core/check_scaffolding.py`

production CLI:

- `cloudanalyzer/cli/main.py`

experiment slices:

- `cloudanalyzer/ca/experiments/check_triage/`
- `cloudanalyzer/ca/experiments/check_baseline_evolution/`
- `cloudanalyzer/ca/experiments/process_docs.py`

tests:

- `cloudanalyzer/tests/test_check_suite.py`
- `cloudanalyzer/tests/test_cli.py`
- `cloudanalyzer/tests/test_process_docs.py`
- `cloudanalyzer/tests/test_check_triage_process.py`
- `cloudanalyzer/tests/test_check_baseline_evolution_process.py`

docs:

- `README.md`
- `docs/commands/analysis.md`
- `docs/tutorial-run-quality-gate.md`
- `docs/experiments.md`
- `docs/decisions.md`
- `docs/interfaces.md`

## 13. 最後のメモ

この handoff の時点では、実装そのものより「どう clean に landing させるか」が主題。

コードはかなり揃っている。
未解決なのは品質ではなく、commit boundary と push timing。

Claude が入ったら、まず diff を確認し、必要なら triage と baseline evolution を切り分け、それからユーザーに commit / push の判断を仰ぐのがよい。
