# Tutorial: Unified Run Quality Gate

mapping / localization / perception の出力を、`cloudanalyzer.yaml` で 1 つの gate にまとめるワークフロー。

## シナリオ

1. map の後処理結果を baseline と比較したい
2. trajectory の ATE / RPE / drift / coverage も同時に見たい
3. run 単位で fail したら report と browser inspection まで残したい

## Step 1: `cloudanalyzer.yaml` を置く

最短なら先に雛形を出せる。

```bash
ca init-check --profile integrated
```

生成された YAML を自分の artifact / trajectory path に合わせて埋める。

```yaml
version: 1
summary_output_json: qa/summary.json

defaults:
  thresholds: [0.05, 0.1, 0.2]
  max_time_delta: 0.05
  report_dir: qa/reports
  json_dir: qa/results

checks:
  - id: mapping-postprocess
    kind: artifact
    source: outputs/map.pcd
    reference: baselines/map_ref.pcd
    gate:
      min_auc: 0.95
      max_chamfer: 0.02

  - id: localization-run
    kind: trajectory
    estimated: outputs/trajectory.csv
    reference: baselines/trajectory_ref.csv
    alignment: rigid
    gate:
      max_ate: 0.5
      max_rpe: 0.2
      max_drift: 1.0
      min_coverage: 0.9

  - id: integrated-run
    kind: run
    map: outputs/map.pcd
    map_reference: baselines/map_ref.pcd
    trajectory: outputs/trajectory.csv
    trajectory_reference: baselines/trajectory_ref.csv
    alignment: rigid
    gate:
      min_auc: 0.95
      max_chamfer: 0.02
      max_ate: 0.5
      max_rpe: 0.2
      max_drift: 1.0
      min_coverage: 0.9
```

`kind: artifact` は mapping だけでなく、深度点群や再構成点群のような perception 側の 3D output にも使える。

## Step 2: ローカルで gate を回す

```bash
ca check cloudanalyzer.yaml
```

出力例:

```text
Project: localization-mapping-perception
Config:   /path/to/cloudanalyzer.yaml
[PASS] mapping-postprocess (artifact): auc=0.9852  chamfer=0.0083
[PASS] localization-run (trajectory): matched=1240  coverage=98.4%  ate=0.1421  rpe=0.0612
[FAIL] integrated-run (run): map_auc=0.9852  traj_ate=0.6214  coverage=88.0%
  Report: /path/to/qa/reports/integrated-run.html
  JSON:   /path/to/qa/results/integrated-run.json

Checks: total=3  gated=3  pass=2  fail=1  info=0

Triage: severity_weighted  failed=1
  1. integrated-run (run): score=0.7420  dims=trajectory_ate, coverage
```

fail したら exit code は `1` になる。

## Step 3: baseline を更新してよいか決める

`summary_output_json` を出しておくと、current candidate を history と比較して `promote / keep / reject` を決められる。

```bash
ca baseline-decision qa/summary.json \
  --history qa/baseline-2026-03-20.json \
  --history qa/baseline-2026-03-27.json
```

出力例:

```text
Candidate: summary
History:   2 summaries
Decision:  keep (stability_window, confidence=0.76)
Reasons:   candidate_not_yet_stable_enough
Labels:    baseline-2026-03-20, baseline-2026-03-27
```

failed candidate は即 `reject` になり、stable に改善した window だけ `promote` になる。

## Step 4: report と browser inspection を使う

`report_dir` と `json_dir` を入れておくと、各 check ごとに HTML report / JSON が出る。

- `artifact`: map / 3D output の AUC, Chamfer, Best F1 と inspection command
- `trajectory`: coverage, ATE, RPE, drift と alignment 情報
- `run`: map + trajectory をまとめた overall gate と drill-down command

`run` の JSON / HTML には、以下の inspect command が含まれる。

```bash
ca web outputs/map.pcd baselines/map_ref.pcd --heatmap \
  --trajectory outputs/trajectory.csv \
  --trajectory-reference baselines/trajectory_ref.csv \
  --trajectory-align-rigid
```

## Step 5: GitHub Actions で回す

新しい workflow は `.github/workflows/config-quality-gate.yml`。

```bash
gh workflow run config-quality-gate.yml \
  -f config_path=cloudanalyzer.yaml \
  -f artifact_name=run-quality-gate
```

この workflow は:

1. `ca check cloudanalyzer.yaml` を実行する
2. summary JSON を保存する
3. per-check の report / JSON を artifact として upload する
4. gated check が 1 つでも fail したら workflow を fail にする

他 repo から使うときは reusable workflow として呼べる。

```yaml
jobs:
  qa:
    uses: rsasaki0109/CloudAnalyzer/.github/workflows/config-quality-gate.yml@main
    with:
      config_path: cloudanalyzer.yaml
```

## Step 6: どの領域にどう使うか

- Mapping: `artifact` で map の後処理 QA、`run` で trajectory と合わせた deploy gate
- Localization: `trajectory` / `trajectory_batch` で run benchmark、`run_batch` で map 付き比較
- Perception: 深度点群、再構成点群、Gaussian Splatting 由来の geometry output を `artifact` として gate

要点は、CloudAnalyzer を point cloud tool としてではなく、**出力検証の contract** として使うこと。
