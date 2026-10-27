# Public Benchmark Packs

CloudAnalyzer の `mapping / localization / perception` QA を、誰でも再生成できる pack 置き場。

このディレクトリには large binary asset は commit せず、builder と generated output の contract だけを置く。

## Generate

```bash
python scripts/build_public_benchmark_pack.py --output benchmarks/public/stanford-bunny-mini
```

Stanford Bunny のダウンロードはソケット通信に30秒のタイムアウトを設定し、
一時的な通信障害、HTTP 408/429/5xx、Content-Length より短い取得結果を最大3回まで試行する。
再試行前には1秒、2秒待ち、部分アーカイブを破棄して先頭から取得する。
404などの恒久的なHTTPエラーやローカルの書き込み失敗は再試行しない。
タイムアウトは処理全体の期限ではなく、配布元が利用できなければ生成は失敗する。
別データへの自動置き換えは行わない。

生成される pack には以下が含まれる:

- `baselines/` reference artifact
- `outputs/` pass / regression candidate
- `configs/` `ca check` 用 config
- `expected/` expected summary JSON
- `reports/` / `results/` generated artifact
- `manifest.json` expected pass/fail manifest

## Run

```bash
ca check benchmarks/public/stanford-bunny-mini/configs/suite-pass.cloudanalyzer.yaml
ca check benchmarks/public/stanford-bunny-mini/configs/suite-regression.cloudanalyzer.yaml
```

`suite-pass` は pass、`suite-regression` は fail を返す想定。

`.github/workflows/public-benchmark-pack.yml` はこの pack を public Stanford Bunny から再生成し、manifest の pass/fail expectation と一致することを CI で確認する。
