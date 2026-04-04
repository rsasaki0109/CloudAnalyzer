# Public Benchmark Packs

CloudAnalyzer の `mapping / localization / perception` QA を、誰でも再生成できる pack 置き場。

このディレクトリには large binary asset は commit せず、builder と generated output の contract だけを置く。

## Generate

```bash
python scripts/build_public_benchmark_pack.py --output benchmarks/public/stanford-bunny-mini
```

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
