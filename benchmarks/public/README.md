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

アーカイブ全体は展開せず、`bunny/reconstruction/bun_zipper.ply` の内容だけを
固定の一時ファイルへコピーする。対象は空でない通常ファイル（最大64 MiB）1個に限り、
重複・リンク・欠落を拒否する。アーカイブの項目数は最大10,000件。
他の項目はディスクへ展開せず、メッシュの内容は変更しない。

取得済みのアーカイブを使う場合はネットワーク不要で生成できる:

```bash
python scripts/build_public_benchmark_pack.py --bunny-archive /data/bunny.tar.gz --output benchmarks/public/local-bunny
```

出力指定を省略した場合の保存先は `benchmarks/public/local-bunny`。
入力は出力ディレクトリ外の通常ファイル（最大128 MiB）に限る。シンボリックリンク、
`--fixture synthetic` やAPIの `source_points` との併用は拒否する。
読み込んだバイト列のSHA-256とサイズを `manifest.json` の `source_archive` に保存し、
sourceを `local-bunny-archive` として直接ダウンロードと区別する。
ハッシュは内容の識別用で、公式配布物との一致を保証するものではない。
既知のアーカイブとの一致を必須にするには `--bunny-sha256 EXPECTED_SHA256` を追加する。
64桁の16進数を指定すると、展開・生成前に同じ読み込み内容のハッシュを比較し、
不一致や不正な形式なら既存出力を変更せずに終了する。大文字の16進数も受け付ける。
一致した期待値は `source_archive.expected_sha256` に記録する。
期待値は利用者が指定するため、一致しても `authenticity_verified` は false のままとする。
既存の出力ディレクトリは生成時に置き換えられるため、入力アーカイブをその中に置かないこと。

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
