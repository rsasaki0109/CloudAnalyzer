# 補正した点群からHD地図を再生成・比較する

MCPの `trial_mapping_motion` で保存した点群候補から、元の点群・HD地図を
残して新しいHD地図を作ります。元の軌跡の参照評価と候補の評価も保持します。
動いた軌跡へ古い車線・接続・監査を移すことはしません。

1. `start_mapping_motion_run(trial_job_dir, finished_job_dir, out_dir,
   max_attempts, reason)` に、候補が使った元ジョブの**完了した所有者run**、
   新しい外部ディレクトリ、明示した新規2〜8試行の予算を渡します。
   元runの固定レイアウト・抽出範囲・軌跡を含むバンドの関連付け方針を引き継ぎ、
   候補の点群で新しい提案を抽出します。オドメトリ・点群融合は再実行しません。
2. `inspect_mapping_run` と `advance_mapping_run` のinspect/draft/finishで
   新しい元点群の断面を読み、include/deferと理由を明示してHD地図を生成します。
   geometryとlane生成は合計2試行を使います。部分範囲は観察済みのstationだけを
   使い、交通・速度・幅の仮説は元runから変えません。
3. `compare_mapping_motion_maps(candidate_job_dir, baseline_report_file,
   candidate_report_file, out_dir, reason)` に、完了した新runと元・候補の保存済み
   軌跡評価のファイルartifactを渡します。評価のsource、reference、保持ID、
   位置合わせ・評価区間、HDの4監査と抽出条件が一致する必要があります。
4. 比較の新ディレクトリは、元の点群とHDを選んだ `selection.json` をrevision 0で
   保存します。`choose_mapping_motion_pair(selection_dir, choice, reason,
   expected_revision)` のchoice=`candidate`で両方の新成果を一緒に選び、
   choice=`baseline`で元の正確な成果物へ戻せます。`inspect_mapping_motion_selection`
   はハッシュを検査して現在の組を返します。古いrun・ジョブ・評価は変更しません。

軌跡が変わると、同じ「10〜20 m」が同じ入力区間にはなりません。比較は保持した
元スキャンIDの対応を使い、各軌跡のXY距離を元オドメトリの共通XY距離へ区分線形で
対応付けます。時刻と連続frame座標も記録します。これは時間的な対応であり、
物理的に同じ道路や車線だと証明するものではありません。停止区間など、保持姿勢間の
XY移動がゼロの場合は曖昧な対応を作らず比較を拒否します。

新旧の生の生成範囲、共通軸の獲得・喪失区間、4監査、経路、点数、参照軌跡の
局所悪化を読みます。返答は区間・要レビューIDを16件、履歴を8件に制限し、全情報は
比較JSONとselection JSONに残します。新地図の車線IDは別物で、古い接続を引き継ぎません。
参照を見て選んだ方針の探索は未使用走行での精度検証ではありません。

既存の出力先・失敗した提案を上書き・自動再実行しません。生成途中の失敗と予算を
保持し、元の組への選択は保存済み成果を参照するだけです。全参照先ディレクトリを
保持してください。採用はレビューするdraftの選択であり、元の品質・範囲・交通の
未達条件は消えません。品質認証や配備を実行する操作ではありません。

実データの根拠: [NCLTの2走行での再生成・比較・復元](../../benchmarks/vector-map/nclt-motion-pairs/README.md)。
