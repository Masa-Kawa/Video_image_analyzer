# 変更履歴

本プロジェクトの主な変更を記録する。日付は `YYYY-MM-DD`（JST）。

## 2026-06-11

### ドキュメント
- **アノテーション編集Webアプリ（`src/annotate/`）を文書化**
  - `docs/annotation_editor.md` を新規作成（起動方法・CLIオプション・UI操作/キーボードショートカット・出力形式〈`_gold.srt` / `_gold_dpo_pairs.jsonl`〉・DPOペアの `change` 種別・術式別フェーズ語彙・API・セキュリティ）。
  - `README.md` に「アノテーション編集Webアプリ（修正用UI）」セクションを追加し、`docs/annotation_editor.md` へリンク。
  - `TUTORIAL.md` の「Step 4: 手動修正」を「方法A: 編集Webアプリ（推奨）/ 方法B: Shotcut」の2本立てに再構成。
- **README をコードベースの現状に同期**
  - プロジェクト構成ツリーに未掲載モジュールを追加（`annotate/`、`bleed_ai/`＋`zeroshot/`、`cavity/`、`anomaly/`、`cholec_phase/`、`surgical_pipeline.py`、`red/` の追加検出器〈`bleed_trend`/`bleed_contact`/`bleed_flow`〉、`phase_segmenter`、action推論スクリプト、`srt_to_mkv` ほか）。
  - 解析エンジン一覧に新解析器の行を追加し、詳細は `docs/analyzers.md` へ誘導。
  - 出力ファイル一覧に `_gold.srt` / `_gold_dpo_pairs.jsonl` を追加。
