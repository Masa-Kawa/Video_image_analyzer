# 変更履歴

本プロジェクトの主な変更を記録する。日付は `YYYY-MM-DD`（JST）。

## 2026-06-11 (3)

### ドキュメント
- **アノテーション編集アプリの実践チュートリアルを新規作成**: `docs/annotation_editor_tutorial.md`（起動・画面の見かた・キーボード操作早見表・実際の修正手順・保存物・再編集・修正履歴・トラブル対処）。`README.md` と `TUTORIAL.md`（Step 4）から誘導リンクを追加。

## 2026-06-11 (2)

### アノテーション編集Webアプリ 機能拡張
- **区間ジャンプ**: `Shift+←/→` で再生ヘッドを前後の区間先頭へ移動＋選択。
- **UIから動画/SRTを読み込み**（`POST /api/open`）。保存済み `_gold.srt` を指定すれば安定IDを保ったまま**再編集**できる。
- **修正履歴**: 保存ごとに「前回保存→今回」の差分を `_gold_history.jsonl` に追記し、`GET /api/history` と履歴ダイアログで一覧。変更0の保存は記録しない。

### レビュー反映（堅牢性・セキュリティ）
- `/api/open` のパスを正規化し `--base-dir`（既定=cwd）配下に限定（パストラバーサル防止）。配下外/絶対パス/空/ディレクトリは 400。
- `/api/history` に CSRF 必須化＋`limit` ページング（`deque(maxlen)` で OOM 回避）。
- 履歴タイムスタンプを UTC 固定。`last_saved` 更新を全書き込み成功後に移動（整合性）。
- 動画キャッシュバスターを mtime ベースに変更。
- フロント: 履歴更新を `await` 化、集計を null プロトタイプ辞書に、未知 change タイプを明示表示。CSS は主要ボタンを `.btn-primary` に共通化しダイアログのレスポンシブを追加。
- テスト: 一時ディレクトリの後始末を共通化、パストラバーサル等のバリデーションを追加（annotate 計25テスト）。

### 開発基盤
- **コミット前フック（lint）を追加**: `scripts/git-hooks/pre-commit` がステージした `.py` に `ruff check --fix` を実行（整形は含めない）。`scripts/install-hooks.sh` で各自有効化。`pyproject.toml` に `[tool.ruff]` 設定と dev 依存 `ruff` を追加。

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
