# 変更履歴

本プロジェクトの主な変更を記録する。日付は `YYYY-MM-DD`（JST）。

## 2026-06-11

### アノテーション編集Webアプリ（機能拡張）
- **区間ジャンプ**: `Shift+←/→` で再生ヘッドを前後の区間先頭へ移動＋選択。
- **UIから動画/SRTを読み込み**（`POST /api/open`）。保存済み `_gold.srt` を指定すれば安定IDを保ったまま**再編集**できる。
- **修正履歴**: 保存ごとに「前回保存→今回」の差分を `_gold_history.jsonl` に追記し、`GET /api/history` と履歴ダイアログで一覧。変更0の保存は記録しない。

### 堅牢性・セキュリティ（レビュー反映）
- `/api/open` のパスを正規化し `--base-dir`（既定=cwd）配下に限定（パストラバーサル防止）。配下外/絶対パス/空/ディレクトリは 400。
- `/api/history` に CSRF 必須化＋`limit` ページング（`deque(maxlen)` で OOM 回避、異常値は 1〜1000 に丸め）。
- 履歴タイムスタンプを UTC（`+00:00`）固定。`last_saved` 更新を全ファイル書き込み成功後に移動（整合性）。
- 動画キャッシュバスターを mtime ベースに変更。
- フロント: 履歴更新を `await` 化、集計を null プロトタイプ辞書＋`hasOwnProperty.call`（互換性）に、未知 change タイプを明示表示、履歴取得失敗を画面に表示。
- CSS: 主要ボタンを `.btn-primary` に共通化、履歴色を `--success` 変数で統一、`dialog` に `max-height:90vh`＋レスポンシブ調整を追加。
- テスト: 一時ディレクトリ／`TestClient` の後始末を共通化、パストラバーサル・`limit` 境界・timestamp 形式などのバリデーションを追加（annotate 計27テスト）。

### 開発基盤（Git フック）
- **pre-commit（lint）**: ステージした `.py`（リネーム含む `ACMR`）に `ruff check --fix` を実行（整形は含めない）。`pyproject.toml` に `[tool.ruff]` と dev 依存 `ruff` を追加。
- **pre-push（安全検査）**: 送信対象コミットを `scripts/safety-check.sh` で走査し、危険ファイル/データの混入を検出して push を中止（PR前に手動実行も可）。
  - BLOCK: 動画/モデル/解析データ/アーカイブ/鍵・証明書/`.env`、`.gitignore`対象なのに追跡中、5MB超、APIキー/秘密鍵らしき文字列。
  - WARN（許可）: 個人の絶対パス、パスワード/キーらしき代入。
- `scripts/install-hooks.sh` で pre-commit と pre-push を一括有効化。回避は `--no-verify`。

### ドキュメント
- **編集アプリ**: 仕様 `docs/annotation_editor.md` と実践チュートリアル `docs/annotation_editor_tutorial.md` を新規作成。`README.md` / `TUTORIAL.md`（Step 4 を「編集アプリ／Shotcut」の2本立てに再構成）から誘導。
- **ディレクトリ地図**: `docs/architecture.md` を新規作成（3層〈入口/解析エンジン/内部部品〉の整理・人が直接動かすプログラム一覧・やりたいこと別早見表）。`README` からリンク。
- **TUTORIAL.md Step 1** に「複数動画をまとめてプロキシ作成」を追記（列挙／ワイルドカード／`--series` の3方法と違い）。
- **README をコードベースの現状に同期**: 構成ツリーに未掲載モジュールを追加（`annotate/`・`bleed_ai/`＋`zeroshot/`・`cavity/`・`anomaly/`・`cholec_phase/`・`surgical_pipeline.py`・`red/` の追加検出器・`phase_segmenter`・action推論・`srt_to_mkv` ほか）、解析エンジン一覧・出力ファイル一覧を更新、Git フック節を追加。
