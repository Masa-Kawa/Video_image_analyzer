# アノテーション編集Webアプリ（修正用UI）

自動生成したSRT（手術フェーズ等の区間ラベル）を、**動画と同期したブラウザUI上で確認・修正**するためのWebアプリです。Shotcutを使わずブラウザ単体で、セグメントの編集・削除・挿入ができます。SRTが無い状態からの**ゼロ作成（ゴールドラベル付け）**も同じUIで扱えます。

実装: `src/annotate/`（FastAPIバックエンド + バニラJSフロントエンド）

---

## 目的と位置づけ

このアプリは単なる字幕エディタではなく、**追加学習データを生成するツール**を兼ねています。

- 解析器が出力した一次ラベル（自動生成SRT）を人が修正する **Human-in-the-loop** の修正端末。
- 各セグメントに**安定ID**を付与し、元SRT（自動生成）と修正後SRTをIDで突き合わせる。
- 保存時に **DPO的な追加学習ペア**（`rejected`=自動生成 / `chosen`=人手修正）を自動出力する。

```
自動生成SRT ──► [編集Webアプリで修正] ──► gold.srt（確定ラベル）
   (rejected)                          └─► dpo_pairs.jsonl（追加学習素材）
```

---

## 起動

```bash
# 既存SRTを編集モードで開く（ブラウザが自動起動）
python -m src.annotate.server \
    --video case001.mp4 \
    --srt out/case001_phase.srt \
    --procedure cholecystectomy --port 8000

# --srt を省略すると新規作成モード（空セグメントから手付け）
python -m src.annotate.server --video case001.mp4 --procedure cholecystectomy
```

起動時の挙動:

1. ブラウザ再生用プロキシ（**480p**）を `ProxyManager` で取得（無ければ生成）。
2. 指定術式のフェーズ語彙を検証（未登録なら即エラー）。
3. FastAPIサーバを起動し、既定でブラウザを自動で開く（`--no-browser` で抑止）。

### CLIオプション

| オプション | 既定 | 説明 |
|------------|------|------|
| `--video` | （必須） | 対象動画ファイル |
| `--srt` | なし | 編集対象の元SRT（省略で新規作成モード） |
| `--procedure` | `cholecystectomy` | フェーズ語彙の選択（`label_sets.py` のキー） |
| `--save-target` | `<video_stem>_gold.srt` | 保存先SRT |
| `--host` | `127.0.0.1` | 待受アドレス（既定はローカルのみ） |
| `--port` | `8000` | ポート |
| `--no-browser` | — | 起動時にブラウザを開かない |

---

## ブラウザUIの使い方

画面は左に動画＋タイムライン、右にフェーズパレット＋セグメント編集パネルが並びます。

### キーボードショートカット

| キー | 動作 |
|------|------|
| `Space` | 再生 / 停止 |
| `←` / `→` | 1秒シーク（戻る / 進む） |
| `I` / `O` | 選択中セグメントの**開始 / 終了**を現在の再生位置にスナップ |
| `1`–`9` | パレットのフェーズを選択中セグメントへ付与（語彙順） |
| `N` | 再生位置に新規セグメントを挿入 |
| `Del` / `Backspace` | 選択中セグメントを削除 |
| `Ctrl+S` | 保存 |

> 時刻入力欄（`HH:MM:SS,mmm`）にフォーカス中はショートカットは無効化されます。

### 典型的な修正フロー

1. タイムラインのセグメントをクリックして選択（リストからも選択可）。
2. 動画を再生し、区間の境界を目視で確認。
3. ずれていれば、再生位置に合わせて `I`（開始）/ `O`（終了）でスナップ、または時刻欄を直接編集。
4. ラベルが誤っていれば `1`–`9` で正しいフェーズに付け替え。
5. 誤検出は `Del` で削除、抜けは `N` で挿入。
6. `Ctrl+S` で保存。未保存の変更がある状態で離脱しようとすると警告が出ます。

---

## 出力ファイル

保存（`POST /api/save`）すると、`save_lock` で直列化したうえで次の2ファイルを書き出します（**元SRTは保持**）。

| ファイル | 説明 |
|----------|------|
| `<video_stem>_gold.srt` | 修正後の確定SRT。安定ID付きメタJSON行を含む（`--save-target` で変更可） |
| `<video_stem>_gold_dpo_pairs.jsonl` | DPO追加学習用ペア（変更があったセグメントのみ） |

### gold.srt の形式

通常の2行構造SRT（1行目=タグ、2行目=メタJSON）で、メタJSONに `id` を持ちます。Shotcut等の従来ツールでもタグ行はそのまま表示できます。時刻はSRT本体が正で、`start_sec`/`end_sec` も下流互換のため補完されます。

### dpo_pairs.jsonl の形式

1行1ペアのJSONL。`change` は `edited` / `inserted` / `deleted` の3種:

```json
{"id": "a1b2c3d4", "procedure": "cholecystectomy", "video": "case001.mp4",
 "change": "edited",
 "rejected": {"phase_name": "CalotTriangleDissection", "start_sec": 120.0, "end_sec": 180.0},
 "chosen":   {"phase_name": "ClippingCutting",        "start_sec": 125.5, "end_sec": 180.0}}
```

| change | 意味 | rejected | chosen |
|--------|------|----------|--------|
| `edited` | ID一致かつ ラベル or 時刻が変化 | 自動生成 | 人手修正 |
| `inserted` | 修正後にのみ存在（新規追加） | `null` | 人手追加 |
| `deleted` | 元にのみ存在（人手で削除） | 自動生成 | `null` |

> **新規作成モード**（`--srt` 省略）では元セグメントが空のため、全件が `inserted`（`rejected: null` = 純ゴールドラベル）として出力されます。

ペアは `chosen`（無ければ `rejected`）の `start_sec` 昇順でソートされます。

---

## 術式別フェーズ語彙（label_sets）

フェーズ名は術式ごとに異なるため、`src/annotate/label_sets.py` の**レジストリ方式**で管理します。

```python
LABEL_SETS = {
    "cholecystectomy": list(CHOLEC80_PHASES),  # 既存のCholec80定義を再利用
}
```

新しい術式に対応するには `LABEL_SETS` にエントリを追加するだけで、サーバ/フロントのコード変更は不要です（パレットボタンと `1`–`9` の割り当てが自動生成されます）。MVPは胆嚢摘出術（cholecystectomy）。

---

## API（参考）

| メソッド / パス | 説明 |
|-----------------|------|
| `GET /` | エディタHTML（CSRFトークン埋め込み済み） |
| `GET /app.js`, `GET /style.css` | フロントエンド静的ファイル |
| `GET /api/session` | 術式・フェーズ語彙・セグメント・動画URL・モード（新規/編集） |
| `GET /media/video` | プロキシ動画配信（Rangeリクエスト/206対応でシーク可能） |
| `POST /api/save` | 修正セグメントを保存し、gold.srt と dpo_pairs.jsonl を書き出す |

---

## セキュリティ

ローカル利用を前提とした最小限の対策が入っています。

- **待受**: 既定で `127.0.0.1` のみ（外部公開する場合は `--host` を明示）。
- **CSRF**: 保存APIは `X-CSRF-Token` ヘッダを要求。HTMLに埋め込んだトークンと `secrets.compare_digest` で照合し、不一致は `403`。
- **重複ID拒否**: 保存時に重複 `id` を検出すると `400`（ペアリングのズレを防ぐ）。
- **同時保存の直列化**: 複数タブからの同時保存によるSRT/JSONL破損を防ぐためロックで直列化。
- **ラベル描画**: サーバ由来の任意文字列ラベルは `innerHTML` 連結を避けて描画（XSS対策）。

---

## 関連

- ワークフロー全体での位置づけ: [TUTORIAL.md](../TUTORIAL.md) の「Step 4: 手動修正とデータ更新」
- 一次ラベルを生成する解析器: [docs/analyzers.md](analyzers.md)、[docs/phase_converter.md](phase_converter.md)
- 安定ID付きSRTの土台: `src/tools/srt_to_jsonl.py` / `src/tools/jsonl_to_srt.py`
