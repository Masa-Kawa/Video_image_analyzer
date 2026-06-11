# ディレクトリ地図と使い方ガイド

このリポジトリの「どこに何があるか」と「人が直接動かすプログラムはどれか」を、
初めての人でも追えるようにまとめた地図です。仕様の詳細ではなく**全体像と入口**に絞っています。

- 解析〜編集の手順 → [../TUTORIAL.md](../TUTORIAL.md)
- 編集アプリの操作 → [annotation_editor_tutorial.md](annotation_editor_tutorial.md)

---

## 3層で覚える

このプロジェクトは「人が動かす入口」「解析エンジン」「内部部品」の3層でできています。
**普段触るのは①だけ**で、②は解析するとき、③は基本触りません。

```
① 人が動かす入口   …… annotate.server / pipeline / tools/*   ← 普段はここだけ
        ↓ 呼び出す
② 解析エンジン     …… red / yolo / cholec_phase / …          ← 解析時に選んで使う
        ↓ 使う
③ 内部部品         …… core / analyzers / label_sets / web    ← 触らない（術式追加時のみ）
```

---

## ディレクトリ地図

```
video_data_analyser/                ← プロジェクトのルート
│
├─ 📖 説明書 ───────────────────────────────────────
│   README.md                       プロジェクト概要・入口一覧
│   TUTORIAL.md                     解析〜編集の全体ワークフロー
│   CHANGELOG.md                    変更履歴
│   docs/                           詳しい資料
│     ├ architecture.md             ★この地図
│     ├ annotation_editor_tutorial.md   編集アプリの操作ガイド
│     ├ annotation_editor.md            編集アプリの仕様・API
│     └ analyzers.md / *_converter.md   各解析器の技術資料
│
├─ ⚙️ 設定・資産 ───────────────────────────────────
│   pyproject.toml / uv.lock        依存・設定（uv で管理）
│   scripts/                        コミット前フック（ruff lint）
│   models/ / yolov8n.pt / third_party/   学習済みモデル・外部リポジトリ
│   out_*/                          ★動画・SRT などの作業データ（gitignore）
│
├─ 🧪 tests/                        テストコード（自動検証・人は普段触らない）
│
└─ 📦 src/                          プログラム本体（↓詳説）
    ├ annotate/   ★編集アプリ一式（GUI）
    ├ tools/       変換・準備ツール
    ├ red/         出血解析（色・差分・流れ）
    ├ bleed_ai/    出血解析（AI・ゼロショット）
    ├ cavity/      腹腔内外判定
    ├ anomaly/     映像異常検知
    ├ cholec_phase/ フェーズ認識AI（Cholec80）
    ├ phase/       フェーズ整形・教師なし分割
    ├ action/      手技/動作・器械認識
    ├ transnet/    シーン境界検出
    ├ yolo/        器械検出
    ├ motion/      動き検出
    ├ mlt/         Shotcut プロジェクト生成
    ├ core/        共通ユーティリティ（内部部品）
    ├ analyzers/   解析器の基底クラス（内部部品）
    ├ pipeline.py          統合パイプライン
    └ surgical_pipeline.py 腹腔鏡向け一括解析
```

> **コマンドの読み方**: `python -m src.annotate.server` は
> 「`src` フォルダ → `annotate` フォルダ → `server.py` を実行」の意味。`.` がフォルダ区切り。

---

## ① 人が直接動かすプログラム（フロントライン）

普段はこれだけで足ります。

### A. GUI（ブラウザで操作）

| コマンド | 何をする |
|---|---|
| `python -m src.annotate.server --video V.mp4 --srt S.srt --procedure cholecystectomy` | ⭐**アノテーション編集アプリ**。動画を見ながらフェーズSRTを手修正。詳細は [annotation_editor_tutorial.md](annotation_editor_tutorial.md) |

### B. パイプライン（一括実行のまとめ役）

| コマンド | 何をする |
|---|---|
| `python -m src.surgical_pipeline` | 腹腔鏡向けに 腹腔内外→出血→フェーズ を一括解析 |
| `python -m src.pipeline` | プロキシ→各解析→MLT生成 の統合パイプライン |

### C. よく使う変換・準備ツール（`src/tools/`）

| コマンド | 何をする |
|---|---|
| `python -m src.tools.proxy_manager` | 動画を軽量化（解析・再生用プロキシ作成） |
| `python -m src.tools.merge_srt` | 複数のSRTを1つに統合 |
| `python -m src.tools.srt_to_jsonl` / `jsonl_to_srt` | SRT⇄JSONL 変換（編集の反映／可視化） |
| `python -m src.tools.csv_to_srt` | CSV時系列 → 字幕 |
| `python -m src.tools.plot_redlog` | CSV → グラフPNG |
| `python -m src.tools.srt_to_mkv` | フェーズSRT → チャプター付きMKV |

---

## ② 解析エンジン（解析するとき選んで動かす）

「この動画を解析したい」ときに走らせるもの。パイプライン経由でも内部的に呼ばれます。
多くは `timeseries`（動画→CSV）と `annotate`（CSV→SRT/JSONL）の2段階で動きます。

| 分野 | 主なコマンド |
|---|---|
| 出血（色・差分・流れ） | `src.red.redlog` / `bleed_detector` / `bleed_spread` / `bleed_trend` / `bleed_contact` / `bleed_flow` |
| 出血（AI） | `src.bleed_ai.detector`、`src.bleed_ai.zeroshot.*`（DINOv2 等の研究系） |
| 腹腔内外 | `src.cavity.cavity_detector` |
| 異常検知 | `src.anomaly.detector` |
| フェーズ | `src.cholec_phase.detector`（学習済みAI）、`src.phase.phase_segmenter`（教師なし） |
| シーン境界 | `src.transnet.transnet_analyzer` / `transnet_to_srt` |
| 器械・手技 | `src.yolo.yolo_analyzer`、`src.action.*_inference` |
| 動き | `src.motion.motion_analyzer` |
| 変換器（先行研究→標準形式） | `src.phase.phase_to_outputs` / `src.action.action_to_outputs` / `src.red.bleed_model_to_outputs` |
| 学習（モデル作成・上級） | `src.cholec_phase.train` / `src.anomaly.train` |

---

## ③ 内部部品（人は普段触らない）

上のプログラムが内部で使う「歯車」。直接は実行しません。

| 種類 | 場所 |
|---|---|
| 共通ユーティリティ | `src/core/time_utils.py`（時刻フォーマット） |
| 解析器の基底クラス | `src/analyzers/base.py` |
| 編集アプリ内部 | `src/annotate/` の `server.py` / `srt_io.py` / `pairing.py` / `web/` |
| モデル定義・データ | 各 `models.py` / `dataset.py` |
| パッケージ印 | 各フォルダの `__init__.py` |

> **例外**: `src/annotate/label_sets.py` は内部部品だが、**新しい術式を追加するときだけ**人が編集する。
> 「術式名 → フェーズ名リスト」の台帳で、1行足すとアプリのパレット・数字キー・術式選択に自動反映される（コマンド実行はしない）。

---

## やりたいこと別 早見表

| やりたいこと | どこを使う |
|---|---|
| 動画を見ながらフェーズSRTを手で直す | `python -m src.annotate.server`（GUI） |
| 新しい術式のフェーズ選択肢を増やす | `src/annotate/label_sets.py` を1行編集 |
| 動画をまとめて解析する | `python -m src.surgical_pipeline` or `src.pipeline` |
| 出血だけ解析したい | `src/red/` の各エンジン |
| 解析を速くしたい（軽量化） | `python -m src.tools.proxy_manager` |
| 複数の解析結果を1つの字幕にまとめる | `python -m src.tools.merge_srt` |
| 編集結果をデータ（JSONL）に反映 | `python -m src.tools.srt_to_jsonl` |
| Shotcut で開けるプロジェクトにする | `python -m src.mlt.mlt_generator` / `src.pipeline` |
| テストを走らせる | `uv run python -m pytest tests/ -v` |

---

## 入口の見分け方（自分で探すとき）

「人が動かせるプログラム」かどうかは、ファイル末尾に次のような記述があるかで分かります。

```python
if __name__ == "__main__":   # ← これがあれば python -m で直接実行できる
    ...
```

逆に、これが無く `import` されるだけのファイルは内部部品です。
全入口を一覧したいときは:

```bash
grep -rl "__main__\|ArgumentParser" src --include='*.py'
```
