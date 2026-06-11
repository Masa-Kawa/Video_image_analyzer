# チュートリアル: 手術動画解析ワークフロー

このチュートリアルでは、手術動画 `case001.mp4` を題材に、解析からShotcutでの確認・修正までの一連の流れを説明します。

## 全体の流れ

```
動画ファイル
  │
  ├─ [Step 1] プロキシ作成（軽量化）
  │
  ├─ [Step 2] 解析エンジンで自動検出
  │    ├── 赤色解析（出血候補）
  │    ├── TransNet（シーン境界）
  │    ├── YOLO（器械認識）
  │    └── モーション（動き検出）
  │
  ├─ [Step 3] SRTマージ → Shotcutで確認
  │
  ├─ [Step 4] 手動修正 → JSONL更新
  │
  └─ [Step 5] MLTプロジェクト生成（オプション）
```

---

## Step 1: プロキシ動画の作成

高解像度の動画をそのまま処理すると時間がかかります。まず軽量なプロキシを作成します。

```bash
python -m src.tools.proxy_manager case001.mp4 \
    --resolution 720p \
    --proxy-dir proxy/
```

`proxy/case001_720p.mp4` が作成されます。以降の解析にはこのファイルを使うと高速です。

### 複数動画をまとめて作成

`proxy_manager` は**複数の動画を一度に**受け取れます。指定方法は3通りです。

```bash
# 1. ファイルを並べて指定
python -m src.tools.proxy_manager case001.mp4 case002.mp4 case003.mp4 \
    --resolution 720p --proxy-dir proxy/

# 2. シェルのワイルドカードでまとめて
python -m src.tools.proxy_manager videos/*.mp4 \
    --resolution 720p --proxy-dir proxy/

# 3. シリーズモード（大量・連続ケース向け・推奨）
python -m src.tools.proxy_manager videos/*.mp4 --series \
    --resolution 720p --proxy-dir proxy/
```

**通常 と `--series` の違い:**

| | 通常（`--series` なし） | `--series` あり |
|---|---|---|
| 壊れた動画があったら | **そこで停止**（中断） | **スキップして続行** |
| 終了時の表示 | — | `✓ N 成功 / ⚠ M 失敗` の集計 |
| 失敗時の終了コード | — | 失敗が1件でもあれば `1` |
| 向いている場面 | 数本・全部正常が前提 | 大量／連続ケース（1本壊れても止めたくない） |

> どちらのモードでも、**既存のプロキシは自動でスキップ**されます（作り直したいときだけ `--force`）。
> 途中で中断しても、同じコマンドを再実行すれば続きから作れます。

### オプション

| フラグ | 説明 | デフォルト |
|--------|------|-----------|
| `videos`（位置引数） | 対象動画。**複数指定可**（`a.mp4 b.mp4` やワイルドカード） | （必須） |
| `--resolution` | `360p` / `480p` / `720p` / `1080p` | `720p` |
| `--proxy-dir` | 出力先ディレクトリ | 元動画と同じ場所 |
| `--series` | 複数動画を一括処理（失敗をスキップして集計） | `False` |
| `--force` | 既存プロキシを上書き | `False` |
| `--gpu` | NVIDIA NVENC（`h264_nvenc`）で**GPUエンコード**。CPUより数倍高速 | `False` |
| `--fps` | プロキシのフレームレート。解析は最大5fpsなので30で十分。長尺は15でさらに半減。`0`で元のfps維持 | `30` |
| `--merge` | 作成した全プロキシを**1本に無劣化連結**（中間ファイルは自動削除） | `False` |
| `--merge-name` | 連結ファイル名（ファイル名部分のみ採用） | `<先頭動画名>_merged_<解像度>.MP4` |
| `--keep-parts` | `--merge`時、分割プロキシも残す | `False` |

### GPU・一本化・長尺動画

**GPUで高速作成 ＋ 分割動画を1本に連結**（レコーダーが分割記録した1手術を、まとめて軽量プロキシ化）:

```bash
python -m src.tools.proxy_manager videos/*.MP4 \
    --resolution 720p --proxy-dir proxy/ --gpu --merge
```

> ⚠️ ワイルドカードは**拡張子の大文字・小文字を一致**させること（Linuxは区別する）。
> 例: ファイルが `.MP4` なら `*.MP4`。`*.mp4` だと展開されず分かりやすいエラーで停止します。

**5〜6時間の長尺動画**はファイルが巨大化します。`--fps 15`（必要なら `--resolution 480p` も）で圧縮します（解析精度は不変）:

```bash
# 6時間動画の目安: 720p/30fps ≒ 7.8GB → 15fps ≒ 4GB → +480p ≒ 2.2GB
python -m src.tools.proxy_manager videos/*.MP4 \
    --resolution 480p --proxy-dir proxy/ --gpu --merge --fps 15
```

---

## Step 2: 解析を実行する

### 2-1. 赤色解析（出血候補の検出）

赤色解析は **2段階** に分かれています。

#### Step 2-1a: 時系列記録

動画全体をスキャンし、フレームごとの赤色率をCSVに記録します。

```bash
python -m src.red.redlog timeseries \
    --video case001.mp4 \
    --outdir out/
```

出力: `out/case001_redlog.csv`

この段階ではイベント判定はしません。CSVさえあれば後の工程は何度でもやり直せます。

#### Step 2-1b: 可視化（オプション）

解析結果を確認するための補助ツールです。

```bash
# CSV → SRT字幕（動画上で数値をリアルタイム表示）
python -m src.tools.csv_to_srt \
    --in-csv out/case001_redlog.csv \
    --out-srt out/case001_metrics.srt

# CSV → PNGグラフ（全体の傾向を一目で把握）
python -m src.tools.plot_redlog \
    --in-csv out/case001_redlog.csv \
    --out-png out/case001_plot.png \
    --thr 0.03
```

#### Step 2-1c: イベント抽出

閾値を指定して出血候補を抽出します。CSVがあるので一瞬で完了します。

```bash
python -m src.red.redlog annotate \
    --csv out/case001_redlog.csv \
    --outdir out/ \
    --thr 0.03
```

出力:
- `out/case001_bleed.srt` — Shotcutで確認するための字幕
- `out/case001_events.jsonl` — イベントデータの正本

閾値は何度でも変えて再実行できます。検出漏れが多ければ下げ、誤検知が多ければ上げてください。

```bash
# 閾値を下げて再実行
python -m src.red.redlog annotate \
    --csv out/case001_redlog.csv \
    --outdir out/ \
    --thr 0.02
```

#### 一括実行（上記をまとめて実行）

```bash
python -m src.red.redlog analyze \
    --video case001.mp4 \
    --outdir out/
```

### 2-2. 改良版出血検出（bleed_detector / bleed_spread）

より精度の高い出血検出アルゴリズムも利用できます。

```bash
# 赤色拡大検出（フレーム間の新規赤色ピクセルを追跡）
python -m src.red.bleed_detector timeseries --video case001.mp4 --outdir out/
python -m src.red.bleed_detector annotate --csv out/case001_bleedlog.csv --outdir out/ --thr 0.005

# 局所拡散検出（8x8グリッドで出血の広がりパターンを検出）
python -m src.red.bleed_spread timeseries --video case001.mp4 --outdir out/
python -m src.red.bleed_spread annotate --csv out/case001_spreadlog.csv --outdir out/ --thr 0.001
```

3つのアルゴリズムの違い:

| アルゴリズム | 得意なケース | 閾値の目安 |
|-------------|-------------|-----------|
| **redlog** | 全体的な赤色率の急変 | `0.02`〜`0.05` |
| **bleed_detector** | 背景が安定した状態での赤色領域の拡大 | `0.003`〜`0.01` |
| **bleed_spread** | 局所的な出血の広がり（カメラ移動との区別） | `0.0005`〜`0.005` |

### 2-3. TransNetシーン境界の変換

TransNetV2の実行結果（JSONL）をSRTに変換します。

```bash
python -m src.transnet.transnet_to_srt \
    --in-jsonl out/case001_transnet.jsonl \
    --out-srt out/case001_cut.srt \
    --pad-ms 100
```

入力JSONLの形式（1行1境界）:
```json
{"t_sec": 615.2, "score": 0.93}
{"t_sec": 720.5, "score": 0.87}
```

### 2-4. YOLO器械認識

YOLO8で手術器械を検出し、器械の組み合わせ変化でシーンを分割します。

```bash
# Step 1: フレームごとの器械検出
python -m src.yolo.yolo_analyzer timeseries \
    --video case001.mp4 \
    --outdir out/ \
    --fps 2.0

# Step 2: シーン分割
python -m src.yolo.yolo_analyzer annotate \
    --jsonl out/case001_yolo_timeseries.jsonl \
    --outdir out/

# 一括実行
python -m src.yolo.yolo_analyzer analyze \
    --video case001.mp4 \
    --outdir out/
```

---

## Step 3: SRTをマージしてShotcutで確認

複数の解析結果を1つのSRTにまとめます。

```bash
python -m src.tools.merge_srt \
    --out out/case001_merged.srt \
    out/case001_bleed.srt \
    out/case001_cut.srt \
    out/case001_scenes.srt
```

### Shotcutでの確認方法

1. Shotcutで**元動画**（または`proxy/`のプロキシ）を開く
2. 統合SRT（`case001_merged.srt`）を字幕トラックにインポート
3. タイムラインでイベントを視覚的に確認

各イベントは `[bleed]`、`[cut]`、`[scene]` などのタグ付きで表示されるため、種類が一目でわかります。

---

## Step 4: 手動修正とデータ更新（Human-in-the-loop）

自動検出には限界があります。最後は人の目で確認・修正します。
修正方法は2通り — **(A) 専用の編集Webアプリ**（フェーズSRTなど区間ラベルの修正に最適）、**(B) Shotcut** での字幕編集です。

### 方法A: アノテーション編集Webアプリ（推奨）

動画と同期したブラウザUIで、セグメントの編集・削除・挿入ができます。Shotcut不要で、修正と同時にDPO追加学習用ペアも生成されます。

```bash
# 自動生成したフェーズSRTを編集モードで開く（ブラウザが自動起動）
python -m src.annotate.server \
    --video case001.mp4 \
    --srt out/case001_phase.srt \
    --procedure cholecystectomy --port 8000
```

保存すると次が出力されます（元SRTは保持）:

- `case001_gold.srt` — 修正後の確定SRT（安定ID付き）
- `case001_gold_dpo_pairs.jsonl` — `rejected`(自動生成)/`chosen`(人手修正) の学習ペア

> `--srt` を省略すると空の状態から手付けする**ゼロ作成（ゴールドラベル）モード**になります。
>
> 👉 **操作方法を一から知りたい場合は、実践ガイド [docs/annotation_editor_tutorial.md](docs/annotation_editor_tutorial.md) を参照**
> （キーボード操作・区間ジャンプ・再編集・修正履歴の使い方を順に説明）。仕様の詳細は [docs/annotation_editor.md](docs/annotation_editor.md)。

### 方法B: Shotcutで修正

1. Shotcutで `out/case001_bleed.srt` を字幕トラックとして開く
2. **誤検知**（出血ではない箇所）→ 字幕ブロックを削除
3. **区間ずれ** → 字幕ブロックの端をドラッグして調整
4. 修正が終わったらSRTを保存（例: `case001_bleed_edited.srt`）

### 修正結果をJSONL正本に反映

```bash
python -m src.tools.srt_to_jsonl \
    --in-srt out/case001_bleed_edited.srt \
    --out-jsonl out/case001_events_final.jsonl
```

SRTの時刻がJSONL内の時間フィールドを上書きするため、Shotcutでの編集がそのままデータに反映されます。

### JSONL → SRT に戻す（再確認用）

```bash
python -m src.tools.jsonl_to_srt \
    --in-jsonl out/case001_events_final.jsonl \
    --out-srt out/case001_bleed_final.srt
```

---

## Step 5: MLTプロジェクト生成（オプション）

解析結果をShotcutプロジェクトファイル（MLT）として直接出力することもできます。
字幕トラックの手動インポートが不要になり、複数動画のシリーズ分析にも対応しています。

```python
from src.mlt.mlt_generator import MLTGenerator

# MLTプロジェクト生成
gen = MLTGenerator("case001.mp4")
gen.add_cuts(scenes)         # TransNetのシーン区間
gen.add_annotation_track(annotations, track_name="出血イベント")
gen.generate("out/case001.mlt")

# → Shotcutで case001.mlt を開くだけでOK
```

---

## ツールリファレンス

### 変換ツール

| コマンド | 入力 | 出力 | 用途 |
|---------|------|------|------|
| `src.tools.jsonl_to_srt` | JSONL | SRT | イベントデータの可視化 |
| `src.tools.srt_to_jsonl` | SRT | JSONL | 手動編集結果のデータ反映 |
| `src.tools.csv_to_srt` | CSV | SRT | 時系列数値の字幕化 |
| `src.tools.merge_srt` | SRT（複数） | SRT | 複数解析結果の統合 |
| `src.tools.plot_redlog` | CSV | PNG | 時系列グラフの描画 |
| `src.tools.proxy_manager` | 動画 | 動画 | プロキシ（軽量版）作成 |

### データの流れ

```
動画
 ├──[解析]──→ CSV（時系列ログ）──[annotate]──→ JSONL（正本）
 │                                              ↓ ↑
 │                                    jsonl_to_srt / srt_to_jsonl
 │                                              ↓ ↑
 │                                          SRT（可視化・編集）
 │                                              ↓
 └──────────────────────────→ Shotcut で確認・修正
```

### SRT時刻フォーマット

本プロジェクト全体で統一されたフォーマット:

- **SRT**: `HH:MM:SS,mmm`（カンマ区切り） 例: `01:23:45,678`
- **MLT**: `HH:MM:SS.mmm`（ピリオド区切り） 例: `01:23:45.678`

---

## テスト

```bash
uv run python -m pytest tests/ -v
```

102件のテストがすべて通過すれば正常です。
