# 手術動画解析 SRT共通レイヤ（TransNet + 出血）

手術動画を複数の解析器（HSV赤色解析 / TransNetカット境界）で分析し、
**SRT字幕ファイル**として出力するPythonツール群です。
Shotcutに取り込んで視覚的に確認しつつ、Human-in-the-loop修正が可能です。

## 設計原則

- 同一動画に対して複数解析器の結果を**絶対時間（SRT）**で重ねる
- 出力は**二層構造**：可視化・修正用（SRT）＋正本・集計用（CSV / JSONL）
- 赤色ログは **5fps**（0.2秒ごと）でサンプリング
- VFR動画でも安定するよう **PTSベース**（PyAV）でフレーム時刻を取得

## 必要要件

- Python 3.8+
- OpenCV, NumPy, PyAV

```bash
pip install -r requirements.txt
```

## 出力ファイル一覧

| ファイル名 | 種別 | 説明 |
|---|---|---|
| ファイル名 | 種別 | 説明 |
|---|---|---|
| `{stem}_redlog.csv` | CSV | 赤色率ログ（0.2秒刻み） |
| `{stem}_metrics.srt` | SRT | 赤色率・変化量の可視化字幕 |
| `{stem}_bleed.srt` | SRT | 出血候補イベント（可視化・編集用） |
| `{stem}_events.jsonl` | JSONL | イベント正本 |
| `{stem}_cut.srt` | SRT | TransNetカット境界 |
| `{stem}_merged.srt` | SRT | 統合SRT（Shotcut投入用） |

## 使い方

### 1. 赤色解析（2段階プロセス）

#### Step 1: 時系列記録

動画から赤色率ログCSVを生成します。

```bash
python -m src.red.redlog timeseries \
    --video case001.mp4 \
    --outdir ./out
```

#### Step 2: 可視化データの作成（オプション）

CSVの数値をSRT字幕に変換し、動画上で確認できます。

```bash
python -m src.tools.csv_to_srt \
    --in-csv ./out/case001_redlog.csv \
    --out-srt ./out/case001_metrics.srt
```

#### Step 3: 出血アノテーション

CSVと閾値を指定してイベントを抽出します。何度でも再実行可能です。

```bash
python -m src.red.redlog annotate \
    --csv ./out/case001_redlog.csv \
    --outdir ./out \
    --thr 0.03
```

#### ※ 一括実行（従来互換）

```bash
python -m src.red.redlog analyze \
    --video case001.mp4 \
    --outdir ./out
```

### 2. イベントデータの相互変換（JSONL ⇔ SRT）

手動編集を行う場合に使用します。

#### JSONL → SRT

```bash
python -m src.tools.jsonl_to_srt \
    --in-jsonl events.jsonl \
    --out-srt bleed.srt
```

#### SRT → JSONL（編集反映）

SRTファイル（Shotcutで編集済み）からJSONL正本を更新します。

```bash
python -m src.tools.srt_to_jsonl \
    --in-srt bleed_edited.srt \
    --out-jsonl events_updated.jsonl
```

### 2. TransNet境界 → SRT変換

TransNetV2の実行結果（JSONL）をSRTに変換します。

```bash
python -m src.transnet.transnet_to_srt \
    --in-jsonl case001_transnet_boundaries.jsonl \
    --out-srt ./out/case001_cut.srt \
    --pad-ms 100
```

**入力JSONLの形式**（1行1境界）:

```json
{"t_sec": 615.2, "score": 0.93}
{"t_sec": 720.5, "score": 0.87}
```

### 3. SRTマージ

複数のSRTファイルを時刻順にマージし、統合SRTを生成します。

```bash
python -m src.tools.merge_srt \
    --out ./out/case001_merged.srt \
    ./out/case001_cut.srt \
    ./out/case001_bleed.srt
```

### 4. Shotcutでの確認

1. Shotcutで**元動画**を開く
2. 統合SRT（`case001_merged.srt`）を字幕トラックにインポート
3. タイムラインでイベントを視覚的に確認・修正

## SRT本文フォーマット

すべてのSRTエントリは **2行構造** で統一されています。

**出血候補**:

```
[bleed] delta_over_threshold
{"type": "bleed_candidate", "metric": "red_ratio", "thr": 0.03, ...}
```

**カット境界**:

```
[cut] transnet
{"type": "cut", "model": "TransNetV2", "score": 0.93}
```

1行目は人間向けタグ、2行目はJSON（機械向け）です。

## CSV列仕様（redlog.csv）

| 列名 | 型 | 説明 |
|---|---|---|
| `t_sec` | float | 秒 |
| `t_srt` | string | SRT形式時刻 |
| `red_ratio` | float | 赤色率（0〜1） |
| `delta` | float | 増加量 |
| `smooth_delta` | float | 平滑化増加量 |
| `reader` | string | 使用した読込方式（pyav / opencv） |

## テスト

```bash
python -m pytest tests/ -v
```

## 拡張予定（次フェーズ）

- `port` / `intrabody` / `extrabody` 状態推定
  - v0: OpenCVベース特徴量
  - v1: AI分類器（CNN/ViT）
