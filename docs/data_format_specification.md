# データフォーマット仕様書

Video Image Analyzer プロジェクトにおけるデータの入出力フォーマットとデータフローについて記述する。

## データフロー概要

本システムでは、動画解析を **(1) 時系列記録** と **(2) アノテーション** の2段階に分離している。また、**(3) 可視化** のためにSRTファイルを活用する。

```mermaid
graph TD
    Video[入力動画] -->|Step 1: timeseries| CSV[赤色率ログ CSV]
    
    CSV -->|tools.csv_to_srt| SRT_Metrics[指標SRT<br>(可視化用)]
    SRT_Metrics -->|Shotcut| HumanCheck[人手確認]
    
    CSV -->|Step 2: annotate| JSONL[イベント正本JSONL]
    JSONL -->|自動変換| SRT_Events[イベントSRT<br>(可視化・編集用)]
    
    SRT_Events -->|Shotcut| HumanEdit[人手修正]
    HumanEdit -->|tools.srt_to_jsonl| JSONL
```

---

## 1. JSONL フォーマット（イベント正本）

**システムの中心となるデータフォーマット（Single Source of Truth）。**
出血イベントやカット境界などの「イベントデータ」は全てこの形式で管理される。
SRTやその他のフォーマットは、このJSONLから生成される派生物である。

- **ファイル名**: `{video_name}_events.jsonl`
- **エンコーディング**: UTF-8

### フィールド定義

1行に1つのイベントオブジェクトを格納する。

```json
{
  "type": "bleed_candidate",
  "metric": "red_ratio",
  "thr": 0.03,
  "k_s": 3.0,
  "smooth_s": 5.0,
  "delta_max": 0.051234,
  "start_sec": 135.6,
  "end_sec": 138.8,
  "start_srt": "00:02:15,600",
  "end_srt": "00:02:18,800"
}
```

| フィールド | 型 | 説明 |
|------------|----|------|
| `type` | string | イベント種別 (`bleed_candidate`, `cut` 等) |
| `start_sec` | float | 開始時刻（秒） |
| `end_sec` | float | 終了時刻（秒） |
| `metric` | string | 検出指標（例: `red_ratio`） |
| `thr` | float | 検出に使用した閾値 |
| `delta_max` | float | 区間内の最大変化量 |

---

## 2. SRT フォーマット（可視化・アノテーション用）

動画編集ソフト（Shotcut等）上でイベントを可視化し、**人手による修正を行うためのインターフェース**。
JSONL正本から生成され、人手修正後は再びJSONLに変換して正本に反映する。

### 2a. イベントSRT (`{video_name}_bleed.srt`)

```srt
1
00:02:15,600 --> 00:02:18,800
[bleed] delta_over_threshold
{"type": "bleed_candidate", "metric": "red_ratio", "thr": 0.03, "delta_max": 0.05}
```

- **タグ行**: `[bleed]`, `[cut]` など、イベント種別を視覚的に示す。
- **JSON行**: イベントのメタデータを含むJSON文字列。
- **時刻情報**: SRTのタイムスタンプが正となり、JSON行内の時刻情報は無視される（Shotcutでの編集を反映するため）。

### 2b. 指標SRT (`{video_name}_metrics.srt`)

後述のCSV（時系列データ）を動画上に可視化するための字幕ファイル。
`src.tools.csv_to_srt` で生成する。

```srt
1
00:00:00,000 --> 00:00:00,200
red=0.0123 Δs=0.0000
```

---

## 3. CSV フォーマット（時系列推移の可視化）

動画の全フレームにおける赤色率や変化量などの**「量的推移」を記録・可視化するためのデータ**。
イベントそのものではなく、イベント検出の根拠となる生データを提供する。
主にグラフ化や、指標SRTへの変換に使用される。

- **ファイル名**: `{video_name}_redlog.csv`
- **エンコーディング**: UTF-8 (BOMなし)

### カラム定義

| 列名 | 型 | 説明 |
|------|----|------|
| `t_sec` | float | 動画先頭からの経過秒数 |
| `t_srt` | string | SRT形式のタイムスタンプ |
| `red_ratio` | float | 計算された赤色率（0.0〜1.0） |
| `delta` | float | 前フレームからの赤色率増分 |
| `smooth_delta` | float | deltaの移動平均（平滑化後の値） |
| `reader` | string | フレーム読み込みに使用したバックエンド |

---

## 4. コマンド体系

### Step 1: 時系列記録

動画からCSVを生成する。この段階ではイベント検出は行わない。

```bash
python -m src.red.redlog timeseries \
    --video input.mp4 \
    --outdir output/ \
    --fps 5.0
```

### 可視化データの作成（オプション）

CSVの値を動画上で確認したい場合に実行する。

```bash
python -m src.tools.csv_to_srt \
    --in-csv output/input_redlog.csv \
    --out-srt output/input_metrics.srt
```

### Step 2: 出血アノテーション

CSVと閾値を指定してイベントを抽出する。閾値を調整して何度でも再実行可能。

```bash
python -m src.red.redlog annotate \
    --csv output/input_redlog.csv \
    --outdir output/ \
    --thr 0.03 \
    --k_s 3.0
```

### JSONL ⇔ SRT 相互変換

手動編集を行う場合に使用する。

```bash
# JSONL -> SRT
python -m src.tools.jsonl_to_srt --in-jsonl events.jsonl --out-srt bleed.srt

# SRT -> JSONL（編集反映）
python -m src.tools.srt_to_jsonl --in-srt bleed_edited.srt --out-jsonl events_updated.jsonl
```
