# データフォーマット定義 (JSONL / SRT / CSV)

本プロジェクトにおける主要なデータフォーマットの定義と役割について記述する。

## 概要

| フォーマット | 役割 | ファイル形式 |
|---|---|---|
| **JSONL** | **イベント正本（Single Source of Truth）** | `{stem}_events.jsonl` |
| **SRT** | **可視化・編集用インターフェース** | `{stem}_bleed.srt`<br>`{stem}_metrics.srt` |
| **CSV** | **時系列データの記録・分析用** | `{stem}_redlog.csv` |

---

## 1. JSONL (Events)

システムの中心となるデータフォーマット。
出血候補やカット点などの「イベント」情報は全てこのJSONLを正本として管理する。
Shotcut等の外部ツールで編集されたSRTは、最終的にこのJSONLに変換して保存する。

**ファイル名**: `{stem}_events.jsonl`

### フィールド定義

1行に1つのイベントオブジェクト（JSON）を格納する。

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
|---|---|---|
| `type` | string | イベント種別 (`bleed_candidate`, `cut` 等) |
| `start_sec` | float | 開始時刻（秒） |
| `end_sec` | float | 終了時刻（秒） |
| `metric` | string | 検出指標（例: `red_ratio`） |
| `thr` | float | 検出に使用した閾値 |
| `delta_max` | float | 区間内の最大変化量 |
| `start_srt` | string | 開始時刻（HH:MM:SS,mmm） |
| `end_srt` | string | 終了時刻（HH:MM:SS,mmm） |

---

## 2. SRT (SubRip Text)

動画プレイヤーや編集ソフト（Shotcut）でイベントを可視化するためのフォーマット。

### 2a. イベントSRT (`{stem}_bleed.srt`)

イベントの区間を可視化し、ラベルとメタデータを表示する。
Shotcutで字幕ブロックを移動・リサイズして編集し、その結果をJSONLにフィードバックできる。

```srt
1
00:02:15,600 --> 00:02:18,800
[bleed] delta_over_threshold
{"type": "bleed_candidate", "metric": "red_ratio", "thr": 0.03, "delta_max": 0.05}
```

- **1行目**: インデックス
- **2行目**: 時間範囲（編集対象）
- **3行目**: 人間用ラベル（`[bleed] ...`）
- **4行目**: 機械用メタデータ（JSON文字列）

### 2b. 指標SRT (`{stem}_metrics.srt`)

時系列データ（CSV）の値を、フレーム単位に近い細かさで字幕として表示する。
グラフの代わりに数値変化を動画上で直接確認するために使用する。

```srt
1
00:00:00,000 --> 00:00:00,200
red=0.0123 Δs=0.0000
```

---

## 3. CSV (Time Series Log)

動画の全フレームにわたる定量的数値を記録したファイル。
イベント検出の計算根拠となる生データ。

**ファイル名**: `{stem}_redlog.csv`

### カラム定義

| 列名 | 型 | 説明 |
|---|---|---|
| `t_sec` | float | 動画先頭からの経過秒数 |
| `t_srt` | string | SRT形式のタイムスタンプ |
| `red_ratio` | float | 赤色率（0.0〜1.0） |
| `delta` | float | 前フレームからの増加量 |
| `smooth_delta` | float | deltaの移動平均（平滑化後の値） |
| `reader` | string | フレーム読込バックエンド（opencv / pyav） |

**補足**:
- `smooth_delta` が閾値（`thr`）を超え続ける区間が、イベント候補となる。
