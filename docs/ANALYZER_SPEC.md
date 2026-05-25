# Video Analyzer 評価器開発仕様書

本ドキュメントは、video_data_analyser プロジェクトに新しい評価器を追加する AI（または開発者）に向けた仕様書である。このプロジェクトでは、すべての評価器が統一されたパターンに従うことで、出力の比較・統合・可視化を容易にしている。

---

## 1. 基本アーキテクチャ: 2ステップパイプライン

すべての評価器は **必ず** 以下の2ステップに分離して実装する。

```
Step 1: record_timeseries()  動画 → CSV（フレームごとの数値指標）
Step 2: annotate_*()         CSV → JSONL（イベント正本） + SRT（字幕）
```

### なぜ分離するか

- **再現性**: CSVが残るため、閾値を変えて再アノテーションできる（動画の再処理が不要）
- **検証性**: CSVを目視やグラフで確認してから閾値を決められる
- **統合性**: 複数の評価器のCSVをタイムスタンプで結合して相関分析できる

---

## 2. Step 1: `record_timeseries()` の仕様

### 関数シグネチャ

```python
def record_timeseries(
    video_path: str,
    outdir: str,
    fps: float = 5.0,       # サンプリングFPS（評価器ごとに適切な値）
    roi_margin: float = 0.08,
    no_roi: bool = False,
    smooth_s: float = 5.0,   # 平滑化窓（秒）
    # ... 評価器固有のパラメータ
) -> dict:
    """
    Returns:
        {"csv": str}  # CSVファイルの絶対パス
        空の dict を返した場合は処理失敗
    """
```

### CSV出力フォーマット

**ファイル名**: `{video_stem}_{analyzer}log.csv`

| 必須カラム | 型 | 説明 |
|---|---|---|
| `t_sec` | float (小数点3桁) | タイムスタンプ（秒） |
| `t_srt` | string | SRT時刻形式 `HH:MM:SS,mmm` |
| `reader` | string | 使用したフレーム読み込み方式（`"pyav"` or `"opencv"`） |

これら3カラムの後に、評価器固有の指標カラムを追加する。

**CSV記述ルール**:
- エンコーディング: UTF-8
- 改行: `\n`
- 1行目はヘッダ行
- 数値は文字列として記録: `t_sec` は `f"{value:.3f}"`, 指標は `f"{value:.6f}"`

**実装例**:
```csv
t_sec,t_srt,red_ratio,newly_red_ratio,bg_stability,red_expansion,smooth_expansion,reader
0.000,00:00:00,000,0.045678,0.000000,1.000000,0.000000,0.000000,pyav
0.200,00:00:00,200,0.048901,0.003200,0.985432,0.003152,0.001576,pyav
```

### 平滑化カラムの慣例

生の指標に対して平滑化した値を持つ場合、カラム名は `smooth_` プレフィックスを付ける。

```
red_expansion  → smooth_expansion
cavity_score   → smooth_cavity
similarity     → smooth_similarity
```

### フレーム読み込み

共通ユーティリティ `iter_frames()` を使用する。

```python
from src.red.redlog import iter_frames

for t_sec, bgr_frame, reader_name in iter_frames(video_path, fps):
    # bgr_frame: np.ndarray (H, W, 3), BGR形式
    # t_sec: float, タイムスタンプ（秒）
    # reader_name: "pyav" or "opencv"
    pass
```

### ROIマスク

腹腔鏡映像は円形の視野を持つため、円形ROIマスクを適用する。

```python
from src.red.redlog import make_circular_roi

# 最初のフレームでROIを初期化
roi_mask = make_circular_roi(height, width, margin=0.08)
# roi_mask: np.ndarray (H, W), bool型。True = ROI内
```

### 平滑化

```python
from src.red.redlog import smooth_center

window_size = max(1, int(round(smooth_s * fps)))
smooth_values = smooth_center(raw_values, window_size)
# 中心移動平均。入力と同じ長さのリストを返す。
```

### 時刻フォーマット

```python
from src.core.time_utils import format_srt_time, parse_srt_time

format_srt_time(615.2)           # → "00:10:15,200"
parse_srt_time("00:10:15,200")   # → 615.2

# 重要: SRT時刻のミリ秒区切りはカンマ（,）であること。ピリオド（.）ではない。
```

---

## 3. Step 2: `annotate_*()` の仕様

### 関数シグネチャ

```python
def annotate_something(
    csv_path: str,
    outdir: str,
    thr: float = 0.5,          # イベント検出閾値
    min_duration_s: float = 1.0,  # 最小イベント持続時間（秒）
    smooth_s: float = 5.0,      # 記録用パラメータ
    # ... 評価器固有のパラメータ
) -> dict:
    """
    Returns:
        {"jsonl": str, "srt": str, "events": int}
    """
```

### JSONL出力フォーマット（イベント正本）

**ファイル名**: `{video_stem}_{event_type}_events.jsonl`

1行1イベント。各行は独立したJSONオブジェクト。空行なし。

**必須フィールド**:

| フィールド | 型 | 説明 |
|---|---|---|
| `type` | string | イベントタイプの識別子（後述のルールに従う） |
| `start_sec` | float | イベント開始時刻（秒） |
| `end_sec` | float | イベント終了時刻（秒） |
| `start_srt` | string | 開始時刻のSRT形式 |
| `end_srt` | string | 終了時刻のSRT形式 |

**推奨フィールド（検出条件の記録用）**:

| フィールド | 型 | 説明 |
|---|---|---|
| `thr` | float | 使用した閾値 |
| `metric` | string | 使用した指標名 |
| `label` | string | 人間向けラベル |

**任意フィールド**: 評価器固有のメタデータを自由に追加してよい。

**JSONL記述ルール**:
- エンコーディング: UTF-8
- `ensure_ascii=False` で日本語もそのまま記録
- イベントは `start_sec` 昇順で出力する

**実装例**:
```jsonl
{"type":"bleed_candidate","metric":"red_expansion","thr":0.005,"k_s":1.0,"smooth_s":5.0,"delta_max":0.012345,"start_sec":123.456,"end_sec":125.789,"start_srt":"00:02:03,456","end_srt":"00:02:05,789"}
{"type":"bleed_candidate","metric":"red_expansion","thr":0.005,"k_s":1.0,"smooth_s":5.0,"delta_max":0.008901,"start_sec":200.0,"end_sec":203.5,"start_srt":"00:03:20,000","end_srt":"00:03:23,500"}
```

### SRT出力フォーマット（Shotcut字幕）

**ファイル名**: `{video_stem}_{event_type}.srt`

SRTファイルは動画編集ソフト（Shotcut）で表示・目視確認するためのもの。

**SRTエントリ構造（1イベント = 4行）**:

```
<連番>\n
<開始時刻> --> <終了時刻>\n
<タグ行>\n
\n
```

**タグ行のフォーマット**: `[カテゴリ] 説明`

```
[bleed] delta_over_threshold
[cavity] INSIDE
[cavity] OUTSIDE
[phase] Dissection
[cut] transnet
```

**重要**: SRTにはタグ行のみを記載する。JSONメタデータ行は含めない。メタデータはJSONLファイルで確認する。

**SRT記述ルール**:
- エンコーディング: UTF-8
- 時刻形式: `HH:MM:SS,mmm --> HH:MM:SS,mmm`（カンマ区切り）
- 連番は1から開始
- エントリ間は空行1行で区切る
- 最後のエントリの後にも空行を入れる

**実装例**:
```srt
1
00:02:03,456 --> 00:02:05,789
[bleed] delta_over_threshold

2
00:03:20,000 --> 00:03:23,500
[bleed] delta_over_threshold

```

---

## 4. イベントタイプ (`type`) の命名規則

`type` フィールドは評価器間でイベントを区別するための識別子。以下のルールに従う:

1. **スネークケース**: `bleed_candidate`, `cavity_inside`, `surgical_phase`
2. **一意性**: 他の評価器の `type` と重複しないこと
3. **タグ行との対応**: `type` に対応するSRTタグ行テンプレートを定義すること

### 既存の type → タグ行マッピング

| `type` | SRTタグ行 |
|---|---|
| `bleed_candidate` | `[bleed] delta_over_threshold` |
| `cavity_inside` | `[cavity] INSIDE` |
| `cavity_outside` | `[cavity] OUTSIDE` |
| `surgical_phase` | `[phase] {phase_name}` |
| `cut` | `[cut] transnet` |

### 新しい type を追加する場合

1. `src/tools/jsonl_to_srt.py` の `TAG_TEMPLATES` 辞書に追加:
```python
TAG_TEMPLATES = {
    "bleed_candidate": "[bleed] delta_over_threshold",
    "cut": "[cut] transnet",
    "your_new_type": "[category] description",
}
```

2. `src/tools/srt_to_jsonl.py` の `TAG_PATTERNS` 辞書に逆変換パターンを追加:
```python
TAG_PATTERNS = {
    re.compile(r"^\[bleed\]"): "bleed_candidate",
    re.compile(r"^\[cut\]"): "cut",
    re.compile(r"^\[category\]"): "your_new_type",
}
```

---

## 5. `analyze_video()` ラッパーの仕様

2ステップを連続実行する便利関数。全評価器に必須。

```python
def analyze_video(
    video_path: str,
    outdir: str,
    fps: float = 5.0,
    # Step 1 のパラメータすべて
    # Step 2 のパラメータすべて
) -> dict:
    """
    Returns:
        {"csv": str, "jsonl": str, "srt": str, "events": int}
        失敗時は空の dict
    """
    result1 = record_timeseries(video_path, outdir, fps, ...)
    if not result1:
        return {}
    result2 = annotate_something(result1["csv"], outdir, ...)
    return {**result1, **result2}
```

---

## 6. CLIエントリポイントの仕様

各評価器はサブコマンド方式のCLIを提供する。

```python
def main() -> int:
    parser = argparse.ArgumentParser(description="...")
    subparsers = parser.add_subparsers(dest="command")

    # timeseries: Step 1
    ts_parser = subparsers.add_parser("timeseries")
    ts_parser.add_argument("--video", required=True)
    ts_parser.add_argument("--outdir", required=True)
    ts_parser.add_argument("--fps", type=float, default=5.0)
    # ... 評価器固有のパラメータ

    # annotate: Step 2
    ann_parser = subparsers.add_parser("annotate")
    ann_parser.add_argument("--csv", required=True)
    ann_parser.add_argument("--outdir", required=True)
    ann_parser.add_argument("--thr", type=float, default=...)
    # ... 評価器固有のパラメータ

    # analyze: Step 1 + 2
    ana_parser = subparsers.add_parser("analyze")
    ana_parser.add_argument("--video", required=True)
    ana_parser.add_argument("--outdir", required=True)
    # ... すべてのパラメータ

    args = parser.parse_args()
    # ... dispatch
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

**実行例**:
```bash
# Step 1 のみ
python -m src.my_analyzer.detector timeseries --video input.mp4 --outdir output/

# Step 2 のみ（閾値を変えて再実行）
python -m src.my_analyzer.detector annotate --csv output/input_mylog.csv --outdir output/ --thr 0.01

# 一括実行
python -m src.my_analyzer.detector analyze --video input.mp4 --outdir output/
```

---

## 7. ディレクトリ構成

```
src/
├── core/
│   └── time_utils.py           # format_srt_time, parse_srt_time（共通）
├── red/
│   └── redlog.py               # iter_frames, make_circular_roi, smooth_center（共通）
├── tools/
│   ├── jsonl_to_srt.py         # JSONL → SRT 変換ツール
│   ├── srt_to_jsonl.py         # SRT → JSONL 変換ツール
│   └── merge_srt.py            # 複数SRTのマージ
├── your_analyzer/              # 新しい評価器のディレクトリ
│   ├── __init__.py
│   └── detector.py             # メインモジュール
└── surgical_pipeline.py        # 統合パイプライン
```

### 出力ファイル命名規則

```
{video_stem}_{analyzer}log.csv           # Step 1: 時系列CSV
{video_stem}_{event_type}_events.jsonl   # Step 2: イベントJSONL（正本）
{video_stem}_{event_type}.srt            # Step 2: イベントSRT（字幕）
```

例（動画名: `surgery.mp4`, 評価器: `smoke_detector`）:
```
surgery_smokelog.csv
surgery_smoke_events.jsonl
surgery_smoke.srt
```

---

## 8. 共通ユーティリティ一覧

### `src.core.time_utils`

| 関数 | シグネチャ | 用途 |
|---|---|---|
| `format_srt_time` | `(seconds: float) -> str` | 秒 → `"HH:MM:SS,mmm"` |
| `parse_srt_time` | `(time_str: str) -> float` | `"HH:MM:SS,mmm"` → 秒 |
| `format_mlt_time` | `(seconds: float) -> str` | 秒 → `"HH:MM:SS.mmm"`（MLT用） |
| `seconds_to_frames` | `(seconds: float, fps: float) -> int` | 秒 → フレーム番号 |
| `frames_to_seconds` | `(frame: int, fps: float) -> float` | フレーム番号 → 秒 |

### `src.red.redlog`

| 関数 | シグネチャ | 用途 |
|---|---|---|
| `iter_frames` | `(video_path: str, fps: float) -> Iterator[(float, np.ndarray, str)]` | フレームイテレータ（PyAV優先, OpenCVフォールバック） |
| `make_circular_roi` | `(height: int, width: int, margin: float) -> np.ndarray` | 円形ROIマスク生成（bool配列） |
| `smooth_center` | `(values: List[float], window: int) -> List[float]` | 中心移動平均（入力と同じ長さ） |
| `compute_red_ratio` | `(frame_bgr: np.ndarray, roi_mask, s_min, v_min) -> float` | HSVベースの赤色率（0〜1） |
| `extract_bleed_events` | `(times, smooth_vals, thr, k_s, fps, smooth_s) -> List[dict]` | 閾値超過の連続区間をイベントとして抽出 |

### `src.tools.jsonl_to_srt`

| 関数 | シグネチャ | 用途 |
|---|---|---|
| `convert` | `(in_jsonl: str, out_srt: str, event_type: Optional[str]) -> int` | JSONL → SRT変換 |

### `src.tools.srt_to_jsonl`

| 関数 | シグネチャ | 用途 |
|---|---|---|
| `convert` | `(in_srt: str, out_jsonl: str) -> int` | SRT → JSONL変換（人手修正の反映） |

---

## 9. 閾値ベースイベント抽出のパターン

多くの評価器では「平滑化した指標が閾値を一定時間超えたらイベント」というパターンを使う。共通関数 `extract_bleed_events()` がこのロジックを提供する。

```python
from src.red.redlog import extract_bleed_events

events = extract_bleed_events(
    times=times,            # List[float] タイムスタンプ
    smooth_deltas=values,   # List[float] 平滑化済み指標
    thr=0.005,              # 閾値
    k_s=1.0,                # 最小連続時間（秒）
    fps=5.0,                # サンプリングFPS
    smooth_s=5.0,           # 記録用（JONLに含める）
)
# → [{"type": "bleed_candidate", "metric": "smooth_delta",
#      "thr": 0.005, "k_s": 1.0, "smooth_s": 5.0,
#      "delta_max": 0.012, "start": 10.0, "end": 12.5}, ...]
```

この関数が返すイベントの `type` は `"bleed_candidate"` 固定なので、必要に応じて上書きする。独自のイベント抽出ロジックを実装してもよい。

---

## 10. 統合パイプラインへの組み込み

`src/surgical_pipeline.py` に新しい評価器を追加する手順:

1. `run_surgical_analysis()` 関数にスキップフラグを追加:
```python
def run_surgical_analysis(
    video_path: str,
    outdir: str,
    # ... 既存パラメータ
    skip_my_analyzer: bool = False,
) -> dict:
```

2. 評価器の呼び出しを追加:
```python
if not skip_my_analyzer:
    print("=== My Analyzer ===")
    my_result = my_analyzer.analyze_video(video_path, outdir, ...)
    all_results["my_analyzer"] = my_result
    srt_files.append(my_result.get("srt"))
```

3. `_build_combined_csv()` に新しいカラムを追加（必要な場合）

---

## 11. テストの書き方

各評価器に対して `tests/test_{analyzer}.py` を作成する。

### 必須テスト項目

1. **個別指標の計算テスト**: 合成画像（`np.zeros` / `np.ones`）で各指標関数の出力を検証
2. **CSV往復テスト**: `record_timeseries` で書いたCSVを `read_*_csv` で正しく読めること
3. **アノテーションテスト**: 合成CSVデータからイベントが正しく抽出されること
4. **SRT形式テスト**: 出力SRTが正しいフォーマットであること

### テスト実行

```bash
cd video_data_analyser && python -m pytest tests/ -v
```

---

## 12. チェックリスト

新しい評価器を実装する際の確認項目:

### ファイル構成
- [ ] `src/{analyzer}/__init__.py` を作成した
- [ ] `src/{analyzer}/detector.py` にメインロジックを実装した
- [ ] `tests/test_{analyzer}.py` にテストを書いた

### Step 1 (`record_timeseries`)
- [ ] `iter_frames()` でフレームを読み込んでいる
- [ ] CSVに `t_sec`, `t_srt`, `reader` の3必須カラムがある
- [ ] `t_sec` は `:.3f`, 指標は `:.6f` でフォーマットしている
- [ ] 平滑化カラムは `smooth_` プレフィックスを付けている
- [ ] `{"csv": str}` を返している
- [ ] フレームが0件の場合、空の `dict` を返している

### Step 2 (`annotate_*`)
- [ ] JSONLの各行に `type`, `start_sec`, `end_sec`, `start_srt`, `end_srt` がある
- [ ] `start_srt` / `end_srt` は `format_srt_time()` で生成している
- [ ] イベントは `start_sec` 昇順で出力している
- [ ] SRTのタグ行は `[カテゴリ] 説明` 形式である
- [ ] SRTにJSONメタデータ行を含めていない（タグ行のみ）
- [ ] `{"jsonl": str, "srt": str, "events": int}` を返している

### タイプ登録
- [ ] `src/tools/jsonl_to_srt.py` の `TAG_TEMPLATES` に追加した
- [ ] `src/tools/srt_to_jsonl.py` の `TAG_PATTERNS` に追加した

### CLI
- [ ] `timeseries`, `annotate`, `analyze` の3サブコマンドを定義した
- [ ] `python -m src.{analyzer}.detector analyze --video X --outdir Y` で実行できる

### テスト
- [ ] 全テストが `python -m pytest tests/ -v` で通る

---

## 13. 実装テンプレート

以下は新しい評価器の最小限の実装テンプレートである。

```python
"""
{評価器名}モジュール

{この評価器が何を検出するかの説明}

2段階の処理:
  Step 1 - record_timeseries(): 動画→CSV
  Step 2 - annotate_{name}():   CSV→JSONL/SRT
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from src.core.time_utils import format_srt_time
from src.red.redlog import iter_frames, make_circular_roi, smooth_center


# ---------------------------------------------------------------------------
# 指標計算
# ---------------------------------------------------------------------------

def compute_my_metric(
    frame_bgr: np.ndarray,
    roi_mask: Optional[np.ndarray] = None,
) -> dict:
    """フレームから指標を算出する。"""
    # TODO: 評価器固有のロジック
    return {
        "metric_a": 0.0,
        "metric_b": 0.0,
    }


# ---------------------------------------------------------------------------
# Step 1: 時系列記録
# ---------------------------------------------------------------------------

def record_timeseries(
    video_path: str,
    outdir: str,
    fps: float = 5.0,
    roi_margin: float = 0.08,
    no_roi: bool = False,
    smooth_s: float = 5.0,
) -> dict:
    """動画をサンプリングしてCSVを出力する。"""
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    stem = Path(video_path).stem

    times: List[float] = []
    metric_a_vals: List[float] = []
    metric_b_vals: List[float] = []
    reader_name = "opencv"
    roi_mask: Optional[np.ndarray] = None
    roi_initialized = False

    for t_sec, bgr, reader in iter_frames(video_path, fps):
        reader_name = reader
        if not roi_initialized:
            h, w = bgr.shape[:2]
            if not no_roi:
                roi_mask = make_circular_roi(h, w, margin=roi_margin)
            roi_initialized = True

        metrics = compute_my_metric(bgr, roi_mask)
        times.append(t_sec)
        metric_a_vals.append(metrics["metric_a"])
        metric_b_vals.append(metrics["metric_b"])

    if not times:
        print("警告: フレームが取得できませんでした。", file=sys.stderr)
        return {}

    # 平滑化
    window = max(1, int(round(smooth_s * fps)))
    smooth_a = smooth_center(metric_a_vals, window)

    # CSV出力
    csv_path = out_path / f"{stem}_mylog.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "t_sec", "t_srt", "metric_a", "metric_b",
            "smooth_a", "reader",
        ])
        for i in range(len(times)):
            writer.writerow([
                f"{times[i]:.3f}",
                format_srt_time(times[i]),
                f"{metric_a_vals[i]:.6f}",
                f"{metric_b_vals[i]:.6f}",
                f"{smooth_a[i]:.6f}",
                reader_name,
            ])

    print(f"CSV  : {csv_path}")
    return {"csv": str(csv_path)}


# ---------------------------------------------------------------------------
# CSV読み込み
# ---------------------------------------------------------------------------

def read_mylog_csv(csv_path: str) -> dict:
    """CSV読み込み。"""
    times, metric_a, metric_b, smooth_a = [], [], [], []
    reader = "unknown"
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            times.append(float(row["t_sec"]))
            metric_a.append(float(row["metric_a"]))
            metric_b.append(float(row["metric_b"]))
            smooth_a.append(float(row["smooth_a"]))
            reader = row.get("reader", "unknown")
    fps = 1.0 / (times[1] - times[0]) if len(times) >= 2 else 5.0
    return {
        "times": times, "metric_a": metric_a,
        "metric_b": metric_b, "smooth_a": smooth_a,
        "reader": reader, "fps": fps,
    }


# ---------------------------------------------------------------------------
# Step 2: アノテーション
# ---------------------------------------------------------------------------

def annotate_my_events(
    csv_path: str,
    outdir: str,
    thr: float = 0.5,
    min_duration_s: float = 1.0,
    smooth_s: float = 5.0,
) -> dict:
    """CSVから閾値ベースでイベントを抽出し、JSONL/SRTを出力する。"""
    data = read_mylog_csv(csv_path)
    times = data["times"]
    values = data["smooth_a"]
    fps = data["fps"]

    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    stem = Path(csv_path).stem.replace("_mylog", "")

    # イベント抽出（閾値超過の連続区間）
    min_samples = max(1, int(round(min_duration_s * fps)))
    events: List[dict] = []
    in_event = False
    start_idx = 0

    for i in range(len(values)):
        if values[i] >= thr and not in_event:
            in_event = True
            start_idx = i
        elif values[i] < thr and in_event:
            in_event = False
            if i - start_idx >= min_samples:
                events.append({
                    "type": "my_event_type",
                    "metric": "metric_a",
                    "thr": thr,
                    "peak": round(max(values[start_idx:i]), 6),
                    "start_sec": times[start_idx],
                    "end_sec": times[i - 1],
                    "start_srt": format_srt_time(times[start_idx]),
                    "end_srt": format_srt_time(times[i - 1]),
                })
    # 末尾処理
    if in_event and len(values) - start_idx >= min_samples:
        events.append({
            "type": "my_event_type",
            "metric": "metric_a",
            "thr": thr,
            "peak": round(max(values[start_idx:]), 6),
            "start_sec": times[start_idx],
            "end_sec": times[-1],
            "start_srt": format_srt_time(times[start_idx]),
            "end_srt": format_srt_time(times[-1]),
        })

    # JSONL出力
    jsonl_path = out_path / f"{stem}_my_events.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")

    # SRT出力
    srt_path = out_path / f"{stem}_my.srt"
    srt_lines: List[str] = []
    for idx, ev in enumerate(events, start=1):
        srt_lines.append(f"{idx}")
        srt_lines.append(f"{ev['start_srt']} --> {ev['end_srt']}")
        srt_lines.append("[my_category] description")
        srt_lines.append("")
    Path(srt_path).write_text("\n".join(srt_lines), encoding="utf-8")

    print(f"JSONL: {jsonl_path}")
    print(f"SRT  : {srt_path}")
    print(f"イベント数: {len(events)}")
    return {"jsonl": str(jsonl_path), "srt": str(srt_path), "events": len(events)}


# ---------------------------------------------------------------------------
# 一括実行
# ---------------------------------------------------------------------------

def analyze_video(
    video_path: str,
    outdir: str,
    fps: float = 5.0,
    roi_margin: float = 0.08,
    no_roi: bool = False,
    smooth_s: float = 5.0,
    thr: float = 0.5,
    min_duration_s: float = 1.0,
) -> dict:
    """2ステップの一括実行。"""
    result1 = record_timeseries(
        video_path, outdir, fps, roi_margin, no_roi, smooth_s,
    )
    if not result1:
        return {}
    result2 = annotate_my_events(
        result1["csv"], outdir, thr, min_duration_s, smooth_s,
    )
    return {**result1, **result2}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="...")
    subparsers = parser.add_subparsers(dest="command")

    ts = subparsers.add_parser("timeseries")
    ts.add_argument("--video", required=True)
    ts.add_argument("--outdir", required=True)
    ts.add_argument("--fps", type=float, default=5.0)

    ann = subparsers.add_parser("annotate")
    ann.add_argument("--csv", required=True)
    ann.add_argument("--outdir", required=True)
    ann.add_argument("--thr", type=float, default=0.5)

    ana = subparsers.add_parser("analyze")
    ana.add_argument("--video", required=True)
    ana.add_argument("--outdir", required=True)
    ana.add_argument("--fps", type=float, default=5.0)
    ana.add_argument("--thr", type=float, default=0.5)

    args = parser.parse_args()
    if args.command == "timeseries":
        record_timeseries(args.video, args.outdir, args.fps)
    elif args.command == "annotate":
        annotate_my_events(args.csv, args.outdir, args.thr)
    elif args.command == "analyze":
        analyze_video(args.video, args.outdir, args.fps, thr=args.thr)
    else:
        parser.print_help()
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

---

## 14. 環境制約

| 項目 | 値 |
|---|---|
| Python | 3.11+ |
| GPU | NVIDIA RTX 4070 (12GB VRAM), CUDA対応 |
| 一時ディレクトリ | `/tmp` |
| モデルキャッシュ | `~/AI/huggingface` |
| テスト実行 | `cd video_data_analyser && python -m pytest tests/ -v` |
| フレーム読み込み | PyAV優先、OpenCVフォールバック |
| 依存ライブラリ | `requirements.txt` に記載。numpy, opencv-python, torch, av が主要 |
