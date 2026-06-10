# 出血検出変換器（bleed_model_to_outputs）

先行研究の **出血検出結果** を読み込み、リポジトリ標準の二層構造
（SRT＋JSONL/CSV）へ変換するツールです。**推論モデル本体は含みません**。
フレーム単位の出血確率や区間ラベルを標準の出血イベントに整形し、他の解析
エンジン（redlog / bleed_detector / bleed_spread / transnet / yolo …）と同じ
タイムライン上に重ねられるようにします。

既存の `src/red/`（redlog, bleed_detector, bleed_spread）は HSV ベースの
**ヒューリスティックな一次検出** です。本変換器はそれらと衝突せず、別サフィックス
`_bleed_model` で出力する **補完的な変換器** です。既存 red/ の挙動・出力は
変更しません（追加のみ）。

対象として想定する先行研究:

- 深層学習ベースの **出血確率** 出力（per-frame スコア）
- **SurgBlood** 風の region(bbox) + point 検出
- **MultiBypass140** の術中有害事象(IAE)ラベル（出血イベント＋5段階 severity）

## 入力形式

### 1. per-frame 出血スコア CSV（確率系列）

| 列 | 必須 | 説明 |
|----|------|------|
| `frame_idx` | △ | フレーム番号（無ければ行順を採番） |
| `timestamp_sec` / `t_sec` | △ | 絶対時刻（秒）。あれば時刻として優先採用 |
| `bleed_prob` | ✅ | 出血確率（0〜1）。`prob` / `score` / `probability` も可 |
| `x,y,w,h` | – | 出血領域 bbox（任意、pass-through） |
| `point_x,point_y` | – | 出血点（任意、pass-through） |

```csv
frame_idx,timestamp_sec,bleed_prob,x,y,w,h
0,0.0,0.05,,,,
1,0.5,0.62,,,,
2,1.0,0.71,120,80,40,30
3,1.5,0.55,,,,
4,2.0,0.20,,,,
```

確率系列はヒステリシス閾値で区間化されます（後述）。

### 2. 区間ラベル形式（MultiBypass140 IAE 風）

出血イベントの開始/終了（秒またはフレーム）＋ `severity`(1..5) を持つ
**JSON または CSV**。JSON はオブジェクト配列、または
`{"events": [...]}`（`annotations` / `intervals` / `labels` も可）。

```json
{"events": [
  {"start_sec": 12.0, "end_sec": 18.5, "severity": 3, "type": "Bleeding"},
  {"start_sec": 40.0, "end_sec": 42.0, "severity": 5, "type": "Bleeding",
   "bbox": [10, 20, 30, 40], "point": [25, 35], "confidence": 0.91}
]}
```

CSV 例:

```csv
start_sec,end_sec,severity,type
12.0,18.5,3,Bleeding
40.0,42.0,5,Bleeding
```

時刻はフレーム指定（`start_frame` / `end_frame`）でも与えられます。その場合は
`--video` の PTS または `--fps` で秒へ解決します。

### 3. 汎用イベント JSON（区間＋種別）

種別フィールド（`type` / `label` / `event` / `category` …）が出血系
（`bleed` / `bleeding` / `hemorrhage` / `出血`）のレコードのみを変換対象とします。
種別の無いレコードは出血として扱います。`--include-all` で全件を変換します。

### 列名エイリアス

| 概念 | 受理するキー |
|------|--------------|
| 開始（秒） | `start_sec`, `start`, `begin`, `start_time`, `t_start`, `onset` |
| 終了（秒） | `end_sec`, `end`, `stop`, `end_time`, `t_end`, `offset` |
| 開始（フレーム） | `start_frame`, `frame_start`, `begin_frame` |
| 終了（フレーム） | `end_frame`, `frame_end`, `stop_frame` |
| severity | `severity`, `sev`, `grade`, `iae_severity` |
| region | `bbox`=[x,y,w,h] / `region`={x,y,w,h} / `x,y,w,h` |
| point | `point`=[x,y] / `point_x,point_y` |
| confidence | `confidence`, `conf`, `score`, `prob`, `probability` |

## 時刻解決

フェーズ変換器と同様、`--video` があれば PyAV の **PTS** から絶対時刻を得ます
（VFR 耐性）。`--video` が無ければ `--fps`（既定 25）でフレーム→秒に換算します。
入力に `timestamp_sec` があればそれを優先します。

## 変換ロジック

- **確率系列（per-frame CSV）**: ヒステリシス閾値で出血イベント区間を生成。
  `bleed_prob >= --thr-on`（既定 0.5）で ON、`< --thr-off`（既定 0.3）で OFF。
  生成区間のうち `--min-duration`（既定 1.0 秒）未満を除去します。`confidence`
  は区間内ピーク確率、`region`/`point` はピークフレームの値を pass-through します。
- **区間ラベル（JSON/CSV）**: モデル/データセットの確定ラベルとして忠実に変換し、
  `severity` を保持します（短区間でも除去しません）。
- `type` は既存 red/ と同じ `bleed_candidate` に揃え、`source` は
  `bleed_model_converter` を付与します。**冪等**（同入力→同出力）。

## 出力（二層構造、別サフィックスで既存と非衝突）

- `{stem}_bleed_model.jsonl` … **正本**。1区間=1行。`start_sec`/`end_sec`/
  `start_srt`/`end_srt`/`duration_sec`, `type`, `source`,
  `severity`(任意), `confidence`(任意), `region`/`point`(任意)。
- `{stem}_bleed_model.srt` … 既存の **2行構造**。1行目 `[bleed] bleeding`
  （severity があれば `[bleed] bleeding(sev=3)`）、2行目=機械向け JSON
  （時間フィールドは含めない＝SRT の時刻が正）。
- `{stem}_bleed_model.csv` … per-frame 入力では
  `frame_idx, t_sec, t_srt, bleed_prob, in_event, event_id` の時系列。
  区間入力では `event_id, start_sec, …, severity, confidence` のイベント単位。

## CLI

```bash
# per-frame 出血確率CSV をヒステリシスで区間化（動画PTSで時刻解決）
python -m src.red.bleed_model_to_outputs \
    --in pred.csv --video case.mp4 --outdir out/ \
    --thr-on 0.5 --thr-off 0.3 --min-duration 1.0

# MultiBypass140 IAE 風の区間ラベル（severity 付き）を変換
python -m src.red.bleed_model_to_outputs \
    --in iae.json --fps 25 --outdir out/

# 生成した _bleed_model.srt を既存の _bleed.srt 等と統合
python -m src.tools.merge_srt --out out/merged.srt \
    out/case_bleed.srt out/pred_bleed_model.srt
```

| オプション | 既定 | 説明 |
|------------|------|------|
| `--in` | （必須） | 入力（per-frame CSV / 区間ラベル JSON・CSV） |
| `--video` | – | 時刻解決用の動画（PyAV PTS、VFR耐性） |
| `--fps` | 25 | 動画が無い場合の換算 fps |
| `--outdir` | （必須） | 出力ディレクトリ |
| `--thr-on` | 0.5 | ヒステリシス ON 閾値（確率系列） |
| `--thr-off` | 0.3 | ヒステリシス OFF 閾値（確率系列） |
| `--min-duration` | 1.0 | 最小継続長フィルタ（秒、確率系列） |
| `--include-all` | off | 汎用JSONで非出血ラベルも含める |
| `--stem` | 入力名 | 出力ファイル名の語幹 |

## 既存ツールとの連携（往復・統合）

- `merge_srt` … `_bleed_model.srt` を既存の `_bleed.srt` / `_bleed_spread.srt`
  等と開始時刻順に統合できます。
- `srt_to_jsonl` … Shotcut で人手修正した SRT を JSONL 正本へ戻せます。
  `type` / `severity` / `region` / `point` / `confidence` / `source` は SRT の
  2行目 JSON に格納されるため、**JSONL→SRT→JSONL の往復で情報が保持** されます。

## BaseAnalyzer 連携

`make_analyzer()` で `BaseAnalyzer` 互換インスタンスを取得できます。
`analyze(video_path, in_path=..., fps=..., thr_on=..., thr_off=...,
min_duration=...)` でイベント列を `AnalysisResult.results` に格納して返します
（`video_path` は時刻解決の PTS にのみ使用）。

## 制約

重み・患者データ・サンプルはコミットしません。ライセンスは MIT。
既存 `src/red/` の挙動・出力は変更しません（追加のみ）。
