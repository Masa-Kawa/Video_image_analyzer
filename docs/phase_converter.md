# フェーズ認識変換器（phase_to_outputs）

先行研究の **手術フェーズ認識結果**（Cholec80 系）を読み込み、リポジトリ標準の
二層構造（SRT＋JSONL/CSV）へ変換するツールです。**推論モデル本体は含みません**。
フレーム単位の予測やアノテーションを「区間（セグメント）」に整形し、
他の解析エンジン（redlog / transnet / yolo …）と同じタイムライン上に重ねられる
ようにします。

対象は Cholec80 公式アノテーション、および TeCNO / Trans-SVNet / LoViT /
Surgformer 等のモデルが出力するフレーム単位の予測です。

## 入力形式

### 1. Cholec80 公式アノテーション（`Frame<TAB>Phase`）

タブ区切り・ヘッダ行あり・既定 25fps。7 フェーズ taxonomy:

```
Frame	Phase
0	Preparation
25	Preparation
50	CalotTriangleDissection
...
```

フェーズ名: `Preparation` / `CalotTriangleDissection` / `ClippingCutting` /
`GallbladderDissection` / `GallbladderPackaging` / `CleaningCoagulation` /
`GallbladderRetraction`

### 2. 汎用 per-frame 予測 CSV

| 列 | 必須 | 説明 |
|----|------|------|
| `frame_idx` | △ | フレーム番号（`timestamp_sec` があれば省略可。その場合は行順を採用） |
| `timestamp_sec` / `t_sec` | △ | 絶対時刻（秒）。あれば時刻として優先採用 |
| `phase_name` または `phase_id` | ✅（一方） | フェーズ名 / ID |
| `confidence` | – | 予測確信度 |

`frame_idx` と `timestamp_sec`（`t_sec`）は少なくとも一方が必要です。
本リポジトリの `*_cholecphaselog.csv`（`t_sec, t_srt, phase_id, phase_name,
prob_*, confidence, reader`）もそのまま入力できます。

```csv
frame_idx,timestamp_sec,phase_id,confidence
0,0.00,0,0.97
1,0.04,0,0.95
2,0.08,1,0.62
```

`phase_id ↔ phase_name` のマッピングは JSON 設定で差し替え可能です
（既定は同梱の `src/phase/maps/cholec80_phases.json`）。

## 時刻解決

- `--video case.mp4` … PyAV の PTS から絶対時刻を得ます（VFR 動画に強い）。
- `--video` 省略時 … `--fps`（既定はフェーズマップの 25fps）で `frame_idx / fps` 換算。
- CSV に `timestamp_sec` があればそれを最優先で採用します。

時刻フォーマットは `src.core.time_utils`（SRT は `HH:MM:SS,mmm`）を再利用します。

## 区間化とフリッカ除去

1. 連続する同一フェーズのフレームを 1 区間に統合（区間は時間的に連続）。
2. `--smooth-window N`（フレーム、既定 0=無効）… 中心 N フレームの多数決で
   孤立した誤分類を平滑化。
3. `--min-duration S`（秒、既定 2.0）… S 秒未満の区間を隣接区間へ吸収。

同じ入力からは常に同じ出力が得られます（冪等）。

## 出力（二層構造）

| ファイル | 役割 |
|----------|------|
| `{stem}_phase.jsonl` | **正本**。1 区間 = 1 行。`type="surgical_phase"`, `label=phase_name`, `source="phase_converter"`, `confidence`（任意）, 時刻フィールドを含む |
| `{stem}_phase.srt` | 2 行構造。1 行目 `[phase] <phase_name>`、2 行目に機械向け JSON（時間フィールドは除外＝SRT の時刻が正） |
| `{stem}_phase.csv` | フェーズ時系列。`--level segment`（既定）/ `frame` / `both` |

`type` は `[phase]` タグに対応する `surgical_phase` を用います
（`jsonl_to_srt` / `srt_to_jsonl` のタグ規約に準拠）。これにより
`{stem}_phase.srt` を Shotcut で人手修正したのち
`python -m src.tools.srt_to_jsonl` で JSONL 正本へ戻す往復編集が可能です。

## 使い方

```bash
# Cholec80 公式アノテーション（25fps、最小継続長 2 秒）
python -m src.phase.phase_to_outputs \
    --in video01-phase.txt --fps 25 --outdir out/

# per-frame 予測 CSV を動画 PTS で時刻解決（VFR 耐性）＋多数決スムージング
python -m src.phase.phase_to_outputs \
    --in pred.csv --video case.mp4 --outdir out/ \
    --min-duration 2.0 --smooth-window 5 --level both

# カスタムのフェーズマップを使う
python -m src.phase.phase_to_outputs \
    --in pred.csv --fps 1 --outdir out/ \
    --phase-map maps/my_phases.json
```

`--help` で全オプションを確認できます。

## パイプライン連携

`src.phase.phase_to_outputs.make_analyzer()` は `BaseAnalyzer` 互換の
変換器インスタンスを返します。`analyze(video_path, in_path=..., fps=..., ...)`
を呼ぶと、区間化結果を `AnalysisResult.results` に格納して返します
（`video_path` は PTS による時刻解決にのみ使用）。
