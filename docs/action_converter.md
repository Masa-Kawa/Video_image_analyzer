# 手技・動作認識変換器（action_to_outputs）

先行研究の **動作/手技認識結果** を読み込み、リポジトリ標準の二層構造
（SRT＋JSONL/CSV）へ変換するツールです。**推論モデル本体は含みません**。
フレーム単位／クリップ単位の予測を「区間（セグメント）」に整形し、他の解析エンジン
（redlog / transnet / yolo / phase …）と同じタイムライン上に重ねられるようにします。

主対象は **CholecT50 の triplet〈instrument, verb, target〉** 認識（Rendezvous 系の
出力等）。加えて、汎用の per-frame / per-clip action（SLAM の action、JIGSAWS gesture、
SAR-RARP50 action 等）もサポートします。

> 動作は同時に複数成立しうる（例: ある瞬間に複数の triplet がアクティブ）ため、
> フェーズ認識（排他的な区間分割）とは異なり、**ラベルごとに独立した区間**として
> 多ラベルの重なりを許容します。

## 入力形式

### 1. CholecT50 triplet 予測（per-frame）

**(a) triplet id（argmax）**

| 列 | 必須 | 説明 |
|----|------|------|
| `frame_idx` | △ | フレーム番号（無ければ行順を採用） |
| `timestamp_sec` / `t_sec` | – | 絶対時刻（秒）。あれば優先採用 |
| `triplet_id` | ✅ | triplet ID（0–99） |
| `confidence` | – | 予測確信度（`--triplet-thr` 未満は不採用） |

同一 `frame_idx` の複数行で **多ラベル**（同時に複数 triplet）を表現できます。

```csv
frame_idx,triplet_id,confidence
0,7,0.95
0,1,0.62
1,7,0.93
```

**(b) 確率ベクトル（100クラス）**

列 `triplet_0 .. triplet_99`（`ivt_*` も可）。`--triplet-thr`（既定 0.5）以上の列を
アクティブな多ラベルとして区間化します。

```csv
frame_idx,triplet_0,triplet_1,...,triplet_99
0,0.01,0.62,...,0.00
```

`triplet_id ↔ (instrument, verb, target)` の分解は JSON 設定で差し替え可能です
（既定は同梱の `src/action/maps/cholect50_triplets.json`＝CholecT50 公式 taxonomy、
6 instruments / 10 verbs / 15 targets, 100 triplets）。

### 2. 汎用 per-frame / per-clip action

| 列 | 必須 | 説明 |
|----|------|------|
| `frame_idx` または `clip_idx` | △ | フレーム/クリップ番号（無ければ行順） |
| `timestamp_sec` / `start_sec` | – | 絶対時刻（秒） |
| `action_name` または `action_id` | ✅（一方） | 動作名 / ID |
| `confidence` | – | 予測確信度 |

`action_0 .. action_K` の確率ベクトル形式も可（`--triplet-thr` を閾値として共用）。

### 3. clip 単位ラベル（SLAM 風 7アクション等）

`clip_idx` ＋ action ラベル。時刻区間は次の優先順位で復元します:
明示の `start_sec`/`end_sec` 列 → `--clip-len`（秒）→ `timestamp` 間隔の中央値。

```csv
clip_idx,action_name
0,navigation
1,navigation
2,transection
```

## 時刻解決

- `--video case.mp4` … PyAV の PTS から絶対時刻を得ます（VFR 動画に強い）。
- `--video` 省略時 … `--fps`（既定は triplet マップの 25fps）で `frame_idx / fps` 換算。
- CSV に `timestamp_sec` があればそれを最優先で採用します。
- clip 入力は clip 長から区間を復元します。

時刻フォーマットは `src.core.time_utils`（SRT は `HH:MM:SS,mmm`）を再利用します。

## 区間化・分解・フリッカ除去

1. ラベルごとに、連続するアクティブなフレーム/クリップを 1 区間に統合。
   多ラベルは独立に区間化され、時間的に重なって構いません。
2. `--smooth-window N`（フレーム/クリップ、既定 0=無効）… ラベルごとに中心 N の
   多数決で孤立した在/不在（フリッカ）を平滑化。
3. `--min-duration S`（秒、既定 0.5）… S 秒未満の短い区間を除去。
4. `--decompose` … triplet を `instrument` / `verb` / `target` の別トラック区間
   （`type="action"`, `role=...`）へ分解して追加出力。共有コンポーネント
   （例: 複数 triplet が共有する `grasper`）は 1 区間に統合されます。

同じ入力からは常に同じ出力が得られます（冪等）。

## 出力（二層構造）

| ファイル | 役割 |
|----------|------|
| `{stem}_action.jsonl` | **正本**。1 区間 = 1 行。`type`（`"triplet"`/`"action"`）, `label`, `triplet_id`/`action_id`（任意）, `components{instrument,verb,target}`（triplet 時）, `role`（分解時）, `confidence`（任意）, `source="action_converter"`, 時刻フィールド |
| `{stem}_action.srt` | 2 行構造。1 行目 `[action] grasper,retract,gallbladder`（triplet は i,v,t をカンマ連結）または `[action] <action_name>`、2 行目に機械向け JSON（時間フィールドは除外＝SRT の時刻が正） |
| `{stem}_action.csv` | 動作時系列。`--level segment`（既定）/ `frame` / `both`。frame は多ラベルを表す long 形式（1 行 = ユニット×アクティブラベル） |

SRT は常に `[action]` タグを用います。2 行目の JSON が `type`（triplet/action）と
`components` を保持するため、`{stem}_action.srt` を Shotcut で人手修正したのち
`python -m src.tools.srt_to_jsonl` で JSONL 正本へ戻す往復編集が可能です
（タグは `[action]` 共通でも、種別はJSON行から正しく復元されます）。

## 使い方

```bash
# CholecT50 triplet 確率ベクトルCSV（triplet_0..99）を動画 PTS で時刻解決＋分解出力
python -m src.action.action_to_outputs \
    --in triplet_pred.csv --video case.mp4 --outdir out/ \
    --triplet-thr 0.5 --decompose --level both

# triplet id（argmax）の per-frame CSV を 25fps で区間化
python -m src.action.action_to_outputs \
    --in triplet_ids.csv --fps 25 --outdir out/ --min-duration 0.5

# SLAM 風 clip 単位 action（clip 長 1 秒）
python -m src.action.action_to_outputs \
    --in clip_actions.csv --clip-len 1.0 --outdir out/

# カスタムの triplet マップを使う
python -m src.action.action_to_outputs \
    --in pred.csv --fps 5 --outdir out/ \
    --triplet-map maps/my_triplets.json

# 生成した _action.srt を既存の _bleed.srt / _cut.srt 等と統合
python -m src.tools.merge_srt --out out/merged.srt \
    out/case_bleed.srt out/triplet_pred_action.srt
```

`--help` で全オプションを確認できます。

## triplet マップ JSON の形式

```json
{
  "name": "cholect50",
  "fps": 25,
  "instruments": ["grasper", "bipolar", "hook", "scissors", "clipper", "irrigator"],
  "verbs": ["grasp", "retract", "dissect", "...", "null_verb"],
  "targets": ["gallbladder", "cystic_plate", "...", "null_target"],
  "triplets": { "0": [0, 2, 1], "1": [0, 2, 0], "...": [0, 0, 0] }
}
```

`triplets` は `triplet_id -> [instrument_id, verb_id, target_id]`。コンポーネント名の
並び順と分解は CAMMA-public/ivtmetrics の公式 `maps.txt` および Rendezvous
（Nwoye et al., MedIA 2022）の taxonomy に準拠しています。

## パイプライン連携

`src.action.action_to_outputs.make_analyzer()` は `BaseAnalyzer` 互換の変換器
インスタンスを返します。`analyze(video_path, in_path=..., fps=..., decompose=..., ...)`
を呼ぶと、区間化結果を `AnalysisResult.results` に格納して返します
（`video_path` は PTS による時刻解決にのみ使用）。
