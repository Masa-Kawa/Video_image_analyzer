# Video Data Analyser

手術動画を複数の解析エンジンで分析し、結果を **SRT字幕** や **MLTプロジェクト** として出力する統合プラットフォームです。
Shotcutに取り込んで視覚的に確認しつつ、Human-in-the-loop で修正できます。

> 旧 `video_data_analyser` + `scene_detector` + `common_video_analyser` を統合したプロジェクトです。

---

> **📢 Public release notice**
> 本リポジトリは第8回日本メディカルAI学会（2026年）の発表に合わせて
> 公開しています。維持リソースの都合上、リポジトリ自体は学会開催の前後
> 約1ヶ月で非公開化する可能性がありますが、公開期間中に取得したコードは
> MITライセンスのもと**永続的に自由に再利用・改変・再配布いただけます**。
> 広く活用していただければ幸いです。
>
> Author: **Masahiko Kawaguchi**, Yokohama Sakae Kyosai Hospital, Department of Surgery
> Contact: 学会発表時の連絡先、もしくは GitHub Issues 経由でお願いします。

## ライセンス・データの扱い

- **コード**: MIT License（[LICENSE](LICENSE) 参照）
- **学習済みモデル (`*.pth`)**: 本リポジトリには**含まれていません**。
  Cholec80 由来の重みは CC-BY-NC-SA 4.0 由来のため、利用者各自で
  [Cholec80 データセット](http://camma.u-strasbg.fr/datasets)を申請・学習してください。
  手順は `src/cholec_phase/train.py` の docstring を参照。
- **動画データ・解析結果サンプル**: 患者情報保護のため**含まれていません**。
- **依拠する外部資産** (各自のライセンスに従ってください):
  - [SelfSupSurg](https://github.com/CAMMA-public/SelfSupSurg) — DINO pretrained ResNet50
  - [TransNet V2](https://github.com/soCzech/TransNetV2) — シーン境界検出
  - [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — 器械検出
  - [Cholec80 dataset](http://camma.u-strasbg.fr/datasets) — フェーズ認識学習データ

## セットアップ

```bash
cd video_data_analyser
uv sync            # 本体の依存関係をインストール
uv sync --group dev  # テスト用（pytest）も含める
```

## プロジェクト構成

```
video_data_analyser/
├── pyproject.toml
├── src/
│   ├── core/               # 共通ユーティリティ
│   │   └── time_utils.py   #   SRT/MLT時刻フォーマット
│   ├── analyzers/           # 解析器の基底クラス
│   │   └── base.py          #   BaseAnalyzer / AnalysisResult
│   ├── red/                 # 赤色（出血）解析 ── 3アルゴリズム
│   │   ├── redlog.py        #   HSV赤色率の時系列記録＋閾値イベント抽出
│   │   ├── bleed_detector.py #   赤色拡大検出（フレーム差分ベース）
│   │   ├── bleed_spread.py  #   グリッドベース局所拡散検出
│   │   └── bleed_model_to_outputs.py # 先行研究の出血検出結果→標準SRT/JSONL/CSV変換器
│   ├── transnet/            # TransNetV2 シーン検出
│   │   ├── inference.py     #   TransNetV2 PyTorchモデル推論
│   │   ├── transnet_analyzer.py # BaseAnalyzerインターフェース
│   │   └── transnet_to_srt.py   # JSONL→SRT境界変換
│   ├── motion/              # モーション検出
│   │   └── motion_analyzer.py #  フレーム差分ベースの動き検出
│   ├── yolo/                # YOLO器械検出
│   │   └── yolo_analyzer.py #   YOLO8手術器械認識＋シーン分割
│   ├── phase/               # 手術フェーズ
│   │   ├── phase_to_outputs.py # 先行研究フェーズ予測→標準SRT/JSONL/CSV変換器
│   │   └── maps/            #   phase_id↔phase_name マップ（Cholec80既定）
│   ├── action/              # 手技・動作認識
│   │   ├── action_to_outputs.py # 先行研究 triplet/action 予測→標準SRT/JSONL/CSV変換器
│   │   └── maps/            #   triplet_id↔(i,v,t) マップ（CholecT50既定）
│   ├── mlt/                 # Shotcut MLTプロジェクト生成
│   │   ├── mlt_generator.py #   MLT XML生成（マルチトラック）
│   │   └── mlt_builder.py   #   柔軟な設定ベースMLTビルダー
│   ├── tools/               # 変換・ユーティリティ
│   │   ├── proxy_manager.py #   プロキシ動画の作成・管理
│   │   ├── jsonl_to_srt.py  #   JSONL → SRT
│   │   ├── srt_to_jsonl.py  #   SRT → JSONL（編集反映）
│   │   ├── merge_srt.py     #   複数SRTマージ
│   │   ├── csv_to_srt.py    #   CSV時系列 → SRT字幕
│   │   └── plot_redlog.py   #   CSV → PNGグラフ
│   └── pipeline.py          # 統合パイプライン（プロキシ→解析→MLT）
├── tests/                   # ユニットテスト（102テスト）
└── docs/                    # 詳細ドキュメント
```

## 設計原則

| 原則 | 説明 |
|------|------|
| **二層構造** | SRT（可視化・編集用）＋ JSONL/CSV（正本・集計用） |
| **絶対時間で重畳** | 複数解析器の結果を同一タイムライン上に重ねる |
| **Human-in-the-loop** | アルゴリズムが一次検出 → 人間がShotcutで修正 → JSONL更新 |
| **PTSベース** | VFR動画でも安定するようPyAVでフレーム時刻を取得 |
| **Unix哲学** | 1ツール1機能。パイプラインで組み合わせる |

## 解析エンジン一覧

| エンジン | モジュール | 説明 | 主な出力 |
|----------|-----------|------|----------|
| **redlog** | `src.red.redlog` | HSV赤色率の時系列記録＋出血候補検出 | `_redlog.csv`, `_bleed.srt` |
| **bleed_detector** | `src.red.bleed_detector` | 赤色拡大検出（背景安定度考慮） | `_bleedlog.csv`, `_bleed_expansion.srt` |
| **bleed_spread** | `src.red.bleed_spread` | 8x8グリッド局所拡散スコアリング | `_spreadlog.csv`, `_bleed_spread.srt` |
| **transnet** | `src.transnet` | TransNetV2によるシーン境界検出 | `_cut.srt`, `_cut.jsonl` |
| **motion** | `src.motion` | フレーム差分ベースの動き検出 | JSON結果 |
| **yolo** | `src.yolo` | YOLO8手術器械認識＋シーン分割 | `_scenes.srt`, `_scenes.jsonl` |
| **phase_converter** | `src.phase.phase_to_outputs` | 先行研究のフェーズ認識結果（Cholec80公式アノテーション / TeCNO・Trans-SVNet・LoViT・Surgformer 等の予測）をフレーム単位→区間に整形し標準フォーマットへ変換 | `_phase.srt`, `_phase.jsonl`, `_phase.csv` |
| **bleed_model_converter** | `src.red.bleed_model_to_outputs` | 先行研究の出血検出結果（深層学習の出血確率、SurgBlood 風 region+point、MultiBypass140 の IAE 出血ラベル＋severity）を、ヒステリシス区間化して標準フォーマットへ変換（補完的な変換器、既存 red/ と非衝突） | `_bleed_model.srt`, `_bleed_model.jsonl`, `_bleed_model.csv` |
| **action_converter** | `src.action.action_to_outputs` | 先行研究の動作/手技認識結果（CholecT50 triplet〈instrument, verb, target〉/ Rendezvous 系、汎用 per-frame/clip action、SLAM action・JIGSAWS gesture・SAR-RARP50 action）をフレーム/クリップ単位→区間に整形し標準フォーマットへ変換。多ラベル重畳・triplet 分解（`--decompose`）対応 | `_action.srt`, `_action.jsonl`, `_action.csv` |

## クイックスタート

```bash
# 1. プロキシ作成（解析高速化のため）
python -m src.tools.proxy_manager case001.mp4 --resolution 720p

# 2. 赤色解析（時系列記録 → イベント抽出）
python -m src.red.redlog timeseries --video case001.mp4 --outdir out/
python -m src.red.redlog annotate --csv out/case001_redlog.csv --outdir out/ --thr 0.03

# 3. TransNet境界をSRTに変換
python -m src.transnet.transnet_to_srt \
    --in-jsonl out/case001_transnet.jsonl --out-srt out/case001_cut.srt

# 4. 複数SRTをマージして統合字幕を作成
python -m src.tools.merge_srt --out out/case001_merged.srt \
    out/case001_bleed.srt out/case001_cut.srt

# 5. Shotcutで元動画 + 統合SRTを開いて確認・修正
```

詳しい手順は [TUTORIAL.md](TUTORIAL.md) を参照してください。

## 出力ファイル一覧

| ファイル名 | 種別 | 説明 |
|------------|------|------|
| `{stem}_events.jsonl` | JSONL | **イベント正本**（Single Source of Truth） |
| `{stem}_bleed.srt` | SRT | 出血候補イベント（可視化・編集用） |
| `{stem}_metrics.srt` | SRT | 赤色率・変化量の可視化字幕 |
| `{stem}_redlog.csv` | CSV | 赤色率ログ（時系列データ） |
| `{stem}_plot.png` | PNG | 赤色率・変化量の時系列グラフ |
| `{stem}_cut.srt` | SRT | TransNetカット境界 |
| `{stem}_scenes.srt` | SRT | YOLO器械シーン区間 |
| `{stem}_phase.jsonl` | JSONL | フェーズ区間の正本（先行研究フェーズ認識の変換結果） |
| `{stem}_phase.srt` | SRT | フェーズ区間（`[phase] <phase_name>`、可視化・編集用） |
| `{stem}_phase.csv` | CSV | フェーズ時系列（`--level` で segment / frame 粒度） |
| `{stem}_bleed_model.jsonl` | JSONL | 先行研究の出血検出を変換した出血イベント正本（`source="bleed_model_converter"`） |
| `{stem}_bleed_model.srt` | SRT | 出血イベント（`[bleed] bleeding`、severity 時 `[bleed] bleeding(sev=N)`） |
| `{stem}_bleed_model.csv` | CSV | per-frame の `bleed_prob`＋イベント該当フラグ/区間ID（区間入力ではイベント単位） |
| `{stem}_action.jsonl` | JSONL | 動作/手技区間の正本（`type="triplet"`/`"action"`, `label`, `components`, `source="action_converter"`） |
| `{stem}_action.srt` | SRT | 動作/手技区間（`[action] grasper,retract,gallbladder` 等、可視化・編集用） |
| `{stem}_action.csv` | CSV | 動作/手技の時系列（`--level` で segment / frame 粒度、frame は多ラベル long 形式） |
| `{stem}_merged.srt` | SRT | 統合SRT（Shotcut投入用） |

## SRT本文フォーマット

すべてのSRTエントリは **2行構造** で統一されています。

```srt
1
00:02:15,600 --> 00:02:18,800
[bleed] delta_over_threshold
{"type": "bleed_candidate", "metric": "red_ratio", "thr": 0.03, "delta_max": 0.05}
```

- 1行目: 人間向けタグ（`[bleed]`, `[cut]`, `[scene]` 等）
- 2行目: 機械向けJSON（時間フィールドは除外、SRTの時刻が正）

## テスト

```bash
uv run python -m pytest tests/ -v
```

## ライセンス

MIT License — see [LICENSE](LICENSE).
Copyright (c) 2026 Masahiko Kawaguchi, Yokohama Sakae Kyosai Hospital
