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

> 「どこに何があるか」「人が直接動かすプログラムはどれか」を 3層（入口／解析エンジン／内部部品）で
> 整理した地図と使い方は [`docs/architecture.md`](docs/architecture.md) を参照してください。

```
video_data_analyser/
├── pyproject.toml
├── src/
│   ├── core/               # 共通ユーティリティ
│   │   └── time_utils.py   #   SRT/MLT時刻フォーマット
│   ├── analyzers/           # 解析器の基底クラス
│   │   └── base.py          #   BaseAnalyzer / AnalysisResult
│   ├── red/                 # 赤色（出血）解析 ── 色・差分・流れベースの複数手法
│   │   ├── redlog.py        #   HSV赤色率の時系列記録＋閾値イベント抽出
│   │   ├── bleed_detector.py #   赤色拡大検出（フレーム差分ベース）
│   │   ├── bleed_spread.py  #   グリッドベース局所拡散検出
│   │   ├── bleed_trend.py   #   赤色率の単調増加トレンド検出
│   │   ├── bleed_contact.py #   既存赤領域からの接触成長検出
│   │   ├── bleed_flow.py    #   オプティカルフロー発散による出血検出
│   │   └── bleed_model_to_outputs.py # 先行研究の出血検出結果→標準SRT/JSONL/CSV変換器
│   ├── bleed_ai/            # AI出血検出（DL分類器＋U-Net分割 / HSVハイブリッド）
│   │   └── zeroshot/        #   ゼロショット出血検出（DINOv2 / BiomedCLIP / SurgVLP 等）
│   ├── cavity/             # 腹腔内外判定（ヒューリスティック）
│   │   └── cavity_detector.py
│   ├── anomaly/            # 自己教師あり映像異常検知（ConvLSTM AE）
│   │   ├── detector.py
│   │   └── train.py        #   正常フレームのみで学習
│   ├── cholec_phase/       # Cholec80ベース手術フェーズ認識（BiLSTM）
│   │   ├── detector.py     #   動画→CSV→JSONL/SRT 推論
│   │   └── train.py        #   学習スクリプト（要Cholec80）
│   ├── transnet/            # TransNetV2 シーン検出
│   │   ├── inference.py     #   TransNetV2 PyTorchモデル推論
│   │   ├── transnet_analyzer.py # BaseAnalyzerインターフェース
│   │   └── transnet_to_srt.py   # JSONL→SRT境界変換
│   ├── motion/              # モーション検出
│   │   └── motion_analyzer.py #  フレーム差分ベースの動き検出
│   ├── yolo/                # YOLO器械検出
│   │   └── yolo_analyzer.py #   YOLO8手術器械認識＋シーン分割
│   ├── phase/               # 手術フェーズ
│   │   ├── phase_segmenter.py # 教師なしフェーズ分割（ラベル不要）
│   │   ├── phase_to_outputs.py # 先行研究フェーズ予測→標準SRT/JSONL/CSV変換器
│   │   └── maps/            #   phase_id↔phase_name マップ（Cholec80既定）
│   ├── action/              # 手技・動作認識
│   │   ├── action_to_outputs.py # 先行研究 triplet/action 予測→標準SRT/JSONL/CSV変換器
│   │   ├── rendezvous_inference.py # Rendezvous triplet 推論→CSV
│   │   ├── surgeonet_inference.py  # SurgeoNet 器械検出→per-frame action CSV
│   │   ├── surgical_yolo_inference.py # 手術器械専用YOLO→action CSV
│   │   └── maps/            #   triplet_id↔(i,v,t) マップ（CholecT50既定）
│   ├── annotate/            # ★ アノテーション編集Webアプリ（修正用UI）
│   │   ├── server.py        #   FastAPIバックエンド（動画同期・保存）
│   │   ├── srt_io.py        #   安定ID付きSRT読み書き
│   │   ├── pairing.py       #   DPOペア（rejected/chosen）生成
│   │   ├── label_sets.py    #   術式別フェーズ語彙レジストリ
│   │   └── web/             #   フロントエンド（index.html / app.js / style.css）
│   ├── mlt/                 # Shotcut MLTプロジェクト生成
│   │   ├── mlt_generator.py #   MLT XML生成（マルチトラック）
│   │   └── mlt_builder.py   #   柔軟な設定ベースMLTビルダー
│   ├── tools/               # 変換・ユーティリティ
│   │   ├── proxy_manager.py #   プロキシ動画の作成・管理
│   │   ├── jsonl_to_srt.py  #   JSONL → SRT
│   │   ├── srt_to_jsonl.py  #   SRT → JSONL（編集反映）
│   │   ├── srt_to_mkv.py    #   フェーズSRT → チャプター付きMKV
│   │   ├── merge_srt.py     #   複数SRTマージ
│   │   ├── csv_to_srt.py    #   CSV時系列 → SRT字幕
│   │   └── plot_redlog.py   #   CSV → PNGグラフ
│   ├── pipeline.py          # 統合パイプライン（プロキシ→解析→MLT）
│   └── surgical_pipeline.py # 腹腔鏡向け一括解析（腹腔内外＋出血＋フェーズ）
├── tests/                   # ユニットテスト
└── docs/                    # 詳細ドキュメント（analyzers.md ほか）
```

## 設計原則

| 原則 | 説明 |
|------|------|
| **二層構造** | SRT（可視化・編集用）＋ JSONL/CSV（正本・集計用） |
| **絶対時間で重畳** | 複数解析器の結果を同一タイムライン上に重ねる |
| **Human-in-the-loop** | アルゴリズムが一次検出 → 人間が編集アプリ/Shotcutで修正 → JSONL更新 |
| **PTSベース** | VFR動画でも安定するようPyAVでフレーム時刻を取得 |
| **Unix哲学** | 1ツール1機能。パイプラインで組み合わせる |

## 解析エンジン一覧

| エンジン | モジュール | 説明 | 主な出力 |
|----------|-----------|------|----------|
| **redlog** | `src.red.redlog` | HSV赤色率の時系列記録＋出血候補検出 | `_redlog.csv`, `_bleed.srt` |
| **bleed_detector** | `src.red.bleed_detector` | 赤色拡大検出（背景安定度考慮） | `_bleedlog.csv`, `_bleed_expansion.srt` |
| **bleed_spread** | `src.red.bleed_spread` | 8x8グリッド局所拡散スコアリング | `_spreadlog.csv`, `_bleed_spread.srt` |
| **bleed_trend / contact / flow** | `src.red.bleed_trend` 他 | 赤色の単調増加・接触成長・オプティカルフロー発散による出血検出 | `_bleed_*.srt` |
| **bleed_ai** | `src.bleed_ai.detector` | DL分類器＋U-Net分割とHSVのハイブリッド出血検出（`zeroshot/` にDINOv2/BiomedCLIP等のゼロショット系） | CSV/SRT |
| **cavity_detector** | `src.cavity.cavity_detector` | 腹腔内外判定（ヒューリスティック、有効区間の特定） | CSV/JSONL/SRT |
| **anomaly** | `src.anomaly.detector` | 自己教師あり映像異常検知（ConvLSTM AE、正常分布からの逸脱） | CSV/SRT |
| **cholec_phase** | `src.cholec_phase.detector` | Cholec80学習済みBiLSTMによる手術フェーズ認識（要モデル） | `_phase.csv/jsonl/srt` |
| **phase_segmenter** | `src.phase.phase_segmenter` | ラベル不要の教師なしフェーズ分割 | JSONL/SRT |
| **transnet** | `src.transnet` | TransNetV2によるシーン境界検出 | `_cut.srt`, `_cut.jsonl` |
| **motion** | `src.motion` | フレーム差分ベースの動き検出 | JSON結果 |
| **yolo** | `src.yolo` | YOLO8手術器械認識＋シーン分割 | `_scenes.srt`, `_scenes.jsonl` |
| **phase_converter** | `src.phase.phase_to_outputs` | 先行研究のフェーズ認識結果（Cholec80公式アノテーション / TeCNO・Trans-SVNet・LoViT・Surgformer 等の予測）をフレーム単位→区間に整形し標準フォーマットへ変換 | `_phase.srt`, `_phase.jsonl`, `_phase.csv` |
| **bleed_model_converter** | `src.red.bleed_model_to_outputs` | 先行研究の出血検出結果（深層学習の出血確率、SurgBlood 風 region+point、MultiBypass140 の IAE 出血ラベル＋severity）を、ヒステリシス区間化して標準フォーマットへ変換（補完的な変換器、既存 red/ と非衝突） | `_bleed_model.srt`, `_bleed_model.jsonl`, `_bleed_model.csv` |
| **action_converter** | `src.action.action_to_outputs` | 先行研究の動作/手技認識結果（CholecT50 triplet〈instrument, verb, target〉/ Rendezvous 系、汎用 per-frame/clip action、SLAM action・JIGSAWS gesture・SAR-RARP50 action）をフレーム/クリップ単位→区間に整形し標準フォーマットへ変換。多ラベル重畳・triplet 分解（`--decompose`）対応 | `_action.srt`, `_action.jsonl`, `_action.csv` |

> **一括実行:** `src.surgical_pipeline` は腹腔鏡向けに「腹腔内外判定→出血検出→フェーズ認識」をまとめて実行します。各解析器（cavity / bleed_ai / anomaly / cholec_phase 等）の技術的な詳細は [`docs/analyzers.md`](docs/analyzers.md) を参照してください。

## クイックスタート

```bash
# 1. プロキシ作成（解析高速化のため。--gpu で高速化、複数→--merge で1本化）
python -m src.tools.proxy_manager case001.mp4 --resolution 720p --gpu

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

## アノテーション編集Webアプリ（修正用UI）

自動生成したSRT（フェーズ等）を、**動画と同期したブラウザUI上で修正**するためのWebアプリです（`src/annotate/`）。Shotcutを使わずブラウザ単体で、セグメントの編集・削除・挿入ができます。SRTが無い状態からの**ゼロ作成（ゴールドラベル付け）**も同じUIで行えます。

```bash
# 既存SRTを編集モードで開く（ブラウザが自動起動）
python -m src.annotate.server \
    --video case001.mp4 \
    --srt out/case001_phase.srt \
    --procedure cholecystectomy --port 8000

# --srt を省略すると新規作成モード（空セグメントから手付け）
python -m src.annotate.server --video case001.mp4 --procedure cholecystectomy
```

**仕組みと出力:**

- 起動時にブラウザ再生用プロキシ（480p）を自動生成し、`/media/video` で配信（シーク対応）。
- 各セグメントに**安定ID**を付与（`src/annotate/srt_io.py`）。元SRTと修正後SRTをIDで突き合わせる。
- 保存すると次の2ファイルを書き出す:
  - `{video_stem}_gold.srt` — 修正後SRT（**元SRTは保持**、`--save-target` で変更可）
  - `{video_stem}_gold_dpo_pairs.jsonl` — DPO的な追加学習用ペア（`rejected`=自動生成 / `chosen`=人手修正）。変更種別 `edited` / `inserted` / `deleted` を記録。新規作成モードでは全件が `inserted`（純ゴールドラベル）。
- 術式ごとのフェーズ語彙は `src/annotate/label_sets.py` のレジストリ方式（`LABEL_SETS` にエントリを追加するだけで新術式に対応）。MVPは胆嚢摘出術（Cholec80フェーズを再利用）。

| オプション | 既定 | 説明 |
|------------|------|------|
| `--video` | （必須） | 対象動画 |
| `--srt` | なし | 編集対象の元SRT（省略で新規作成モード） |
| `--procedure` | `cholecystectomy` | フェーズ語彙の選択 |
| `--save-target` | `<video>_gold.srt` | 保存先SRT |
| `--host` / `--port` | `127.0.0.1` / `8000` | 待受アドレス |
| `--no-browser` | — | 起動時にブラウザを開かない |

> セキュリティ: 保存APIはCSRFトークンを要求し、既定で `127.0.0.1` のみ待受（ローカル利用前提）。

操作を一から学ぶ実践ガイドは [`docs/annotation_editor_tutorial.md`](docs/annotation_editor_tutorial.md)、
UI操作・出力形式・APIの詳細は [`docs/annotation_editor.md`](docs/annotation_editor.md) を参照してください。

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
| `{stem}_gold.srt` | SRT | アノテーション編集アプリで修正/作成した確定SRT（安定ID付き） |
| `{stem}_gold_dpo_pairs.jsonl` | JSONL | DPO追加学習用ペア（`rejected`/`chosen`＋変更種別） |

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

## Git フック（lint と push前の安全検査）

2種類のフックを用意しています。ネイティブ Git フックはクローン後に共有されないため、
各自で一度だけ有効化します:

```bash
uv sync --group dev            # ruff を含む開発依存を導入
bash scripts/install-hooks.sh  # pre-commit と pre-push を有効化
```

### pre-commit（lint）

コミット時に、変更した Python ファイルへ自動で [ruff](https://docs.astral.sh/ruff/) の lint
（未使用 import・未定義名・構文/論理エラー等の検出）を走らせます。自動修正できた分は再ステージされ、
直せない問題が残るとコミットが中断されます。整形（`ruff format`）は既存スタイルを大きく変えるため
**含めていません**（lint のみ）。

- フック本体: `scripts/git-hooks/pre-commit`
- 手動実行: `ruff check src/ tests/` ／ 一時回避: `git commit --no-verify`

### pre-push（危険ファイル/データの混入チェック）

push 直前に、送ろうとしているコミットを走査し、**公開してはいけないものが混ざっていないか**を検査します。
1件でも該当（BLOCK）すれば push を中止します。**PR を出す前の最終関門**としても手動で実行できます。

検出対象（BLOCK）:
- 動画(`*.mp4` 等)・モデル/重み(`*.pth/*.pt/*.onnx` 等)・解析データ(`*.csv/*.srt/*.jsonl`)・アーカイブ・鍵/証明書(`*.pem/*.key`)・`.env`
- `.gitignore` 対象なのに追跡されているファイル
- 5MB を超える大容量ファイル
- 中身に含まれる API キー/秘密鍵らしき文字列（AWS/GitHub/Google/Slack/OpenAI 形式・PEM 秘密鍵）

WARN（表示のみ・push は許可）: 個人の絶対パス `/home/<user>/`、パスワード/キーらしき代入。

```bash
bash scripts/safety-check.sh          # 未pushコミットを手動検査（PR前の確認に）
bash scripts/safety-check.sh main..HEAD  # 範囲指定
```

- フック本体: `scripts/git-hooks/pre-push` ／ 検査ロジック: `scripts/safety-check.sh`
- 一時回避（誤検知時）: `git push --no-verify`

## ライセンス

MIT License — see [LICENSE](LICENSE).
Copyright (c) 2026 Masahiko Kawaguchi, Yokohama Sakae Kyosai Hospital
