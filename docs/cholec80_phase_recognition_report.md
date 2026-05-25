# Cholec80 を用いた腹腔鏡手術フェーズ認識モデル 開発レポート

作成日: 2026-04-22  
対象プロジェクト: `video_data_analyser`

---

## 1. はじめに

腹腔鏡下胆嚢摘出術（Laparoscopic Cholecystectomy）の動画解析において、術中のどの手術フェーズにいるかを自動認識することは、出血検出・術式評価・教育支援など多くの下流タスクに共通して必要な基盤技術である。

本プロジェクトでは、Cholec80 データセット（腹腔鏡手術動画 80 本、フレームレベルのフェーズアノテーション付き）を用いた教師あり機械学習により、手術動画からリアルタイムにフェーズを認識するモデルを開発・評価した。

---

## 2. 目的

### 主目的

手術動画（MP4）を入力として、以下を自動生成する評価器を実装する。

- **CSV**: フレームごとの手術フェーズ確率（時系列ログ）
- **JSONL**: フェーズ区間イベント（正本データ）
- **SRT**: Shotcut 用フェーズ字幕ファイル

### 対象フェーズ（Cholec80 標準 7 フェーズ）

| ID | フェーズ名 | 説明 |
|---|---|---|
| P1 | Preparation | 術前準備、ポート挿入前 |
| P2 | CalotTriangleDissection | カロー三角の剥離 |
| P3 | ClippingCutting | クリッピングと切断 |
| P4 | GallbladderDissection | 胆嚢の剥離 |
| P5 | GallbladderPackaging | 胆嚢の袋詰め |
| P6 | CleaningCoagulation | 洗浄・止血 |
| P7 | GallbladderRetraction | 胆嚢の摘出 |

---

## 3. 準備

### 3.1 データセット: Cholec80

| 項目 | 内容 |
|---|---|
| 提供元 | University of Strasbourg (CAMMA Lab) |
| 動画数 | 80 本（腹腔鏡下胆嚢摘出術） |
| 収録レート | 25 fps |
| アノテーション | フレームごとのフェーズラベル（全フレーム）、ツール使用バイナリ |
| ライセンス | CC BY-NC-SA 4.0 |
| 動画長 | 最短 12 分 / 最長 100 分 / 平均 38.5 分 |
| 合計時間 | 約 51 時間（3,076 分） |
| ダウンロードサイズ | 70 GB (zip) / 71 GB (展開後) |
| 展開先 | `/path/to/cholec80/` |

**フェーズ分布（全80動画）:**

| フェーズ | フレーム数（1fps換算） | 割合 |
|---|---|---|
| Preparation | 8,574 | 4.6% |
| CalotTriangleDissection | 74,826 | **40.5%** |
| ClippingCutting | 14,080 | 7.6% |
| GallbladderDissection | 58,433 | **31.7%** |
| GallbladderPackaging | 7,618 | 4.1% |
| CleaningCoagulation | 14,335 | 7.8% |
| GallbladderRetraction | 6,712 | 3.6% |
| **合計** | **184,578** | 100% |

CalotTriangleDissection（40.5%）と GallbladderDissection（31.7%）で全体の 72% を占めるクラス不均衡なデータセットである。

### 3.2 事前学習済み重み: SelfSupSurg (DINO)

Backbone として、Cholec80 で自己教師あり学習された ResNet50 重みを使用した。

| 項目 | 内容 |
|---|---|
| 手法 | DINO (Self-Distillation with No Labels) |
| Backbone | ResNet50 |
| 学習データ | Cholec80 全動画（自己教師あり、アノテーション不使用） |
| 出力 | 2,048 次元特徴ベクトル |
| 提供元 | CAMMA Lab, University of Strasbourg |
| 保存先 | `$TORCH_HOME/hub/checkpoints/model_final_checkpoint_dino_surg.torch` |

手術映像に特化した特徴空間を持つため、ImageNet 事前学習よりも手術フェーズの弁別に有利であることが期待される。

### 3.3 実行環境

| 項目 | 内容 |
|---|---|
| GPU | NVIDIA RTX 4070 (12 GB VRAM) |
| CPU | AMD（換装後） |
| Python | 3.11 |
| フレームワーク | PyTorch + torchvision |

---

## 4. 方法

### 4.1 アーキテクチャ

2 段のニューラルネットワークを組み合わせた構成を採用した。

```
[動画フレーム]
     ↓  (1fps サンプリング)
[ResNet50 Backbone]      ← SelfSupSurg DINO 重み（凍結）
     ↓  2,048次元特徴ベクトル
[Bidirectional LSTM]     ← hidden_dim=512, num_layers=2
     ↓  1,024次元ベクトル（双方向 × 512）
[Dropout(0.3) → FC(256) → ReLU → Dropout(0.3) → FC(7)]
     ↓
[フェーズ確率 × 7クラス]
```

**設計の意図:**

- **Backbone 凍結**: SelfSupSurg の特徴空間を保持しつつ、LSTM のみを学習することで過学習を防ぐ
- **Bidirectional LSTM**: 前後の時間文脈を双方向に参照し、フェーズ遷移を捉える
- **パラメータ数**: 約 17,057,799（BiLSTM + Classifier のみ、Backbone 除く）

### 4.2 学習パイプライン

2 段階の処理で効率的に学習する。

#### Stage 1: 特徴量の事前抽出

全動画をオフラインで特徴抽出し、`.npy` ファイルに保存する。これにより Stage 2 の学習ループで Backbone の推論コストをゼロにできる。

```bash
python -m src.cholec_phase.train extract-features \
    --cholec80-dir /path/to/cholec80 \
    --outdir ./features \
    --device cuda \
    --sample-fps 1.0 \
    --backbone-method dino
```

| 設定 | 値 |
|---|---|
| サンプリングレート | 1 fps（25 fps から間引き） |
| 出力形式 | `video{id}_features.npy`（N × 2048） + `video{id}_labels.npy`（N,） |
| 処理時間 | 約 80 分（80 動画） |
| 出力サイズ | 723 MB（160 ファイル） |

#### Stage 2: BiLSTM の学習

事前抽出済み特徴量でシーケンスモデルを学習する。

```bash
python -m src.cholec_phase.train train \
    --feature-dir ./features \
    --outdir ./models \
    --device cuda \
    --epochs 50 \
    --batch-size 8 \
    --lr 1e-3 \
    --hidden-dim 512 \
    --num-layers 2 \
    --seq-len 300 \
    --stride 150
```

| 設定 | 値 |
|---|---|
| データ分割 | Train: video01-40 の 80%（32 動画）、Val: 20%（8 動画）|
| テストセット | video41-80（40 動画、論文慣例） |
| シーケンス長 | 300 フレーム（300 秒 = 5 分）のスライディングウィンドウ |
| ストライド | 150 フレーム（50% オーバーラップ）|
| 損失関数 | CrossEntropyLoss（ignore_index=-1 でパディング除外） |
| 最適化 | Adam（lr=1e-3, weight_decay=1e-5） |
| スケジューラ | CosineAnnealingLR |
| Mixed precision | fp16（torch.amp.autocast） |
| 勾配クリッピング | max_norm=5.0 |
| 処理時間 | 50 エポック × 1.5 秒 ≒ 約 80 秒 |

### 4.3 推論パイプライン（2 ステップ）

ANALYZER_SPEC.md に準拠した 2 ステップ方式を採用する。

**Step 1: `record_timeseries()` — 動画 → CSV**

フレームを 1 fps でサンプリングし、各フレームに対してフェーズ確率ベクトルを記録する。

| CSV カラム | 説明 |
|---|---|
| `t_sec`, `t_srt` | タイムスタンプ |
| `phase_id` | 予測フェーズ ID (0-6) |
| `phase_name` | 予測フェーズ名 |
| `prob_Preparation` 〜 `prob_GallbladderRetraction` | 各フェーズの確率 |
| `confidence` | 最大確率値 |
| `reader` | フレーム読み込み方式 |

**Step 2: `annotate_phases()` — CSV → JSONL/SRT**

フレームレベルのフェーズ ID を区間にまとめる。

1. **中央値フィルタ**: 10 秒窓でフェーズ ID の最頻値をとる（ノイズ除去）
2. **短区間マージ**: 30 秒未満の区間を前後のフェーズに吸収
3. **区間化**: 連続する同一フェーズを 1 イベントとしてまとめる

---

## 5. できたものとその使い方

### 5.1 実装ファイル構成

```
src/cholec_phase/
├── __init__.py          Cholec80 標準7フェーズ定数
├── models.py            PhaseFeatureExtractor, PhaseRecognitionModel, PhaseModelManager
├── dataset.py           PhaseFeatureDataset, Cholec80アノテーション読み込み
├── train.py             Stage 1 特徴抽出 + Stage 2 LSTM学習 + 評価
└── detector.py          2ステップ推論パイプライン + CLI

models/（学習済み成果物）
└── ./models/
    ├── phase_model_best.pth      ← ベストモデル（val_acc=91.5%）
    ├── phase_model_final.pth     ← 最終モデル
    └── training_history.json     ← 学習曲線データ
```

### 5.2 コマンドライン使用例

#### a) 学習済みモデルで手術動画を解析（一括実行）

```bash
cd video_data_analyser
python -m src.cholec_phase.detector analyze \
    --video /path/to/surgery.mp4 \
    --outdir /path/to/output \
    --model ./models/phase_model_best.pth \
    --fps 1.0 \
    --min-phase-s 30 \
    --smooth-s 10
```

#### b) Step 1 のみ（フェーズ確率 CSV を生成）

```bash
python -m src.cholec_phase.detector timeseries \
    --video /path/to/surgery.mp4 \
    --outdir /path/to/output \
    --model ./models/phase_model_best.pth
```

#### c) Step 2 のみ（閾値・平滑化パラメータを変えて再アノテーション）

```bash
python -m src.cholec_phase.detector annotate \
    --csv /path/to/output/surgery_cholecphaselog.csv \
    --outdir /path/to/output \
    --min-phase-s 60 \
    --smooth-s 15
```

#### d) Cholec80 再学習（データが揃った場合）

```bash
# Stage 1: 特徴抽出（80動画、約80分）
python -m src.cholec_phase.train extract-features \
    --cholec80-dir /path/to/cholec80 \
    --outdir ./features

# Stage 2: 学習（約80秒）
python -m src.cholec_phase.train train \
    --feature-dir ./features \
    --outdir ./models \
    --epochs 50

# 評価（video41-80）
python -m src.cholec_phase.train evaluate \
    --feature-dir ./features \
    --model ./models/phase_model_best.pth
```

### 5.3 出力ファイル

解析結果として以下の 3 ファイルが生成される。

| ファイル | 用途 |
|---|---|
| `{stem}_cholecphaselog.csv` | フレームごとの確率（閾値調整用） |
| `{stem}_surgical_phase_events.jsonl` | フェーズ区間イベント（正本・機械可読） |
| `{stem}_surgical_phase.srt` | Shotcut 用字幕ファイル（目視確認用） |

SRT 形式:
```srt
5
00:03:42,222 --> 00:17:14,033
[phase] CalotTriangleDissection

6
00:17:15,034 --> 00:18:26,105
[phase] ClippingCutting
```

JSONL 形式（1行1イベント）:
```json
{"type": "surgical_phase", "phase_id": 1, "phase_name": "CalotTriangleDissection",
 "mean_confidence": 0.922, "duration_sec": 812.0,
 "start_sec": 222.222, "end_sec": 1034.033,
 "start_srt": "00:03:42,222", "end_srt": "00:17:14,033"}
```

---

## 6. 評価

### 6.1 定量評価（Cholec80 テストセット: video41-80）

| 指標 | 値 |
|---|---|
| **全体精度（フレームレベル）** | **87.1%** |
| Val 精度（video01-40 の 20%） | 91.5% |

**フェーズ別精度:**

| フェーズ | テスト精度 |
|---|---|
| Preparation | 77.4% |
| **CalotTriangleDissection** | **90.8%** |
| ClippingCutting | 67.1% |
| **GallbladderDissection** | **90.7%** |
| GallbladderPackaging | 83.1% |
| CleaningCoagulation | 76.9% |
| **GallbladderRetraction** | **89.1%** |

ClippingCutting（67.1%）と Preparation（77.4%）の精度が相対的に低い。ClippingCutting は短時間（全体の 7.6%）かつ CalotTriangleDissection と視覚的に類似しているため混同しやすい。

### 6.2 定性評価（LapC_EvalDemo.MP4）

動画: `LapC_EvalDemo.MP4`（34 分、2,038 フレーム）

**認識結果:**

| # | 時刻 | フェーズ | 確信度 | 所見 |
|---|---|---|---|---|
| 1 | 0:00 - 0:57 | GallbladderRetraction | 0.54 | 低確信度（術前の非典型映像か） |
| 2 | 0:57 - 1:46 | Preparation | 0.52 | 低確信度 |
| 3 | 1:47 - 2:34 | GallbladderRetraction | 0.53 | 低確信度 |
| 4 | 2:35 - 3:41 | Preparation | 0.65 | |
| **5** | **3:42 - 17:14** | **CalotTriangleDissection** | **0.92** | 長い主要フェーズを正確に検出 |
| 6 | 17:15 - 18:26 | ClippingCutting | 0.90 | |
| 7 | 18:27 - 19:20 | CalotTriangleDissection | 0.90 | クリップ後の追加剥離 |
| 8 | 19:21 - 20:43 | ClippingCutting | 0.89 | |
| 9 | 20:44 - 21:53 | CalotTriangleDissection | 0.86 | |
| 10 | 21:54 - 23:20 | ClippingCutting | 0.87 | |
| **11** | **23:21 - 29:47** | **GallbladderDissection** | **0.92** | 胆嚢剥離を高精度で検出 |
| 12 | 29:47 - 33:16 | CleaningCoagulation | 0.78 | |
| 13 | 33:16 - 33:59 | GallbladderRetraction | 0.58 | 術後映像で低確信度 |

**考察:**

- 術式の中核フェーズ（CalotTriangleDissection、GallbladderDissection）は確信度 0.92 と高精度で認識できている
- 序盤（0〜3 分）は確信度が 0.52〜0.65 と低く不安定。手術開始直後のセットアップや非定型映像がトレーニングデータと乖離している可能性がある
- CalotTriangleDissection と ClippingCutting の反復（6→7→8→9→10）は実際の術式（剥離と切断を繰り返す）と整合しており、フェーズ遷移のパターンが正しく学習されている

### 6.3 既存の教師なし手法との比較

| 指標 | 教師なし（phase_segmenter.py） | 教師あり（本モデル） |
|---|---|---|
| フェーズ定義 | 独自の 7 フェーズ（ヒューリスティック） | Cholec80 標準 7 フェーズ |
| 精度 | 定量評価なし | **87.1%**（フレームレベル） |
| 推論方式 | K-means クラスタリング + ルールベースマッピング | BiLSTM + Softmax |
| アノテーション要否 | 不要 | Cholec80 アノテーション必要 |
| 汎用性 | 動画依存 | Cholec80 で学習済み（胆嚢摘出術に特化） |

---

## 7. 今後の課題

1. **ClippingCutting 精度向上**: データ拡張または重み付き損失でクラス不均衡を緩和
2. **序盤の安定化**: 文脈フレーム数（現在 30 フレーム）を増やし、開始直後の予測精度を改善
3. **術式適用範囲の拡大**: 他の腹腔鏡手術（虫垂切除術等）への転移学習
4. **出血検出との統合**: `surgical_phase` の認識結果を `anomaly_candidate` の文脈情報として利用し、フェーズ依存の閾値設定を実現する

---

*このレポートは `video_data_analyser` プロジェクトの開発記録として `docs/` に保管する。*
