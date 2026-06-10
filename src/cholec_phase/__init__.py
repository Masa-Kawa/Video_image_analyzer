"""
Cholec80ベース教師あり手術フェーズ認識モジュール

Cholec80データセット（80本の腹腔鏡下胆嚢摘出術動画、フレームレベルのフェーズアノテーション付き）
を用いて学習した手術フェーズ認識モデルによる評価器。

Cholec80 標準7フェーズ:
  P1: Preparation
  P2: CalotTriangleDissection
  P3: ClippingCutting
  P4: GallbladderDissection
  P5: GallbladderPackaging
  P6: CleaningCoagulation
  P7: GallbladderRetraction

アーキテクチャ:
  - Backbone: ResNet50 (SelfSupSurg DINO pretrained on Cholec80, frozen)
  - Temporal: Bidirectional LSTM
  - Head: FC → 7クラス分類

2段階の処理:
  Step 1 - record_timeseries(): 動画 → CSV（フレームごとのフェーズ確率）
  Step 2 - annotate_phases():   CSV → JSONL/SRT（フェーズ区間アノテーション）
"""

# Cholec80 標準フェーズ定義
CHOLEC80_PHASES = [
    "Preparation",
    "CalotTriangleDissection",
    "ClippingCutting",
    "GallbladderDissection",
    "GallbladderPackaging",
    "CleaningCoagulation",
    "GallbladderRetraction",
]

NUM_PHASES = len(CHOLEC80_PHASES)
