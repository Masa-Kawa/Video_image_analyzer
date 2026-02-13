# プロジェクト概要とディレクトリ構造

## video_data_analyser

手術動画解析のためのデータ処理パイプラインを提供するリポジトリ。
動画解析、データ変換、SRT操作のためのツール群で構成される。

### ディレクトリ構造

```
video_data_analyser/
├── src/                      # ソースコード
│   ├── red/                  # 赤色解析モジュール
│   │   └── redlog.py         # 赤色率計算・イベント抽出ロジック
│   │
│   ├── transnet/             # TransNet連携モジュール
│   │   └── transnet_to_srt.py # TransNet境界データのSRT変換
│   │
│   └── tools/                # 汎用ツール・変換スクリプト
│       ├── csv_to_srt.py     # CSV(時系列) -> SRT(字幕) 変換
│       ├── srt_to_jsonl.py   # SRT(編集済) -> JSONL(データ) 変換
│       ├── jsonl_to_srt.py   # JSONL(データ) -> SRT(字幕) 変換
│       └── merge_srt.py      # 複数SRTのマージ
│
├── docs/                     # ドキュメント
│   ├── dataformat.md         # データフォーマット定義 (JSONL/SRT/CSV)
│   ├── tutorial.md           # 具体的な使用例・ワークフロー
│   └── overview.md           # (本書) ディレクトリ構造
│
├── tests/                    # ユニットテスト
│   ├── test_redlog.py
│   ├── test_transnet_to_srt.py
│   ├── test_csv_to_srt.py
│   └── ...
│
├── requirements.txt          # 依存ライブラリ (opencv-python, numpy, av)
└── README.md                 # プロジェクトのトップレベル説明
```

### モジュールの役割

#### 1. 解析モジュール (`src/red`, `src/transnet`)
動画や外部AIの出力から、生の解析データを生成する。
- **redlog**: 動画 → CSV（時系列） → JSONL/SRT（イベント）
- **transnet**: 外部JSONL → SRT（境界）

#### 2. 変換ツール (`src/tools`)
異なるデータフォーマット間の相互変換を担当する。
特に **SRT** を介した人間による介入（Human-in-the-loop）を支援するための変換機能が充実している。

- **csv_to_srt**: 数値データの可視化
- **srt_to_jsonl** / **jsonl_to_srt**: 編集ツール(Shotcut)とデータ正本(JSONL)のブリッジ

### 設計思想

- **Unix哲学**: 一つのツールは一つのことをうまくやる。パイプラインで繋ぐ。
- **Single Source of Truth**: データの実体は JSONL/CSV にあり、SRT はあくまで「ビュー」である。
- **Human-in-the-loop**: アルゴリズムは完璧ではないため、人間が修正しやすいワークフローを提供する。
