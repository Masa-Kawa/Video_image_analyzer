# チュートリアル: 手術動画解析ワークフロー

このチュートリアルでは、実際の動画を使用して赤色解析からイベントアノテーション、そしてShotcutでの確認までの一連の流れを体験します。

## シナリオ: 出血イベントの検出と確認

ある症例動画 (`case001.mp4`) に対して、以下の手順で解析を行います。

1. **基本解析**: 赤色率の時系列データを取得する
2. **可視化**: 解析結果をSRT字幕で動画に重ねて確認する
3. **イベント抽出**: 閾値を調整して出血候補を検出する
4. **マニュアル修正**: Shotcutで誤検知を修正し、データを更新する

---

## Step 1: 赤色率の時系列記録 (Time Series Logging)

まずは動画全体をスキャンし、フレームごとの赤色率をCSVに記録します。
この段階ではイベント判定は行いません。

```bash
# 出力先ディレクトリを作成
mkdir -p output

# 解析実行（デフォルト5fps）
python -m src.red.redlog timeseries \
    --video case001.mp4 \
    --outdir output
```

**出力結果**:
- `output/case001_redlog.csv`: 全フレームの赤色率ログ

---

## Step 2: 解析データの可視化 (Visualization)

CSVの数値だけでは直感的にわかりにくいため、SRT字幕に変換して動画上で確認します。
また、グラフ（PNG画像）として全体の傾向を一目で把握することもできます。

```bash
# CSV -> SRT変換（動画上で数値を確認）
python -m src.tools.csv_to_srt \
    --in-csv output/case001_redlog.csv \
    --out-srt output/case001_metrics.srt

# CSV -> PNGグラフ（赤色率と変化量の時系列プロット）
python -m src.tools.plot_redlog \
    --in-csv output/case001_redlog.csv \
    --out-png output/case001_plot.png \
    --thr 0.03
```

**SRTでの確認方法**:
1. Shotcutで `case001.mp4` を開く
2. `output/case001_metrics.srt` を字幕トラックに追加する
3. 動画を再生すると、画面下部にリアルタイムで赤色率（`red=...`）が表示されます

**PNGグラフ**: `output/case001_plot.png` を開くと、上パネルに赤色率（red_ratio）、下パネルに変化量（smooth_delta）と閾値ラインが表示されます。

---

## Step 3: 出血候補の抽出 (Event Annotation)

赤色率の変化量（`smooth_delta`）に基づいて、出血と思われる箇所を抽出します。
CSVがあるため、動画を再読み込みせずに一瞬で完了します。

```bash
# 閾値 0.03 で抽出
python -m src.red.redlog annotate \
    --csv output/case001_redlog.csv \
    --outdir output \
    --thr 0.03
```

**出力結果**:
- `output/case001_bleed.srt`: 検出されたイベント区間の字幕
- `output/case001_events.jsonl`: イベントデータの正本

Shotcutで確認し、もし検出漏れが多い場合は閾値を下げて再実行します。

```bash
# 閾値を 0.02 に下げて再実行（上書き）
python -m src.red.redlog annotate \
    --csv output/case001_redlog.csv \
    --outdir output \
    --thr 0.02
```

---

## Step 4: マニュアル修正とデータ更新 (Human-in-the-loop)

自動検出には限界があります。最後は人の目で確認・修正します。

1. Shotcutで `output/case001_bleed.srt` を開く。
2. 誤検知（出血ではないのに検出された箇所）の字幕ブロックを削除する。
3. 区間がずれている場合は、字幕ブロックの端をドラッグして調整する。
4. 修正が終わったら、SRTを保存する（例: `case001_bleed_edited.srt`）。

### 修正結果をデータ正本（JSONL）に反映

編集後のSRTからJSONLを更新します。

```bash
python -m src.tools.srt_to_jsonl \
    --in-srt output/case001_bleed_edited.srt \
    --out-jsonl output/case001_events_final.jsonl
```

これで、機械による一次スクリーニングと、人間による確定を経た高品質なデータセットが完成しました。
