"""
Optical Flow Divergence 出血検出モジュール (bleed_flow)

腹腔鏡手術動画において、オプティカルフローの発散（divergence）を計算し、
「放射状に広がる動き」を出血候補として検出する。

原理:
  出血は出血源から放射状に広がる。
  オプティカルフローベクトル場 F = (u, v) の空間勾配から発散を計算:
    div(F) = ∂u/∂x + ∂v/∂y

  正の発散 = 湧き出し（expansion） → 出血の本質的特徴
  負の発散 = 吸い込み（contraction）
  ゼロ付近 = 並進移動のみ

  カメラ移動による見かけの発散を除去するため:
    1. フレーム全体の中央値発散（global divergence）を推定
    2. 局所発散から global divergence を差し引く
    3. 正の残差 = 真の局所的湧き出し = 出血候補

先行モジュールとの違い:
  - redlog.py: HSV色空間の赤色率 → 「赤さ」= 炎症組織も誤検出
  - bleed_detector.py: 新規赤化画素 → 「背景が安定」前提が手術動画で破綻
  - bleed_spread.py: グリッド拡散 → 色依存 + 粗粒度
  - bleed_flow.py (本モジュール): オプティカルフロー → 「動き」のみ

2段階の処理:
  Step 1 - record_timeseries(): 動画→CSV（発散スコアの時系列記録）
  Step 2 - annotate_bleed():    CSV→JSONL/SRT（閾値ベースの出血アノテーション）

出力: CSV（発散スコアログ）、SRT（出血候補イベント）、JSONL（イベント正本）

フローバックエンド:
  --flow-model raft_small (デフォルト): GPU-RAFT（torchvision）
  --flow-model raft_large: 高精度RAFT（低速）
  --flow-model farneback: CPU Farneback（後方互換）
  --device cpu: 強制CPU（farneback にフォールバック）

Refs:
  - Farneback G. "Two-Frame Motion Estimation Based on Polynomial Expansion" (2003)
  - Teed & Deng, "RAFT: Recurrent All-Pairs Field Transforms for Optical Flow" (2020)
  - torchvision.models.optical_flow
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    import torch
    from torchvision.models.optical_flow import raft_large, raft_small
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from src.core.time_utils import format_srt_time
from src.red.redlog import (
    make_circular_roi,
    iter_frames,
    smooth_center,
)


# ---------------------------------------------------------------------------
# オプティカルフロー発散の計算
# ---------------------------------------------------------------------------

def compute_flow_divergence(
    flow: np.ndarray,
    roi_mask: Optional[np.ndarray] = None,
) -> Tuple[float, float, np.ndarray]:
    """
    オプティカルフローフィールドから発散（divergence）を計算する。

    Args:
        flow: (H, W, 2) オプティカルフロー (u, v)
        roi_mask: (H, W) bool ROIマスク。Noneなら全領域。

    Returns:
        (mean_positive_divergence, max_divergence, divergence_map)
        - mean_positive_divergence: 正の発散のみの平均（負は0扱い）
        - max_divergence: 発散の最大値
        - divergence_map: (H, W) float64 発散マップ（デバッグ用）
    """
    # Sobel微分で勾配を計算
    # ∂u/∂x: flow[:,:,0] の x方向勾配
    du_dx = cv2.Sobel(flow[:, :, 0], cv2.CV_64F, 1, 0, ksize=3)
    # ∂v/∂y: flow[:,:,1] の y方向勾配
    dv_dy = cv2.Sobel(flow[:, :, 1], cv2.CV_64F, 0, 1, ksize=3)

    divergence = du_dx + dv_dy  # (H, W) = ∂u/∂x + ∂v/∂y

    if roi_mask is not None:
        # ROI外は無視
        divergence = divergence * roi_mask.astype(np.float64)
        n_pixels = int(np.count_nonzero(roi_mask))
        if n_pixels == 0:
            return 0.0, 0.0, divergence
        # global divergence = ROI内の中央値（平均よりロバスト）
        global_div = np.median(divergence[roi_mask])
    else:
        n_pixels = divergence.size
        global_div = np.median(divergence)

    # 局所発散 = 生発散 - global divergence（カメラ移動補正）
    local_divergence = divergence - global_div

    # 正の発散のみを集計（負 = 収縮、非出血）
    positive_div = np.maximum(local_divergence, 0.0)
    mean_positive = float(np.sum(positive_div) / n_pixels)
    max_div = float(np.max(local_divergence))

    return mean_positive, max_div, local_divergence


def compute_flow_divergence_detailed(
    flow: np.ndarray,
    roi_mask: Optional[np.ndarray] = None,
    percentile_threshold: float = 90.0,
) -> dict:
    """
    オプティカルフロー発散の詳細統計を計算する。

    Args:
        flow: (H, W, 2) オプティカルフロー
        roi_mask: ROIマスク
        percentile_threshold: 上位パーセンタイル閾値（デフォルト90）

    Returns:
        dict with keys:
        - mean_positive_div: 正の発散平均
        - max_div: 最大発散
        - p90_div: 90パーセンタイル発散
        - area_high_div: 上位パーセンタイルを超える画素数
        - global_div: 全体の中央値発散（カメラ移動量の代理）
    """
    du_dx = cv2.Sobel(flow[:, :, 0], cv2.CV_64F, 1, 0, ksize=3)
    dv_dy = cv2.Sobel(flow[:, :, 1], cv2.CV_64F, 0, 1, ksize=3)
    divergence = du_dx + dv_dy

    if roi_mask is not None:
        divergence_masked = divergence[roi_mask]
        n_total = int(np.count_nonzero(roi_mask))
        if n_total == 0:
            return {
                "mean_positive_div": 0.0, "max_div": 0.0,
                "p90_div": 0.0, "area_high_div": 0,
                "global_div": 0.0,
            }
    else:
        divergence_masked = divergence.flatten()
        n_total = divergence_masked.size

    global_div = float(np.median(divergence_masked))
    local_div = divergence_masked - global_div

    positive = np.maximum(local_div, 0.0)
    mean_positive = float(np.mean(positive))
    max_div = float(np.max(local_div))
    p90 = float(np.percentile(local_div, percentile_threshold))
    area_high = int(np.sum(local_div > np.percentile(local_div, percentile_threshold)))

    return {
        "mean_positive_div": mean_positive,
        "max_div": max_div,
        "p90_div": p90,
        "area_high_div": area_high,
        "global_div": global_div,
    }


# ---------------------------------------------------------------------------
# フレームイテレータ (flow用: 連続2フレーム必要)
# ---------------------------------------------------------------------------

def iter_frame_pairs(video_path: str, fps: float):
    """
    連続する2フレームのペアをイテレートする。

    オプティカルフロー計算には前フレームと現フレームのペアが必要。

    Yields:
        (t_sec, prev_bgr, curr_bgr, reader_name)
        最初のフレームでは prev_bgr=None でスキップされる。
    """
    prev_frame = None
    prev_t = 0.0
    reader_name = "unknown"

    for t_sec, bgr, reader in iter_frames(video_path, fps):
        reader_name = reader
        if prev_frame is not None:
            yield prev_t, prev_frame, bgr, reader_name
        prev_frame = bgr
        prev_t = t_sec


# ---------------------------------------------------------------------------
# RAFT GPU オプティカルフロー
# ---------------------------------------------------------------------------

_RAFT_MODEL_CACHE: dict = {}


def _load_raft_model(model_name: str, device: str):
    """RAFTモデルをロード（初回のみ、以降はキャッシュから返す）。"""
    key = (model_name, device)
    if key not in _RAFT_MODEL_CACHE:
        print(f"RAFTモデルをロード中: {model_name} → {device}", file=sys.stderr)
        if model_name == "raft_large":
            model = raft_large(pretrained=True)
        else:
            model = raft_small(pretrained=True)
        _RAFT_MODEL_CACHE[key] = model.eval().to(device)
    return _RAFT_MODEL_CACHE[key]


def _bgr_list_to_tensor(frames_bgr: list, device: str) -> "torch.Tensor":
    """BGR uint8 フレームリスト → RAFT 用 float テンソル (N, 3, H, W) in [0,1]。"""
    tensors = []
    for bgr in frames_bgr:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        tensors.append(t)
    return torch.stack(tensors).to(device)


def _raft_flow_batch(
    prev_frames: list,
    curr_frames: list,
    model,
    device: str,
) -> list:
    """
    RAFT でフレームペアのバッチ処理を行い (H, W, 2) numpy 配列のリストを返す。

    出力形状・単位は cv2.calcOpticalFlowFarneback と同一（ピクセル変位）。
    RAFT 出力 (N, 2, H, W) を (N, H, W, 2) に変換して返す。
    """
    prev_batch = _bgr_list_to_tensor(prev_frames, device)
    curr_batch = _bgr_list_to_tensor(curr_frames, device)
    with torch.no_grad():
        flow_preds = model(prev_batch, curr_batch)
    # flow_preds[-1]: (N, 2, H, W) → (N, H, W, 2)
    flows_np = flow_preds[-1].permute(0, 2, 3, 1).cpu().numpy()
    return [flows_np[i] for i in range(len(prev_frames))]


# ---------------------------------------------------------------------------
# Step 1: 時系列記録
# ---------------------------------------------------------------------------

def record_timeseries(
    video_path: str,
    outdir: str,
    fps: float = 5.0,
    roi_margin: float = 0.08,
    no_roi: bool = False,
    downsample_ratio: float = 0.5,
    flow_scale: float = 0.5,
    device: str = "cuda",
    flow_model: str = "raft_small",
    batch_size: int = 8,
) -> dict:
    """
    Step 1: 動画からオプティカルフロー発散の時系列をCSVに記録する。

    Args:
        video_path: 入力動画ファイルパス
        outdir: 出力ディレクトリ
        fps: サンプリングFPS（デフォルト5）
        roi_margin: 円形ROIマージン
        no_roi: TrueならROIを無効化
        downsample_ratio: フロー計算時の入力フレーム縮小率（高速化）
        flow_scale: Farnebackの pyr_scale パラメータ（farneback使用時のみ）
        device: 演算デバイス ("cuda" | "cpu")
        flow_model: フローバックエンド ("raft_small" | "raft_large" | "farneback")
        batch_size: RAFT バッチサイズ（RAFT使用時のみ）

    Returns:
        {"csv": CSVファイルパス}
    """
    # フローバックエンドを決定
    use_raft = (
        _TORCH_AVAILABLE
        and flow_model in ("raft_small", "raft_large")
        and device != "cpu"
    )
    if use_raft:
        if not torch.cuda.is_available():
            print("警告: CUDA使用不可。Farnebackにフォールバック。", file=sys.stderr)
            use_raft = False
        else:
            raft_model = _load_raft_model(flow_model, device)

    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    stem = Path(video_path).stem
    reader_name = "unknown"
    roi_mask: Optional[np.ndarray] = None
    roi_initialized = False

    # --- 書き出し準備 ---
    csv_path = out_path / f"{stem}_flowlog.csv"
    f = open(csv_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(f)
    writer.writerow([
        "t_sec", "t_srt",
        "mean_positive_div", "max_div", "p90_div",
        "area_high_div", "global_div", "reader"
    ])

    n_samples = 0

    def _write_stats_row(t_sec: float, stats: dict, rdr: str) -> None:
        writer.writerow([
            f"{t_sec:.3f}",
            format_srt_time(t_sec),
            f"{stats['mean_positive_div']:.8f}",
            f"{stats['max_div']:.8f}",
            f"{stats['p90_div']:.8f}",
            stats['area_high_div'],
            f"{stats['global_div']:.8f}",
            rdr,
        ])

    def _prepare_small_frames(prev_bgr, curr_bgr):
        """ダウンサンプル + ROI縮小。"""
        if downsample_ratio < 1.0:
            small_h = int(prev_bgr.shape[0] * downsample_ratio)
            small_w = int(prev_bgr.shape[1] * downsample_ratio)
            prev_s = cv2.resize(prev_bgr, (small_w, small_h))
            curr_s = cv2.resize(curr_bgr, (small_w, small_h))
            if roi_mask is not None:
                roi_s = cv2.resize(
                    roi_mask.astype(np.uint8), (small_w, small_h)
                ).astype(bool)
            else:
                roi_s = None
        else:
            prev_s, curr_s = prev_bgr, curr_bgr
            roi_s = roi_mask
        return prev_s, curr_s, roi_s

    try:
        if use_raft:
            # ---- RAFT パス: バッチ処理 ----
            batch_t: List[float] = []
            batch_prev: List[np.ndarray] = []
            batch_curr: List[np.ndarray] = []
            batch_roi: List[Optional[np.ndarray]] = []
            batch_reader: List[str] = []

            def _flush_raft_batch() -> None:
                nonlocal n_samples
                flows = _raft_flow_batch(batch_prev, batch_curr, raft_model, device)
                for t, flow, roi_s, rdr in zip(batch_t, flows, batch_roi, batch_reader):
                    stats = compute_flow_divergence_detailed(flow, roi_s)
                    _write_stats_row(t, stats, rdr)
                    n_samples += 1
                    if n_samples % 100 == 0:
                        print(f"  ... {n_samples} サンプル処理済み", file=sys.stderr)
                batch_t.clear()
                batch_prev.clear()
                batch_curr.clear()
                batch_roi.clear()
                batch_reader.clear()

            for t_sec, prev_bgr, curr_bgr, reader in iter_frame_pairs(video_path, fps):
                reader_name = reader

                if not roi_initialized:
                    h, w = prev_bgr.shape[:2]
                    if not no_roi:
                        roi_mask = make_circular_roi(h, w, margin=roi_margin)
                    roi_initialized = True

                prev_s, curr_s, roi_s = _prepare_small_frames(prev_bgr, curr_bgr)

                batch_t.append(t_sec)
                batch_prev.append(prev_s)
                batch_curr.append(curr_s)
                batch_roi.append(roi_s)
                batch_reader.append(reader)

                if len(batch_prev) >= batch_size:
                    _flush_raft_batch()

            if batch_prev:
                _flush_raft_batch()

        else:
            # ---- Farneback パス: 元のストリーミング処理（後方互換） ----
            for t_sec, prev_bgr, curr_bgr, reader in iter_frame_pairs(video_path, fps):
                reader_name = reader

                if not roi_initialized:
                    h, w = prev_bgr.shape[:2]
                    if not no_roi:
                        roi_mask = make_circular_roi(h, w, margin=roi_margin)
                    roi_initialized = True

                prev_s, curr_s, roi_s = _prepare_small_frames(prev_bgr, curr_bgr)

                # グレースケール化（Farneback はグレースケール入力）
                prev_gray = cv2.cvtColor(prev_s, cv2.COLOR_BGR2GRAY)
                curr_gray = cv2.cvtColor(curr_s, cv2.COLOR_BGR2GRAY)

                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, curr_gray,
                    None,
                    pyr_scale=flow_scale,
                    levels=3,
                    winsize=15,
                    iterations=3,
                    poly_n=5,
                    poly_sigma=1.2,
                    flags=0,
                )

                stats = compute_flow_divergence_detailed(flow, roi_s)
                _write_stats_row(t_sec, stats, reader_name)
                n_samples += 1
                if n_samples % 100 == 0:
                    print(f"  ... {n_samples} サンプル処理済み", file=sys.stderr)

    finally:
        f.close()

    if n_samples == 0:
        print("警告: フレームが取得できませんでした。", file=sys.stderr)
        return {}

    backend = f"RAFT({flow_model})" if use_raft else "Farneback"
    print(f"CSV  : {csv_path}  ({n_samples} サンプル, backend={backend})")
    return {"csv": str(csv_path)}


# ---------------------------------------------------------------------------
# CSV読み込み
# ---------------------------------------------------------------------------

def read_flowlog_csv(csv_path: str) -> dict:
    """
    発散ログCSVを読み込む。

    Args:
        csv_path: CSVファイルパス

    Returns:
        {
            "times": [...], "mean_positive_divs": [...],
            "max_divs": [...], "p90_divs": [...],
            "area_high_divs": [...], "global_divs": [...],
            "reader": "...", "fps": float,
        }
    """
    times: List[float] = []
    mean_positive_divs: List[float] = []
    max_divs: List[float] = []
    p90_divs: List[float] = []
    area_high_divs: List[int] = []
    global_divs: List[float] = []
    reader_name = "unknown"

    with open(csv_path, "r", encoding="utf-8") as f:
        reader_obj = csv.DictReader(f)
        for row in reader_obj:
            times.append(float(row["t_sec"]))
            mean_positive_divs.append(float(row["mean_positive_div"]))
            max_divs.append(float(row["max_div"]))
            p90_divs.append(float(row.get("p90_div", 0)))
            area_high_divs.append(int(row.get("area_high_div", 0)))
            global_divs.append(float(row.get("global_div", 0)))
            reader_name = row.get("reader", "unknown")

    if len(times) >= 2:
        est_fps = 1.0 / (times[1] - times[0])
    else:
        est_fps = 5.0

    return {
        "times": times,
        "mean_positive_divs": mean_positive_divs,
        "max_divs": max_divs,
        "p90_divs": p90_divs,
        "area_high_divs": area_high_divs,
        "global_divs": global_divs,
        "reader": reader_name,
        "fps": est_fps,
    }


# ---------------------------------------------------------------------------
# 出血イベント抽出
# ---------------------------------------------------------------------------

def extract_flow_bleed_events(
    times: List[float],
    scores: List[float],
    thr: float,
    k_s: float,
    fps: float,
    score_name: str = "flow_divergence",
) -> List[dict]:
    """
    時系列スコアから閾値ベースで出血イベントを抽出する。

    既存 redlog.py の extract_bleed_events() と同様のロジック。

    Args:
        times: 時刻リスト
        scores: スコアリスト
        thr: 閾値
        k_s: 連続条件（秒）— この秒数以上連続した超過を1イベントとする
        fps: サンプリングFPS
        score_name: スコア名（JSONL内のキー）

    Returns:
        イベントリスト [{type, metric, thr, start_sec, end_sec, ...}, ...]
    """
    events = []
    k = max(1, int(k_s * fps))
    i = 0
    n = len(times)

    while i < n:
        if scores[i] > thr:
            j = i
            while j < n and scores[j] > thr:
                j += 1
            duration = j - i
            if duration >= k:
                events.append({
                    "type": "bleed_candidate",
                    "metric": score_name,
                    "thr": thr,
                    "start_sec": round(times[i], 3),
                    "end_sec": round(times[min(j, n - 1)], 3),
                    "duration_sec": round(times[min(j, n - 1)] - times[i], 3),
                    "start_srt": format_srt_time(times[i]),
                    "end_srt": format_srt_time(times[min(j, n - 1)]),
                    "max_score": round(max(scores[i:j]), 8),
                    "mean_score": round(sum(scores[i:j]) / duration, 8),
                })
            i = j
        else:
            i += 1

    return events


# ---------------------------------------------------------------------------
# Step 2: 出血アノテーション
# ---------------------------------------------------------------------------

def annotate_bleed(
    csv_path: str,
    outdir: str,
    thr: float = 0.005,
    k_s: float = 3.0,
    use_metric: str = "mean_positive_div",
) -> dict:
    """
    Step 2: 発散ログCSVから出血イベントを抽出し、SRT/JSONLを出力する。

    Args:
        csv_path: flowlog CSVファイルパス
        outdir: 出力ディレクトリ
        thr: 出血候補閾値
        k_s: 連続条件（秒）
        use_metric: 使用するスコア指標
                    "mean_positive_div" | "max_div" | "p90_div"

    Returns:
        {"srt": SRTパス, "jsonl": JSONLパス}
    """
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    data = read_flowlog_csv(csv_path)

    stem = Path(csv_path).stem.replace("_flowlog", "")

    if use_metric == "max_div":
        scores = data["max_divs"]
    elif use_metric == "p90_div":
        scores = data["p90_divs"]
    else:
        scores = data["mean_positive_divs"]

    events = extract_flow_bleed_events(
        data["times"], scores, thr=thr, k_s=k_s,
        fps=data["fps"], score_name=use_metric,
    )

    # --- JSONL出力 ---
    jsonl_path = out_path / f"{stem}_bleed_flow_events.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for ev in events:
            json.dump(ev, f, ensure_ascii=False)
            f.write("\n")

    # --- SRT出力 ---
    srt_path = out_path / f"{stem}_bleed_flow.srt"
    with open(srt_path, "w", encoding="utf-8") as f:
        for idx, ev in enumerate(events, 1):
            f.write(f"{idx}\n")
            f.write(f"{ev['start_srt']} --> {ev['end_srt']}\n")
            # 1行目: 人間向けタグ
            f.write(f"[bleed_flow] {use_metric} over threshold\n")
            # 2行目: 機械向けJSON
            meta = {
                "type": ev["type"],
                "metric": ev["metric"],
                "thr": ev["thr"],
                "max_score": ev["max_score"],
                "mean_score": ev["mean_score"],
            }
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")
            f.write("\n")

    print(f"SRT  : {srt_path}  ({len(events)} イベント)")
    print(f"JSONL: {jsonl_path}")

    return {"srt": str(srt_path), "jsonl": str(jsonl_path), "events": len(events)}


# ---------------------------------------------------------------------------
# 統合パイプライン
# ---------------------------------------------------------------------------

def analyze_video(
    video_path: str,
    outdir: str,
    fps: float = 5.0,
    roi_margin: float = 0.08,
    no_roi: bool = False,
    thr: float = 0.005,
    k_s: float = 3.0,
    use_metric: str = "mean_positive_div",
    downsample_ratio: float = 0.5,
    device: str = "cuda",
    flow_model: str = "raft_small",
    batch_size: int = 8,
) -> dict:
    """
    動画を解析し、CSV・SRT・JSONLを出力する（2ステップの一括実行）。

    Args:
        video_path: 入力動画ファイルパス
        outdir: 出力ディレクトリ
        fps: サンプリングFPS（デフォルト5）
        roi_margin: 円形ROIマージン
        no_roi: TrueならROIを無効化
        thr: 出血候補閾値
        k_s: 連続条件（秒）
        use_metric: スコア指標
        downsample_ratio: フレーム縮小率
        device: 演算デバイス ("cuda" | "cpu")
        flow_model: フローバックエンド ("raft_small" | "raft_large" | "farneback")
        batch_size: RAFT バッチサイズ

    Returns:
        出力ファイルパスの辞書
    """
    # Step 1
    result1 = record_timeseries(
        video_path=video_path,
        outdir=outdir,
        fps=fps,
        roi_margin=roi_margin,
        no_roi=no_roi,
        downsample_ratio=downsample_ratio,
        device=device,
        flow_model=flow_model,
        batch_size=batch_size,
    )
    if not result1:
        return {}

    # Step 2
    result2 = annotate_bleed(
        csv_path=result1["csv"],
        outdir=outdir,
        thr=thr,
        k_s=k_s,
        use_metric=use_metric,
    )

    return {**result1, **result2}


# ---------------------------------------------------------------------------
# CLI エントリポイント
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Optical Flow Divergence 出血検出"
    )
    sub = parser.add_subparsers(dest="command")

    # --- timeseries ---
    p_ts = sub.add_parser("timeseries", help="Step 1: 発散時系列をCSVに記録")
    p_ts.add_argument("--video", required=True)
    p_ts.add_argument("--outdir", required=True)
    p_ts.add_argument("--fps", type=float, default=5.0)
    p_ts.add_argument("--roi-margin", type=float, default=0.08)
    p_ts.add_argument("--no-roi", action="store_true")
    p_ts.add_argument("--downsample", type=float, default=0.5,
                      help="フレーム縮小率（0.5=半分, 1.0=等倍）")
    p_ts.add_argument("--device", default="cuda",
                      help="演算デバイス (cuda | cpu)")
    p_ts.add_argument("--flow-model", default="raft_small",
                      choices=["raft_small", "raft_large", "farneback"],
                      help="フローバックエンド (デフォルト: raft_small)")
    p_ts.add_argument("--batch-size", type=int, default=8,
                      help="RAFT バッチサイズ（RAFT使用時のみ）")

    # --- annotate ---
    p_an = sub.add_parser("annotate", help="Step 2: CSVから出血イベントを抽出")
    p_an.add_argument("--csv", required=True)
    p_an.add_argument("--outdir", required=True)
    p_an.add_argument("--thr", type=float, default=0.005)
    p_an.add_argument("--k-s", type=float, default=3.0)
    p_an.add_argument("--metric", default="mean_positive_div",
                      choices=["mean_positive_div", "max_div", "p90_div"])

    # --- analyze (shortcut) ---
    p_al = sub.add_parser("analyze", help="一括実行 (timeseries → annotate)")
    p_al.add_argument("--video", required=True)
    p_al.add_argument("--outdir", required=True)
    p_al.add_argument("--fps", type=float, default=5.0)
    p_al.add_argument("--roi-margin", type=float, default=0.08)
    p_al.add_argument("--no-roi", action="store_true")
    p_al.add_argument("--thr", type=float, default=0.005)
    p_al.add_argument("--k-s", type=float, default=3.0)
    p_al.add_argument("--metric", default="mean_positive_div",
                      choices=["mean_positive_div", "max_div", "p90_div"])
    p_al.add_argument("--downsample", type=float, default=0.5)
    p_al.add_argument("--device", default="cuda",
                      help="演算デバイス (cuda | cpu)")
    p_al.add_argument("--flow-model", default="raft_small",
                      choices=["raft_small", "raft_large", "farneback"],
                      help="フローバックエンド (デフォルト: raft_small)")
    p_al.add_argument("--batch-size", type=int, default=8,
                      help="RAFT バッチサイズ（RAFT使用時のみ）")

    return parser


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "timeseries":
        record_timeseries(
            video_path=args.video,
            outdir=args.outdir,
            fps=args.fps,
            roi_margin=args.roi_margin,
            no_roi=args.no_roi,
            downsample_ratio=args.downsample,
            device=args.device,
            flow_model=args.flow_model,
            batch_size=args.batch_size,
        )
    elif args.command == "annotate":
        annotate_bleed(
            csv_path=args.csv,
            outdir=args.outdir,
            thr=args.thr,
            k_s=args.k_s,
            use_metric=args.metric,
        )
    elif args.command == "analyze":
        analyze_video(
            video_path=args.video,
            outdir=args.outdir,
            fps=args.fps,
            roi_margin=args.roi_margin,
            no_roi=args.no_roi,
            thr=args.thr,
            k_s=args.k_s,
            use_metric=args.metric,
            downsample_ratio=args.downsample,
            device=args.device,
            flow_model=args.flow_model,
            batch_size=args.batch_size,
        )
    else:
        parser.print_help()
