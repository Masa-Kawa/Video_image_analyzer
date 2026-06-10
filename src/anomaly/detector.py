"""
Self-Supervised Video Anomaly Detection モジュール

正常フレームだけで学習した ConvLSTM Autoencoder により、
出血・煙・カメラ衝突等を「正常分布からの逸脱」として検出する。

2段階の処理:
  Step 1 - record_timeseries(): 動画→CSV（フレームごとの異常スコア）
  Step 2 - annotate_anomaly():  CSV→JSONL/SRT（異常イベント検出）

特徴:
  - ConvLSTM Autoencoder による次フレーム予測
  - 予測誤差 = 異常スコア (MSE)
  - 色変化スコア（赤色領域の急変検出）も併用
  - 学習済み重みなしでもピクセル差分フォールバックで動作
  - Mixed precision (fp16) 対応

出力: CSV（時系列ログ）、JSONL（イベント正本）、SRT（字幕）
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch

from src.core.time_utils import format_srt_time
from src.red.redlog import iter_frames, make_circular_roi, smooth_center
from src.anomaly.models import (
    AnomalyModelManager,
    preprocess_frame_anomaly,
    ANOMALY_FRAME_SIZE,
)
from src.tools.proxy_manager import ProxyManager

# SelfSupSurg デフォルト設定
DEFAULT_SELFSUP_METHOD = "dino"

# プロキシ解像度
PROXY_RESOLUTION = "480p"


# ---------------------------------------------------------------------------
# 色変化スコア（赤色領域の急変）
# ---------------------------------------------------------------------------


def compute_color_change_score(
    prev_bgr: np.ndarray,
    curr_bgr: np.ndarray,
    roi_mask: Optional[np.ndarray] = None,
) -> float:
    """
    前後フレーム間の赤色チャンネル変化量を計算する。

    出血は赤色の急増を伴うため、R チャンネルの変化量を
    異常スコアの補助指標として使用する。

    Returns:
        float: 赤色変化スコア (0.0〜1.0)
    """
    # HSV の H チャンネルで赤色領域を抽出
    hsv_prev = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2HSV)
    hsv_curr = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2HSV)

    # 赤色マスク（H: 0-10 or 170-179）
    def red_mask(hsv: np.ndarray) -> np.ndarray:
        lo1 = cv2.inRange(hsv, np.array([0, 50, 40]), np.array([10, 255, 255]))
        lo2 = cv2.inRange(hsv, np.array([170, 50, 40]), np.array([179, 255, 255]))
        return (lo1 | lo2).astype(np.float32) / 255.0

    prev_red = red_mask(hsv_prev)
    curr_red = red_mask(hsv_curr)

    diff = np.abs(curr_red - prev_red)

    if roi_mask is not None:
        roi_f = roi_mask.astype(np.float32)
        total = float(np.sum(roi_f))
        if total == 0:
            return 0.0
        return float(np.sum(diff * roi_f) / total)

    total = diff.shape[0] * diff.shape[1]
    if total == 0:
        return 0.0
    return float(np.sum(diff) / total)


# ---------------------------------------------------------------------------
# プロキシ動画生成
# ---------------------------------------------------------------------------


def _ensure_proxy(
    video_path: str,
    outdir: str,
    resolution: str = PROXY_RESOLUTION,
) -> str:
    """プロキシ動画を生成し、そのパスを返す。既存なら再利用する。"""
    import subprocess

    proxy_dir = Path(outdir)
    proxy_dir.mkdir(parents=True, exist_ok=True)

    mgr = ProxyManager(proxy_dir=proxy_dir)
    proxy_path = mgr.get_proxy_path(video_path, resolution)

    if mgr.proxy_exists(video_path, resolution):
        print(f"プロキシ既存: {proxy_path}")
        return str(proxy_path)

    target_w, target_h = ProxyManager.RESOLUTIONS[resolution]
    print(f"プロキシ生成中 ({resolution} = {target_w}x{target_h}): {video_path}")

    scale_filter = (
        f"scale={target_w}:{target_h}"
        f":force_original_aspect_ratio=decrease,"
        f"pad=ceil(iw/2)*2:ceil(ih/2)*2"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", scale_filter,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-an",
        str(proxy_path),
    ]

    subprocess.run(cmd, check=True, capture_output=True)
    print(f"プロキシ生成完了: {proxy_path}")
    return str(proxy_path)


# ---------------------------------------------------------------------------
# Step 1: 時系列記録
# ---------------------------------------------------------------------------


def record_timeseries(
    video_path: str,
    outdir: str,
    fps: float = 5.0,
    roi_margin: float = 0.08,
    no_roi: bool = False,
    smooth_s: float = 5.0,
    device: str = "cuda",
    weights_path: Optional[str] = None,
    seq_len: int = 4,
    frame_size: int = ANOMALY_FRAME_SIZE,
    proxy_resolution: str = PROXY_RESOLUTION,
    no_proxy: bool = False,
    selfsup_weights: Optional[str] = None,
    selfsup_method: str = DEFAULT_SELFSUP_METHOD,
    auto_download_selfsup: bool = False,
) -> dict:
    """
    Step 1: 動画をサンプリングしてCSVを出力する。

    3つの動作モード:
      1. "selfsup":  SelfSupSurg (DINO on Cholec80) 特徴距離 → 学習不要、推奨
      2. "convlstm": ConvLSTM Autoencoder 予測誤差 → 自前学習が必要
      3. "fallback": ピクセル差分 → モデル不要だが精度低い

    各フレームに対して以下の指標を計算:
      - anomaly_score: 予測誤差 or 特徴距離
      - color_change: 赤色チャンネル変化量
      - combined_score: anomaly_score + color_change の加重合計
      - smooth_anomaly, smooth_color, smooth_combined: 平滑化値

    Args:
        video_path: 入力動画ファイルパス
        outdir: 出力ディレクトリ
        fps: サンプリングFPS（デフォルト: 5.0）
        roi_margin: 円形ROIマージン
        no_roi: TrueならROI無効化
        smooth_s: 平滑化窓サイズ（秒）
        device: 計算デバイス ("cuda" or "cpu")
        weights_path: ConvLSTM 学習済みモデル重みパス
        seq_len: ConvLSTM 入力シーケンス長
        frame_size: ConvLSTM 入力フレームサイズ
        proxy_resolution: プロキシ解像度
        no_proxy: Trueならプロキシを使わない
        selfsup_weights: SelfSupSurg 重みパス（Noneで自動検索）
        selfsup_method: SSL手法 ("dino", "moco_v2", "simclr", "swav")
        auto_download_selfsup: SelfSupSurg 重みの自動ダウンロード

    Returns:
        {"csv": CSVファイルパス}
    """
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    stem = Path(video_path).stem

    # プロキシ動画の生成・使用
    analysis_path = video_path
    if not no_proxy:
        analysis_path = _ensure_proxy(video_path, outdir, proxy_resolution)

    # モデル初期化
    model_mgr = AnomalyModelManager(
        device=device,
        weights_path=weights_path,
        seq_len=seq_len,
        frame_size=frame_size,
        selfsup_weights_path=selfsup_weights,
        selfsup_method=selfsup_method,
        auto_download_selfsup=auto_download_selfsup,
    )

    mode = model_mgr.mode  # "selfsup", "convlstm", or "fallback"

    # データ収集
    times: List[float] = []
    anomaly_scores: List[float] = []
    color_changes: List[float] = []
    combined_scores: List[float] = []
    reader_name = "opencv"

    roi_mask: Optional[np.ndarray] = None
    roi_initialized = False
    prev_bgr: Optional[np.ndarray] = None
    frame_buffer: List[torch.Tensor] = []
    frame_count = 0

    # 重み: モデルの種類に応じて配分
    if mode == "selfsup":
        w_anomaly = 0.7
    elif mode == "convlstm":
        w_anomaly = 0.7
    else:
        w_anomaly = 0.3
    w_color = 1.0 - w_anomaly

    for t_sec, bgr, reader in iter_frames(analysis_path, fps):
        reader_name = reader
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"  処理中: {frame_count} フレーム / t={t_sec:.1f}s",
                  file=sys.stderr)

        # ROI初期化
        if not roi_initialized:
            h, w = bgr.shape[:2]
            if not no_roi:
                roi_mask = make_circular_roi(h, w, margin=roi_margin)
            roi_initialized = True

        # 色変化スコア
        if prev_bgr is not None:
            cc = compute_color_change_score(prev_bgr, bgr, roi_mask)
        else:
            cc = 0.0

        # 異常スコア（モードに応じて計算方法を切替）
        if mode == "selfsup":
            a_score = model_mgr.compute_selfsup_score(bgr)
        else:
            # ConvLSTM or fallback
            tensor = preprocess_frame_anomaly(bgr, frame_size)
            frame_buffer.append(tensor)
            if len(frame_buffer) > seq_len:
                seq = frame_buffer[-seq_len - 1 : -1]
                target = frame_buffer[-1]
                a_score, _ = model_mgr.predict_and_score(seq, target)
            else:
                a_score = 0.0

        # 合成スコア
        combined = w_anomaly * a_score + w_color * cc

        times.append(t_sec)
        anomaly_scores.append(a_score)
        color_changes.append(cc)
        combined_scores.append(combined)
        prev_bgr = bgr.copy()

    if not times:
        print("警告: フレームが取得できませんでした。", file=sys.stderr)
        return {}

    # 平滑化
    window = max(1, int(round(smooth_s * fps)))
    smooth_anomaly = smooth_center(anomaly_scores, window)
    smooth_color = smooth_center(color_changes, window)
    smooth_combined = smooth_center(combined_scores, window)

    # CSV出力
    csv_path = out_path / f"{stem}_anomalylog.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "t_sec", "t_srt",
            "anomaly_score", "color_change", "combined_score",
            "smooth_anomaly", "smooth_color", "smooth_combined",
            "reader",
        ])
        for i in range(len(times)):
            writer.writerow([
                f"{times[i]:.3f}",
                format_srt_time(times[i]),
                f"{anomaly_scores[i]:.6f}",
                f"{color_changes[i]:.6f}",
                f"{combined_scores[i]:.6f}",
                f"{smooth_anomaly[i]:.6f}",
                f"{smooth_color[i]:.6f}",
                f"{smooth_combined[i]:.6f}",
                reader_name,
            ])

    print(f"CSV  : {csv_path}")
    print(f"モード: {mode} (w_anomaly={w_anomaly}, w_color={w_color})")
    return {"csv": str(csv_path)}


# ---------------------------------------------------------------------------
# CSV読み込み
# ---------------------------------------------------------------------------


def read_anomalylog_csv(csv_path: str) -> dict:
    """
    異常検出ログCSVを読み込む。

    Returns:
        {"times": [...], "anomaly_scores": [...], "color_changes": [...],
         "combined_scores": [...], "smooth_anomaly": [...],
         "smooth_color": [...], "smooth_combined": [...],
         "reader": str, "fps": float}
    """
    times: List[float] = []
    anomaly_scores: List[float] = []
    color_changes: List[float] = []
    combined_scores: List[float] = []
    smooth_anomaly: List[float] = []
    smooth_color: List[float] = []
    smooth_combined: List[float] = []
    reader = "unknown"

    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            times.append(float(row["t_sec"]))
            anomaly_scores.append(float(row["anomaly_score"]))
            color_changes.append(float(row["color_change"]))
            combined_scores.append(float(row["combined_score"]))
            smooth_anomaly.append(float(row["smooth_anomaly"]))
            smooth_color.append(float(row["smooth_color"]))
            smooth_combined.append(float(row["smooth_combined"]))
            reader = row.get("reader", "unknown")

    fps = 1.0 / (times[1] - times[0]) if len(times) >= 2 else 5.0

    return {
        "times": times,
        "anomaly_scores": anomaly_scores,
        "color_changes": color_changes,
        "combined_scores": combined_scores,
        "smooth_anomaly": smooth_anomaly,
        "smooth_color": smooth_color,
        "smooth_combined": smooth_combined,
        "reader": reader,
        "fps": fps,
    }


# ---------------------------------------------------------------------------
# Step 2: アノテーション
# ---------------------------------------------------------------------------


def annotate_anomaly(
    csv_path: str,
    outdir: str,
    thr: float = 0.01,
    min_duration_s: float = 1.0,
    smooth_s: float = 5.0,
    metric: str = "smooth_combined",
    max_duration_s: float = 300.0,
) -> dict:
    """
    Step 2: CSVから閾値ベースで異常イベントを抽出し、JSONL/SRTを出力する。

    Args:
        csv_path: 入力CSVファイル（異常ログ）
        outdir: 出力ディレクトリ
        thr: 異常検出閾値
        min_duration_s: 最小イベント持続時間（秒）
        smooth_s: 記録用パラメータ
        metric: 使用する指標名
            "smooth_anomaly", "smooth_color", "smooth_combined"（デフォルト）
        max_duration_s: 最大イベント持続時間（秒）。超過は除外。0で無制限

    Returns:
        {"jsonl": str, "srt": str, "events": int}
    """
    data = read_anomalylog_csv(csv_path)
    times = data["times"]
    fps = data["fps"]

    # 指標の選択
    metric_map = {
        "smooth_anomaly": data["smooth_anomaly"],
        "smooth_color": data["smooth_color"],
        "smooth_combined": data["smooth_combined"],
    }
    values = metric_map.get(metric, data["smooth_combined"])

    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    stem = Path(csv_path).stem.replace("_anomalylog", "")

    # イベント抽出（閾値超過の連続区間）
    min_samples = max(1, int(round(min_duration_s * fps)))
    raw_events: List[dict] = []
    in_event = False
    start_idx = 0

    for i in range(len(values)):
        if values[i] >= thr and not in_event:
            in_event = True
            start_idx = i
        elif values[i] < thr and in_event:
            in_event = False
            if i - start_idx >= min_samples:
                _add_event(
                    raw_events, times, values, data,
                    start_idx, i, thr, min_duration_s, smooth_s, metric,
                )

    # 末尾処理
    if in_event and len(values) - start_idx >= min_samples:
        _add_event(
            raw_events, times, values, data,
            start_idx, len(values), thr, min_duration_s, smooth_s, metric,
        )

    # max_duration_s フィルタ
    if max_duration_s > 0:
        events = []
        for ev in raw_events:
            if ev["duration_s"] <= max_duration_s:
                events.append(ev)
            else:
                print(f"  除外: {ev['start_srt']}〜{ev['end_srt']} "
                      f"({ev['duration_s']:.1f}s > {max_duration_s}s)",
                      file=sys.stderr)
        excluded = len(raw_events) - len(events)
        if excluded:
            print(f"  {excluded}件を長時間イベントとして除外",
                  file=sys.stderr)
    else:
        events = raw_events

    # JSONL出力（正本）
    jsonl_path = out_path / f"{stem}_anomaly_events.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")

    # SRT出力（JSONLから変換）
    srt_path = out_path / f"{stem}_anomaly.srt"
    from src.tools.jsonl_to_srt import convert as jsonl_to_srt_convert
    jsonl_to_srt_convert(
        in_jsonl=str(jsonl_path),
        out_srt=str(srt_path),
        event_type="anomaly_candidate",
    )

    print(f"JSONL: {jsonl_path} （正本）")
    print(f"SRT  : {srt_path} （JSONLから変換）")
    max_dur_str = f", max_dur={max_duration_s}s" if max_duration_s > 0 else ""
    print(f"イベント数: {len(events)} （thr={thr}, metric={metric}"
          f"{max_dur_str}）")

    return {
        "jsonl": str(jsonl_path),
        "srt": str(srt_path),
        "events": len(events),
    }


def _add_event(
    events: List[dict],
    times: List[float],
    values: List[float],
    data: dict,
    start_idx: int,
    end_idx: int,
    thr: float,
    min_duration_s: float,
    smooth_s: float,
    metric: str,
) -> None:
    """イベント辞書を構築してリストに追加する。"""
    seg = values[start_idx:end_idx]
    peak_value = max(seg)

    # イベント区間の平均色変化量
    cc = data["color_changes"][start_idx:end_idx]
    mean_color_change = sum(cc) / len(cc) if cc else 0.0

    # イベント区間の平均異常スコア
    a_scores = data["anomaly_scores"][start_idx:end_idx]
    mean_anomaly = sum(a_scores) / len(a_scores) if a_scores else 0.0

    events.append({
        "type": "anomaly_candidate",
        "metric": metric,
        "thr": thr,
        "min_duration_s": min_duration_s,
        "smooth_s": smooth_s,
        "peak_value": round(peak_value, 6),
        "mean_anomaly": round(mean_anomaly, 6),
        "mean_color_change": round(mean_color_change, 6),
        "duration_s": round(times[end_idx - 1] - times[start_idx], 3),
        "start_sec": times[start_idx],
        "end_sec": times[end_idx - 1],
        "start_srt": format_srt_time(times[start_idx]),
        "end_srt": format_srt_time(times[end_idx - 1]),
    })


# ---------------------------------------------------------------------------
# 一括実行
# ---------------------------------------------------------------------------


def analyze_video(
    video_path: str,
    outdir: str,
    fps: float = 5.0,
    roi_margin: float = 0.08,
    no_roi: bool = False,
    smooth_s: float = 5.0,
    device: str = "cuda",
    weights_path: Optional[str] = None,
    seq_len: int = 4,
    frame_size: int = ANOMALY_FRAME_SIZE,
    thr: float = 0.01,
    min_duration_s: float = 1.0,
    metric: str = "smooth_combined",
    max_duration_s: float = 300.0,
    proxy_resolution: str = PROXY_RESOLUTION,
    no_proxy: bool = False,
    selfsup_weights: Optional[str] = None,
    selfsup_method: str = DEFAULT_SELFSUP_METHOD,
    auto_download_selfsup: bool = False,
) -> dict:
    """2ステップの一括実行。"""
    result1 = record_timeseries(
        video_path=video_path,
        outdir=outdir,
        fps=fps,
        roi_margin=roi_margin,
        no_roi=no_roi,
        smooth_s=smooth_s,
        device=device,
        weights_path=weights_path,
        seq_len=seq_len,
        frame_size=frame_size,
        proxy_resolution=proxy_resolution,
        no_proxy=no_proxy,
        selfsup_weights=selfsup_weights,
        selfsup_method=selfsup_method,
        auto_download_selfsup=auto_download_selfsup,
    )
    if not result1:
        return {}

    result2 = annotate_anomaly(
        csv_path=result1["csv"],
        outdir=outdir,
        thr=thr,
        min_duration_s=min_duration_s,
        smooth_s=smooth_s,
        metric=metric,
        max_duration_s=max_duration_s,
    )
    return {**result1, **result2}


# ---------------------------------------------------------------------------
# CLI エントリポイント
# ---------------------------------------------------------------------------


def main() -> int:
    """コマンドラインエントリポイント（サブコマンド方式）"""
    parser = argparse.ArgumentParser(
        description="Self-Supervised Video Anomaly Detection（出血・煙・急動検出）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # SelfSupSurg (DINO on Cholec80) で解析（推奨、学習不要）
  python -m src.anomaly.detector analyze --video input.mp4 --outdir output/ \\
      --use-selfsup --auto-download

  # 既にダウンロード済みの SelfSupSurg 重みを使用
  python -m src.anomaly.detector analyze --video input.mp4 --outdir output/ \\
      --use-selfsup

  # ConvLSTM 学習済みモデルを使用
  python -m src.anomaly.detector analyze --video input.mp4 --outdir output/ \\
      --weights models/anomaly_convlstm.pth

  # フォールバック（モデルなし）で一括解析
  python -m src.anomaly.detector analyze --video input.mp4 --outdir output/

  # Step 2 のみ（閾値を変えて再実行）
  python -m src.anomaly.detector annotate --csv output/input_anomalylog.csv \\
      --outdir output/ --thr 0.005 --metric smooth_anomaly
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="実行コマンド")

    # --- timeseries ---
    ts = subparsers.add_parser("timeseries", help="Step 1: 時系列記録（動画→CSV）")
    ts.add_argument("--video", required=True, help="入力動画ファイルパス")
    ts.add_argument("--outdir", required=True, help="出力ディレクトリ")
    ts.add_argument("--fps", type=float, default=5.0,
                     help="サンプリングFPS（デフォルト: 5）")
    ts.add_argument("--roi-margin", type=float, default=0.08,
                     help="円形ROIマージン（デフォルト: 0.08）")
    ts.add_argument("--no-roi", action="store_true", help="ROI無効化")
    ts.add_argument("--smooth-s", type=float, default=5.0,
                     help="平滑化窓（秒、デフォルト: 5）")
    ts.add_argument("--device", default="cuda",
                     help="計算デバイス（デフォルト: cuda）")
    ts.add_argument("--weights", default=None,
                     help="学習済みモデル重みファイルパス")
    ts.add_argument("--seq-len", type=int, default=4,
                     help="入力シーケンス長（デフォルト: 4）")
    ts.add_argument("--frame-size", type=int, default=ANOMALY_FRAME_SIZE,
                     help=f"フレームサイズ（デフォルト: {ANOMALY_FRAME_SIZE}）")
    ts.add_argument("--proxy-resolution", default=PROXY_RESOLUTION,
                     choices=["360p", "480p", "720p"],
                     help=f"プロキシ解像度（デフォルト: {PROXY_RESOLUTION}）")
    ts.add_argument("--no-proxy", action="store_true",
                     help="プロキシを使わず元動画を直接解析")
    ts.add_argument("--use-selfsup", action="store_true",
                     help="SelfSupSurg (DINO on Cholec80) を使用")
    ts.add_argument("--selfsup-weights", default=None,
                     help="SelfSupSurg 重みファイルパス")
    ts.add_argument("--selfsup-method", default=DEFAULT_SELFSUP_METHOD,
                     choices=["dino", "moco_v2", "simclr", "swav"],
                     help=f"SSL手法（デフォルト: {DEFAULT_SELFSUP_METHOD}）")
    ts.add_argument("--auto-download", action="store_true",
                     help="SelfSupSurg 重みを自動ダウンロード")

    # --- annotate ---
    ann = subparsers.add_parser("annotate", help="Step 2: アノテーション（CSV→JSONL/SRT）")
    ann.add_argument("--csv", required=True, help="入力CSVファイル")
    ann.add_argument("--outdir", required=True, help="出力ディレクトリ")
    ann.add_argument("--thr", type=float, default=0.01,
                      help="異常検出閾値（デフォルト: 0.01）")
    ann.add_argument("--min-duration-s", type=float, default=1.0,
                      help="最小イベント持続時間（秒、デフォルト: 1.0）")
    ann.add_argument("--smooth-s", type=float, default=5.0,
                      help="平滑化窓（秒、デフォルト: 5）")
    ann.add_argument("--metric", default="smooth_combined",
                      choices=["smooth_anomaly", "smooth_color", "smooth_combined"],
                      help="使用する指標名（デフォルト: smooth_combined）")
    ann.add_argument("--max-duration-s", type=float, default=300.0,
                      help="最大イベント持続時間（秒、デフォルト: 300）。0で無制限")

    # --- analyze ---
    ana = subparsers.add_parser("analyze", help="一括実行（timeseries + annotate）")
    ana.add_argument("--video", required=True, help="入力動画ファイルパス")
    ana.add_argument("--outdir", required=True, help="出力ディレクトリ")
    ana.add_argument("--fps", type=float, default=5.0,
                      help="サンプリングFPS（デフォルト: 5）")
    ana.add_argument("--roi-margin", type=float, default=0.08,
                      help="円形ROIマージン")
    ana.add_argument("--no-roi", action="store_true", help="ROI無効化")
    ana.add_argument("--smooth-s", type=float, default=5.0,
                      help="平滑化窓（秒）")
    ana.add_argument("--device", default="cuda", help="計算デバイス")
    ana.add_argument("--weights", default=None,
                      help="学習済みモデル重みファイルパス")
    ana.add_argument("--seq-len", type=int, default=4,
                      help="入力シーケンス長")
    ana.add_argument("--frame-size", type=int, default=ANOMALY_FRAME_SIZE,
                      help=f"フレームサイズ（デフォルト: {ANOMALY_FRAME_SIZE}）")
    ana.add_argument("--proxy-resolution", default=PROXY_RESOLUTION,
                      choices=["360p", "480p", "720p"],
                      help=f"プロキシ解像度（デフォルト: {PROXY_RESOLUTION}）")
    ana.add_argument("--no-proxy", action="store_true",
                      help="プロキシを使わず元動画を直接解析")
    ana.add_argument("--use-selfsup", action="store_true",
                      help="SelfSupSurg (DINO on Cholec80) を使用")
    ana.add_argument("--selfsup-weights", default=None,
                      help="SelfSupSurg 重みファイルパス")
    ana.add_argument("--selfsup-method", default=DEFAULT_SELFSUP_METHOD,
                      choices=["dino", "moco_v2", "simclr", "swav"],
                      help=f"SSL手法（デフォルト: {DEFAULT_SELFSUP_METHOD}）")
    ana.add_argument("--auto-download", action="store_true",
                      help="SelfSupSurg 重みを自動ダウンロード")
    ana.add_argument("--thr", type=float, default=0.01,
                      help="異常検出閾値（デフォルト: 0.01）")
    ana.add_argument("--min-duration-s", type=float, default=1.0,
                      help="最小イベント持続時間（秒、デフォルト: 1.0）")
    ana.add_argument("--metric", default="smooth_combined",
                      choices=["smooth_anomaly", "smooth_color", "smooth_combined"],
                      help="使用する指標名（デフォルト: smooth_combined）")
    ana.add_argument("--max-duration-s", type=float, default=300.0,
                      help="最大イベント持続時間（秒、デフォルト: 300）。0で無制限")

    args = parser.parse_args()

    if args.command == "timeseries":
        record_timeseries(
            video_path=args.video,
            outdir=args.outdir,
            fps=args.fps,
            roi_margin=args.roi_margin,
            no_roi=args.no_roi,
            smooth_s=args.smooth_s,
            device=args.device,
            weights_path=args.weights,
            seq_len=args.seq_len,
            frame_size=args.frame_size,
            proxy_resolution=args.proxy_resolution,
            no_proxy=args.no_proxy,
            selfsup_weights=args.selfsup_weights if args.use_selfsup else None,
            selfsup_method=args.selfsup_method,
            auto_download_selfsup=args.auto_download or args.use_selfsup,
        )
    elif args.command == "annotate":
        annotate_anomaly(
            csv_path=args.csv,
            outdir=args.outdir,
            thr=args.thr,
            min_duration_s=args.min_duration_s,
            smooth_s=args.smooth_s,
            metric=args.metric,
            max_duration_s=args.max_duration_s,
        )
    elif args.command == "analyze":
        analyze_video(
            video_path=args.video,
            outdir=args.outdir,
            fps=args.fps,
            roi_margin=args.roi_margin,
            no_roi=args.no_roi,
            smooth_s=args.smooth_s,
            device=args.device,
            weights_path=args.weights,
            seq_len=args.seq_len,
            frame_size=args.frame_size,
            thr=args.thr,
            min_duration_s=args.min_duration_s,
            metric=args.metric,
            max_duration_s=args.max_duration_s,
            proxy_resolution=args.proxy_resolution,
            no_proxy=args.no_proxy,
            selfsup_weights=args.selfsup_weights if args.use_selfsup else None,
            selfsup_method=args.selfsup_method,
            auto_download_selfsup=args.auto_download or args.use_selfsup,
        )
    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
