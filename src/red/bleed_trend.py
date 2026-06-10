"""
赤色トレンド検出モジュール（bleed_trend）

出血は数秒間にわたって赤色が一方向に増加する。
カメラ移動は赤色率が上下に揺らぐ（行って戻る）。
この違いを利用し、スライディングウィンドウ内の赤色率に対して
線形回帰を当てはめ、「正の傾き × 決定係数」を出血指標とする。

指標:
  trend_score = max(0, slope) × R²

  slope: 窓内の赤色率の線形回帰傾き（/秒）
    - 正 = 赤が増加中, 負 = 赤が減少中
  R²: 決定係数（0〜1）
    - 1に近い = 直線的に変化（出血パターン）
    - 0に近い = ランダムに揺らぐ（カメラ移動パターン）

出血: slope > 0 かつ R² が高い → trend_score が高い
カメラ移動: slope は正負に揺れ、R² が低い → trend_score が低い

2段階の処理:
  Step 1 - record_timeseries(): 動画→CSV
  Step 2 - annotate_bleed():    CSV→JSONL/SRT
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from src.core.time_utils import format_srt_time
from src.red.redlog import (
    make_circular_roi,
    iter_frames,
    smooth_center,
    compute_red_ratio,
    extract_bleed_events,
)


# ---------------------------------------------------------------------------
# トレンドスコア計算
# ---------------------------------------------------------------------------

def compute_trend(values: List[float], fps: float, window_s: float = 5.0) -> List[float]:
    """
    スライディングウィンドウ内の線形回帰で trend_score を計算する。

    Args:
        values: 赤色率の時系列
        fps: サンプリングFPS
        window_s: 回帰ウィンドウ幅（秒）

    Returns:
        trend_score のリスト（入力と同じ長さ）
    """
    n = len(values)
    half_w = max(1, int(round(window_s * fps / 2)))
    scores = []

    for i in range(n):
        lo = max(0, i - half_w)
        hi = min(n, i + half_w + 1)
        seg = values[lo:hi]

        if len(seg) < 3:
            scores.append(0.0)
            continue

        # 線形回帰: y = a*x + b
        x = np.arange(len(seg), dtype=np.float64)
        y = np.array(seg, dtype=np.float64)

        x_mean = np.mean(x)
        y_mean = np.mean(y)

        ss_xx = np.sum((x - x_mean) ** 2)
        ss_xy = np.sum((x - x_mean) * (y - y_mean))
        ss_yy = np.sum((y - y_mean) ** 2)

        if ss_xx < 1e-12:
            scores.append(0.0)
            continue

        slope_per_sample = ss_xy / ss_xx

        # 傾きを /秒 単位に変換
        slope = slope_per_sample * fps

        # R²
        if ss_yy < 1e-12:
            r_squared = 0.0
        else:
            r_squared = (ss_xy ** 2) / (ss_xx * ss_yy)
            r_squared = max(0.0, min(1.0, r_squared))

        # trend_score: 正の傾きのみ
        if slope > 0:
            scores.append(slope * r_squared)
        else:
            scores.append(0.0)

    return scores


# ---------------------------------------------------------------------------
# Step 1: 時系列記録
# ---------------------------------------------------------------------------

def record_timeseries(
    video_path: str,
    outdir: str,
    fps: float = 5.0,
    s_min: int = 60,
    v_min: int = 40,
    roi_margin: float = 0.08,
    no_roi: bool = False,
    trend_window_s: float = 5.0,
    smooth_s: float = 3.0,
) -> dict:
    """
    Step 1: 動画をサンプリングしてCSVを出力する。

    Args:
        video_path: 入力動画ファイルパス
        outdir: 出力ディレクトリ
        fps: サンプリングFPS
        s_min: HSV彩度最小値
        v_min: HSV明度最小値
        roi_margin: 円形ROIマージン
        no_roi: TrueならROIを無効化
        trend_window_s: トレンド計算ウィンドウ（秒）
        smooth_s: trend_score の平滑化窓（秒）

    Returns:
        {"csv": CSVファイルパス}
    """
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    stem = Path(video_path).stem

    times: List[float] = []
    ratios: List[float] = []
    reader_name = "opencv"
    roi_mask: Optional[np.ndarray] = None
    roi_initialized = False

    for t_sec, bgr, reader in iter_frames(video_path, fps):
        reader_name = reader
        if not roi_initialized:
            h, w = bgr.shape[:2]
            if not no_roi:
                roi_mask = make_circular_roi(h, w, margin=roi_margin)
            roi_initialized = True

        ratio = compute_red_ratio(bgr, roi_mask, s_min=s_min, v_min=v_min)
        times.append(t_sec)
        ratios.append(ratio)

    if not times:
        print("警告: フレームが取得できませんでした。", file=sys.stderr)
        return {}

    # トレンドスコア計算
    trend_scores = compute_trend(ratios, fps, window_s=trend_window_s)

    # 平滑化
    window = max(1, int(round(smooth_s * fps)))
    smooth_trends = smooth_center(trend_scores, window)

    # CSV出力
    csv_path = out_path / f"{stem}_trendlog.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "t_sec", "t_srt", "red_ratio",
            "trend_score", "smooth_trend", "reader",
        ])
        for i in range(len(times)):
            writer.writerow([
                f"{times[i]:.3f}",
                format_srt_time(times[i]),
                f"{ratios[i]:.6f}",
                f"{trend_scores[i]:.6f}",
                f"{smooth_trends[i]:.6f}",
                reader_name,
            ])

    print(f"CSV  : {csv_path}")
    return {"csv": str(csv_path)}


# ---------------------------------------------------------------------------
# CSV読み込み
# ---------------------------------------------------------------------------

def read_trendlog_csv(csv_path: str) -> dict:
    """トレンドログCSVを読み込む。"""
    times: List[float] = []
    ratios: List[float] = []
    trend_scores: List[float] = []
    smooth_trends: List[float] = []
    reader = "unknown"

    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            times.append(float(row["t_sec"]))
            ratios.append(float(row["red_ratio"]))
            trend_scores.append(float(row["trend_score"]))
            smooth_trends.append(float(row["smooth_trend"]))
            reader = row.get("reader", "unknown")

    fps = 1.0 / (times[1] - times[0]) if len(times) >= 2 else 5.0
    return {
        "times": times, "ratios": ratios,
        "trend_scores": trend_scores, "smooth_trends": smooth_trends,
        "reader": reader, "fps": fps,
    }


# ---------------------------------------------------------------------------
# Step 2: アノテーション
# ---------------------------------------------------------------------------

def annotate_bleed(
    csv_path: str,
    outdir: str,
    thr: float = 0.01,
    k_s: float = 2.0,
    smooth_s: float = 3.0,
) -> dict:
    """
    Step 2: CSVから閾値ベースで出血イベントを抽出する。

    Args:
        csv_path: 入力CSV
        outdir: 出力ディレクトリ
        thr: trend_score の閾値
        k_s: 最小連続時間（秒）
        smooth_s: 記録用

    Returns:
        {"jsonl": path, "srt": path, "events": int}
    """
    data = read_trendlog_csv(csv_path)
    times = data["times"]
    smooth_trends = data["smooth_trends"]
    fps = data["fps"]

    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    stem = Path(csv_path).stem.replace("_trendlog", "")

    events = extract_bleed_events(times, smooth_trends, thr, k_s, fps, smooth_s)
    for ev in events:
        ev["metric"] = "trend_score"

    # JSONL
    jsonl_path = out_path / f"{stem}_bleed_trend_events.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for ev in events:
            line = {
                "type": ev["type"],
                "metric": ev["metric"],
                "thr": ev["thr"],
                "k_s": ev["k_s"],
                "smooth_s": ev["smooth_s"],
                "delta_max": ev["delta_max"],
                "start_sec": ev["start"],
                "end_sec": ev["end"],
                "start_srt": format_srt_time(ev["start"]),
                "end_srt": format_srt_time(ev["end"]),
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    # SRT
    from src.tools.jsonl_to_srt import convert as jsonl_to_srt_convert
    srt_path = out_path / f"{stem}_bleed_trend.srt"
    jsonl_to_srt_convert(str(jsonl_path), str(srt_path), event_type="bleed_candidate")

    print(f"JSONL: {jsonl_path}")
    print(f"SRT  : {srt_path}")
    print(f"イベント数: {len(events)} (thr={thr}, k_s={k_s})")
    return {"jsonl": str(jsonl_path), "srt": str(srt_path), "events": len(events)}


# ---------------------------------------------------------------------------
# 一括実行
# ---------------------------------------------------------------------------

def analyze_video(
    video_path: str,
    outdir: str,
    fps: float = 5.0,
    s_min: int = 60,
    v_min: int = 40,
    roi_margin: float = 0.08,
    no_roi: bool = False,
    trend_window_s: float = 5.0,
    smooth_s: float = 3.0,
    thr: float = 0.01,
    k_s: float = 2.0,
) -> dict:
    """2ステップの一括実行。"""
    result1 = record_timeseries(
        video_path, outdir, fps, s_min, v_min,
        roi_margin, no_roi, trend_window_s, smooth_s,
    )
    if not result1:
        return {}
    result2 = annotate_bleed(result1["csv"], outdir, thr, k_s, smooth_s)
    return {**result1, **result2}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="赤色トレンド検出（線形回帰ベースの出血検出）"
    )
    subparsers = parser.add_subparsers(dest="command")

    ts = subparsers.add_parser("timeseries")
    ts.add_argument("--video", required=True)
    ts.add_argument("--outdir", required=True)
    ts.add_argument("--fps", type=float, default=5.0)
    ts.add_argument("--trend-window-s", type=float, default=5.0)
    ts.add_argument("--smooth-s", type=float, default=3.0)

    ann = subparsers.add_parser("annotate")
    ann.add_argument("--csv", required=True)
    ann.add_argument("--outdir", required=True)
    ann.add_argument("--thr", type=float, default=0.01)
    ann.add_argument("--k-s", type=float, default=2.0)

    ana = subparsers.add_parser("analyze")
    ana.add_argument("--video", required=True)
    ana.add_argument("--outdir", required=True)
    ana.add_argument("--fps", type=float, default=5.0)
    ana.add_argument("--trend-window-s", type=float, default=5.0)
    ana.add_argument("--smooth-s", type=float, default=3.0)
    ana.add_argument("--thr", type=float, default=0.01)
    ana.add_argument("--k-s", type=float, default=2.0)

    args = parser.parse_args()
    if args.command == "timeseries":
        record_timeseries(args.video, args.outdir, args.fps,
                          trend_window_s=args.trend_window_s,
                          smooth_s=args.smooth_s)
    elif args.command == "annotate":
        annotate_bleed(args.csv, args.outdir, args.thr, args.k_s)
    elif args.command == "analyze":
        analyze_video(args.video, args.outdir, args.fps,
                      trend_window_s=args.trend_window_s,
                      smooth_s=args.smooth_s,
                      thr=args.thr, k_s=args.k_s)
    else:
        parser.print_help()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
