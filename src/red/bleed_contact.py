"""
接触成長検出モジュール（bleed_contact）

出血は既存の赤色領域から隣接方向に広がる（点源拡散）。
カメラ移動で新たに赤い領域が見えるとき、それは既存の赤と
空間的に連続しているとは限らない。

この違いを利用し、新規赤化画素のうち「既存赤色画素に隣接しているもの」
の割合（接触成長率）を出血指標とする。

指標:
  contact_ratio = 接触新規赤画素数 / 全新規赤画素数
  contact_area  = 接触新規赤画素数 / ROI画素数
  bleed_score   = contact_area × contact_ratio

  接触新規赤画素: 今回赤 AND 前回赤でない AND 前回赤の画素に隣接（8近傍）

出血: 赤の縁から外側に広がる → contact_ratio が高い + contact_area が正
カメラ移動: 画面端から赤い領域が入ってくる → 既存赤と離れている → contact_ratio が低い

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
    extract_bleed_events,
)
from src.red.bleed_detector import make_red_mask


# ---------------------------------------------------------------------------
# 接触成長スコア計算
# ---------------------------------------------------------------------------

def compute_contact_growth(
    prev_bgr: np.ndarray,
    curr_bgr: np.ndarray,
    roi_mask: Optional[np.ndarray] = None,
    s_min: int = 60,
    v_min: int = 40,
) -> dict:
    """
    連続する2フレームから接触成長指標を算出する。

    Args:
        prev_bgr: 前フレーム（BGR）
        curr_bgr: 現フレーム（BGR）
        roi_mask: 円形ROI。Noneなら全画素。
        s_min: 彩度最小値
        v_min: 明度最小値

    Returns:
        {
            "red_ratio": float,       # 現フレームの赤色率
            "newly_red_count": int,   # 新規赤化画素数
            "contact_count": int,     # 接触新規赤画素数
            "contact_ratio": float,   # 接触成長率（0〜1）
            "contact_area": float,    # 接触面積率（ROIに対する比）
            "bleed_score": float,     # contact_area × contact_ratio
        }
    """
    red_prev = make_red_mask(prev_bgr, s_min, v_min)
    red_curr = make_red_mask(curr_bgr, s_min, v_min)

    if roi_mask is not None:
        roi_u8 = roi_mask.astype(np.uint8) * 255
        red_prev = red_prev & roi_u8
        red_curr = red_curr & roi_u8
        total_pixels = int(np.count_nonzero(roi_mask))
    else:
        total_pixels = curr_bgr.shape[0] * curr_bgr.shape[1]

    if total_pixels == 0:
        return {
            "red_ratio": 0.0, "newly_red_count": 0,
            "contact_count": 0, "contact_ratio": 0.0,
            "contact_area": 0.0, "bleed_score": 0.0,
        }

    red_ratio = int(np.count_nonzero(red_curr)) / total_pixels

    # 新規赤化画素: 今回赤 AND 前回赤でない
    newly_red = (red_curr > 0) & (red_prev == 0)
    if roi_mask is not None:
        newly_red = newly_red & roi_mask
    newly_red_count = int(np.count_nonzero(newly_red))

    if newly_red_count == 0:
        return {
            "red_ratio": red_ratio, "newly_red_count": 0,
            "contact_count": 0, "contact_ratio": 0.0,
            "contact_area": 0.0, "bleed_score": 0.0,
        }

    # 前回赤の8近傍膨張 → 赤の縁
    kernel = np.ones((3, 3), dtype=np.uint8)
    prev_dilated = cv2.dilate(red_prev, kernel, iterations=1)

    # 接触新規赤画素: 新規赤 AND 前回赤の隣接画素
    contact = newly_red & (prev_dilated > 0)
    contact_count = int(np.count_nonzero(contact))

    contact_ratio = contact_count / newly_red_count
    contact_area = contact_count / total_pixels

    bleed_score = contact_area * contact_ratio

    return {
        "red_ratio": red_ratio,
        "newly_red_count": newly_red_count,
        "contact_count": contact_count,
        "contact_ratio": contact_ratio,
        "contact_area": contact_area,
        "bleed_score": bleed_score,
    }


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
    smooth_s: float = 3.0,
) -> dict:
    """
    Step 1: 動画をサンプリングしてCSVを出力する。

    Returns:
        {"csv": CSVファイルパス}
    """
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    stem = Path(video_path).stem

    times: List[float] = []
    red_ratios: List[float] = []
    contact_ratios: List[float] = []
    contact_areas: List[float] = []
    bleed_scores: List[float] = []
    reader_name = "opencv"
    roi_mask: Optional[np.ndarray] = None
    roi_initialized = False
    prev_bgr: Optional[np.ndarray] = None

    for t_sec, bgr, reader in iter_frames(video_path, fps):
        reader_name = reader
        if not roi_initialized:
            h, w = bgr.shape[:2]
            if not no_roi:
                roi_mask = make_circular_roi(h, w, margin=roi_margin)
            roi_initialized = True

        if prev_bgr is None:
            from src.red.redlog import compute_red_ratio
            ratio = compute_red_ratio(bgr, roi_mask, s_min=s_min, v_min=v_min)
            times.append(t_sec)
            red_ratios.append(ratio)
            contact_ratios.append(0.0)
            contact_areas.append(0.0)
            bleed_scores.append(0.0)
        else:
            metrics = compute_contact_growth(
                prev_bgr, bgr, roi_mask, s_min=s_min, v_min=v_min,
            )
            times.append(t_sec)
            red_ratios.append(metrics["red_ratio"])
            contact_ratios.append(metrics["contact_ratio"])
            contact_areas.append(metrics["contact_area"])
            bleed_scores.append(metrics["bleed_score"])

        prev_bgr = bgr.copy()

    if not times:
        print("警告: フレームが取得できませんでした。", file=sys.stderr)
        return {}

    # 平滑化
    window = max(1, int(round(smooth_s * fps)))
    smooth_scores = smooth_center(bleed_scores, window)

    # CSV
    csv_path = out_path / f"{stem}_contactlog.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "t_sec", "t_srt", "red_ratio",
            "contact_ratio", "contact_area",
            "bleed_score", "smooth_bleed", "reader",
        ])
        for i in range(len(times)):
            writer.writerow([
                f"{times[i]:.3f}",
                format_srt_time(times[i]),
                f"{red_ratios[i]:.6f}",
                f"{contact_ratios[i]:.6f}",
                f"{contact_areas[i]:.6f}",
                f"{bleed_scores[i]:.6f}",
                f"{smooth_scores[i]:.6f}",
                reader_name,
            ])

    print(f"CSV  : {csv_path}")
    return {"csv": str(csv_path)}


# ---------------------------------------------------------------------------
# CSV読み込み
# ---------------------------------------------------------------------------

def read_contactlog_csv(csv_path: str) -> dict:
    """接触成長ログCSVを読み込む。"""
    times: List[float] = []
    red_ratios: List[float] = []
    contact_ratios: List[float] = []
    contact_areas: List[float] = []
    bleed_scores: List[float] = []
    smooth_scores: List[float] = []
    reader = "unknown"

    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            times.append(float(row["t_sec"]))
            red_ratios.append(float(row["red_ratio"]))
            contact_ratios.append(float(row["contact_ratio"]))
            contact_areas.append(float(row["contact_area"]))
            bleed_scores.append(float(row["bleed_score"]))
            smooth_scores.append(float(row["smooth_bleed"]))
            reader = row.get("reader", "unknown")

    fps = 1.0 / (times[1] - times[0]) if len(times) >= 2 else 5.0
    return {
        "times": times, "red_ratios": red_ratios,
        "contact_ratios": contact_ratios, "contact_areas": contact_areas,
        "bleed_scores": bleed_scores, "smooth_scores": smooth_scores,
        "reader": reader, "fps": fps,
    }


# ---------------------------------------------------------------------------
# Step 2: アノテーション
# ---------------------------------------------------------------------------

def annotate_bleed(
    csv_path: str,
    outdir: str,
    thr: float = 0.0001,
    k_s: float = 2.0,
    smooth_s: float = 3.0,
) -> dict:
    """
    Step 2: CSVから閾値ベースで出血イベントを抽出する。

    Args:
        csv_path: 入力CSV
        outdir: 出力ディレクトリ
        thr: bleed_score の閾値
        k_s: 最小連続時間（秒）
        smooth_s: 記録用

    Returns:
        {"jsonl": path, "srt": path, "events": int}
    """
    data = read_contactlog_csv(csv_path)
    times = data["times"]
    smooth_scores = data["smooth_scores"]
    fps = data["fps"]

    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    stem = Path(csv_path).stem.replace("_contactlog", "")

    events = extract_bleed_events(times, smooth_scores, thr, k_s, fps, smooth_s)
    for ev in events:
        ev["metric"] = "contact_bleed"

    # JSONL
    jsonl_path = out_path / f"{stem}_bleed_contact_events.jsonl"
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
    srt_path = out_path / f"{stem}_bleed_contact.srt"
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
    smooth_s: float = 3.0,
    thr: float = 0.0001,
    k_s: float = 2.0,
) -> dict:
    """2ステップの一括実行。"""
    result1 = record_timeseries(
        video_path, outdir, fps, s_min, v_min,
        roi_margin, no_roi, smooth_s,
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
        description="接触成長検出（隣接拡散ベースの出血検出）"
    )
    subparsers = parser.add_subparsers(dest="command")

    ts = subparsers.add_parser("timeseries")
    ts.add_argument("--video", required=True)
    ts.add_argument("--outdir", required=True)
    ts.add_argument("--fps", type=float, default=5.0)
    ts.add_argument("--smooth-s", type=float, default=3.0)

    ann = subparsers.add_parser("annotate")
    ann.add_argument("--csv", required=True)
    ann.add_argument("--outdir", required=True)
    ann.add_argument("--thr", type=float, default=0.0001)
    ann.add_argument("--k-s", type=float, default=2.0)

    ana = subparsers.add_parser("analyze")
    ana.add_argument("--video", required=True)
    ana.add_argument("--outdir", required=True)
    ana.add_argument("--fps", type=float, default=5.0)
    ana.add_argument("--smooth-s", type=float, default=3.0)
    ana.add_argument("--thr", type=float, default=0.0001)
    ana.add_argument("--k-s", type=float, default=2.0)

    args = parser.parse_args()
    if args.command == "timeseries":
        record_timeseries(args.video, args.outdir, args.fps,
                          smooth_s=args.smooth_s)
    elif args.command == "annotate":
        annotate_bleed(args.csv, args.outdir, args.thr, args.k_s)
    elif args.command == "analyze":
        analyze_video(args.video, args.outdir, args.fps,
                      smooth_s=args.smooth_s,
                      thr=args.thr, k_s=args.k_s)
    else:
        parser.print_help()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
