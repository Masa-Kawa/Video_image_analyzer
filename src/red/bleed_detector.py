"""
赤色拡大検出モジュール（bleed_detector）

腹腔鏡手術動画において、「背景が安定しているのに赤色領域が拡大する」
現象を出血候補として検出する。

従来の redlog.py（smooth_delta ベース）では赤色率の全体的な変化量しか
見ないため、カメラ移動と出血の区別ができなかった。本モジュールでは
フレーム間の「新規赤化画素」と「背景安定度」を組み合わせた
red_expansion 指標で出血をより正確に検出する。

アルゴリズム:
  フレーム N-1 → N を比較:
    1. newly_red     = (N で赤) AND (N-1 で赤でない) の画素
    2. bg_diff       = 非赤領域のフレーム間平均差分（背景の動き量）
    3. bg_stability  = 1.0 - min(bg_diff / norm_factor, 1.0)
    4. red_expansion = newly_red_ratio × bg_stability

2段階の処理:
  Step 1 - record_timeseries(): 動画→CSV（赤色拡大率の時系列記録）
  Step 2 - annotate_bleed():    CSV→JSONL/SRT（閾値ベースの出血アノテーション）

出力: CSV（赤色拡大率ログ）、SRT（出血候補イベント）、JSONL（イベント正本）
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

# --- 既存 redlog.py の共通関数を再利用 ---
from src.red.redlog import (
    format_srt_time,
    make_circular_roi,
    iter_frames,
    smooth_center,
    extract_bleed_events,
)


# ---------------------------------------------------------------------------
# 赤色マスク生成
# ---------------------------------------------------------------------------

def make_red_mask(
    frame_bgr: np.ndarray,
    s_min: int = 60,
    v_min: int = 40,
) -> np.ndarray:
    """
    BGRフレームからHSVベースの赤色マスクを生成する。

    Args:
        frame_bgr: BGR画像（OpenCV形式）
        s_min: 彩度最小値
        v_min: 明度最小値

    Returns:
        赤色マスク（uint8, 0 or 255）
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, np.array([0, s_min, v_min]), np.array([10, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([170, s_min, v_min]), np.array([179, 255, 255]))
    return mask1 | mask2


# ---------------------------------------------------------------------------
# 赤色拡大率の計算
# ---------------------------------------------------------------------------

def compute_red_expansion(
    prev_bgr: np.ndarray,
    curr_bgr: np.ndarray,
    roi_mask: Optional[np.ndarray] = None,
    s_min: int = 60,
    v_min: int = 40,
    bg_norm_factor: float = 30.0,
) -> dict:
    """
    連続する2フレームから赤色拡大指標を算出する。

    Args:
        prev_bgr: 前フレーム（BGR）
        curr_bgr: 現フレーム（BGR）
        roi_mask: 円形ROI（Trueの画素のみ集計）。Noneなら全画素。
        s_min: 彩度最小値
        v_min: 明度最小値
        bg_norm_factor: 背景差分の正規化係数（画素値の平均差分がこの値で1.0になる）

    Returns:
        {
            "red_ratio": float,         # 現フレームの赤色率
            "newly_red_ratio": float,   # 新規赤化画素率
            "bg_stability": float,      # 背景安定度（0〜1）
            "red_expansion": float,     # 出血指標（newly_red_ratio × bg_stability）
        }
    """
    # 赤色マスク
    red_prev = make_red_mask(prev_bgr, s_min, v_min)
    red_curr = make_red_mask(curr_bgr, s_min, v_min)

    # ROI適用
    if roi_mask is not None:
        roi_u8 = roi_mask.astype(np.uint8) * 255
        red_prev = red_prev & roi_u8
        red_curr = red_curr & roi_u8
        total_pixels = int(np.count_nonzero(roi_mask))
    else:
        total_pixels = curr_bgr.shape[0] * curr_bgr.shape[1]

    if total_pixels == 0:
        return {
            "red_ratio": 0.0,
            "newly_red_ratio": 0.0,
            "bg_stability": 1.0,
            "red_expansion": 0.0,
        }

    # 現フレームの赤色率
    red_ratio = int(np.count_nonzero(red_curr)) / total_pixels

    # 新規赤化画素: 今回赤 AND 前回赤でない
    newly_red = red_curr & (~red_prev)
    if roi_mask is not None:
        newly_red = newly_red & roi_u8
    newly_red_ratio = int(np.count_nonzero(newly_red)) / total_pixels

    # 背景安定度: 非赤領域のフレーム間差分
    # グレースケールで差分を計算
    gray_prev = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    gray_curr = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)
    frame_diff = cv2.absdiff(gray_prev, gray_curr)

    # 非赤領域のマスク（前後どちらでも赤でない領域）
    non_red = ~(red_prev | red_curr)
    if roi_mask is not None:
        non_red = non_red & roi_u8

    non_red_count = int(np.count_nonzero(non_red))
    if non_red_count > 0:
        bg_diff = float(np.sum(frame_diff[non_red > 0])) / non_red_count
    else:
        # 全画素が赤の場合、背景差分は計算不能 → 安定と仮定
        bg_diff = 0.0

    bg_stability = 1.0 - min(bg_diff / bg_norm_factor, 1.0)

    # 赤色拡大指標
    red_expansion = newly_red_ratio * bg_stability

    return {
        "red_ratio": red_ratio,
        "newly_red_ratio": newly_red_ratio,
        "bg_stability": bg_stability,
        "red_expansion": red_expansion,
    }


# ---------------------------------------------------------------------------
# メインパイプライン
# ---------------------------------------------------------------------------

def record_timeseries(
    video_path: str,
    outdir: str,
    fps: float = 5.0,
    s_min: int = 60,
    v_min: int = 40,
    roi_margin: float = 0.08,
    no_roi: bool = False,
    smooth_s: float = 5.0,
    bg_norm_factor: float = 30.0,
) -> dict:
    """
    Step 1: 動画をサンプリングしてCSVを出力する（時系列のみ）。

    Args:
        video_path: 入力動画ファイルパス
        outdir: 出力ディレクトリ
        fps: サンプリングFPS（デフォルト5）
        s_min: HSV彩度最小値
        v_min: HSV明度最小値
        roi_margin: 円形ROIマージン
        no_roi: TrueならROIを無効化
        smooth_s: 平滑化窓サイズ（秒）
        bg_norm_factor: 背景差分の正規化係数

    Returns:
        {"csv": CSVファイルパス}
    """
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    stem = Path(video_path).stem

    # --- サンプリング ---
    times: List[float] = []
    red_ratios: List[float] = []
    newly_red_ratios: List[float] = []
    bg_stabilities: List[float] = []
    red_expansions: List[float] = []
    reader_name = "opencv"
    roi_mask: Optional[np.ndarray] = None
    roi_initialized = False
    prev_bgr: Optional[np.ndarray] = None

    for t_sec, bgr, reader in iter_frames(video_path, fps):
        reader_name = reader

        # 最初のフレームでROIマスクを初期化
        if not roi_initialized:
            h, w = bgr.shape[:2]
            if not no_roi:
                roi_mask = make_circular_roi(h, w, margin=roi_margin)
            roi_initialized = True

        if prev_bgr is None:
            # 最初のフレーム: 比較対象がないので初期値
            from src.red.redlog import compute_red_ratio
            ratio = compute_red_ratio(bgr, roi_mask, s_min=s_min, v_min=v_min)
            times.append(t_sec)
            red_ratios.append(ratio)
            newly_red_ratios.append(0.0)
            bg_stabilities.append(1.0)
            red_expansions.append(0.0)
        else:
            metrics = compute_red_expansion(
                prev_bgr, bgr, roi_mask,
                s_min=s_min, v_min=v_min,
                bg_norm_factor=bg_norm_factor,
            )
            times.append(t_sec)
            red_ratios.append(metrics["red_ratio"])
            newly_red_ratios.append(metrics["newly_red_ratio"])
            bg_stabilities.append(metrics["bg_stability"])
            red_expansions.append(metrics["red_expansion"])

        prev_bgr = bgr.copy()

    if len(times) == 0:
        print("警告: フレームが取得できませんでした。", file=sys.stderr)
        return {}

    # --- smooth_expansion ---
    window_size = max(1, int(round(smooth_s * fps)))
    smooth_expansions = smooth_center(red_expansions, window_size)

    # ===================== CSV出力 =====================
    csv_path = out_path / f"{stem}_bleedlog.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "t_sec", "t_srt", "red_ratio",
            "newly_red_ratio", "bg_stability",
            "red_expansion", "smooth_expansion", "reader",
        ])
        for i in range(len(times)):
            writer.writerow([
                f"{times[i]:.3f}",
                format_srt_time(times[i]),
                f"{red_ratios[i]:.6f}",
                f"{newly_red_ratios[i]:.6f}",
                f"{bg_stabilities[i]:.6f}",
                f"{red_expansions[i]:.6f}",
                f"{smooth_expansions[i]:.6f}",
                reader_name,
            ])

    print(f"CSV  : {csv_path}")
    return {"csv": str(csv_path)}


# ---------------------------------------------------------------------------
# CSV読み込み
# ---------------------------------------------------------------------------

def read_bleedlog_csv(csv_path: str) -> dict:
    """
    赤色拡大ログCSVを読み込む。

    Args:
        csv_path: CSVファイルパス

    Returns:
        {"times": [...], "red_ratios": [...], "newly_red_ratios": [...],
         "bg_stabilities": [...], "red_expansions": [...],
         "smooth_expansions": [...], "reader": "...", "fps": float}
    """
    times: List[float] = []
    red_ratios: List[float] = []
    newly_red_ratios: List[float] = []
    bg_stabilities: List[float] = []
    red_expansions: List[float] = []
    smooth_expansions: List[float] = []
    reader = "unknown"

    with open(csv_path, "r", encoding="utf-8") as f:
        reader_obj = csv.DictReader(f)
        for row in reader_obj:
            times.append(float(row["t_sec"]))
            red_ratios.append(float(row["red_ratio"]))
            newly_red_ratios.append(float(row["newly_red_ratio"]))
            bg_stabilities.append(float(row["bg_stability"]))
            red_expansions.append(float(row["red_expansion"]))
            smooth_expansions.append(float(row["smooth_expansion"]))
            reader = row.get("reader", "unknown")

    # fpsの推定
    if len(times) >= 2:
        fps = 1.0 / (times[1] - times[0])
    else:
        fps = 5.0

    return {
        "times": times,
        "red_ratios": red_ratios,
        "newly_red_ratios": newly_red_ratios,
        "bg_stabilities": bg_stabilities,
        "red_expansions": red_expansions,
        "smooth_expansions": smooth_expansions,
        "reader": reader,
        "fps": fps,
    }


# ---------------------------------------------------------------------------
# 出血アノテーション
# ---------------------------------------------------------------------------

def annotate_bleed(
    csv_path: str,
    outdir: str,
    thr: float = 0.005,
    k_s: float = 1.0,
    smooth_s: float = 5.0,
) -> dict:
    """
    Step 2: CSVから閾値ベースで出血イベントを抽出し、JSONL/SRTを出力する。

    Args:
        csv_path: 入力CSVファイルパス（赤色拡大ログ）
        outdir: 出力ディレクトリ
        thr: 出血候補閾値（red_expansion ベース、デフォルト: 0.005）
        k_s: 連続条件（秒、デフォルト: 1.0）
        smooth_s: 平滑化窓サイズ（秒）

    Returns:
        {"jsonl": JSONLファイルパス, "srt": SRTファイルパス, "events": イベント数}
    """
    data = read_bleedlog_csv(csv_path)
    times = data["times"]
    smooth_expansions = data["smooth_expansions"]
    fps = data["fps"]

    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    stem = Path(csv_path).stem.replace("_bleedlog", "")

    # ===================== イベント抽出 =====================
    # extract_bleed_events を再利用（smooth_deltas の代わりに smooth_expansions）
    events = extract_bleed_events(times, smooth_expansions, thr, k_s, fps, smooth_s)

    # イベントの metric を上書き
    for ev in events:
        ev["metric"] = "red_expansion"

    # ===================== JSONL出力（正本） =====================
    jsonl_path = out_path / f"{stem}_bleed_events.jsonl"
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

    # ===================== SRT出力 =====================
    from src.tools.jsonl_to_srt import convert as jsonl_to_srt_convert
    srt_path = out_path / f"{stem}_bleed_expansion.srt"
    jsonl_to_srt_convert(
        in_jsonl=str(jsonl_path),
        out_srt=str(srt_path),
        event_type="bleed_candidate",
    )

    print(f"JSONL: {jsonl_path} （正本）")
    print(f"SRT  : {srt_path} （JSONLから変換）")
    print(f"イベント数: {len(events)} （thr={thr}, k_s={k_s}）")

    return {
        "jsonl": str(jsonl_path),
        "srt": str(srt_path),
        "events": len(events),
    }


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
    smooth_s: float = 5.0,
    bg_norm_factor: float = 30.0,
    thr: float = 0.005,
    k_s: float = 1.0,
) -> dict:
    """
    動画を解析し、CSV・SRT・JSONLを出力する（2ステップの一括実行）。

    Args:
        video_path: 入力動画ファイルパス
        outdir: 出力ディレクトリ
        fps: サンプリングFPS（デフォルト5）
        s_min: HSV彩度最小値
        v_min: HSV明度最小値
        roi_margin: 円形ROIマージン
        no_roi: TrueならROIを無効化
        smooth_s: 平滑化窓サイズ（秒）
        bg_norm_factor: 背景差分の正規化係数
        thr: 出血候補閾値
        k_s: 連続条件（秒）

    Returns:
        出力ファイルパスの辞書
    """
    result1 = record_timeseries(
        video_path=video_path,
        outdir=outdir,
        fps=fps,
        s_min=s_min,
        v_min=v_min,
        roi_margin=roi_margin,
        no_roi=no_roi,
        smooth_s=smooth_s,
        bg_norm_factor=bg_norm_factor,
    )

    if not result1:
        return {}

    result2 = annotate_bleed(
        csv_path=result1["csv"],
        outdir=outdir,
        thr=thr,
        k_s=k_s,
        smooth_s=smooth_s,
    )

    return {**result1, **result2}


# ---------------------------------------------------------------------------
# CLI エントリポイント
# ---------------------------------------------------------------------------

def main() -> int:
    """コマンドラインエントリポイント（サブコマンド方式）"""
    parser = argparse.ArgumentParser(
        description="手術動画の赤色拡大解析（出血検出）"
    )
    subparsers = parser.add_subparsers(dest="command", help="実行コマンド")

    # --- サブコマンド: timeseries ---
    ts_parser = subparsers.add_parser(
        "timeseries", help="Step 1: 時系列記録（動画→CSV）"
    )
    ts_parser.add_argument("--video", required=True, help="入力動画ファイルパス")
    ts_parser.add_argument("--outdir", required=True, help="出力ディレクトリ")
    ts_parser.add_argument("--fps", type=float, default=5.0,
                           help="サンプリングFPS（デフォルト: 5）")
    ts_parser.add_argument("--s-min", type=int, default=60,
                           help="HSV彩度最小値（デフォルト: 60）")
    ts_parser.add_argument("--v-min", type=int, default=40,
                           help="HSV明度最小値（デフォルト: 40）")
    ts_parser.add_argument("--roi-margin", type=float, default=0.08,
                           help="円形ROIマージン（デフォルト: 0.08）")
    ts_parser.add_argument("--no-roi", action="store_true",
                           help="ROIを無効にする")
    ts_parser.add_argument("--smooth-s", type=float, default=5.0,
                           help="平滑化窓（秒、デフォルト: 5）")
    ts_parser.add_argument("--bg-norm", type=float, default=30.0,
                           help="背景差分の正規化係数（デフォルト: 30）")

    # --- サブコマンド: annotate ---
    ann_parser = subparsers.add_parser(
        "annotate", help="Step 2: 出血アノテーション（CSV→JSONL/SRT）"
    )
    ann_parser.add_argument("--csv", required=True,
                            help="入力CSVファイル（赤色拡大ログ）")
    ann_parser.add_argument("--outdir", required=True, help="出力ディレクトリ")
    ann_parser.add_argument("--thr", type=float, default=0.005,
                            help="出血候補閾値（デフォルト: 0.005）")
    ann_parser.add_argument("--k-s", type=float, default=1.0,
                            help="連続条件（秒、デフォルト: 1.0）")
    ann_parser.add_argument("--smooth-s", type=float, default=5.0,
                            help="平滑化窓（秒、デフォルト: 5）")

    # --- サブコマンド: analyze ---
    ana_parser = subparsers.add_parser(
        "analyze", help="一括実行（timeseries + annotate）"
    )
    ana_parser.add_argument("--video", required=True, help="入力動画ファイルパス")
    ana_parser.add_argument("--outdir", required=True, help="出力ディレクトリ")
    ana_parser.add_argument("--fps", type=float, default=5.0,
                            help="サンプリングFPS（デフォルト: 5）")
    ana_parser.add_argument("--s-min", type=int, default=60,
                            help="HSV彩度最小値（デフォルト: 60）")
    ana_parser.add_argument("--v-min", type=int, default=40,
                            help="HSV明度最小値（デフォルト: 40）")
    ana_parser.add_argument("--roi-margin", type=float, default=0.08,
                            help="円形ROIマージン（デフォルト: 0.08）")
    ana_parser.add_argument("--no-roi", action="store_true",
                            help="ROIを無効にする")
    ana_parser.add_argument("--smooth-s", type=float, default=5.0,
                            help="平滑化窓（秒、デフォルト: 5）")
    ana_parser.add_argument("--bg-norm", type=float, default=30.0,
                            help="背景差分の正規化係数（デフォルト: 30）")
    ana_parser.add_argument("--thr", type=float, default=0.005,
                            help="出血候補閾値（デフォルト: 0.005）")
    ana_parser.add_argument("--k-s", type=float, default=1.0,
                            help="連続条件（秒、デフォルト: 1.0）")

    args = parser.parse_args()

    if args.command == "timeseries":
        record_timeseries(
            video_path=args.video,
            outdir=args.outdir,
            fps=args.fps,
            s_min=args.s_min,
            v_min=args.v_min,
            roi_margin=args.roi_margin,
            no_roi=args.no_roi,
            smooth_s=args.smooth_s,
            bg_norm_factor=args.bg_norm,
        )
    elif args.command == "annotate":
        annotate_bleed(
            csv_path=args.csv,
            outdir=args.outdir,
            thr=args.thr,
            k_s=args.k_s,
            smooth_s=args.smooth_s,
        )
    elif args.command == "analyze":
        analyze_video(
            video_path=args.video,
            outdir=args.outdir,
            fps=args.fps,
            s_min=args.s_min,
            v_min=args.v_min,
            roi_margin=args.roi_margin,
            no_roi=args.no_roi,
            smooth_s=args.smooth_s,
            bg_norm_factor=args.bg_norm,
            thr=args.thr,
            k_s=args.k_s,
        )
    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
