"""
局所赤色拡散検出モジュール（bleed_spread）

フレームをグリッド分割し、各セルの赤色率を独立追跡する。
「一部のセルだけ赤色が増加し、他のセルは変化しない」パターンを
出血候補として検出する。

アルゴリズム:
  1. フレームを grid_size × grid_size に分割
  2. 各セルの赤色率を算出 → cell_ratios[grid][grid]
  3. 前フレームとの差分 → cell_deltas[grid][grid]
  4. cell_deltas の標準偏差を計算:
     - 標準偏差 小 = 均等変化（カメラ移動）→ 検出しない
     - 標準偏差 大 = 局所変化（出血候補）→ 検出する
  5. spread_score = delta_std × max_cell_delta

2段階の処理:
  Step 1 - record_timeseries(): 動画→CSV（局所拡散スコアの時系列記録）
  Step 2 - annotate_bleed():    CSV→JSONL/SRT（閾値ベースの出血アノテーション）

出力: CSV（拡散スコアログ）、SRT（出血候補イベント）、JSONL（イベント正本）
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
    compute_red_ratio,
)


# ---------------------------------------------------------------------------
# セル別赤色率の計算
# ---------------------------------------------------------------------------

def compute_cell_ratios(
    frame_bgr: np.ndarray,
    grid_size: int = 8,
    roi_mask: Optional[np.ndarray] = None,
    s_min: int = 60,
    v_min: int = 40,
) -> np.ndarray:
    """
    フレームをグリッド分割し、各セルの赤色率を算出する。

    Args:
        frame_bgr: BGR画像（OpenCV形式）
        grid_size: グリッドの分割数（デフォルト8 → 8×8=64セル）
        roi_mask: 円形ROI。Noneなら全画素。
        s_min: 彩度最小値
        v_min: 明度最小値

    Returns:
        赤色率の2D配列（grid_size × grid_size, 各要素 0〜1）
    """
    h, w = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # 赤色マスク
    mask1 = cv2.inRange(hsv, np.array([0, s_min, v_min]), np.array([10, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([170, s_min, v_min]), np.array([179, 255, 255]))
    red_mask = mask1 | mask2

    if roi_mask is not None:
        roi_u8 = roi_mask.astype(np.uint8) * 255
        red_mask = red_mask & roi_u8

    cell_ratios = np.zeros((grid_size, grid_size), dtype=np.float64)
    cell_h = h / grid_size
    cell_w = w / grid_size

    for r in range(grid_size):
        for c in range(grid_size):
            y0 = int(r * cell_h)
            y1 = int((r + 1) * cell_h)
            x0 = int(c * cell_w)
            x1 = int((c + 1) * cell_w)

            cell_red = red_mask[y0:y1, x0:x1]
            if roi_mask is not None:
                cell_roi = roi_mask[y0:y1, x0:x1]
                total = int(np.count_nonzero(cell_roi))
            else:
                total = (y1 - y0) * (x1 - x0)

            if total > 0:
                cell_ratios[r, c] = int(np.count_nonzero(cell_red)) / total
            else:
                cell_ratios[r, c] = 0.0

    return cell_ratios


# ---------------------------------------------------------------------------
# 拡散スコアの計算
# ---------------------------------------------------------------------------

def compute_spread_score(
    prev_cells: np.ndarray,
    curr_cells: np.ndarray,
) -> dict:
    """
    連続する2フレームのセル赤色率から拡散スコアを算出する。

    Args:
        prev_cells: 前フレームのセル赤色率（grid × grid）
        curr_cells: 現フレームのセル赤色率（grid × grid）

    Returns:
        {
            "max_cell_delta": float,   # セル差分の最大値（正の方向のみ）
            "delta_std": float,        # セル差分の標準偏差
            "spread_score": float,     # 拡散スコア（delta_std × max_cell_delta）
            "n_rising_cells": int,     # 赤色率が上昇したセルの数
        }
    """
    cell_deltas = curr_cells - prev_cells

    # 正の差分のみに注目（赤色が増加したセル）
    positive_deltas = np.maximum(cell_deltas, 0.0)

    max_cell_delta = float(np.max(positive_deltas))
    delta_std = float(np.std(cell_deltas))

    # 有意に上昇したセルの数（差分 > 0.01）
    n_rising = int(np.sum(positive_deltas > 0.01))

    # 拡散スコア: 標準偏差が大きい（局所変化）× 最大差分が大きい（明確な変化）
    spread_score = delta_std * max_cell_delta

    return {
        "max_cell_delta": max_cell_delta,
        "delta_std": delta_std,
        "spread_score": spread_score,
        "n_rising_cells": n_rising,
    }


# ---------------------------------------------------------------------------
# メインパイプライン
# ---------------------------------------------------------------------------

def record_timeseries(
    video_path: str,
    outdir: str,
    fps: float = 5.0,
    grid_size: int = 8,
    s_min: int = 60,
    v_min: int = 40,
    roi_margin: float = 0.08,
    no_roi: bool = False,
    smooth_s: float = 5.0,
) -> dict:
    """
    Step 1: 動画をサンプリングしてCSVを出力する（時系列のみ）。

    Args:
        video_path: 入力動画ファイルパス
        outdir: 出力ディレクトリ
        fps: サンプリングFPS（デフォルト5）
        grid_size: グリッド分割数（デフォルト8）
        s_min: HSV彩度最小値
        v_min: HSV明度最小値
        roi_margin: 円形ROIマージン
        no_roi: TrueならROIを無効化
        smooth_s: 平滑化窓サイズ（秒）

    Returns:
        {"csv": CSVファイルパス}
    """
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    stem = Path(video_path).stem

    # --- サンプリング ---
    times: List[float] = []
    red_ratios: List[float] = []
    max_cell_deltas: List[float] = []
    delta_stds: List[float] = []
    spread_scores: List[float] = []
    n_rising_cells_list: List[int] = []
    reader_name = "opencv"
    roi_mask: Optional[np.ndarray] = None
    roi_initialized = False
    prev_cells: Optional[np.ndarray] = None

    for t_sec, bgr, reader in iter_frames(video_path, fps):
        reader_name = reader

        # 最初のフレームでROIマスクを初期化
        if not roi_initialized:
            h, w = bgr.shape[:2]
            if not no_roi:
                roi_mask = make_circular_roi(h, w, margin=roi_margin)
            roi_initialized = True

        # 全体の赤色率
        ratio = compute_red_ratio(bgr, roi_mask, s_min=s_min, v_min=v_min)

        # セル別赤色率
        curr_cells = compute_cell_ratios(
            bgr, grid_size=grid_size, roi_mask=roi_mask,
            s_min=s_min, v_min=v_min,
        )

        if prev_cells is None:
            # 最初のフレーム
            times.append(t_sec)
            red_ratios.append(ratio)
            max_cell_deltas.append(0.0)
            delta_stds.append(0.0)
            spread_scores.append(0.0)
            n_rising_cells_list.append(0)
        else:
            metrics = compute_spread_score(prev_cells, curr_cells)
            times.append(t_sec)
            red_ratios.append(ratio)
            max_cell_deltas.append(metrics["max_cell_delta"])
            delta_stds.append(metrics["delta_std"])
            spread_scores.append(metrics["spread_score"])
            n_rising_cells_list.append(metrics["n_rising_cells"])

        prev_cells = curr_cells.copy()

    if len(times) == 0:
        print("警告: フレームが取得できませんでした。", file=sys.stderr)
        return {}

    # --- smooth_spread ---
    window_size = max(1, int(round(smooth_s * fps)))
    smooth_spreads = smooth_center(spread_scores, window_size)

    # ===================== CSV出力 =====================
    csv_path = out_path / f"{stem}_spreadlog.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "t_sec", "t_srt", "red_ratio",
            "max_cell_delta", "delta_std",
            "spread_score", "smooth_spread",
            "n_rising_cells", "reader",
        ])
        for i in range(len(times)):
            writer.writerow([
                f"{times[i]:.3f}",
                format_srt_time(times[i]),
                f"{red_ratios[i]:.6f}",
                f"{max_cell_deltas[i]:.6f}",
                f"{delta_stds[i]:.6f}",
                f"{spread_scores[i]:.6f}",
                f"{smooth_spreads[i]:.6f}",
                n_rising_cells_list[i],
                reader_name,
            ])

    print(f"CSV  : {csv_path}")
    return {"csv": str(csv_path)}


# ---------------------------------------------------------------------------
# CSV読み込み
# ---------------------------------------------------------------------------

def read_spreadlog_csv(csv_path: str) -> dict:
    """
    拡散スコアログCSVを読み込む。

    Args:
        csv_path: CSVファイルパス

    Returns:
        {"times": [...], "red_ratios": [...], "max_cell_deltas": [...],
         "delta_stds": [...], "spread_scores": [...],
         "smooth_spreads": [...], "n_rising_cells": [...],
         "reader": "...", "fps": float}
    """
    times: List[float] = []
    red_ratios: List[float] = []
    max_cell_deltas: List[float] = []
    delta_stds: List[float] = []
    spread_scores: List[float] = []
    smooth_spreads: List[float] = []
    n_rising_cells: List[int] = []
    reader = "unknown"

    with open(csv_path, "r", encoding="utf-8") as f:
        reader_obj = csv.DictReader(f)
        for row in reader_obj:
            times.append(float(row["t_sec"]))
            red_ratios.append(float(row["red_ratio"]))
            max_cell_deltas.append(float(row["max_cell_delta"]))
            delta_stds.append(float(row["delta_std"]))
            spread_scores.append(float(row["spread_score"]))
            smooth_spreads.append(float(row["smooth_spread"]))
            n_rising_cells.append(int(row["n_rising_cells"]))
            reader = row.get("reader", "unknown")

    # fpsの推定
    if len(times) >= 2:
        fps = 1.0 / (times[1] - times[0])
    else:
        fps = 5.0

    return {
        "times": times,
        "red_ratios": red_ratios,
        "max_cell_deltas": max_cell_deltas,
        "delta_stds": delta_stds,
        "spread_scores": spread_scores,
        "smooth_spreads": smooth_spreads,
        "n_rising_cells": n_rising_cells,
        "reader": reader,
        "fps": fps,
    }


# ---------------------------------------------------------------------------
# 出血アノテーション
# ---------------------------------------------------------------------------

def annotate_bleed(
    csv_path: str,
    outdir: str,
    thr: float = 0.001,
    k_s: float = 1.0,
    smooth_s: float = 5.0,
) -> dict:
    """
    Step 2: CSVから閾値ベースで出血イベントを抽出し、JSONL/SRTを出力する。

    Args:
        csv_path: 入力CSVファイルパス（拡散スコアログ）
        outdir: 出力ディレクトリ
        thr: 出血候補閾値（spread_score ベース、デフォルト: 0.001）
        k_s: 連続条件（秒、デフォルト: 1.0）
        smooth_s: 平滑化窓サイズ（秒）

    Returns:
        {"jsonl": JSONLファイルパス, "srt": SRTファイルパス, "events": イベント数}
    """
    data = read_spreadlog_csv(csv_path)
    times = data["times"]
    smooth_spreads = data["smooth_spreads"]
    fps = data["fps"]

    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    stem = Path(csv_path).stem.replace("_spreadlog", "")

    # ===================== イベント抽出 =====================
    events = extract_bleed_events(times, smooth_spreads, thr, k_s, fps, smooth_s)

    # イベントの metric を上書き
    for ev in events:
        ev["metric"] = "spread_score"

    # ===================== JSONL出力（正本） =====================
    jsonl_path = out_path / f"{stem}_spread_events.jsonl"
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
    srt_path = out_path / f"{stem}_bleed_spread.srt"
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
    grid_size: int = 8,
    s_min: int = 60,
    v_min: int = 40,
    roi_margin: float = 0.08,
    no_roi: bool = False,
    smooth_s: float = 5.0,
    thr: float = 0.001,
    k_s: float = 1.0,
) -> dict:
    """動画を解析し、CSV・SRT・JSONLを出力する（2ステップの一括実行）。"""
    result1 = record_timeseries(
        video_path=video_path,
        outdir=outdir,
        fps=fps,
        grid_size=grid_size,
        s_min=s_min,
        v_min=v_min,
        roi_margin=roi_margin,
        no_roi=no_roi,
        smooth_s=smooth_s,
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
        description="手術動画の局所赤色拡散解析（出血検出）"
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
    ts_parser.add_argument("--grid-size", type=int, default=8,
                           help="グリッド分割数（デフォルト: 8）")
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

    # --- サブコマンド: annotate ---
    ann_parser = subparsers.add_parser(
        "annotate", help="Step 2: 出血アノテーション（CSV→JSONL/SRT）"
    )
    ann_parser.add_argument("--csv", required=True,
                            help="入力CSVファイル（拡散スコアログ）")
    ann_parser.add_argument("--outdir", required=True, help="出力ディレクトリ")
    ann_parser.add_argument("--thr", type=float, default=0.001,
                            help="出血候補閾値（デフォルト: 0.001）")
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
    ana_parser.add_argument("--grid-size", type=int, default=8,
                            help="グリッド分割数（デフォルト: 8）")
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
    ana_parser.add_argument("--thr", type=float, default=0.001,
                            help="出血候補閾値（デフォルト: 0.001）")
    ana_parser.add_argument("--k-s", type=float, default=1.0,
                            help="連続条件（秒、デフォルト: 1.0）")

    args = parser.parse_args()

    if args.command == "timeseries":
        record_timeseries(
            video_path=args.video,
            outdir=args.outdir,
            fps=args.fps,
            grid_size=args.grid_size,
            s_min=args.s_min,
            v_min=args.v_min,
            roi_margin=args.roi_margin,
            no_roi=args.no_roi,
            smooth_s=args.smooth_s,
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
            grid_size=args.grid_size,
            s_min=args.s_min,
            v_min=args.v_min,
            roi_margin=args.roi_margin,
            no_roi=args.no_roi,
            smooth_s=args.smooth_s,
            thr=args.thr,
            k_s=args.k_s,
        )
    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
