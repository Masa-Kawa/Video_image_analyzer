"""
赤色解析モジュール（HSV / 5fps）

腹腔鏡手術動画の赤色率をサンプリングし、出血候補イベントを検出する。
出力: CSV（赤色率ログ）、SRT（出血候補イベント）、JSONL（イベント正本）
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# SRT 時間フォーマット
# ---------------------------------------------------------------------------

def format_srt_time(seconds: float) -> str:
    """秒数を HH:MM:SS,mmm 形式に変換する"""
    if seconds < 0:
        seconds = 0.0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds % 1) * 1000))
    if ms >= 1000:
        ms = 999
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# ---------------------------------------------------------------------------
# 円形ROIマスク生成
# ---------------------------------------------------------------------------

def make_circular_roi(height: int, width: int, margin: float = 0.08) -> np.ndarray:
    """
    円形ROIマスクを生成する。

    Args:
        height: フレームの高さ
        width: フレームの幅
        margin: 外周マージン（0〜0.5）

    Returns:
        boolマスク（ROI内がTrue）
    """
    cy, cx = height / 2.0, width / 2.0
    radius = min(height, width) * (0.5 - margin)
    y_grid, x_grid = np.ogrid[:height, :width]
    dist = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)
    return dist <= radius


# ---------------------------------------------------------------------------
# 赤色率の計算
# ---------------------------------------------------------------------------

def compute_red_ratio(
    frame_bgr: np.ndarray,
    roi_mask: Optional[np.ndarray],
    s_min: int = 60,
    v_min: int = 40,
) -> float:
    """
    フレームからHSVベースの赤色率を算出する。

    Args:
        frame_bgr: BGR画像（OpenCV形式）
        roi_mask: 円形ROI（Trueの画素のみ集計）。Noneなら全画素。
        s_min: 彩度最小値
        v_min: 明度最小値

    Returns:
        赤色率（0〜1）
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # 赤色マスク: H in [0..10] OR H in [170..179]
    mask1 = cv2.inRange(hsv, np.array([0, s_min, v_min]), np.array([10, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([170, s_min, v_min]), np.array([179, 255, 255]))
    red_mask = mask1 | mask2

    if roi_mask is not None:
        roi_u8 = roi_mask.astype(np.uint8) * 255
        red_mask = red_mask & roi_u8
        total_pixels = int(np.count_nonzero(roi_mask))
    else:
        total_pixels = frame_bgr.shape[0] * frame_bgr.shape[1]

    if total_pixels == 0:
        return 0.0

    red_pixels = int(np.count_nonzero(red_mask))
    return red_pixels / total_pixels


# ---------------------------------------------------------------------------
# 平滑化（中心移動平均）
# ---------------------------------------------------------------------------

def smooth_center(values: List[float], window: int) -> List[float]:
    """
    中心移動平均で平滑化する。

    Args:
        values: 入力信号
        window: 窓サイズ（奇数でなくても可、内部で対称に切り取る）

    Returns:
        平滑化信号（同じ長さ）
    """
    n = len(values)
    if n == 0 or window <= 1:
        return list(values)

    half = window // 2
    result = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        result.append(sum(values[lo:hi]) / (hi - lo))
    return result


# ---------------------------------------------------------------------------
# イベント抽出
# ---------------------------------------------------------------------------

def extract_bleed_events(
    times: List[float],
    smooth_deltas: List[float],
    thr: float,
    k_s: float,
    fps: float,
    smooth_s: float,
) -> List[dict]:
    """
    smooth_delta > thr が k_s秒以上連続する区間を出血候補イベントとして抽出する。

    Returns:
        イベント辞書のリスト（start, end, delta_max, ...）
    """
    events: List[dict] = []
    n = len(smooth_deltas)
    if n == 0:
        return events

    min_samples = max(1, int(round(k_s * fps)))
    in_event = False
    start_idx = 0

    for i in range(n):
        above = smooth_deltas[i] > thr
        if above and not in_event:
            in_event = True
            start_idx = i
        elif not above and in_event:
            in_event = False
            length = i - start_idx
            if length >= min_samples:
                _add_event(events, times, smooth_deltas, start_idx, i, thr, k_s, smooth_s)

    # 末尾で区間が終わるケース
    if in_event:
        length = n - start_idx
        if length >= min_samples:
            _add_event(events, times, smooth_deltas, start_idx, n, thr, k_s, smooth_s)

    return events


def _add_event(
    events: List[dict],
    times: List[float],
    smooth_deltas: List[float],
    start_idx: int,
    end_idx: int,
    thr: float,
    k_s: float,
    smooth_s: float,
) -> None:
    """イベント辞書をリストに追加するヘルパー"""
    delta_max = max(smooth_deltas[start_idx:end_idx])
    events.append({
        "type": "bleed_candidate",
        "metric": "red_ratio",
        "thr": thr,
        "k_s": k_s,
        "smooth_s": smooth_s,
        "delta_max": round(delta_max, 6),
        "start": times[start_idx],
        "end": times[end_idx - 1],
    })


# ---------------------------------------------------------------------------
# フレーム読取り（PyAV / OpenCV フォールバック）
# ---------------------------------------------------------------------------

def _iter_frames_pyav(video_path: str, fps: float):
    """PyAVでPTSベースのフレームを取得するジェネレータ"""
    import av  # pylint: disable=import-outside-toplevel

    container = av.open(video_path)
    stream = container.streams.video[0]
    time_base = float(stream.time_base)

    # サンプル間隔（秒）
    interval = 1.0 / fps
    next_t = 0.0

    for frame in container.decode(stream):
        pts_sec = frame.pts * time_base if frame.pts is not None else None
        if pts_sec is None:
            continue
        if pts_sec >= next_t:
            bgr = frame.to_ndarray(format="bgr24")
            yield pts_sec, bgr
            next_t = pts_sec + interval

    container.close()


def _iter_frames_opencv(video_path: str, fps: float):
    """OpenCVでPOS_MSECベースのフレームを取得するジェネレータ"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"動画ファイルを開けません: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps <= 0:
        src_fps = 30.0

    # サンプル間隔（フレーム単位）
    step = max(1, int(round(src_fps / fps)))
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            t_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            yield t_sec, frame
        idx += 1

    cap.release()


def iter_frames(video_path: str, fps: float):
    """
    フレームイテレータ。PyAVを優先し、なければOpenCVにフォールバック。

    Yields:
        (t_sec, bgr_frame, reader_name)
    """
    try:
        for t, bgr in _iter_frames_pyav(video_path, fps):
            yield t, bgr, "pyav"
        return
    except Exception:
        pass

    for t, bgr in _iter_frames_opencv(video_path, fps):
        yield t, bgr, "opencv"


# ---------------------------------------------------------------------------
# メインパイプライン
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
    thr: float = 0.03,
    k_s: float = 3.0,
) -> dict:
    """
    動画を解析し、CSV・SRT・JSONLを出力する。

    Args:
        video_path: 入力動画ファイルパス
        outdir: 出力ディレクトリ
        fps: サンプリングFPS（デフォルト5）
        s_min: HSV彩度最小値
        v_min: HSV明度最小値
        roi_margin: 円形ROIマージン
        no_roi: TrueならROIを無効化
        smooth_s: 平滑化窓サイズ（秒）
        thr: 出血候補閾値
        k_s: 連続条件（秒）

    Returns:
        出力ファイルパスの辞書
    """
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    stem = Path(video_path).stem

    # --- サンプリング ---
    times: List[float] = []
    ratios: List[float] = []
    reader_name = "opencv"
    roi_mask: Optional[np.ndarray] = None
    roi_initialized = False

    for t_sec, bgr, reader in iter_frames(video_path, fps):
        reader_name = reader

        # 最初のフレームでROIマスクを初期化
        if not roi_initialized:
            h, w = bgr.shape[:2]
            if not no_roi:
                roi_mask = make_circular_roi(h, w, margin=roi_margin)
            roi_initialized = True

        ratio = compute_red_ratio(bgr, roi_mask, s_min=s_min, v_min=v_min)
        times.append(t_sec)
        ratios.append(ratio)

    if len(times) == 0:
        print("警告: フレームが取得できませんでした。", file=sys.stderr)
        return {}

    # --- delta ---
    deltas = [0.0] + [ratios[i] - ratios[i - 1] for i in range(1, len(ratios))]

    # --- smooth_delta ---
    window_size = max(1, int(round(smooth_s * fps)))
    smooth_deltas = smooth_center(deltas, window_size)

    # ===================== CSV出力 =====================
    csv_path = out_path / f"{stem}_redlog.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["t_sec", "t_srt", "red_ratio", "delta", "smooth_delta", "reader"])
        for i in range(len(times)):
            writer.writerow([
                f"{times[i]:.3f}",
                format_srt_time(times[i]),
                f"{ratios[i]:.6f}",
                f"{deltas[i]:.6f}",
                f"{smooth_deltas[i]:.6f}",
                reader_name,
            ])

    # ===================== イベント抽出 =====================
    events = extract_bleed_events(times, smooth_deltas, thr, k_s, fps, smooth_s)

    # ===================== JSONL出力（正本） =====================
    jsonl_path = out_path / f"{stem}_events.jsonl"
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

    # ===================== SRT出力（JSONLから変換） =====================
    from src.tools.jsonl_to_srt import convert as jsonl_to_srt_convert
    srt_path = out_path / f"{stem}_bleed.srt"
    jsonl_to_srt_convert(
        in_jsonl=str(jsonl_path),
        out_srt=str(srt_path),
        event_type="bleed_candidate",
    )

    print(f"CSV  : {csv_path}")
    print(f"SRT  : {srt_path} （JSONLから変換）")
    print(f"JSONL: {jsonl_path} （正本）")
    print(f"イベント数: {len(events)}")

    return {
        "csv": str(csv_path),
        "srt": str(srt_path),
        "jsonl": str(jsonl_path),
        "events": len(events),
    }


# ---------------------------------------------------------------------------
# CLI エントリポイント
# ---------------------------------------------------------------------------

def main() -> int:
    """コマンドラインエントリポイント"""
    parser = argparse.ArgumentParser(
        description="手術動画の赤色解析（HSV / 5fps）"
    )
    parser.add_argument("--video", required=True, help="入力動画ファイルパス")
    parser.add_argument("--outdir", required=True, help="出力ディレクトリ")
    parser.add_argument("--fps", type=float, default=5.0, help="サンプリングFPS（デフォルト: 5）")
    parser.add_argument("--s-min", type=int, default=60, help="HSV彩度最小値（デフォルト: 60）")
    parser.add_argument("--v-min", type=int, default=40, help="HSV明度最小値（デフォルト: 40）")
    parser.add_argument("--roi-margin", type=float, default=0.08, help="円形ROIマージン（デフォルト: 0.08）")
    parser.add_argument("--no-roi", action="store_true", help="ROIを無効にする")
    parser.add_argument("--smooth-s", type=float, default=5.0, help="平滑化窓（秒、デフォルト: 5）")
    parser.add_argument("--thr", type=float, default=0.03, help="出血候補閾値（デフォルト: 0.03）")
    parser.add_argument("--k-s", type=float, default=3.0, help="連続条件（秒、デフォルト: 3）")

    args = parser.parse_args()

    analyze_video(
        video_path=args.video,
        outdir=args.outdir,
        fps=args.fps,
        s_min=args.s_min,
        v_min=args.v_min,
        roi_margin=args.roi_margin,
        no_roi=args.no_roi,
        smooth_s=args.smooth_s,
        thr=args.thr,
        k_s=args.k_s,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
