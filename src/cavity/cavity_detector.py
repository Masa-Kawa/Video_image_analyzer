"""
腹腔内外判定モジュール（cavity_detector）

腹腔鏡手術動画のフレームが「腹腔内（inside）」か「腹腔外（outside）」かを
判定する。ラベルなしデータで動作するヒューリスティックベースの手法。

判定指標（4つのスコアを統合）:
  1. vignette_score  : 周辺暗域の検出（腹腔鏡の円形視野）
  2. tissue_ratio    : 組織色（赤/茶/黄）の占有率
  3. brightness_bimodality : 明るさの二峰性（中央明・周辺暗）
  4. hue_concentration : 色相の集中度（腹腔内は限定的な色相）

統合スコア: cavity_score = 重み付き平均 → 閾値でinside/outside判定

2段階の処理:
  Step 1 - record_timeseries(): 動画→CSV（cavity指標の時系列記録）
  Step 2 - annotate_cavity():   CSV→JSONL/SRT（inside/outside区間アノテーション）

出力: CSV（指標ログ）、SRT（inside/outside区間）、JSONL（イベント正本）
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
from src.red.redlog import make_circular_roi, iter_frames, smooth_center


# ---------------------------------------------------------------------------
# 個別指標の計算
# ---------------------------------------------------------------------------

def compute_vignette_score(
    frame_bgr: np.ndarray,
    ring_width: float = 0.15,
    gray: Optional[np.ndarray] = None,
) -> float:
    """
    周辺暗域スコア: フレーム外周リングの暗さを測定する。

    腹腔鏡はスコープの円形視野のため、映像の周辺部が暗い。
    外周リングの平均明度が低く、中央部との差が大きいほどスコアが高い。

    Args:
        frame_bgr: BGR画像
        ring_width: 外周リングの幅（画像短辺に対する比率）

    Returns:
        vignette_score: 0.0（暗域なし）〜 1.0（強い暗域）
    """
    if gray is None:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 中央領域と外周リングのマスク
    cy, cx = h / 2.0, w / 2.0
    short_side = min(h, w)
    outer_radius = short_side * 0.5
    inner_radius = short_side * (0.5 - ring_width)

    y_grid, x_grid = np.ogrid[:h, :w]
    dist = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)

    ring_mask = (dist > inner_radius) & (dist <= outer_radius)
    center_mask = dist <= inner_radius * 0.6

    ring_pixels = gray[ring_mask]
    center_pixels = gray[center_mask]

    if len(ring_pixels) == 0 or len(center_pixels) == 0:
        return 0.0

    ring_mean = float(np.mean(ring_pixels))
    center_mean = float(np.mean(center_pixels))

    if center_mean < 1.0:
        return 0.0

    # 中央と外周の明度差の比率
    # 腹腔内: ring_mean << center_mean → 比率が大きい → スコアが高い
    darkness_ratio = max(0.0, 1.0 - ring_mean / center_mean)

    # 外周リングの絶対的な暗さも考慮
    # ring_mean < 40 で暗いと判定
    absolute_darkness = max(0.0, 1.0 - ring_mean / 80.0)

    score = 0.6 * darkness_ratio + 0.4 * absolute_darkness
    return min(1.0, max(0.0, score))


def compute_tissue_ratio(
    frame_bgr: np.ndarray,
    roi_mask: Optional[np.ndarray] = None,
    hsv: Optional[np.ndarray] = None,
) -> float:
    """
    組織色占有率: フレーム内の赤/茶/黄色系（組織色）の占有率。

    腹腔内は臓器・脂肪・組織で赤/茶/黄が支配的。
    腹腔外は皮膚色、ドレープ（青/緑）、器械（銀色）が多い。

    Args:
        frame_bgr: BGR画像
        roi_mask: ROIマスク（Noneなら全画素）

    Returns:
        tissue_ratio: 0.0〜1.0
    """
    if hsv is None:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h_ch = hsv[:, :, 0]
    s_ch = hsv[:, :, 1]
    v_ch = hsv[:, :, 2]

    # 組織色の定義:
    # 赤: H in [0, 15] or [165, 179], S > 30, V > 30
    # 茶/オレンジ: H in [10, 25], S > 30, V > 30
    # 黄: H in [20, 35], S > 30, V > 30
    # → まとめて H in [0, 35] or [165, 179], S > 30, V > 30
    saturation_ok = (s_ch > 30) & (v_ch > 30)
    tissue_hue = ((h_ch <= 35) | (h_ch >= 165)) & saturation_ok

    if roi_mask is not None:
        tissue_hue = tissue_hue & roi_mask
        total = int(np.count_nonzero(roi_mask))
    else:
        total = frame_bgr.shape[0] * frame_bgr.shape[1]

    if total == 0:
        return 0.0

    return int(np.count_nonzero(tissue_hue)) / total


def compute_brightness_bimodality(
    frame_bgr: np.ndarray,
    gray: Optional[np.ndarray] = None,
) -> float:
    """
    明るさの二峰性スコア: 腹腔内は中央が明るく周辺が暗い二峰分布。

    明度ヒストグラムの分散を用いて判定する。
    暗い画素（V < 50）の比率が高く、かつ中間以上の明るさの画素もある場合、
    腹腔内の特徴的な二峰分布と判定。

    Args:
        frame_bgr: BGR画像

    Returns:
        bimodality_score: 0.0〜1.0
    """
    if gray is None:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    dark_ratio = float(np.mean(gray < 50))
    bright_ratio = float(np.mean(gray > 100))

    # 二峰性: 暗い画素も明るい画素も一定割合存在する
    # 腹腔内: dark_ratio > 0.2 かつ bright_ratio > 0.2
    if dark_ratio < 0.05 or bright_ratio < 0.05:
        return 0.0

    # 両方のピークが存在する度合い
    bimodality = min(dark_ratio, bright_ratio) / max(dark_ratio, bright_ratio)
    # 暗い画素の比率も加味（腹腔内は20-60%が暗い）
    dark_penalty = 1.0 if 0.15 <= dark_ratio <= 0.70 else 0.5

    return min(1.0, bimodality * dark_penalty)


def compute_hue_concentration(
    frame_bgr: np.ndarray,
    roi_mask: Optional[np.ndarray] = None,
    hsv: Optional[np.ndarray] = None,
) -> float:
    """
    色相集中度: 有効画素の色相がどれだけ狭い範囲に集中しているか。

    腹腔内は赤〜黄の限られた色相帯に集中する。
    腹腔外は多様な色相（ドレープの青/緑、皮膚、器械の銀）が分散する。

    Args:
        frame_bgr: BGR画像
        roi_mask: ROIマスク

    Returns:
        hue_concentration: 0.0（分散）〜 1.0（集中）
    """
    if hsv is None:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # 彩度・明度が低い画素は色相が不安定なので除外
    valid = (hsv[:, :, 1] > 30) & (hsv[:, :, 2] > 30)
    if roi_mask is not None:
        valid = valid & roi_mask

    hues = hsv[:, :, 0][valid]

    if len(hues) < 100:
        return 0.0

    # 色相ヒストグラム（180ビン、HSVのHは0-179）
    hist, _ = np.histogram(hues, bins=36, range=(0, 180))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 0.0

    hist_norm = hist / total

    # エントロピーベースの集中度
    # 最大エントロピー = log2(36) ≈ 5.17
    eps = 1e-10
    entropy = -np.sum(hist_norm * np.log2(hist_norm + eps))
    max_entropy = np.log2(36)

    # 集中度 = 1 - (entropy / max_entropy)
    concentration = 1.0 - entropy / max_entropy
    return max(0.0, min(1.0, concentration))


# ---------------------------------------------------------------------------
# 統合スコア
# ---------------------------------------------------------------------------

def compute_cavity_score(
    frame_bgr: np.ndarray,
    roi_mask: Optional[np.ndarray] = None,
    weights: Optional[dict] = None,
) -> dict:
    """
    腹腔内外判定の統合スコアを算出する。

    Args:
        frame_bgr: BGR画像
        roi_mask: ROIマスク
        weights: 各指標の重み（デフォルト: 均等）

    Returns:
        {
            "vignette_score": float,
            "tissue_ratio": float,
            "brightness_bimodality": float,
            "hue_concentration": float,
            "cavity_score": float,  # 統合スコア（0=outside, 1=inside）
        }
    """
    if weights is None:
        weights = {
            "vignette": 0.35,
            "tissue": 0.30,
            "bimodality": 0.15,
            "hue_concentration": 0.20,
        }

    # Gray/HSV は全指標で共有するため、ここで1回ずつだけ変換する
    # （従来は各関数内で BGR→Gray×2・BGR→HSV×2 と重複していた）
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    vignette = compute_vignette_score(frame_bgr, gray=gray)
    tissue = compute_tissue_ratio(frame_bgr, roi_mask, hsv=hsv)
    bimodality = compute_brightness_bimodality(frame_bgr, gray=gray)
    hue_conc = compute_hue_concentration(frame_bgr, roi_mask, hsv=hsv)

    cavity_score = (
        weights["vignette"] * vignette
        + weights["tissue"] * tissue
        + weights["bimodality"] * bimodality
        + weights["hue_concentration"] * hue_conc
    )

    return {
        "vignette_score": vignette,
        "tissue_ratio": tissue,
        "brightness_bimodality": bimodality,
        "hue_concentration": hue_conc,
        "cavity_score": cavity_score,
    }


# ---------------------------------------------------------------------------
# メインパイプライン
# ---------------------------------------------------------------------------

def record_timeseries(
    video_path: str,
    outdir: str,
    fps: float = 2.0,
    roi_margin: float = 0.08,
    no_roi: bool = False,
    smooth_s: float = 3.0,
) -> dict:
    """
    Step 1: 動画をサンプリングしてCSVを出力する（cavity指標の時系列）。

    Args:
        video_path: 入力動画ファイルパス
        outdir: 出力ディレクトリ
        fps: サンプリングFPS（デフォルト2 — cavity判定は低FPSで十分）
        roi_margin: 円形ROIマージン
        no_roi: TrueならROIを無効化
        smooth_s: 平滑化窓サイズ（秒）

    Returns:
        {"csv": CSVファイルパス}
    """
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    stem = Path(video_path).stem

    times: List[float] = []
    vignette_scores: List[float] = []
    tissue_ratios: List[float] = []
    bimodalities: List[float] = []
    hue_concentrations: List[float] = []
    cavity_scores: List[float] = []
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

        metrics = compute_cavity_score(bgr, roi_mask)

        times.append(t_sec)
        vignette_scores.append(metrics["vignette_score"])
        tissue_ratios.append(metrics["tissue_ratio"])
        bimodalities.append(metrics["brightness_bimodality"])
        hue_concentrations.append(metrics["hue_concentration"])
        cavity_scores.append(metrics["cavity_score"])

    if len(times) == 0:
        print("警告: フレームが取得できませんでした。", file=sys.stderr)
        return {}

    # 平滑化
    window_size = max(1, int(round(smooth_s * fps)))
    smooth_cavity = smooth_center(cavity_scores, window_size)

    # CSV出力
    csv_path = out_path / f"{stem}_cavitylog.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "t_sec", "t_srt",
            "vignette_score", "tissue_ratio",
            "brightness_bimodality", "hue_concentration",
            "cavity_score", "smooth_cavity",
            "reader",
        ])
        for i in range(len(times)):
            writer.writerow([
                f"{times[i]:.3f}",
                format_srt_time(times[i]),
                f"{vignette_scores[i]:.6f}",
                f"{tissue_ratios[i]:.6f}",
                f"{bimodalities[i]:.6f}",
                f"{hue_concentrations[i]:.6f}",
                f"{cavity_scores[i]:.6f}",
                f"{smooth_cavity[i]:.6f}",
                reader_name,
            ])

    print(f"CSV  : {csv_path}")
    return {"csv": str(csv_path)}


# ---------------------------------------------------------------------------
# CSV読み込み
# ---------------------------------------------------------------------------

def read_cavitylog_csv(csv_path: str) -> dict:
    """
    cavity指標ログCSVを読み込む。

    Returns:
        {"times": [...], "cavity_scores": [...], "smooth_cavity": [...],
         "vignette_scores": [...], "tissue_ratios": [...],
         "bimodalities": [...], "hue_concentrations": [...],
         "reader": str, "fps": float}
    """
    times: List[float] = []
    vignette_scores: List[float] = []
    tissue_ratios: List[float] = []
    bimodalities: List[float] = []
    hue_concentrations: List[float] = []
    cavity_scores: List[float] = []
    smooth_cavity: List[float] = []
    reader = "unknown"

    with open(csv_path, "r", encoding="utf-8") as f:
        reader_obj = csv.DictReader(f)
        for row in reader_obj:
            times.append(float(row["t_sec"]))
            vignette_scores.append(float(row["vignette_score"]))
            tissue_ratios.append(float(row["tissue_ratio"]))
            bimodalities.append(float(row["brightness_bimodality"]))
            hue_concentrations.append(float(row["hue_concentration"]))
            cavity_scores.append(float(row["cavity_score"]))
            smooth_cavity.append(float(row["smooth_cavity"]))
            reader = row.get("reader", "unknown")

    fps = 1.0 / (times[1] - times[0]) if len(times) >= 2 else 2.0

    return {
        "times": times,
        "vignette_scores": vignette_scores,
        "tissue_ratios": tissue_ratios,
        "bimodalities": bimodalities,
        "hue_concentrations": hue_concentrations,
        "cavity_scores": cavity_scores,
        "smooth_cavity": smooth_cavity,
        "reader": reader,
        "fps": fps,
    }


# ---------------------------------------------------------------------------
# 腹腔内外アノテーション
# ---------------------------------------------------------------------------

def annotate_cavity(
    csv_path: str,
    outdir: str,
    thr: float = 0.35,
    min_duration_s: float = 3.0,
    smooth_s: float = 3.0,
) -> dict:
    """
    Step 2: CSVから閾値ベースでinside/outside区間を抽出し、JSONL/SRTを出力する。

    smooth_cavity >= thr の連続区間を「inside」として検出する。

    Args:
        csv_path: 入力CSVファイルパス
        outdir: 出力ディレクトリ
        thr: inside判定閾値（デフォルト: 0.35）
        min_duration_s: 最小区間長（秒、デフォルト: 3.0）
        smooth_s: Step1で適用した平滑化窓サイズ。ここでは記録用にメタデータへ出力する。

    Returns:
        {"jsonl": path, "srt": path, "events": int, "smooth_s": float}
    """
    data = read_cavitylog_csv(csv_path)
    times = data["times"]
    smooth_vals = data["smooth_cavity"]
    fps = data["fps"]

    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    stem = Path(csv_path).stem.replace("_cavitylog", "")

    # 区間抽出: smooth_cavity >= thr → inside
    min_samples = max(1, int(round(min_duration_s * fps)))
    events: List[dict] = []

    in_region = False
    start_idx = 0

    for i in range(len(smooth_vals)):
        above = smooth_vals[i] >= thr
        if above and not in_region:
            in_region = True
            start_idx = i
        elif not above and in_region:
            in_region = False
            if i - start_idx >= min_samples:
                max_score = max(smooth_vals[start_idx:i])
                events.append({
                    "type": "cavity_inside",
                    "label": "inside",
                    "thr": thr,
                    "score_max": round(max_score, 6),
                    "start": times[start_idx],
                    "end": times[i - 1],
                })

    # 末尾処理
    if in_region and len(smooth_vals) - start_idx >= min_samples:
        max_score = max(smooth_vals[start_idx:])
        events.append({
            "type": "cavity_inside",
            "label": "inside",
            "thr": thr,
            "score_max": round(max_score, 6),
            "start": times[start_idx],
            "end": times[-1],
        })

    # outside区間も生成（insideの隙間）
    outside_events: List[dict] = []
    prev_end = 0.0
    for ev in events:
        if ev["start"] > prev_end + 1.0 / fps:
            gap_duration = ev["start"] - prev_end
            if gap_duration >= min_duration_s:
                outside_events.append({
                    "type": "cavity_outside",
                    "label": "outside",
                    "thr": thr,
                    "score_max": 0.0,
                    "start": prev_end,
                    "end": ev["start"],
                })
        prev_end = ev["end"]

    # 最後のinside以降もoutside
    if times and prev_end < times[-1] - min_duration_s:
        outside_events.append({
            "type": "cavity_outside",
            "label": "outside",
            "thr": thr,
            "score_max": 0.0,
            "start": prev_end,
            "end": times[-1],
        })

    all_events = sorted(events + outside_events, key=lambda e: e["start"])

    # JSONL出力
    jsonl_path = out_path / f"{stem}_cavity_events.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for ev in all_events:
            line = {
                "type": ev["type"],
                "label": ev["label"],
                "thr": ev["thr"],
                "score_max": ev["score_max"],
                "start_sec": ev["start"],
                "end_sec": ev["end"],
                "start_srt": format_srt_time(ev["start"]),
                "end_srt": format_srt_time(ev["end"]),
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    # SRT出力
    srt_path = out_path / f"{stem}_cavity.srt"
    srt_lines: List[str] = []
    for idx, ev in enumerate(all_events, start=1):
        start_srt = format_srt_time(ev["start"])
        end_srt = format_srt_time(ev["end"])
        label = ev["label"].upper()
        srt_lines.append(f"{idx}")
        srt_lines.append(f"{start_srt} --> {end_srt}")
        srt_lines.append(f"[cavity] {label}")
        srt_lines.append("")

    Path(srt_path).write_text("\n".join(srt_lines), encoding="utf-8")

    # Step1 の平滑化窓を再現用メタデータとしてサイドカー出力
    meta_path = out_path / f"{stem}_cavity_meta.json"
    meta = {
        "thr": thr,
        "min_duration_s": min_duration_s,
        "smooth_s": smooth_s,
        "fps": fps,
        "events": len(all_events),
        "inside": len(events),
        "outside": len(outside_events),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                         encoding="utf-8")

    print(f"JSONL: {jsonl_path} （正本）")
    print(f"SRT  : {srt_path}")
    print(f"イベント数: {len(all_events)} "
          f"（inside={len(events)}, outside={len(outside_events)}, "
          f"smooth_s={smooth_s}）")

    return {
        "jsonl": str(jsonl_path),
        "srt": str(srt_path),
        "events": len(all_events),
        "smooth_s": smooth_s,
    }


# ---------------------------------------------------------------------------
# 一括実行
# ---------------------------------------------------------------------------

def analyze_video(
    video_path: str,
    outdir: str,
    fps: float = 2.0,
    roi_margin: float = 0.08,
    no_roi: bool = False,
    smooth_s: float = 3.0,
    thr: float = 0.35,
    min_duration_s: float = 3.0,
) -> dict:
    """動画を解析し、CSV・SRT・JSONLを出力する（2ステップの一括実行）。"""
    result1 = record_timeseries(
        video_path=video_path,
        outdir=outdir,
        fps=fps,
        roi_margin=roi_margin,
        no_roi=no_roi,
        smooth_s=smooth_s,
    )

    if not result1:
        return {}

    result2 = annotate_cavity(
        csv_path=result1["csv"],
        outdir=outdir,
        thr=thr,
        min_duration_s=min_duration_s,
        smooth_s=smooth_s,
    )

    return {**result1, **result2}


# ---------------------------------------------------------------------------
# CLI エントリポイント
# ---------------------------------------------------------------------------

def main() -> int:
    """コマンドラインエントリポイント（サブコマンド方式）"""
    parser = argparse.ArgumentParser(
        description="腹腔鏡手術動画の腹腔内外判定"
    )
    subparsers = parser.add_subparsers(dest="command", help="実行コマンド")

    # --- timeseries ---
    ts_parser = subparsers.add_parser(
        "timeseries", help="Step 1: 時系列記録（動画→CSV）"
    )
    ts_parser.add_argument("--video", required=True)
    ts_parser.add_argument("--outdir", required=True)
    ts_parser.add_argument("--fps", type=float, default=2.0)
    ts_parser.add_argument("--roi-margin", type=float, default=0.08)
    ts_parser.add_argument("--no-roi", action="store_true")
    ts_parser.add_argument("--smooth-s", type=float, default=3.0)

    # --- annotate ---
    ann_parser = subparsers.add_parser(
        "annotate", help="Step 2: 内外アノテーション（CSV→JSONL/SRT）"
    )
    ann_parser.add_argument("--csv", required=True)
    ann_parser.add_argument("--outdir", required=True)
    ann_parser.add_argument("--thr", type=float, default=0.35)
    ann_parser.add_argument("--min-duration", type=float, default=3.0)
    ann_parser.add_argument("--smooth-s", type=float, default=3.0)

    # --- analyze ---
    ana_parser = subparsers.add_parser(
        "analyze", help="一括実行（timeseries + annotate）"
    )
    ana_parser.add_argument("--video", required=True)
    ana_parser.add_argument("--outdir", required=True)
    ana_parser.add_argument("--fps", type=float, default=2.0)
    ana_parser.add_argument("--roi-margin", type=float, default=0.08)
    ana_parser.add_argument("--no-roi", action="store_true")
    ana_parser.add_argument("--smooth-s", type=float, default=3.0)
    ana_parser.add_argument("--thr", type=float, default=0.35)
    ana_parser.add_argument("--min-duration", type=float, default=3.0)

    args = parser.parse_args()

    if args.command == "timeseries":
        record_timeseries(
            video_path=args.video, outdir=args.outdir,
            fps=args.fps, roi_margin=args.roi_margin,
            no_roi=args.no_roi, smooth_s=args.smooth_s,
        )
    elif args.command == "annotate":
        annotate_cavity(
            csv_path=args.csv, outdir=args.outdir,
            thr=args.thr, min_duration_s=args.min_duration,
            smooth_s=args.smooth_s,
        )
    elif args.command == "analyze":
        analyze_video(
            video_path=args.video, outdir=args.outdir,
            fps=args.fps, roi_margin=args.roi_margin,
            no_roi=args.no_roi, smooth_s=args.smooth_s,
            thr=args.thr, min_duration_s=args.min_duration,
        )
    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
