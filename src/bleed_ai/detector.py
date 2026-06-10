"""
AI出血検出モジュール（Deep Learning / HSV ハイブリッド）

腹腔鏡手術動画の出血を、深層学習モデル（ResNet-18分類器 + U-Net分割）
とHSVベース色検出のハイブリッドで検出する。

2段階の処理:
  Step 1 - record_timeseries(): 動画→CSV（フレームごとの出血指標）
  Step 2 - annotate_bleed_ai(): CSV→JSONL/SRT（出血イベント検出）

特徴:
  - ResNet-18 (ImageNet pretrained) + HSVブレンドによる出血確率
  - U-Net / HSVマスクによる出血領域セグメンテーション
  - オプティカルフローによる動き解析
  - 複合重症度スコア (severity)
  - 学習済み重みなしでも HSV フォールバックで動作
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

from src.core.time_utils import format_srt_time
from src.red.redlog import iter_frames, make_circular_roi, smooth_center
from src.bleed_ai.models import ModelManager, hsv_blood_area_and_source
from src.tools.proxy_manager import ProxyManager

# プロキシ解像度
PROXY_RESOLUTION = "480p"  # 854x480


# ---------------------------------------------------------------------------
# オプティカルフロー
# ---------------------------------------------------------------------------

_FLOW_MAX_DIM = 320  # optical flow計算の最大解像度（高速化のため）


def compute_flow_magnitude(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    roi_mask: Optional[np.ndarray] = None,
) -> float:
    """
    Farneback optical flow の平均移動量を計算する。

    出血は局所的な拡散（低〜中程度のflow）で発生し、
    カメラ移動は全体的な高flowで発生する。

    性能最適化: フレームを最大320pxにダウンサンプルして計算する。
    """
    h, w = prev_gray.shape[:2]

    # ダウンサンプル（計算量をO(n^2)削減）
    scale = 1.0
    if max(h, w) > _FLOW_MAX_DIM:
        scale = _FLOW_MAX_DIM / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        prev_small = cv2.resize(prev_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
        curr_small = cv2.resize(curr_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
        if roi_mask is not None:
            roi_small = cv2.resize(
                roi_mask.astype(np.uint8), (new_w, new_h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        else:
            roi_small = None
    else:
        prev_small = prev_gray
        curr_small = curr_gray
        roi_small = roi_mask

    flow = cv2.calcOpticalFlowFarneback(
        prev_small, curr_small,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    mag = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)

    # スケール補正: ダウンサンプルした分flowも縮小されるので戻す
    if scale < 1.0:
        mag = mag / scale

    if roi_small is not None:
        mag_in_roi = mag[roi_small]
        if len(mag_in_roi) == 0:
            return 0.0
        return float(np.mean(mag_in_roi))
    return float(np.mean(mag))


# ---------------------------------------------------------------------------
# プロキシ動画生成
# ---------------------------------------------------------------------------

def _ensure_proxy(
    video_path: str,
    outdir: str,
    resolution: str = PROXY_RESOLUTION,
) -> str:
    """
    プロキシ動画を生成し、そのパスを返す。既存なら再利用する。

    ProxyManagerのscaleフィルタが幅奇数でエラーになる場合があるため、
    直接ffmpegコマンドで偶数化パディングを行う。

    Args:
        video_path: 元動画パス
        outdir: プロキシ出力ディレクトリ
        resolution: プロキシ解像度（"360p", "480p", "720p"）

    Returns:
        プロキシ動画のパス
    """
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

    # scale + pad で偶数化（libx264は幅・高さが2の倍数でないとエラー）
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
        "-an",  # 音声不要（解析用）
        str(proxy_path),
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except FileNotFoundError as e:
        raise RuntimeError(
            "ffmpeg が見つかりません。インストールして PATH に含めてください。"
        ) from e
    except subprocess.CalledProcessError as e:
        # 途中まで書かれた不完全なプロキシを削除（ProxyManager と同等の堅牢性）
        if proxy_path.exists():
            try:
                proxy_path.unlink()
            except OSError:
                pass
        stderr = e.stderr.decode("utf-8", errors="replace") if e.stderr else ""
        raise RuntimeError(
            f"プロキシ生成に失敗しました（ffmpeg, exit={e.returncode}）: {video_path}\n"
            f"{stderr.strip()[-800:]}"
        ) from e

    print(f"プロキシ生成完了: {proxy_path}")
    return str(proxy_path)


# ---------------------------------------------------------------------------
# 重症度スコア
# ---------------------------------------------------------------------------

def compute_severity(
    bleed_prob: float,
    area_ratio: float,
    flow_mag: float,
    flow_suppress_thr: float = 8.0,
) -> float:
    """
    複合重症度スコアを計算する。

    severity = bleed_prob * area_ratio * flow_suppression

    flow_suppression: 大きなカメラ動きがある場合は重症度を抑制する。
      flow_mag < thr → 1.0、flow_mag >= thr → 減衰。

    Returns:
        float: 0.0 〜 1.0
    """
    if flow_suppress_thr <= 0:
        # 閾値0以下は分母にできず抑制も定義できないため、抑制なし（0除算回避）
        flow_factor = 1.0
    elif flow_mag >= flow_suppress_thr:
        flow_factor = max(0.0, 1.0 - (flow_mag - flow_suppress_thr) / flow_suppress_thr)
    else:
        flow_factor = 1.0

    # area_ratio は通常 0〜0.5 程度なので 2倍してスケール
    scaled_area = min(1.0, area_ratio * 2.0)

    severity = bleed_prob * scaled_area * flow_factor
    return min(1.0, severity)


# ---------------------------------------------------------------------------
# Step 1: 時系列記録
# ---------------------------------------------------------------------------

def record_timeseries(
    video_path: str,
    outdir: str,
    fps: float = 10.0,
    roi_margin: float = 0.08,
    no_roi: bool = False,
    smooth_s: float = 5.0,
    device: str = "cuda",
    use_deep: bool = False,
    classifier_weights: Optional[str] = None,
    segmenter_weights: Optional[str] = None,
    flow_suppress_thr: float = 8.0,
    proxy_resolution: str = PROXY_RESOLUTION,
    no_proxy: bool = False,
) -> dict:
    """
    Step 1: 動画をサンプリングしてCSVを出力する。

    デフォルトではプロキシ動画（低解像度）を自動生成し、
    プロキシ上で解析を行う。--no-proxy で無効化可能。

    各フレームに対して以下の指標を計算:
      - bleed_prob: 出血確率 (0-1)
      - bleed_area: 出血領域面積比 (0-1)
      - source_x, source_y: 出血源推定位置 (0-1, 正規化)
      - flow_mag: オプティカルフロー平均移動量
      - severity: 複合重症度スコア (0-1)
      - smooth_bleed_prob, smooth_severity: 平滑化値

    Args:
        video_path: 入力動画ファイルパス
        outdir: 出力ディレクトリ
        fps: サンプリングFPS（デフォルト10）
        roi_margin: 円形ROIマージン
        no_roi: TrueならROI無効化
        smooth_s: 平滑化窓サイズ（秒）
        device: 計算デバイス ("cuda" or "cpu")
        use_deep: 深層学習モデル使用フラグ
        classifier_weights: 分類器重みファイルパス
        segmenter_weights: セグメンテーション重みファイルパス
        flow_suppress_thr: フロー抑制閾値
        proxy_resolution: プロキシ解像度（"360p","480p","720p"）
        no_proxy: Trueならプロキシを使わず元動画を直接解析

    Returns:
        {"csv": CSVファイルパス}
    """
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    # CSVファイル名は元動画のstemを使用
    stem = Path(video_path).stem

    # プロキシ動画の生成・使用
    analysis_path = video_path
    if not no_proxy:
        analysis_path = _ensure_proxy(video_path, outdir, proxy_resolution)

    # モデル初期化
    mgr = ModelManager(
        device=device,
        use_deep=use_deep,
        classifier_weights=classifier_weights,
        segmenter_weights=segmenter_weights,
    )

    # データ収集
    times: List[float] = []
    bleed_probs: List[float] = []
    bleed_areas: List[float] = []
    source_xs: List[float] = []
    source_ys: List[float] = []
    flow_mags: List[float] = []
    severities: List[float] = []
    reader_name = "opencv"

    roi_mask: Optional[np.ndarray] = None
    roi_initialized = False
    prev_gray: Optional[np.ndarray] = None
    frame_count = 0

    for t_sec, bgr, reader in iter_frames(analysis_path, fps):
        reader_name = reader
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"  処理中: {frame_count} フレーム / t={t_sec:.1f}s",
                  file=sys.stderr)

        # ROI初期化（最初のフレーム）
        if not roi_initialized:
            h, w = bgr.shape[:2]
            if not no_roi:
                roi_mask = make_circular_roi(h, w, margin=roi_margin)
            roi_initialized = True

        # 分類: 出血確率
        prob = mgr.classify(bgr, roi_mask)

        # セグメンテーション: 出血領域 + 出血源
        seg = mgr.segment(bgr, roi_mask)

        # オプティカルフロー
        curr_gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            flow = compute_flow_magnitude(prev_gray, curr_gray, roi_mask)
        else:
            flow = 0.0
        prev_gray = curr_gray

        # 重症度スコア
        sev = compute_severity(prob, seg["area_ratio"], flow, flow_suppress_thr)

        times.append(t_sec)
        bleed_probs.append(prob)
        bleed_areas.append(seg["area_ratio"])
        source_xs.append(seg["source_x"])
        source_ys.append(seg["source_y"])
        flow_mags.append(flow)
        severities.append(sev)

    if not times:
        print("警告: フレームが取得できませんでした。", file=sys.stderr)
        return {}

    # 平滑化
    window = max(1, int(round(smooth_s * fps)))
    smooth_probs = smooth_center(bleed_probs, window)
    smooth_sevs = smooth_center(severities, window)
    smooth_areas = smooth_center(bleed_areas, window)

    # CSV出力
    csv_path = out_path / f"{stem}_bleedailog.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "t_sec", "t_srt",
            "bleed_prob", "bleed_area", "source_x", "source_y",
            "flow_mag", "severity",
            "smooth_bleed_prob", "smooth_severity", "smooth_area",
            "reader",
        ])
        for i in range(len(times)):
            writer.writerow([
                f"{times[i]:.3f}",
                format_srt_time(times[i]),
                f"{bleed_probs[i]:.6f}",
                f"{bleed_areas[i]:.6f}",
                f"{source_xs[i]:.6f}",
                f"{source_ys[i]:.6f}",
                f"{flow_mags[i]:.6f}",
                f"{severities[i]:.6f}",
                f"{smooth_probs[i]:.6f}",
                f"{smooth_sevs[i]:.6f}",
                f"{smooth_areas[i]:.6f}",
                reader_name,
            ])

    print(f"CSV  : {csv_path}")
    return {"csv": str(csv_path)}


# ---------------------------------------------------------------------------
# CSV読み込み
# ---------------------------------------------------------------------------

def _estimate_fps(times: List[float], default: float = 10.0) -> float:
    """タイムスタンプ列から FPS を頑健に推定する。

    隣接差の中央値の逆数を用いる。重複タイムスタンプや非単調な行があっても
    0/負の間隔は無視するため ZeroDivisionError を起こさない。
    """
    if len(times) < 2:
        return default
    diffs = np.diff(np.asarray(times, dtype=float))
    positive = diffs[diffs > 0]
    if positive.size == 0:
        return default
    median_dt = float(np.median(positive))
    return 1.0 / median_dt if median_dt > 0 else default


# 必須の数値列（reader はオプション）
_CSV_NUMERIC_COLUMNS = [
    "t_sec", "bleed_prob", "bleed_area", "source_x", "source_y",
    "flow_mag", "severity", "smooth_bleed_prob", "smooth_severity", "smooth_area",
]


def read_bleedailog_csv(csv_path: str) -> dict:
    """
    出血AIログCSVを読み込む。

    Returns:
        {"times": [...], "bleed_probs": [...], "bleed_areas": [...],
         "source_xs": [...], "source_ys": [...], "flow_mags": [...],
         "severities": [...], "smooth_probs": [...], "smooth_sevs": [...],
         "smooth_areas": [...], "reader": str, "fps": float}
    """
    times: List[float] = []
    bleed_probs: List[float] = []
    bleed_areas: List[float] = []
    source_xs: List[float] = []
    source_ys: List[float] = []
    flow_mags: List[float] = []
    severities: List[float] = []
    smooth_probs: List[float] = []
    smooth_sevs: List[float] = []
    smooth_areas: List[float] = []
    reader = "unknown"

    with open(csv_path, "r", encoding="utf-8") as f:
        dr = csv.DictReader(f)
        if dr.fieldnames is None:
            raise ValueError(f"CSV が空、またはヘッダー行がありません: {csv_path}")
        missing = [c for c in _CSV_NUMERIC_COLUMNS if c not in dr.fieldnames]
        if missing:
            raise ValueError(
                f"CSV に必須列がありません: {', '.join(missing)} ({csv_path})"
            )

        skipped = 0
        for lineno, row in enumerate(dr, start=2):  # 2行目=最初のデータ行
            try:
                vals = {c: float(row[c]) for c in _CSV_NUMERIC_COLUMNS}
            except (KeyError, ValueError, TypeError):
                # 欠損・空セル・非数値の行はスキップして処理を継続する
                skipped += 1
                continue
            times.append(vals["t_sec"])
            bleed_probs.append(vals["bleed_prob"])
            bleed_areas.append(vals["bleed_area"])
            source_xs.append(vals["source_x"])
            source_ys.append(vals["source_y"])
            flow_mags.append(vals["flow_mag"])
            severities.append(vals["severity"])
            smooth_probs.append(vals["smooth_bleed_prob"])
            smooth_sevs.append(vals["smooth_severity"])
            smooth_areas.append(vals["smooth_area"])
            reader = row.get("reader") or reader

        if skipped:
            print(f"警告: 不正な {skipped} 行をスキップしました: {csv_path}",
                  file=sys.stderr)

    if not times:
        raise ValueError(f"CSV に有効なデータ行がありません: {csv_path}")

    fps = _estimate_fps(times)

    return {
        "times": times,
        "bleed_probs": bleed_probs,
        "bleed_areas": bleed_areas,
        "source_xs": source_xs,
        "source_ys": source_ys,
        "flow_mags": flow_mags,
        "severities": severities,
        "smooth_probs": smooth_probs,
        "smooth_sevs": smooth_sevs,
        "smooth_areas": smooth_areas,
        "reader": reader,
        "fps": fps,
    }


# ---------------------------------------------------------------------------
# Step 2: アノテーション
# ---------------------------------------------------------------------------

def _compute_area_delta(
    smooth_areas: List[float],
    fps: float,
    baseline_s: float = 60.0,
) -> List[float]:
    """
    面積変化量（area_delta）を計算する。

    短期平滑化面積から長期ベースライン（baseline_s秒窓の移動平均）を引く。
    正の値 = ベースラインからの面積増加 = 出血の疑い。
    負の値はゼロにクランプ（面積減少は出血ではない）。

    Args:
        smooth_areas: 短期平滑化済み面積比のリスト
        fps: サンプリングFPS
        baseline_s: ベースライン算出窓サイズ（秒、デフォルト60）

    Returns:
        area_delta値のリスト（0以上）
    """
    baseline_window = max(1, int(round(baseline_s * fps)))
    baseline = smooth_center(smooth_areas, baseline_window)
    delta = [max(0.0, s - b) for s, b in zip(smooth_areas, baseline)]
    return delta


def annotate_bleed_ai(
    csv_path: str,
    outdir: str,
    thr: float = 0.10,
    min_duration_s: float = 2.0,
    smooth_s: float = 5.0,
    metric: str = "area_delta",
    max_duration_s: float = 300.0,
    baseline_s: float = 60.0,
) -> dict:
    """
    Step 2: CSVから閾値ベースで出血イベントを抽出し、JSONL/SRTを出力する。

    Args:
        csv_path: 入力CSVファイル（出血AIログ）
        outdir: 出力ディレクトリ
        thr: 出血候補閾値
        min_duration_s: 最小イベント持続時間（秒）
        smooth_s: 記録用パラメータ
        metric: 使用する指標名
            "smooth_severity", "smooth_bleed_prob", "smooth_area",
            "area_delta"（デフォルト）
        max_duration_s: 最大イベント持続時間（秒、デフォルト300）。
            これを超えるイベントは誤検出として除外される。
            0で無制限。
        baseline_s: area_delta用ベースライン窓（秒、デフォルト60）

    Returns:
        {"jsonl": str, "srt": str, "events": int}
    """
    data = read_bleedailog_csv(csv_path)
    times = data["times"]
    fps = data["fps"]

    # area_deltaを算出
    area_delta = _compute_area_delta(data["smooth_areas"], fps, baseline_s)

    # 指標の選択
    metric_map = {
        "smooth_severity": data["smooth_sevs"],
        "smooth_bleed_prob": data["smooth_probs"],
        "smooth_area": data["smooth_areas"],
        "area_delta": area_delta,
    }
    values = metric_map.get(metric, area_delta)

    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    stem = Path(csv_path).stem.replace("_bleedailog", "")

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

    # max_duration_s フィルタ: 長すぎるイベントは誤検出として除外
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
    jsonl_path = out_path / f"{stem}_bleed_ai_events.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")

    # SRT出力（JSONLから変換）
    srt_path = out_path / f"{stem}_bleed_ai.srt"
    from src.tools.jsonl_to_srt import convert as jsonl_to_srt_convert
    jsonl_to_srt_convert(
        in_jsonl=str(jsonl_path),
        out_srt=str(srt_path),
        event_type="bleed_ai_candidate",
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

    # イベント区間の平均面積比・最大面積比
    areas = data["bleed_areas"][start_idx:end_idx]
    peak_area = max(areas) if areas else 0.0
    mean_area = sum(areas) / len(areas) if areas else 0.0

    events.append({
        "type": "bleed_ai_candidate",
        "metric": metric,
        "thr": thr,
        "min_duration_s": min_duration_s,
        "smooth_s": smooth_s,
        "peak_value": round(peak_value, 6),
        "peak_area": round(peak_area, 6),
        "mean_area": round(mean_area, 6),
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
    fps: float = 10.0,
    roi_margin: float = 0.08,
    no_roi: bool = False,
    smooth_s: float = 5.0,
    device: str = "cuda",
    use_deep: bool = False,
    classifier_weights: Optional[str] = None,
    segmenter_weights: Optional[str] = None,
    flow_suppress_thr: float = 8.0,
    thr: float = 0.10,
    min_duration_s: float = 2.0,
    metric: str = "area_delta",
    max_duration_s: float = 300.0,
    baseline_s: float = 60.0,
    proxy_resolution: str = PROXY_RESOLUTION,
    no_proxy: bool = False,
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
        use_deep=use_deep,
        classifier_weights=classifier_weights,
        segmenter_weights=segmenter_weights,
        flow_suppress_thr=flow_suppress_thr,
        proxy_resolution=proxy_resolution,
        no_proxy=no_proxy,
    )
    if not result1:
        return {}

    result2 = annotate_bleed_ai(
        csv_path=result1["csv"],
        outdir=outdir,
        thr=thr,
        min_duration_s=min_duration_s,
        smooth_s=smooth_s,
        metric=metric,
        max_duration_s=max_duration_s,
        baseline_s=baseline_s,
    )
    return {**result1, **result2}


# ---------------------------------------------------------------------------
# CLI エントリポイント
# ---------------------------------------------------------------------------

def main() -> int:
    """コマンドラインエントリポイント（サブコマンド方式）"""
    parser = argparse.ArgumentParser(
        description="AI出血検出（Deep Learning / HSV ハイブリッド）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # HSVフォールバックで一括解析
  python -m src.bleed_ai.detector analyze --video input.mp4 --outdir output/

  # 深層学習モデル使用
  python -m src.bleed_ai.detector analyze --video input.mp4 --outdir output/ \\
      --use-deep --classifier-weights model/classifier.pth

  # Step 1 のみ
  python -m src.bleed_ai.detector timeseries --video input.mp4 --outdir output/

  # Step 2 のみ（閾値を変えて再実行）
  python -m src.bleed_ai.detector annotate --csv output/input_bleedailog.csv \\
      --outdir output/ --thr 0.03 --metric smooth_severity
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="実行コマンド")

    # 共通引数は親パーサーに集約し、各サブコマンドで parents= 再利用する（DRY）。
    # 片方のみ更新による不整合を防ぎ、デフォルト/choices/help を一元管理する。

    # 全コマンド共通
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--outdir", required=True, help="出力ディレクトリ")
    common.add_argument("--smooth-s", type=float, default=5.0,
                        help="平滑化窓（秒、デフォルト: 5）")

    # 動画キャプチャ系（timeseries / analyze で使用）
    capture = argparse.ArgumentParser(add_help=False)
    capture.add_argument("--video", required=True, help="入力動画ファイルパス")
    capture.add_argument("--fps", type=float, default=10.0,
                         help="サンプリングFPS（デフォルト: 10）")
    capture.add_argument("--roi-margin", type=float, default=0.08,
                         help="円形ROIマージン（デフォルト: 0.08）")
    capture.add_argument("--no-roi", action="store_true", help="ROI無効化")
    capture.add_argument("--device", default="cuda",
                         help="計算デバイス（デフォルト: cuda）")
    capture.add_argument("--use-deep", action="store_true",
                         help="深層学習モデルを使用する")
    capture.add_argument("--classifier-weights", default=None,
                         help="分類器重みファイルパス")
    capture.add_argument("--segmenter-weights", default=None,
                         help="セグメンテーション重みファイルパス")
    capture.add_argument("--flow-suppress-thr", type=float, default=8.0,
                         help="フロー抑制閾値（デフォルト: 8.0）")
    capture.add_argument("--proxy-resolution", default=PROXY_RESOLUTION,
                         choices=["360p", "480p", "720p"],
                         help=f"プロキシ解像度（デフォルト: {PROXY_RESOLUTION}）")
    capture.add_argument("--no-proxy", action="store_true",
                         help="プロキシを使わず元動画を直接解析")

    # イベント検出系（annotate / analyze で使用）
    event = argparse.ArgumentParser(add_help=False)
    event.add_argument("--thr", type=float, default=0.10,
                       help="出血候補閾値（デフォルト: 0.10）")
    event.add_argument("--min-duration-s", type=float, default=2.0,
                       help="最小イベント持続時間（秒、デフォルト: 2.0）")
    event.add_argument("--metric", default="area_delta",
                       choices=["smooth_severity", "smooth_bleed_prob",
                                "smooth_area", "area_delta"],
                       help="使用する指標名（デフォルト: area_delta）")
    event.add_argument("--max-duration-s", type=float, default=300.0,
                       help="最大イベント持続時間（秒、デフォルト: 300）。"
                            "超過イベントは誤検出として除外。0で無制限")
    event.add_argument("--baseline-s", type=float, default=60.0,
                       help="area_delta用ベースライン窓（秒、デフォルト: 60）")

    # --- timeseries ---
    subparsers.add_parser("timeseries", parents=[capture, common],
                          help="Step 1: 時系列記録（動画→CSV）")

    # --- annotate ---
    ann = subparsers.add_parser("annotate", parents=[event, common],
                                help="Step 2: アノテーション（CSV→JSONL/SRT）")
    ann.add_argument("--csv", required=True, help="入力CSVファイル")

    # --- analyze ---
    subparsers.add_parser("analyze", parents=[capture, event, common],
                          help="一括実行（timeseries + annotate）")

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
            use_deep=args.use_deep,
            classifier_weights=args.classifier_weights,
            segmenter_weights=args.segmenter_weights,
            flow_suppress_thr=args.flow_suppress_thr,
            proxy_resolution=args.proxy_resolution,
            no_proxy=args.no_proxy,
        )
    elif args.command == "annotate":
        annotate_bleed_ai(
            csv_path=args.csv,
            outdir=args.outdir,
            thr=args.thr,
            min_duration_s=args.min_duration_s,
            smooth_s=args.smooth_s,
            metric=args.metric,
            max_duration_s=args.max_duration_s,
            baseline_s=args.baseline_s,
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
            use_deep=args.use_deep,
            classifier_weights=args.classifier_weights,
            segmenter_weights=args.segmenter_weights,
            flow_suppress_thr=args.flow_suppress_thr,
            thr=args.thr,
            min_duration_s=args.min_duration_s,
            metric=args.metric,
            max_duration_s=args.max_duration_s,
            baseline_s=args.baseline_s,
            proxy_resolution=args.proxy_resolution,
            no_proxy=args.no_proxy,
        )
    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
