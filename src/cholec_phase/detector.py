"""
Cholec80 ベース手術フェーズ認識 評価器

ANALYZER_SPEC.md に準拠した2ステップパイプライン:
  Step 1 - record_timeseries(): 動画 → CSV（フレームごとのフェーズ確率）
  Step 2 - annotate_phases():   CSV → JSONL/SRT（フェーズ区間アノテーション）

CSV カラム:
  t_sec, t_srt, phase_id, phase_name, prob_Preparation, prob_CalotTriangleDissection,
  prob_ClippingCutting, prob_GallbladderDissection, prob_GallbladderPackaging,
  prob_CleaningCoagulation, prob_GallbladderRetraction, confidence, reader

JSONL type: "surgical_phase"
SRT tag: "[phase] {phase_name}"

CLI:
  python -m src.cholec_phase.detector timeseries --video input.mp4 --outdir output/
  python -m src.cholec_phase.detector annotate --csv output/input_cholecphaselog.csv --outdir output/
  python -m src.cholec_phase.detector analyze --video input.mp4 --outdir output/
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.core.time_utils import format_srt_time
from src.red.redlog import iter_frames, smooth_center
from src.cholec_phase import CHOLEC80_PHASES, NUM_PHASES
from src.cholec_phase.models import PhaseModelManager
from src.tools.proxy_manager import ProxyManager

PROXY_RESOLUTION = "480p"


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
    fps: float = 1.0,
    device: str = "cuda",
    model_path: Optional[str] = None,
    backbone_weights: Optional[str] = None,
    backbone_method: str = "dino",
    auto_download_backbone: bool = True,
    hidden_dim: int = 512,
    num_layers: int = 2,
    context_frames: int = 30,
    proxy_resolution: str = PROXY_RESOLUTION,
    no_proxy: bool = False,
) -> dict:
    """
    Step 1: 動画をサンプリングしてフェーズ確率CSVを出力する。

    学習済みモデルがある場合:
      各フレームのフェーズ確率を BiLSTM で予測

    学習済みモデルがない場合:
      ResNet50特徴量のみ抽出（フェーズ確率は均等分布）
      → 後で学習済みモデルとCSVを組み合わせて再推論可能

    Args:
        video_path: 入力動画ファイルパス
        outdir: 出力ディレクトリ
        fps: サンプリングFPS（デフォルト: 1.0）
        device: 計算デバイス
        model_path: 学習済み BiLSTM モデルパス
        backbone_weights: SelfSupSurg 重みパス
        backbone_method: SelfSupSurg 手法
        auto_download_backbone: backbone重みの自動ダウンロード
        hidden_dim: BiLSTM 隠れ次元
        num_layers: BiLSTM 層数
        context_frames: BiLSTM への入力文脈フレーム数
        proxy_resolution: プロキシ解像度
        no_proxy: Trueならプロキシを使わない

    Returns:
        {"csv": CSVファイルパス}
    """
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    stem = Path(video_path).stem

    # プロキシ動画
    analysis_path = video_path
    if not no_proxy:
        analysis_path = _ensure_proxy(video_path, outdir, proxy_resolution)

    # モデル初期化
    mgr = PhaseModelManager(
        device=device,
        model_path=model_path,
        backbone_weights=backbone_weights,
        backbone_method=backbone_method,
        auto_download_backbone=auto_download_backbone,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        context_frames=context_frames,
    )

    has_model = mgr.is_ready

    # フレーム処理
    times: List[float] = []
    phase_ids: List[int] = []
    phase_names: List[str] = []
    all_probs: List[np.ndarray] = []
    confidences: List[float] = []
    reader_name = "opencv"
    frame_count = 0

    for t_sec, bgr, reader in iter_frames(analysis_path, fps):
        if frame_count == 0:
            reader_name = reader  # reader はフレーム間で不変なので初回のみ取得
        frame_count += 1

        if frame_count % 60 == 0:
            print(f"  処理中: {frame_count} フレーム / t={t_sec:.1f}s",
                  file=sys.stderr)

        pid, pname, probs = mgr.predict_frame(bgr)

        times.append(t_sec)
        phase_ids.append(pid)
        phase_names.append(pname)
        all_probs.append(probs)
        confidences.append(float(np.max(probs)) if probs.sum() > 0 else 0.0)

    if not times:
        print("警告: フレームが取得できませんでした。", file=sys.stderr)
        return {}

    # CSV出力
    csv_path = out_path / f"{stem}_cholecphaselog.csv"
    prob_headers = [f"prob_{name}" for name in CHOLEC80_PHASES]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["t_sec", "t_srt", "phase_id", "phase_name"]
            + prob_headers
            + ["confidence", "reader"]
        )
        for i in range(len(times)):
            row = [
                f"{times[i]:.3f}",
                format_srt_time(times[i]),
                str(phase_ids[i]),
                phase_names[i],
            ]
            for p in all_probs[i]:
                row.append(f"{p:.6f}")
            row.append(f"{confidences[i]:.6f}")
            row.append(reader_name)
            writer.writerow(row)

    mode_str = "BiLSTM" if has_model else "特徴抽出のみ"
    print(f"CSV  : {csv_path}")
    print(f"モード: {mode_str}, {frame_count} フレーム")
    return {"csv": str(csv_path)}


# ---------------------------------------------------------------------------
# CSV読み込み
# ---------------------------------------------------------------------------


def read_cholecphaselog_csv(csv_path: str) -> dict:
    """
    cholecphaselog CSV を読み込む。

    Returns:
        {"times": [...], "phase_ids": [...], "phase_names": [...],
         "probs": np.ndarray (N, 7), "confidences": [...],
         "reader": str, "fps": float}
    """
    times: List[float] = []
    phase_ids: List[int] = []
    phase_names: List[str] = []
    probs_list: List[List[float]] = []
    confidences: List[float] = []
    reader = "unknown"

    with open(csv_path, "r", encoding="utf-8") as f:
        dr = csv.DictReader(f)
        if dr.fieldnames is None:
            raise ValueError(f"CSV が空、またはヘッダー行がありません: {csv_path}")
        missing = [c for c in ("t_sec", "phase_id", "phase_name")
                   if c not in dr.fieldnames]
        if missing:
            raise ValueError(
                f"CSV に必須列がありません: {', '.join(missing)} ({csv_path})"
            )

        skipped = 0
        for lineno, row in enumerate(dr, start=2):  # 2行目=最初のデータ行
            try:
                t = float(row["t_sec"])
                pid = int(row["phase_id"])
                frame_probs = [float(row.get(f"prob_{name}", 0.0))
                               for name in CHOLEC80_PHASES]
                conf = float(row.get("confidence", 0.0))
            except (KeyError, ValueError, TypeError):
                skipped += 1  # 欠損・非数値・空セルの行はスキップして継続
                continue
            times.append(t)
            phase_ids.append(pid)
            phase_names.append(row.get("phase_name", ""))
            probs_list.append(frame_probs)
            confidences.append(conf)
            reader = row.get("reader") or reader

        if skipped:
            print(f"警告: 不正な {skipped} 行をスキップしました: {csv_path}",
                  file=sys.stderr)

    if not times:
        raise ValueError(f"CSV に有効なデータ行がありません: {csv_path}")

    # 重複/非単調タイムスタンプでも 0除算しないよう、正の間隔の中央値から推定
    fps = 1.0
    if len(times) >= 2:
        diffs = np.diff(np.asarray(times, dtype=float))
        positive = diffs[diffs > 0]
        if positive.size:
            fps = 1.0 / float(np.median(positive))
    probs = np.array(probs_list)

    return {
        "times": times,
        "phase_ids": phase_ids,
        "phase_names": phase_names,
        "probs": probs,
        "confidences": confidences,
        "reader": reader,
        "fps": fps,
    }


# ---------------------------------------------------------------------------
# Step 2: アノテーション
# ---------------------------------------------------------------------------


def annotate_phases(
    csv_path: str,
    outdir: str,
    min_phase_s: float = 10.0,
    smooth_s: float = 5.0,
) -> dict:
    """
    Step 2: CSVからフェーズ区間を抽出し、JSONL/SRTを出力する。

    フェーズ区間の抽出ロジック:
      1. フレームレベルの phase_id シーケンスを取得
      2. 短いフェーズ（min_phase_s 未満）を前後のフェーズにマージ
      3. 連続する同一フェーズを1つの区間にまとめる

    Args:
        csv_path: 入力CSVファイル
        outdir: 出力ディレクトリ
        min_phase_s: 最小フェーズ持続時間（秒）
        smooth_s: フェーズID平滑化窓（秒、中央値フィルタ）

    Returns:
        {"jsonl": str, "srt": str, "phases": int}
    """
    data = read_cholecphaselog_csv(csv_path)
    times = data["times"]
    phase_ids = data["phase_ids"]
    fps = data["fps"]
    probs = data["probs"]

    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    stem = Path(csv_path).stem.replace("_cholecphaselog", "")

    if not times:
        return {"jsonl": "", "srt": "", "phases": 0}

    # 中央値フィルタでフェーズIDを平滑化
    smoothed_ids = _mode_smooth_phases(phase_ids, fps, smooth_s)

    # 短いフェーズをマージ
    merged_ids = _merge_short_phases(smoothed_ids, times, min_phase_s)

    # 連続する同一フェーズを区間にまとめる
    events = _extract_phase_intervals(merged_ids, times, probs)

    # JSONL出力
    jsonl_path = out_path / f"{stem}_surgical_phase_events.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")

    # SRT出力
    srt_path = out_path / f"{stem}_surgical_phase.srt"
    srt_lines: List[str] = []
    for idx, ev in enumerate(events, start=1):
        srt_lines.append(f"{idx}")
        srt_lines.append(f"{ev['start_srt']} --> {ev['end_srt']}")
        srt_lines.append(f"[phase] {ev['phase_name']}")
        srt_lines.append("")
    Path(srt_path).write_text("\n".join(srt_lines), encoding="utf-8")

    print(f"JSONL: {jsonl_path} （正本）")
    print(f"SRT  : {srt_path}")
    print(f"フェーズ区間数: {len(events)}")

    for ev in events:
        print(f"  {ev['start_srt']} - {ev['end_srt']}: "
              f"{ev['phase_name']} ({ev['duration_sec']:.0f}s, "
              f"conf={ev['mean_confidence']:.3f})")

    return {
        "jsonl": str(jsonl_path),
        "srt": str(srt_path),
        "phases": len(events),
    }


def _mode_smooth_phases(
    phase_ids: List[int],
    fps: float,
    smooth_s: float,
) -> List[int]:
    """最頻値（多数決）フィルタでフェーズIDを平滑化する。

    フェーズIDは順序を持たないカテゴリ値のため、中央値ではなくウィンドウ内の
    最頻値で平滑化する（中央値はカテゴリに対して意味を持たない）。
    """
    if smooth_s <= 0 or len(phase_ids) < 3:
        return list(phase_ids)

    window = max(3, int(round(smooth_s * fps)))
    if window % 2 == 0:
        window += 1  # 奇数にする

    half = window // 2
    result = list(phase_ids)

    for i in range(half, len(phase_ids) - half):
        segment = phase_ids[i - half:i + half + 1]
        # ウィンドウ内の最頻値（多数決）
        counts = {}
        for pid in segment:
            counts[pid] = counts.get(pid, 0) + 1
        result[i] = max(counts, key=counts.get)

    return result


def _merge_short_phases(
    phase_ids: List[int],
    times: List[float],
    min_phase_s: float,
) -> List[int]:
    """短いフェーズ区間を前後のフェーズにマージする。"""
    if min_phase_s <= 0 or len(phase_ids) < 2:
        return list(phase_ids)

    result = list(phase_ids)

    changed = True
    max_iter = 10
    while changed and max_iter > 0:
        changed = False
        max_iter -= 1

        # 区間を検出
        intervals = []
        start = 0
        for i in range(1, len(result)):
            if result[i] != result[start]:
                intervals.append((start, i, result[start]))
                start = i
        intervals.append((start, len(result), result[start]))

        # 短い区間を隣接フェーズにマージ
        for s, e, pid in intervals:
            duration = times[min(e - 1, len(times) - 1)] - times[s]
            if duration >= min_phase_s or len(intervals) <= 1:
                continue
            # 先頭区間は「前」が無いので次の区間へ、それ以外は前の区間へマージ
            if s == 0:
                new_pid = result[e] if e < len(result) else pid
            else:
                new_pid = result[s - 1]
            # 実際に値が変わる場合のみ置換（無変化での無駄な再反復を防ぐ）
            if new_pid != pid:
                for j in range(s, e):
                    result[j] = new_pid
                changed = True

    return result


def _extract_phase_intervals(
    phase_ids: List[int],
    times: List[float],
    probs: np.ndarray,
) -> List[dict]:
    """フレームレベルのフェーズIDから区間イベントを抽出する。"""
    events: List[dict] = []

    if not phase_ids:
        return events

    start_idx = 0
    current_pid = phase_ids[0]

    for i in range(1, len(phase_ids)):
        if phase_ids[i] != current_pid:
            # 区間終了
            _add_phase_event(events, times, probs, phase_ids,
                             start_idx, i, current_pid)
            start_idx = i
            current_pid = phase_ids[i]

    # 最後の区間
    _add_phase_event(events, times, probs, phase_ids,
                     start_idx, len(phase_ids), current_pid)

    return events


def _add_phase_event(
    events: List[dict],
    times: List[float],
    probs: np.ndarray,
    phase_ids: List[int],
    start_idx: int,
    end_idx: int,
    phase_id: int,
) -> None:
    """フェーズイベントを構築してリストに追加する。"""
    start_sec = times[start_idx]
    end_sec = times[min(end_idx - 1, len(times) - 1)]
    duration = end_sec - start_sec

    # 外部CSV/モデル異常出力で phase_id が値域外でも IndexError にならないよう防御
    if 0 <= phase_id < NUM_PHASES:
        phase_name = CHOLEC80_PHASES[phase_id]
    else:
        print(f"警告: phase_id={phase_id} が値域[0,{NUM_PHASES}) 外です。"
              "'Unknown' として記録します。", file=sys.stderr)
        phase_name = "Unknown"

    # 区間内の平均確率
    seg_probs = probs[start_idx:end_idx]
    mean_conf = float(np.mean(np.max(seg_probs, axis=1))) if len(seg_probs) > 0 else 0.0

    events.append({
        "type": "surgical_phase",
        "phase_id": phase_id,
        "phase_name": phase_name,
        "mean_confidence": round(mean_conf, 4),
        "duration_sec": round(duration, 3),
        "start_sec": round(start_sec, 3),
        "end_sec": round(end_sec, 3),
        "start_srt": format_srt_time(start_sec),
        "end_srt": format_srt_time(end_sec),
    })


# ---------------------------------------------------------------------------
# 一括実行
# ---------------------------------------------------------------------------


def analyze_video(
    video_path: str,
    outdir: str,
    fps: float = 1.0,
    device: str = "cuda",
    model_path: Optional[str] = None,
    backbone_weights: Optional[str] = None,
    backbone_method: str = "dino",
    auto_download_backbone: bool = True,
    hidden_dim: int = 512,
    num_layers: int = 2,
    context_frames: int = 30,
    min_phase_s: float = 10.0,
    smooth_s: float = 5.0,
    proxy_resolution: str = PROXY_RESOLUTION,
    no_proxy: bool = False,
) -> dict:
    """2ステップの一括実行。"""
    result1 = record_timeseries(
        video_path=video_path,
        outdir=outdir,
        fps=fps,
        device=device,
        model_path=model_path,
        backbone_weights=backbone_weights,
        backbone_method=backbone_method,
        auto_download_backbone=auto_download_backbone,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        context_frames=context_frames,
        proxy_resolution=proxy_resolution,
        no_proxy=no_proxy,
    )
    if not result1:
        return {}

    result2 = annotate_phases(
        csv_path=result1["csv"],
        outdir=outdir,
        min_phase_s=min_phase_s,
        smooth_s=smooth_s,
    )

    return {**result1, **result2}


# ---------------------------------------------------------------------------
# CLI エントリポイント
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cholec80ベース教師あり手術フェーズ認識",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 学習済みモデルで一括解析
  python -m src.cholec_phase.detector analyze --video input.mp4 --outdir output/ \\
      --model models/phase_model_best.pth

  # 特徴抽出のみ（モデルなし）
  python -m src.cholec_phase.detector timeseries --video input.mp4 --outdir output/

  # Step 2 のみ（パラメータ変更）
  python -m src.cholec_phase.detector annotate --csv output/input_cholecphaselog.csv \\
      --outdir output/ --min-phase-s 30
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="実行コマンド")

    # --- timeseries ---
    ts = subparsers.add_parser("timeseries", help="Step 1: フェーズ確率CSV（動画→CSV）")
    ts.add_argument("--video", required=True)
    ts.add_argument("--outdir", required=True)
    ts.add_argument("--fps", type=float, default=1.0)
    ts.add_argument("--device", default="cuda")
    ts.add_argument("--model", default=None, help="学習済みBiLSTMモデル")
    ts.add_argument("--backbone-weights", default=None)
    ts.add_argument("--backbone-method", default="dino",
                    choices=["dino", "moco_v2", "simclr", "swav"])
    ts.add_argument("--no-auto-download", action="store_true")
    ts.add_argument("--hidden-dim", type=int, default=512)
    ts.add_argument("--num-layers", type=int, default=2)
    ts.add_argument("--context-frames", type=int, default=30)
    ts.add_argument("--proxy-resolution", default=PROXY_RESOLUTION,
                    choices=["360p", "480p", "720p"])
    ts.add_argument("--no-proxy", action="store_true")

    # --- annotate ---
    ann = subparsers.add_parser("annotate", help="Step 2: フェーズ区間（CSV→JSONL/SRT）")
    ann.add_argument("--csv", required=True)
    ann.add_argument("--outdir", required=True)
    ann.add_argument("--min-phase-s", type=float, default=10.0,
                     help="最小フェーズ持続時間（秒、デフォルト: 10）")
    ann.add_argument("--smooth-s", type=float, default=5.0,
                     help="フェーズ平滑化窓（秒、デフォルト: 5）")

    # --- analyze ---
    ana = subparsers.add_parser("analyze", help="一括実行（timeseries + annotate）")
    ana.add_argument("--video", required=True)
    ana.add_argument("--outdir", required=True)
    ana.add_argument("--fps", type=float, default=1.0)
    ana.add_argument("--device", default="cuda")
    ana.add_argument("--model", default=None, help="学習済みBiLSTMモデル")
    ana.add_argument("--backbone-weights", default=None)
    ana.add_argument("--backbone-method", default="dino",
                    choices=["dino", "moco_v2", "simclr", "swav"])
    ana.add_argument("--no-auto-download", action="store_true")
    ana.add_argument("--hidden-dim", type=int, default=512)
    ana.add_argument("--num-layers", type=int, default=2)
    ana.add_argument("--context-frames", type=int, default=30)
    ana.add_argument("--min-phase-s", type=float, default=10.0)
    ana.add_argument("--smooth-s", type=float, default=5.0)
    ana.add_argument("--proxy-resolution", default=PROXY_RESOLUTION,
                    choices=["360p", "480p", "720p"])
    ana.add_argument("--no-proxy", action="store_true")

    args = parser.parse_args()

    if args.command == "timeseries":
        record_timeseries(
            video_path=args.video,
            outdir=args.outdir,
            fps=args.fps,
            device=args.device,
            model_path=args.model,
            backbone_weights=args.backbone_weights,
            backbone_method=args.backbone_method,
            auto_download_backbone=not args.no_auto_download,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            context_frames=args.context_frames,
            proxy_resolution=args.proxy_resolution,
            no_proxy=args.no_proxy,
        )
    elif args.command == "annotate":
        annotate_phases(
            csv_path=args.csv,
            outdir=args.outdir,
            min_phase_s=args.min_phase_s,
            smooth_s=args.smooth_s,
        )
    elif args.command == "analyze":
        analyze_video(
            video_path=args.video,
            outdir=args.outdir,
            fps=args.fps,
            device=args.device,
            model_path=args.model,
            backbone_weights=args.backbone_weights,
            backbone_method=args.backbone_method,
            auto_download_backbone=not args.no_auto_download,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            context_frames=args.context_frames,
            min_phase_s=args.min_phase_s,
            smooth_s=args.smooth_s,
            proxy_resolution=args.proxy_resolution,
            no_proxy=args.no_proxy,
        )
    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
