"""
YOLO8-based surgical instrument detection for scene analysis.

2段階の処理:
  Step 1 - record_timeseries(): 動画→JSONL（instrument検出の時系列記録）
  Step 2 - annotate_scenes():   JSONL→SRT（instrument組み合わせ変化でシーン分割）

出力: JSONL（イベント正本）、SRT（Shotcut可視化用）
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.core.time_utils import format_srt_time


# ---------------------------------------------------------------------------
# COCO / 手術器械クラスIDマッピング
# ---------------------------------------------------------------------------

# COCO pre-trained model classes (ultralytics YOLOv8, 0-indexed COCO80).
# Surgical-instrument-relevant IDs: knife=43（scalpel相当）, scissors=76。
# 注: cls_id は box.cls 由来の標準 COCO80 インデックスなので、ここも標準に揃える。
SURGICAL_INSTRUMENTS: Dict[int, str] = {
    0: "person",
    42: "fork",
    43: "knife",       # scalpel相当
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot_dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    56: "chair",
    57: "couch",
    58: "potted_plant",
    59: "bed",
    60: "dining_table",
    61: "toilet",
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell_phone",
    68: "microwave",
    69: "oven",
    70: "toaster",
    71: "sink",
    72: "refrigerator",
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy_bear",
    78: "hair_drier",
    79: "toothbrush",
}

# 手術器械に近い COCO クラス（infer_phase が参照する knife / scissors）。
# SURGICAL_INSTRUMENTS と同じ標準 COCO80 インデックスに揃える。
INSTRUMENT_COCONames = {
    43: "knife",
    76: "scissors",
}


# ---------------------------------------------------------------------------
# Frame iteration (reuse from redlog)
# ---------------------------------------------------------------------------

def _iter_frames_pyav(video_path: str, fps: float):
    """PyAVでPTSベースのフレームを取得するジェネレータ"""
    import av
    # 例外発生時・ジェネレータの途中破棄時も container を確実に解放する
    # （PyAV コンテナ/ファイルディスクリプタのリーク防止）。
    container = av.open(video_path)
    try:
        stream = container.streams.video[0]
        time_base = float(stream.time_base)
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
    finally:
        container.close()


def _iter_frames_opencv(video_path: str, fps: float):
    """OpenCVでPOS_MSECベースのフレームを取得するジェネレータ"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"動画ファイルを開けません: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps <= 0:
        src_fps = 30.0

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
    """フレームイテレータ。PyAVを優先し、なければOpenCVにフォールバック。

    PyAV が失敗した場合（未インストール、コーデック非対応、破損ファイル等）は
    理由を stderr に出力してから OpenCV にフォールバックする。理由を握り潰すと
    運用時の切り分けが困難になるため、必ずログを残す。

    注意: PyAV が1フレーム以上 yield した後に失敗した場合は途中までのフレームが
    既に消費者へ渡っており、OpenCV で先頭から再取得すると重複が生じる。そのため
    フォールバックは「PyAV が1フレームも返さずに失敗した場合」に限定する。
    """
    produced = False
    try:
        for t, bgr in _iter_frames_pyav(video_path, fps):
            produced = True
            yield t, bgr, "pyav"
        return
    except Exception as e:
        if produced:
            # 途中まで PyAV で処理済み。重複を避けるためフォールバックせず送出する。
            print(f"[yolo_analyzer] PyAV decode failed partway through "
                  f"({type(e).__name__}: {e}); aborting", file=sys.stderr)
            raise
        print(f"[yolo_analyzer] PyAV unavailable or failed "
              f"({type(e).__name__}: {e}); falling back to OpenCV",
              file=sys.stderr)

    for t, bgr in _iter_frames_opencv(video_path, fps):
        yield t, bgr, "opencv"


# ---------------------------------------------------------------------------
# YOLO model wrapper
# ---------------------------------------------------------------------------

class YOLOAnalyzer:
    """YOLO8-based surgical instrument detection."""

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        sample_fps: float = 2.0,
        min_scene_duration: float = 3.0,
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.sample_fps = sample_fps
        self.min_scene_duration = min_scene_duration
        self.device = device
        self.model = None

    def _load_model(self):
        """Lazy load YOLO model."""
        if self.model is None:
            from ultralytics import YOLO
            self.model = YOLO(self.model_name)
            if self.device == "cuda" and hasattr(self.model, "to"):
                self.model.to(self.device)

    def _detect_frame(self, bgr_frame: np.ndarray) -> List[Dict]:
        """Run YOLO inference on a single frame."""
        self._load_model()
        results = self.model(
            bgr_frame,
            conf=self.conf_thres,
            iou=self.iou_thres,
            verbose=False,
        )
        detections = []
        if results and len(results) > 0:
            r = results[0]
            if r.boxes is not None:
                for box in r.boxes:
                    cls_id = int(box.cls.item())
                    conf = float(box.conf.item())
                    det = {
                        "class_id": cls_id,
                        "class_name": SURGICAL_INSTRUMENTS.get(cls_id, f"class_{cls_id}"),
                        "confidence": conf,
                    }
                    detections.append(det)
        return detections


# ---------------------------------------------------------------------------
# Phase inference from instruments
# ---------------------------------------------------------------------------

def infer_phase(instruments: List[str]) -> Tuple[str, List[str]]:
    """
    Detect instrument setから外科的フェーズを推定する。

    Returns:
        (phase_name, primary_instruments)
    """
    inst_set = set(instruments)

    # Cautery / coagulation: bleeding (red) + lack of clear instruments
    # Harmonic / ultrasonic scalpel: knife-like + high confidence
    has_knife = any("knife" in i for i in instruments)
    has_scissors = any("scissors" in i for i in instruments)
    has_person = "person" in instruments

    # Phase estimation based on instrument combination
    if has_knife and not has_scissors:
        return "dissection", [i for i in instruments if "knife" in i]
    elif has_scissors:
        return "cutting", [i for i in instruments if "scissors" in i]
    elif has_person and not has_knife and not has_scissors:
        return "manipulation", [i for i in instruments if i != "person"]
    else:
        return "neutral", instruments


# ---------------------------------------------------------------------------
# Scene clustering
# ---------------------------------------------------------------------------

def cluster_scenes(
    times: List[float],
    instrument_sets: List[List[str]],
    min_duration: float = 3.0,
) -> List[dict]:
    """
    Connective tissue-like clustering: group consecutive frames
    with similar instrument sets into scenes.

    Args:
        times: List of timestamps (sec)
        instrument_sets: List of detected instrument lists per frame
        min_duration: Minimum scene duration (sec)

    Returns:
        List of scene event dicts
    """
    if len(times) == 0:
        return []

    scenes = []
    current_scene = {
        "start": times[0],
        "end": times[0],
        "instruments": instrument_sets[0] if instrument_sets else [],
        "count": 1,
    }

    for i in range(1, len(times)):
        t = times[i]
        instruments = instrument_sets[i] if instrument_sets else []

        # Determine if this frame continues the current scene
        # Scene changes when instrument set changes meaningfully
        prev_set = set(current_scene["instruments"])
        curr_set = set(instruments)

        # Check if there's a significant change
        # (new instruments added or removed)
        changed = (prev_set != curr_set)

        if not changed:
            # Continue current scene
            current_scene["end"] = t
            current_scene["count"] += 1
        else:
            # Finalize current scene if long enough
            duration = current_scene["end"] - current_scene["start"]
            if duration >= min_duration:
                phase, primary = infer_phase(current_scene["instruments"])
                current_scene["phase"] = phase
                current_scene["primary_instruments"] = primary
                scenes.append(current_scene)

            # Start new scene
            current_scene = {
                "start": t,
                "end": t,
                "instruments": instruments,
                "count": 1,
            }

    # Don't forget the last scene
    duration = current_scene["end"] - current_scene["start"]
    if duration >= min_duration:
        phase, primary = infer_phase(current_scene["instruments"])
        current_scene["phase"] = phase
        current_scene["primary_instruments"] = primary
        scenes.append(current_scene)

    return scenes


# ---------------------------------------------------------------------------
# Step 1: record_timeseries
# ---------------------------------------------------------------------------

def record_timeseries(
    video_path: str,
    outdir: str,
    model_name: str = "yolov8n.pt",
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    sample_fps: float = 2.0,
    device: str = "cuda",
) -> dict:
    """
    Step 1: 動画をサンプリングしてinstrument検出JSONLを出力する。

    Args:
        video_path: 入力動画ファイルパス
        outdir: 出力ディレクトリ
        model_name: YOLOモデル名
        conf_thres: 信頼度閾値
        iou_thres: NMS IoU閾値
        sample_fps: サンプリングFPS
        device: 計算デバイス (cuda/cpu)

    Returns:
        {"jsonl": JSONLファイルパス}
    """
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    stem = Path(video_path).stem

    analyzer = YOLOAnalyzer(
        model_name=model_name,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        sample_fps=sample_fps,
        device=device,
    )

    times: List[float] = []
    detections_list: List[List[dict]] = []

    print(f"YOLO解析中: {video_path}", file=sys.stderr)
    print(f"モデル: {model_name}, サンプルFPS: {sample_fps}", file=sys.stderr)

    for t_sec, bgr, reader in iter_frames(video_path, sample_fps):
        dets = analyzer._detect_frame(bgr)
        times.append(t_sec)
        detections_list.append(dets)

        # Progress indicator
        if len(times) % 50 == 0:
            print(f"  フレーム処理中: {len(times)} ({t_sec:.1f}s)", file=sys.stderr)

    if len(times) == 0:
        print("警告: フレームが取得できませんでした。", file=sys.stderr)
        return {}

    # JSONL出力
    jsonl_path = out_path / f"{stem}_yolo_timeseries.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for t, dets in zip(times, detections_list):
            record = {
                "t_sec": round(t, 3),
                "t_srt": format_srt_time(t),
                "detections": dets,
                "instruments": [d["class_name"] for d in dets],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"JSONL: {jsonl_path}")
    return {"jsonl": str(jsonl_path)}


# ---------------------------------------------------------------------------
# Step 2: annotate_scenes
# ---------------------------------------------------------------------------

def annotate_scenes(
    jsonl_path: str,
    outdir: str,
    min_scene_duration: float = 3.0,
) -> dict:
    """
    Step 2: JSONLからinstrument組み合わせ変化でシーン分割し、
            JSONL/SRTを出力する。

    Args:
        jsonl_path: 入力JSONLファイルパス（record_timeseriesの出力）
        outdir: 出力ディレクトリ
        min_scene_duration: 最小シーン継続時間（秒）

    Returns:
        {"jsonl": JSONLファイルパス, "srt": SRTファイルパス, "scenes": シーン数}
    """
    # JSONL読込
    times = []
    instrument_sets = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            times.append(obj["t_sec"])
            instrument_sets.append(obj.get("instruments", []))

    if len(times) == 0:
        return {}

    # シーンクラスタリング
    scenes = cluster_scenes(times, instrument_sets, min_duration=min_scene_duration)

    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    stem = Path(jsonl_path).stem.replace("_yolo_timeseries", "")

    # JSONL出力（正本）
    jsonl_out = out_path / f"{stem}_scenes.jsonl"
    with open(jsonl_out, "w", encoding="utf-8") as f:
        for i, scene in enumerate(scenes, start=1):
            record = {
                "type": "instrument_scene",
                "scene_id": i,
                "phase": scene["phase"],
                "primary_instruments": scene["primary_instruments"],
                "all_instruments": list(set(scene["instruments"])),
                "start_sec": round(scene["start"], 3),
                "end_sec": round(scene["end"], 3),
                "start_srt": format_srt_time(scene["start"]),
                "end_srt": format_srt_time(scene["end"]),
                "duration_sec": round(scene["end"] - scene["start"], 3),
                "frame_count": scene["count"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # SRT出力
    from src.tools.jsonl_to_srt import convert as jsonl_to_srt_convert
    srt_out = out_path / f"{stem}_scenes.srt"
    jsonl_to_srt_convert(
        in_jsonl=str(jsonl_out),
        out_srt=str(srt_out),
        event_type="instrument_scene",
    )

    print(f"JSONL: {jsonl_out} （正本）")
    print(f"SRT  : {srt_out} （Shotcut用）")
    print(f"シーン数: {len(scenes)}")

    return {
        "jsonl": str(jsonl_out),
        "srt": str(srt_out),
        "scenes": len(scenes),
    }


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def analyze_video(
    video_path: str,
    outdir: str,
    model_name: str = "yolov8n.pt",
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    sample_fps: float = 2.0,
    min_scene_duration: float = 3.0,
    device: str = "cuda",
) -> dict:
    """
    動画を解析し、シーンSRT/JSONLを出力する（2ステップの一括実行）。

    Args:
        video_path: 入力動画ファイルパス
        outdir: 出力ディレクトリ
        model_name: YOLOモデル名
        conf_thres: 信頼度閾値
        iou_thres: NMS IoU閾値
        sample_fps: サンプリングFPS
        min_scene_duration: 最小シーン継続時間（秒）
        device: 計算デバイス

    Returns:
        出力ファイルパスの辞書
    """
    # Step 1: 時系列記録
    result1 = record_timeseries(
        video_path=video_path,
        outdir=outdir,
        model_name=model_name,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        sample_fps=sample_fps,
        device=device,
    )

    if not result1:
        return {}

    # Step 2: シーンアノテーション
    result2 = annotate_scenes(
        jsonl_path=result1["jsonl"],
        outdir=outdir,
        min_scene_duration=min_scene_duration,
    )

    return {**result1, **result2}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> int:
    """コマンドラインエントリポイント（サブコマンド方式）"""
    parser = argparse.ArgumentParser(
        description="YOLO8手術器械検出・シーン分析"
    )
    subparsers = parser.add_subparsers(dest="command", help="実行コマンド")

    # --- timeseries ---
    ts_parser = subparsers.add_parser(
        "timeseries", help="Step 1: 時系列記録（動画→JSONL）"
    )
    ts_parser.add_argument("--video", required=True, help="入力動画ファイルパス")
    ts_parser.add_argument("--outdir", required=True, help="出力ディレクトリ")
    ts_parser.add_argument("--model", default="yolov8n.pt",
                          help="YOLOモデル名（デフォルト: yolov8n.pt）")
    ts_parser.add_argument("--conf", type=float, default=0.25,
                          help="信頼度閾値（デフォルト: 0.25）")
    ts_parser.add_argument("--iou", type=float, default=0.45,
                          help="NMS IoU閾値（デフォルト: 0.45）")
    ts_parser.add_argument("--fps", type=float, default=2.0,
                          help="サンプリングFPS（デフォルト: 2.0）")
    ts_parser.add_argument("--device", default="cuda",
                          help="計算デバイス（cuda/cpu, デフォルト: cuda）")

    # --- annotate ---
    ann_parser = subparsers.add_parser(
        "annotate", help="Step 2: シーンアノテーション（JSONL→SRT）"
    )
    ann_parser.add_argument("--jsonl", required=True,
                           help="入力JSONLファイル（timeseries出力）")
    ann_parser.add_argument("--outdir", required=True, help="出力ディレクトリ")
    ann_parser.add_argument("--min-scene", type=float, default=3.0,
                           help="最小シーン継続時間（秒, デフォルト: 3.0）")

    # --- analyze ---
    ana_parser = subparsers.add_parser(
        "analyze", help="一括実行（timeseries + annotate）"
    )
    ana_parser.add_argument("--video", required=True, help="入力動画ファイルパス")
    ana_parser.add_argument("--outdir", required=True, help="出力ディレクトリ")
    ana_parser.add_argument("--model", default="yolov8n.pt",
                           help="YOLOモデル名（デフォルト: yolov8n.pt）")
    ana_parser.add_argument("--conf", type=float, default=0.25,
                           help="信頼度閾値（デフォルト: 0.25）")
    ana_parser.add_argument("--iou", type=float, default=0.45,
                           help="NMS IoU閾値（デフォルト: 0.45）")
    ana_parser.add_argument("--fps", type=float, default=2.0,
                           help="サンプリングFPS（デフォルト: 2.0）")
    ana_parser.add_argument("--min-scene", type=float, default=3.0,
                           help="最小シーン継続時間（秒, デフォルト: 3.0）")
    ana_parser.add_argument("--device", default="cuda",
                           help="計算デバイス（cuda/cpu, デフォルト: cuda）")

    args = parser.parse_args()

    if args.command == "timeseries":
        record_timeseries(
            video_path=args.video,
            outdir=args.outdir,
            model_name=args.model,
            conf_thres=args.conf,
            iou_thres=args.iou,
            sample_fps=args.fps,
            device=args.device,
        )
    elif args.command == "annotate":
        annotate_scenes(
            jsonl_path=args.jsonl,
            outdir=args.outdir,
            min_scene_duration=args.min_scene,
        )
    elif args.command == "analyze":
        analyze_video(
            video_path=args.video,
            outdir=args.outdir,
            model_name=args.model,
            conf_thres=args.conf,
            iou_thres=args.iou,
            sample_fps=args.fps,
            min_scene_duration=args.min_scene,
            device=args.device,
        )
    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
