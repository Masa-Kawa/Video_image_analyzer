#!/usr/bin/env python3
"""
手術器械専用YOLO検出（surgical-tool-detection / mrnkim）

mrnkim/surgical-tool-detection の best.pt を使用し、腹腔鏡動画から
7 種類の手術器械（Bipolar, Clipper, Grasper, Hook, Irrigator, Scissors, Specimen Bag）
を検出する。推論結果は action_to_outputs 用の per-frame action CSV へ書き出す。

使い方:
  python -m src.action.surgical_yolo_inference \
      --video out_lapc_eval2/LapC_EvalDemo2_480p.MP4 \
      --weights third_party/surgical-tool-detection/backend/best.pt \
      --out out_lapc_eval2/LapC_EvalDemo2_surgical_yolo.csv \
      --fps 2 \
      --conf 0.25
"""

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

# 手術器械クラス名（model.names と一致）
INSTRUMENT_NAMES: Dict[int, str] = {
    0: "Bipolar",
    1: "Clipper",
    2: "Grasper",
    3: "Hook",
    4: "Irrigator",
    5: "Scissors",
    6: "Specimen Bag",
}


def iter_frames(video_path: str, sample_fps: float):
    """OpenCV で動画を sample_fps で間引いてフレームを返す。"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps <= 0:
        src_fps = 30.0

    step = max(1, int(round(src_fps / sample_fps)))
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


def run_inference(
    video_path: str,
    weights_path: str,
    out_csv: str,
    sample_fps: float = 2.0,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    device: str = "cuda",
) -> int:
    from ultralytics import YOLO

    print(f"[INFO] Loading model: {weights_path}")
    model = YOLO(weights_path)
    if device == "cuda" and hasattr(model, "to"):
        model.to(device)

    # モデルのクラス名を表示
    print(f"[INFO] Classes: {model.names}")

    rows: List[dict] = []
    frame_idx = 0
    start_time = time.time()

    print(f"[INFO] Processing video: {video_path} @ {sample_fps} fps")

    for t_sec, frame in iter_frames(video_path, sample_fps):
        results = model.predict(
            frame,
            conf=conf_thres,
            iou=iou_thres,
            verbose=False,
            device=device,
        )

        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                class_name = model.names.get(cls_id, INSTRUMENT_NAMES.get(cls_id, f"class_{cls_id}"))
                rows.append({
                    "frame_idx": frame_idx,
                    "timestamp_sec": round(t_sec, 3),
                    "action_name": class_name,
                    "confidence": round(conf, 4),
                })

        frame_idx += 1
        if frame_idx % 100 == 0:
            elapsed = time.time() - start_time
            print(f"  processed {frame_idx} frames ({elapsed:.1f}s)")

    # CSV 書き出し
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["frame_idx", "timestamp_sec", "action_name", "confidence"])
        writer.writeheader()
        writer.writerows(rows)

    total_elapsed = time.time() - start_time
    print(f"[INFO] Done. {len(rows)} detections written to {out_csv} ({total_elapsed:.1f}s)")
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description="Surgical instrument detection using mrnkim's YOLO model")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--weights", required=True, help="Path to best.pt")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--fps", type=float, default=2.0, help="Sampling fps (default: 2)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold (default: 0.45)")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    args = parser.parse_args()

    run_inference(
        video_path=args.video,
        weights_path=args.weights,
        out_csv=args.out,
        sample_fps=args.fps,
        conf_thres=args.conf,
        iou_thres=args.iou,
        device=args.device,
    )


if __name__ == "__main__":
    main()
