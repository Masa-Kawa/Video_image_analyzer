#!/usr/bin/env python3
"""
SurgeoNet 手術器械検出推論スクリプト

SurgeoNet (m_1152/best.pt) を使用し、腹腔鏡動画から 13 種類の手術器械を
検出して action_to_outputs 用 per-frame action CSV を出力する。

クラス一覧:
  1: Overholt Clamp, 2: Metz. Scissor, 3: Sur. Scissor, 4: Needle Holder,
  5: Sur. Forceps, 6: Atr. Forceps, 7: Scalpel, 8: Retractor,
  9: Hook, 10: Lig. Clamp, 11: Peri. Clamp, 12: Bowl, 13: Tong

使い方:
  python -m src.action.surgeonet_inference \
      --video out_lapc_eval/LapC_EvalDemo_480p.MP4 \
      --weights third_party/SurgeoNet/yolo/runs/pose/m_1152/weights/best.pt \
      --out out_lapc_eval/LapC_EvalDemo_surgeonet.csv \
      --fps 2 --conf 0.25 --imgsz 1152
"""

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List

import cv2


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
    imgsz: int = 1152,
    device: str = "cuda",
) -> int:
    from ultralytics import YOLO

    print(f"[INFO] Loading SurgeoNet model: {weights_path}")
    model = YOLO(weights_path)
    if device == "cuda" and hasattr(model, "to"):
        model.to(device)

    print(f"[INFO] Task: {model.task}, Classes: {model.names}")
    print(f"[INFO] Input size: {imgsz}, Sampling fps: {sample_fps}")

    rows: List[dict] = []
    frame_idx = 0
    start_time = time.time()

    print(f"[INFO] Processing: {video_path}")

    for t_sec, frame in iter_frames(video_path, sample_fps):
        results = model.predict(
            frame,
            conf=conf_thres,
            iou=iou_thres,
            imgsz=imgsz,
            verbose=False,
            device=device,
        )

        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                cls_id = int(box.cls.item())
                if cls_id == 0:
                    continue  # Background はスキップ
                conf = float(box.conf.item())
                class_name = model.names.get(cls_id, f"class_{cls_id}")
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
    print(f"[INFO] Done. {len(rows)} detections -> {out_csv} ({total_elapsed:.1f}s)")
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description="SurgeoNet surgical instrument detection")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--weights", required=True, help="Path to SurgeoNet .pt")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--fps", type=float, default=2.0, help="Sampling fps")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=1152, help="Model input size")
    parser.add_argument("--device", default="cuda", help="Device")
    args = parser.parse_args()

    run_inference(
        video_path=args.video,
        weights_path=args.weights,
        out_csv=args.out,
        sample_fps=args.fps,
        conf_thres=args.conf,
        iou_thres=args.iou,
        imgsz=args.imgsz,
        device=args.device,
    )


if __name__ == "__main__":
    main()
