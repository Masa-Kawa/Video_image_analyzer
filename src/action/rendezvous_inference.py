#!/usr/bin/env python3
"""
Rendezvous (CAMMA) triplet 推論スクリプト
動画を1 fps でサンプリングし、Rendezvous モデルで triplet 予測を行い、
action_to_outputs 用の CSV（frame_idx, triplet_id, confidence）を出力する。

使い方:
  python -m src.action.rendezvous_inference \
      --video out_lapc_eval/LapC_EvalDemo_480p.MP4 \
      --weights models/rendezvous_crossval_k1.pth \
      --out out_lapc_eval/LapC_EvalDemo_triplet_pred.csv \
      --fps 1
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Rendezvous コードパスを追加
RENDEZVOUS_ROOT = Path(__file__).resolve().parent.parent.parent / "third_party" / "rendezvous" / "pytorch"
sys.path.insert(0, str(RENDEZVOUS_ROOT))

import network as rdv_network  # noqa: E402


# ---------------------------------------------------------------------------
# 前処理（Rendezvous 公式と同じ: 256x448, mean-normalize）
# ---------------------------------------------------------------------------

def preprocess_frame(frame_bgr: np.ndarray) -> torch.Tensor:
    """OpenCV BGR フレーム → モデル入力 Tensor [1,3,256,448]"""
    # BGR → RGB
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    # リサイズ
    img = img.resize((448, 256), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32)
    # mean normalization（ImageNet 統計値）
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr / 255.0 - mean) / std
    arr = arr.transpose(2, 0, 1)  # HWC → CHW
    return torch.from_numpy(arr).unsqueeze(0).float()


# ---------------------------------------------------------------------------
# モデルロード
# ---------------------------------------------------------------------------

def load_model(weights_path: str, device: str = "cuda") -> rdv_network.Rendezvous:
    model = rdv_network.Rendezvous(
        basename="resnet18",
        num_tool=6,
        num_verb=10,
        num_target=15,
        num_triplet=100,
        layer_size=8,
        num_heads=4,
        d_model=128,
        hr_output=False,
        use_ln=True,  # crossval k1 は layernorm
    )
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f"[INFO] Model loaded from {weights_path}")
    return model


# ---------------------------------------------------------------------------
# 推論ループ
# ---------------------------------------------------------------------------

def run_inference(
    video_path: str,
    model: rdv_network.Rendezvous,
    out_csv: str,
    sample_fps: float = 1.0,
    device: str = "cuda",
    batch_size: int = 32,
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skip = max(1, int(round(video_fps / sample_fps)))
    expected = total_frames // skip
    print(f"[INFO] Video: {total_frames} frames @ {video_fps:.2f} fps")
    print(f"[INFO] Sampling every {skip} frame(s) ≈ {sample_fps} fps")
    print(f"[INFO] Expected samples: ~{expected}")

    # バッチ処理用バッファ
    batch_imgs: list[torch.Tensor] = []
    batch_meta: list[tuple[int, float]] = []  # (frame_idx, timestamp_sec)

    rows: list[dict] = []
    frame_idx = 0
    start_time = time.time()

    def _flush_batch():
        nonlocal rows
        if not batch_imgs:
            return
        x = torch.cat(batch_imgs, dim=0).to(device)
        with torch.no_grad():
            _, _, _, triplet_logits = model(x)
            probs = torch.sigmoid(triplet_logits).cpu().numpy()  # [B, 100]
        for meta, prob_vec in zip(batch_meta, probs):
            fidx, tsec = meta
            for tid in range(100):
                p = float(prob_vec[tid])
                if p > 0.0:  # すべて書き出してもOK（後で閾値処理する）
                    rows.append({
                        "frame_idx": fidx,
                        "timestamp_sec": round(tsec, 3),
                        "triplet_id": tid,
                        "confidence": round(p, 6),
                    })
        batch_imgs.clear()
        batch_meta.clear()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % skip == 0:
            tsec = frame_idx / video_fps if video_fps > 0 else 0.0
            tensor = preprocess_frame(frame)
            batch_imgs.append(tensor)
            batch_meta.append((frame_idx, tsec))
            if len(batch_imgs) >= batch_size:
                _flush_batch()
        frame_idx += 1
        if frame_idx % 500 == 0:
            elapsed = time.time() - start_time
            print(f"  processed {frame_idx}/{total_frames} frames ({elapsed:.1f}s)")

    _flush_batch()
    cap.release()

    # CSV 書き出し
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["frame_idx", "timestamp_sec", "triplet_id", "confidence"])
        writer.writeheader()
        writer.writerows(rows)

    total_elapsed = time.time() - start_time
    print(f"[INFO] Done. {len(rows)} predictions written to {out_csv} ({total_elapsed:.1f}s)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Rendezvous triplet inference on a custom video")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--weights", required=True, help="Path to rendezvous .pth checkpoint")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--fps", type=float, default=1.0, help="Sampling fps (default: 1)")
    parser.add_argument("--device", default="cuda", help="torch device")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    args = parser.parse_args()

    model = load_model(args.weights, device=args.device)
    run_inference(
        video_path=args.video,
        model=model,
        out_csv=args.out,
        sample_fps=args.fps,
        device=args.device,
        batch_size=args.batch,
    )


if __name__ == "__main__":
    main()
