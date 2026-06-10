"""DINOv2 frozen 特徴 + kNN 異常検知による出血候補検出。

戦略:
  1. 全フレームを DINOv2 (frozen) で特徴抽出
  2. 動画前半（normal_window_s）を "normal" 参照集合とする
  3. 各フレームの異常スコア = 正常集合への kNN 平均距離
  4. min-max 正規化して [0, 1] のスコアを得る

依存:
    pip install timm  （torch.hub 経由でも動作）

メモ:
  - 出血以外の異常（カメラ運動、煙、白飛び等）も拾う可能性あり。
  - 既存 zero-shot モデル（SurgVLP 等）の補完として使うのが筋。
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

from .base import ZeroShotScorer


class DINOv2AnomalyScorer(ZeroShotScorer):
    name = "dinov2_anomaly"

    def __init__(
        self,
        device: str = "cuda",
        model_name: str = "dinov2_vitb14",
        normal_window_s: Tuple[float, float] = (0.0, 360.0),
        knn_k: int = 20,
    ):
        """
        Args:
            model_name: torch.hub の DINOv2 モデル名
                ("dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14")
            normal_window_s: 正常参照に使う時間範囲（秒）
            knn_k: kNN の k
        """
        super().__init__(device)
        self.model_name = model_name
        self.normal_window_s = normal_window_s
        self.knn_k = knn_k
        self._model = None
        self._features: List[np.ndarray] = []
        self._times: List[float] = []

    def load(self) -> None:
        if self._model is not None:
            return
        # torch.hub から DINOv2 をロード
        model = torch.hub.load(
            "facebookresearch/dinov2", self.model_name,
            pretrained=True, trust_repo=True,
        )
        model = model.to(self.device).eval()
        self._model = model

    def _preprocess(self, bgr: np.ndarray) -> torch.Tensor:
        """BGR → DINOv2 入力テンソル (3, 224, 224)。"""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # 224x224 にリサイズ（patch 14 の倍数）
        rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(rgb).float() / 255.0
        x = x.permute(2, 0, 1)  # HWC -> CHW
        # ImageNet 正規化
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        x = (x - mean) / std
        return x

    @torch.no_grad()
    def score_frames(self, bgr_frames: List[np.ndarray]) -> List[float]:
        """ZeroShotScorer 互換: 特徴を蓄積する（スコアは後で確定）。"""
        if not bgr_frames:
            return []
        batch = torch.stack([self._preprocess(b) for b in bgr_frames])
        batch = batch.to(self.device)
        feats = self._model(batch)  # (B, D)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        feats_np = feats.cpu().numpy()
        for f in feats_np:
            self._features.append(f)
        # 仮のスコア（後で score_video が確定値に上書き）
        return [0.0] * len(bgr_frames)

    # -----------------------------------------------------------------
    # score_video を override: 特徴蓄積 → 異常スコア計算 → CSV書き出し
    # -----------------------------------------------------------------
    def score_video(
        self,
        video_path: str,
        outdir: str,
        fps: float = 1.0,
        batch_size: int = 16,
        roi_margin: float = 0.08,
        no_roi: bool = False,
        smooth_s: float = 5.0,
    ) -> str:
        from pathlib import Path
        import csv as _csv
        from src.core.time_utils import format_srt_time
        from src.red.redlog import (
            iter_frames, make_circular_roi, smooth_center,
        )

        out_path = Path(outdir)
        out_path.mkdir(parents=True, exist_ok=True)
        stem = Path(video_path).stem
        csv_path = out_path / f"{stem}_{self.name}.csv"

        self.load()
        self._features = []
        self._times = []

        # フレーム走査 + 特徴抽出
        buf_t: List[float] = []
        buf_f: List[np.ndarray] = []
        roi_mask: Optional[np.ndarray] = None
        roi_initialized = False

        for t_sec, bgr, _r in iter_frames(video_path, fps):
            if not roi_initialized:
                h, w = bgr.shape[:2]
                if not no_roi:
                    roi_mask = make_circular_roi(h, w, margin=roi_margin)
                roi_initialized = True
            frame = bgr
            if roi_mask is not None:
                frame = bgr.copy()
                frame[~roi_mask] = 0
            buf_t.append(t_sec)
            buf_f.append(frame)
            if len(buf_f) >= batch_size:
                self.score_frames(buf_f)
                self._times.extend(buf_t)
                buf_t.clear()
                buf_f.clear()
        if buf_f:
            self.score_frames(buf_f)
            self._times.extend(buf_t)

        if not self._times:
            raise RuntimeError("フレーム取得に失敗")

        F = np.stack(self._features, axis=0)  # (N, D)
        T = np.array(self._times)

        # 正常参照集合
        a, b = self.normal_window_s
        normal_mask = (T >= a) & (T <= b)
        if normal_mask.sum() < self.knn_k + 1:
            raise RuntimeError(
                f"正常参照フレームが不足: {normal_mask.sum()} < k+1={self.knn_k+1}"
            )
        normal_feats = F[normal_mask]
        print(f"正常参照: {normal_mask.sum()} フレーム ({a:.0f}-{b:.0f}s)")

        # kNN 平均距離（コサイン: 正規化済みなので 1 - dot）
        sims = F @ normal_feats.T  # (N, M) cosine 類似度（正規化済み）
        # 自己フレームは除外（normal範囲内のフレームでは自己が距離0になるので注意）
        # 単純化: top-k 類似度の平均を距離化
        top_sims = np.partition(sims, -self.knn_k, axis=1)[:, -self.knn_k:]
        mean_top_sim = top_sims.mean(axis=1)
        anomaly = 1.0 - mean_top_sim  # 0=正常、大=異常

        # min-max 正規化
        rng = anomaly.max() - anomaly.min()
        if rng > 1e-9:
            scores = (anomaly - anomaly.min()) / rng
        else:
            scores = anomaly * 0.0

        # 平滑化
        sample_fps = 1.0 / (T[1] - T[0]) if len(T) >= 2 else fps
        window = max(1, int(round(smooth_s * sample_fps)))
        smooth = smooth_center(scores.tolist(), window)

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w_ = _csv.writer(f)
            w_.writerow(["t_sec", "t_srt", "score", "smooth_score"])
            for t, s, ss in zip(T, scores, smooth):
                w_.writerow([
                    f"{t:.3f}", format_srt_time(float(t)),
                    f"{s:.6f}", f"{ss:.6f}",
                ])
        print(f"CSV: {csv_path} ({len(T)} frames)")

        # 特徴とタイムスタンプを保存（後続の bootstrap で利用）
        feat_path = out_path / f"{stem}_dinov2_features.npz"
        np.savez(feat_path, features=F.astype(np.float32), times=T)
        print(f"Features: {feat_path} (shape={F.shape})")

        return str(csv_path)


def main() -> int:
    import argparse
    from .base import csv_to_events

    p = argparse.ArgumentParser(description="DINOv2 frozen 異常検知")
    p.add_argument("--video", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--model", default="dinov2_vitb14",
                   choices=["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14"])
    p.add_argument("--fps", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="cuda")
    p.add_argument("--normal-start", type=float, default=0.0)
    p.add_argument("--normal-end", type=float, default=360.0)
    p.add_argument("--knn-k", type=int, default=20)
    p.add_argument("--thr", type=float, default=0.5)
    p.add_argument("--min-duration-s", type=float, default=2.0)
    p.add_argument("--no-roi", action="store_true")
    args = p.parse_args()

    scorer = DINOv2AnomalyScorer(
        device=args.device,
        model_name=args.model,
        normal_window_s=(args.normal_start, args.normal_end),
        knn_k=args.knn_k,
    )
    csv_path = scorer.score_video(
        video_path=args.video,
        outdir=args.outdir,
        fps=args.fps,
        batch_size=args.batch_size,
        no_roi=args.no_roi,
    )
    csv_to_events(
        csv_path=csv_path, outdir=args.outdir,
        thr=args.thr, min_duration_s=args.min_duration_s,
    )
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
