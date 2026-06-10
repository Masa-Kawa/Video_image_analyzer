"""BiomedCLIP zero-shot 出血スコアラ。

HuggingFace の microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 を
open_clip 経由でロードする。

依存:
    pip install open_clip_torch transformers

テキストプロンプトは「出血あり」と「出血なし」の2クラス対比で、
softmax 後の「出血あり」確率を出力する。
"""

from __future__ import annotations

from typing import List

import cv2
import numpy as np
import open_clip
import torch
from PIL import Image

from .base import ZeroShotScorer


# 「出血あり」を表すプロンプト群 — 大出血から微小出血まで幅広くカバー
POS_PROMPTS = [
    "laparoscopic view with fresh blood oozing from tissue",
    "endoscopic image showing active bleeding with red blood on surgical surface",
    "minimally invasive surgery with blood pooling around tissue",
    "laparoscopic surgery showing hemorrhage and red blood spreading",
    "blood oozing from a cut vessel during laparoscopic procedure",
    "red blood covering the surgical field in laparoscopy",
]
# 「出血なし」を表すプロンプト群
NEG_PROMPTS = [
    "laparoscopic view of clean dry tissue without blood",
    "endoscopic image of instruments and tissue with no bleeding",
    "normal laparoscopic surgery field with no hemorrhage",
    "laparoscopy showing yellow fat and pink tissue without blood",
    "dry surgical field with surgical instruments in laparoscopy",
]


class BiomedCLIPScorer(ZeroShotScorer):
    name = "biomedclip"

    HF_MODEL = (
        "hf-hub:microsoft/"
        "BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )

    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._text_features = None  # (2, D) [pos_mean, neg_mean]

    def load(self) -> None:
        # Double-checked locking: 高速パス（既ロード）はロックなし、
        # 未ロード時のみロックを取り、再確認してから初期化する。
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            self._load_locked()

    def _load_locked(self) -> None:
        model, _, preprocess = open_clip.create_model_and_transforms(
            self.HF_MODEL
        )
        tokenizer = open_clip.get_tokenizer(self.HF_MODEL)
        model = model.to(self.device).eval()

        with torch.no_grad():
            pos_tok = tokenizer(POS_PROMPTS).to(self.device)
            neg_tok = tokenizer(NEG_PROMPTS).to(self.device)
            pos = model.encode_text(pos_tok)
            neg = model.encode_text(neg_tok)
            pos = pos / pos.norm(dim=-1, keepdim=True)
            neg = neg / neg.norm(dim=-1, keepdim=True)
            txt = torch.stack([pos.mean(0), neg.mean(0)], dim=0)
            txt = txt / txt.norm(dim=-1, keepdim=True)

        self._model = model
        self._preprocess = preprocess
        self._tokenizer = tokenizer
        self._text_features = txt  # (2, D)

    def _to_pil_batch(self, frames: List[np.ndarray]) -> torch.Tensor:
        tensors = []
        for bgr in frames:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            tensors.append(self._preprocess(pil))
        return torch.stack(tensors, dim=0).to(self.device)

    @torch.no_grad()
    def score_frames(self, bgr_frames: List[np.ndarray]) -> List[float]:
        if not bgr_frames:
            return []
        batch = self._to_pil_batch(bgr_frames)
        feats = self._model.encode_image(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        # logit_scale は open_clip 内部実装に依存する属性。モデル差し替えや
        # バージョン差で欠落/型不一致でも落ちないよう安全に取得し、
        # 取れなければ CLIP 標準の 100.0（=ln(100) 相当）にフォールバックする。
        logit_scale = self._safe_logit_scale()
        logits = logit_scale * feats @ self._text_features.t()  # (B, 2)
        probs = logits.softmax(dim=-1)
        return probs[:, 0].detach().cpu().tolist()

    _DEFAULT_LOGIT_SCALE = 100.0  # CLIP 標準（temperature=0.01 相当）

    def _safe_logit_scale(self) -> float:
        """open_clip の logit_scale を安全に取得（欠落/異常時は既定値）。"""
        ls = getattr(self._model, "logit_scale", None)
        if ls is None:
            return self._DEFAULT_LOGIT_SCALE
        try:
            return float(ls.exp().item()) if torch.is_tensor(ls) else float(ls)
        except (RuntimeError, ValueError, TypeError):
            return self._DEFAULT_LOGIT_SCALE

    def unload(self) -> None:
        """GPU 上のモデル・テキスト特徴を解放する（長時間稼働/多重生成時の OOM 対策）。"""
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._text_features = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __del__(self):
        # ベストエフォートで解放（GC 時の例外は握りつぶす）
        try:
            self.unload()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description="BiomedCLIP zero-shot 出血検出")
    p.add_argument("--video", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--fps", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="cuda")
    p.add_argument("--thr", type=float, default=0.5)
    p.add_argument("--min-duration-s", type=float, default=2.0)
    p.add_argument("--no-roi", action="store_true")
    args = p.parse_args()

    from .base import csv_to_events, validate_input_file, validated_outdir

    # 入力動画の存在と出力先パスを呼び出し前に検証する
    video = validate_input_file(args.video, "動画")
    outdir = validated_outdir(args.outdir)

    scorer = BiomedCLIPScorer(device=args.device)
    csv_path = scorer.score_video(
        video_path=video,
        outdir=outdir,
        fps=args.fps,
        batch_size=args.batch_size,
        no_roi=args.no_roi,
    )
    csv_to_events(
        csv_path=csv_path,
        outdir=outdir,
        thr=args.thr,
        min_duration_s=args.min_duration_s,
    )
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
