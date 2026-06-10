"""SurgVLP / PeskaVLP zero-shot 出血スコアラ。

公式リポジトリ: https://github.com/CAMMA-public/SurgVLP

セットアップ（プロジェクト外で）:
    git clone https://github.com/CAMMA-public/SurgVLP.git
    pip install -e ./SurgVLP

重みは初回呼び出し時に著者ホスティングからダウンロードされる。
config ファイル（config_surgvlp.py 等）の場所は SURGVLP_CONFIG_DIR
環境変数で上書きできる（未設定時は surgvlp パッケージ同梱の tests/ を参照）。

依存が不足する場合、`load()` で明示的なエラーを出す。
"""

from __future__ import annotations

import os
from typing import List

import cv2
import numpy as np
import torch

from .base import ZeroShotScorer


# SurgVLP/PeskaVLP は手術ナレーション風プロンプトで学習されているため、
# 第一人称の手術記述スタイルを採用する（test/zero_shot.py の prompts に倣う）。
POS_PROMPTS = [
    "I see active bleeding from the tissue with red blood pooling in the surgical field",
    "There is hemorrhage and the surgical field is filled with fresh blood",
    "Blood is oozing from the cut tissue and I need to control the bleeding",
    "I observe a pool of blood around the dissection site",
    "The surgical site is bleeding and red blood spreads across the field",
    "I see fresh bright red blood covering the operative field",
]
NEG_PROMPTS = [
    "I see clean tissue and surgical instruments without any bleeding",
    "The surgical field is dry and I dissect the tissue with the hook",
    "I use the grasper to manipulate tissue in a clean operative field",
    "The dissection site is clean without blood and I continue the procedure",
    "I observe normal anatomy and yellow fat without any hemorrhage",
    "The operative field is dry and I proceed with the laparoscopic surgery",
]


class SurgVLPScorer(ZeroShotScorer):
    name = "surgvlp"

    def __init__(
        self,
        device: str = "cuda",
        backbone: str = "PeskaVLP",
    ):
        """
        Args:
            backbone: "SurgVLP", "HecVL", "PeskaVLP" のいずれか
        """
        super().__init__(device)
        self.backbone = backbone
        self._model = None
        self._preprocess = None
        self._text_features = None

    def load(self) -> None:
        # Double-checked locking: 既ロードならロックなしで即返し、未ロード時のみ
        # ロックを取り再確認してから初期化する（複数スレッドの二重ロード防止）。
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            self._load_locked()

    def _load_locked(self) -> None:
        try:
            import surgvlp
            from mmengine.config import Config
        except ImportError as e:
            raise ImportError(
                "SurgVLP がインストールされていません。\n"
                "  git clone https://github.com/CAMMA-public/SurgVLP.git\n"
                "  pip install -e ./SurgVLP\n"
                f"元エラー: {e}"
            )

        cfg_name = {
            "SurgVLP": "config_surgvlp.py",
            "HecVL": "config_hecvl.py",
            "PeskaVLP": "config_peskavlp.py",
        }[self.backbone]

        # configs は tests/ 配下にある（著者リポジトリ構造）
        # 優先度: env > $repo/tests。'..' を含むため normpath で正規化する。
        default_cfg_dir = os.path.normpath(os.path.join(
            os.path.dirname(surgvlp.__file__), "..", "tests"
        ))
        cfg_dir = os.environ.get("SURGVLP_CONFIG_DIR", default_cfg_dir)
        cfg_path = os.path.normpath(os.path.join(cfg_dir, cfg_name))
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(
                f"SurgVLP config が見つかりません: {cfg_path}\n"
                "SURGVLP_CONFIG_DIR を設定してください"
            )
        configs = Config.fromfile(cfg_path)["config"]

        model, preprocess = surgvlp.load(
            configs.model_config, device=self.device,
        )
        model.eval()

        with torch.no_grad():
            pos_tok = surgvlp.tokenize(POS_PROMPTS, device=self.device)
            neg_tok = surgvlp.tokenize(NEG_PROMPTS, device=self.device)
            pos_out = model(None, pos_tok, mode="text")
            neg_out = model(None, neg_tok, mode="text")
            pos = pos_out["text_emb"]
            neg = neg_out["text_emb"]
            pos = pos / pos.norm(dim=-1, keepdim=True)
            neg = neg / neg.norm(dim=-1, keepdim=True)
            txt = torch.stack([pos.mean(0), neg.mean(0)], dim=0)
            txt = txt / txt.norm(dim=-1, keepdim=True)

        self._model = model
        self._preprocess = preprocess
        self._text_features = txt

    def _to_batch(self, frames: List[np.ndarray]) -> torch.Tensor:
        from PIL import Image

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
        batch = self._to_batch(bgr_frames)
        out = self._model(batch, None, mode="video")
        feats = out["img_emb"] if isinstance(out, dict) else out
        feats = feats / feats.norm(dim=-1, keepdim=True)
        sim = feats @ self._text_features.t()  # (B, 2)
        probs = (sim * 100.0).softmax(dim=-1)
        return probs[:, 0].detach().cpu().tolist()


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description="SurgVLP/PeskaVLP zero-shot 出血検出")
    p.add_argument("--video", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--backbone", default="PeskaVLP",
                   choices=["SurgVLP", "HecVL", "PeskaVLP"])
    p.add_argument("--fps", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="cuda")
    p.add_argument("--thr", type=float, default=0.5)
    p.add_argument("--min-duration-s", type=float, default=2.0)
    p.add_argument("--no-roi", action="store_true")
    args = p.parse_args()

    from .base import csv_to_events

    scorer = SurgVLPScorer(device=args.device, backbone=args.backbone)
    scorer.name = f"surgvlp_{args.backbone.lower()}"
    csv_path = scorer.score_video(
        video_path=args.video,
        outdir=args.outdir,
        fps=args.fps,
        batch_size=args.batch_size,
        no_roi=args.no_roi,
    )
    csv_to_events(
        csv_path=csv_path,
        outdir=args.outdir,
        thr=args.thr,
        min_duration_s=args.min_duration_s,
    )
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
