"""
Cholec80 手術フェーズ認識モデル

アーキテクチャ:
  - Backbone: ResNet50 (SelfSupSurg DINO pretrained on Cholec80)
    - 2048次元の特徴ベクトルを抽出
    - 学習時は凍結（fine-tune オプションあり）
  - Temporal Model: Bidirectional LSTM
    - 特徴ベクトルのシーケンスから時間的文脈を学習
    - 手術の時間的流れを捉える
  - Classification Head: FC → 7クラス
    - Cholec80 標準7フェーズに分類

推論パイプライン:
  1. フレーム → ResNet50 → 2048次元特徴
  2. 特徴シーケンス → BiLSTM → 時間文脈付き表現
  3. 表現 → FC → 7クラス確率

VRAM: ~6GB (推論), ~8GB (学習)
"""

import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.cholec_phase import CHOLEC80_PHASES, NUM_PHASES
from src.anomaly.models import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    SELFSUP_WEIGHTS_URLS,
    _load_selfsup_resnet50,
    preprocess_frame_resnet,
)


# ---------------------------------------------------------------------------
# Feature Extractor (ResNet50 backbone)
# ---------------------------------------------------------------------------


class PhaseFeatureExtractor:
    """
    SelfSupSurg DINO ResNet50 による特徴抽出器。

    anomaly.models.SelfSupSurgExtractor と同じ重みを使うが、
    フェーズ認識に特化したインターフェースを提供する。

    Usage:
        extractor = PhaseFeatureExtractor(device="cuda")
        feat = extractor.extract(frame_bgr)  # (2048,)
        feats = extractor.extract_batch(frames_bgr)  # (N, 2048)
    """

    def __init__(
        self,
        device: str = "cuda",
        weights_path: Optional[str] = None,
        method: str = "dino",
        auto_download: bool = True,
    ):
        self.device = torch.device(
            device if torch.cuda.is_available() and device != "cpu" else "cpu"
        )
        self.method = method
        self.model: Optional[nn.Module] = None
        self._ready = False

        resolved_path = self._resolve_weights(weights_path, method, auto_download)
        if resolved_path:
            try:
                self.model = _load_selfsup_resnet50(resolved_path, self.device)
                self._ready = True
                print(f"PhaseFeatureExtractor ({method}) loaded: {resolved_path}")
            except Exception as e:
                print(f"PhaseFeatureExtractor load failed: {e}", file=sys.stderr)

        if not self._ready:
            print("PhaseFeatureExtractor: ImageNet ResNet50 にフォールバック",
                  file=sys.stderr)
            self._init_imagenet_resnet()

    def _init_imagenet_resnet(self):
        """ImageNet 事前学習 ResNet50 にフォールバック"""
        from torchvision.models import resnet50, ResNet50_Weights

        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        model.fc = nn.Identity()
        model.to(self.device)
        model.eval()
        self.model = model
        self._ready = True

    @staticmethod
    def _resolve_weights(
        weights_path: Optional[str],
        method: str,
        auto_download: bool,
    ) -> Optional[str]:
        """重みファイルのパスを解決する。"""
        if weights_path and Path(weights_path).exists():
            return weights_path

        cache_dir = Path.home() / "AI" / "huggingface" / "selfsupsurg"
        filename = f"model_final_checkpoint_{method}_surg.torch"
        cached_path = cache_dir / filename

        if cached_path.exists():
            return str(cached_path)

        torch_home = Path(os.environ.get("TORCH_HOME", Path.home() / ".cache" / "torch"))
        torch_cache = torch_home / "hub" / "checkpoints" / filename
        if torch_cache.exists():
            return str(torch_cache)

        if not auto_download:
            return None

        url = SELFSUP_WEIGHTS_URLS.get(method)
        if not url:
            return None

        print(f"SelfSupSurg ({method}) 重みをダウンロード中...")
        cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            torch.hub.download_url_to_file(url, str(cached_path))
            print(f"ダウンロード完了: {cached_path}")
            return str(cached_path)
        except Exception as e:
            print(f"ダウンロード失敗: {e}", file=sys.stderr)
            return None

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def feature_dim(self) -> int:
        return 2048

    @torch.no_grad()
    def extract(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        1フレームから 2048次元特徴ベクトルを抽出する。

        Args:
            frame_bgr: (H, W, 3) BGR uint8

        Returns:
            (2048,) numpy array
        """
        tensor = preprocess_frame_resnet(frame_bgr).unsqueeze(0).to(self.device)
        with torch.amp.autocast(self.device.type, enabled=self.device.type == "cuda"):
            feat = self.model(tensor)
        return feat.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def extract_batch(
        self,
        frames_bgr: List[np.ndarray],
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        複数フレームからバッチで特徴抽出する。

        Args:
            frames_bgr: BGR フレームのリスト
            batch_size: バッチサイズ

        Returns:
            (N, 2048) numpy array
        """
        all_feats = []
        for i in range(0, len(frames_bgr), batch_size):
            batch = frames_bgr[i:i + batch_size]
            tensors = torch.stack([
                preprocess_frame_resnet(f) for f in batch
            ]).to(self.device)
            with torch.amp.autocast(
                self.device.type, enabled=self.device.type == "cuda"
            ):
                feats = self.model(tensors)
            all_feats.append(feats.cpu().numpy())
        return np.concatenate(all_feats, axis=0)


# ---------------------------------------------------------------------------
# Phase Recognition Model (BiLSTM + Classifier)
# ---------------------------------------------------------------------------


class PhaseRecognitionModel(nn.Module):
    """
    手術フェーズ認識モデル（BiLSTM + FC）。

    入力: 特徴ベクトルのシーケンス (B, T, 2048)
    出力: フェーズ確率のシーケンス (B, T, 7)

    ResNet50 backbone は含まない（特徴抽出済みの入力を受け取る）。
    これにより学習時に特徴抽出を1回だけ行い、LSTM学習を高速化できる。

    Args:
        feature_dim: 入力特徴次元（デフォルト: 2048）
        hidden_dim: LSTM 隠れ次元（デフォルト: 512）
        num_layers: LSTM 層数（デフォルト: 2）
        num_classes: 分類クラス数（デフォルト: 7）
        dropout: ドロップアウト率（デフォルト: 0.3）
        bidirectional: 双方向か（デフォルト: True）
    """

    def __init__(
        self,
        feature_dim: int = 2048,
        hidden_dim: int = 512,
        num_layers: int = 2,
        num_classes: int = NUM_PHASES,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, feature_dim) 特徴ベクトルシーケンス
            lengths: (B,) 各バッチのシーケンス長（パディング時に使用）

        Returns:
            (B, T, num_classes) フェーズロジット
        """
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False,
            )
            lstm_out, _ = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out, batch_first=True,
            )
        else:
            lstm_out, _ = self.lstm(x)

        logits = self.classifier(lstm_out)
        return logits

    def predict_proba(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        ソフトマックス確率を返す。

        Returns:
            (B, T, num_classes) フェーズ確率
        """
        logits = self.forward(x, lengths)
        return F.softmax(logits, dim=-1)


# ---------------------------------------------------------------------------
# Model Manager
# ---------------------------------------------------------------------------


class PhaseModelManager:
    """
    フェーズ認識モデルの管理クラス。

    特徴抽出器（ResNet50）とフェーズ認識モデル（BiLSTM）を統合し、
    推論パイプラインを提供する。

    Usage:
        mgr = PhaseModelManager(model_path="phase_model.pth")
        phase_id, phase_name, probs = mgr.predict_frame(frame_bgr)
    """

    def __init__(
        self,
        device: str = "cuda",
        model_path: Optional[str] = None,
        backbone_weights: Optional[str] = None,
        backbone_method: str = "dino",
        auto_download_backbone: bool = True,
        hidden_dim: int = 512,
        num_layers: int = 2,
        context_frames: int = 30,
    ):
        self.device = torch.device(
            device if torch.cuda.is_available() and device != "cpu" else "cpu"
        )
        self.context_frames = context_frames
        self._feature_buffer: List[np.ndarray] = []

        # Backbone
        self.extractor = PhaseFeatureExtractor(
            device=device,
            weights_path=backbone_weights,
            method=backbone_method,
            auto_download=auto_download_backbone,
        )

        # Phase model
        self.phase_model: Optional[PhaseRecognitionModel] = None
        self._model_ready = False

        if model_path and Path(model_path).exists():
            try:
                self.phase_model = PhaseRecognitionModel(
                    feature_dim=2048,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                )
                state = torch.load(model_path, map_location=self.device)
                if "model_state_dict" in state:
                    self.phase_model.load_state_dict(state["model_state_dict"])
                else:
                    self.phase_model.load_state_dict(state)
                self.phase_model.to(self.device)
                self.phase_model.eval()
                self._model_ready = True
                print(f"PhaseRecognitionModel loaded: {model_path}")
            except Exception as e:
                print(f"PhaseRecognitionModel load failed: {e}", file=sys.stderr)

        if not self._model_ready:
            print("PhaseModelManager: 学習済みモデルなし。"
                  "特徴抽出のみ可能（フェーズ予測には学習が必要）。",
                  file=sys.stderr)

    @property
    def is_ready(self) -> bool:
        return self._model_ready and self.extractor.is_ready

    def reset_state(self) -> None:
        """内部状態（特徴バッファ）をリセットする。"""
        self._feature_buffer.clear()

    @torch.no_grad()
    def predict_frame(
        self,
        frame_bgr: np.ndarray,
    ) -> Tuple[int, str, np.ndarray]:
        """
        1フレームのフェーズを予測する。

        内部で特徴バッファを保持し、BiLSTMに文脈を与える。

        Args:
            frame_bgr: (H, W, 3) BGR uint8

        Returns:
            (phase_id, phase_name, probs[7])
        """
        feat = self.extractor.extract(frame_bgr)
        self._feature_buffer.append(feat)

        if not self._model_ready:
            return 0, CHOLEC80_PHASES[0], np.zeros(NUM_PHASES)

        # 文脈窓を構築
        ctx = self._feature_buffer[-self.context_frames:]
        seq = torch.from_numpy(np.array(ctx)).float().unsqueeze(0).to(self.device)

        probs = self.phase_model.predict_proba(seq)  # (1, T, 7)
        last_probs = probs[0, -1].cpu().numpy()  # 最新フレームの確率

        phase_id = int(np.argmax(last_probs))
        phase_name = CHOLEC80_PHASES[phase_id]

        return phase_id, phase_name, last_probs

    @torch.no_grad()
    def predict_sequence(
        self,
        features: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        特徴シーケンス全体のフェーズを予測する（オフライン推論）。

        Args:
            features: (N, 2048) 特徴ベクトルの配列

        Returns:
            (phase_ids[N], probs[N, 7])
        """
        if not self._model_ready:
            ids = np.zeros(len(features), dtype=int)
            probs = np.zeros((len(features), NUM_PHASES))
            return ids, probs

        seq = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
        prob_seq = self.phase_model.predict_proba(seq)  # (1, N, 7)
        probs = prob_seq[0].cpu().numpy()
        phase_ids = np.argmax(probs, axis=1)

        return phase_ids, probs
