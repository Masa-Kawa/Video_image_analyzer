"""
ConvLSTM Autoencoder for self-supervised video anomaly detection.

正常フレームシーケンスから次フレームを予測するモデル。
予測誤差が大きい = 異常（出血・煙・急激なカメラ動き等）。

Architecture:
  - ConvLSTM Encoder: フレームシーケンスを時空間特徴に圧縮
  - CNN Decoder: 特徴マップから次フレームを再構成
  - 入力: (B, T, 3, H, W) RGB normalized [0, 1]
  - 出力: (B, 3, H, W) 予測次フレーム

追加モード: SelfSupSurg (DINO pretrained on Cholec80)
  - ResNet50 で 2048次元特徴ベクトルを抽出
  - フレーム間の特徴距離で異常スコアを計算
  - 学習不要で手術映像に特化した特徴が使える

RTX 4070 (12GB VRAM) で動作可能:
  - ConvLSTM: フレームサイズ 128x128, VRAM ~4GB (学習), ~2GB (推論)
  - SelfSupSurg: ResNet50 224x224, VRAM ~2GB (推論のみ)
"""

import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# ConvLSTM Cell
# ---------------------------------------------------------------------------


class ConvLSTMCell(nn.Module):
    """
    Convolutional LSTM Cell.

    通常の LSTM の全結合層を畳み込みに置換し、
    空間構造を保持しながら時系列を処理する。
    """

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2

        # input gate, forget gate, cell gate, output gate を一括計算
        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size,
            padding=padding,
            bias=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: (B, C_in, H, W)
            state: (h, c) each (B, C_hidden, H, W), or None for zero-init

        Returns:
            h: (B, C_hidden, H, W)
            (h, c): new state
        """
        if state is None:
            b, _, h, w = x.shape
            device = x.device
            h_prev = torch.zeros(b, self.hidden_channels, h, w, device=device)
            c_prev = torch.zeros(b, self.hidden_channels, h, w, device=device)
        else:
            h_prev, c_prev = state

        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv(combined)

        i, f, g, o = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        c = f * c_prev + i * g
        h = o * torch.tanh(c)

        return h, (h, c)


# ---------------------------------------------------------------------------
# ConvLSTM (multi-step)
# ---------------------------------------------------------------------------


class ConvLSTM(nn.Module):
    """
    Multi-layer ConvLSTM.

    複数のConvLSTMCellを積み重ね、時系列フレームを処理する。
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: List[int],
        kernel_size: int = 3,
    ):
        super().__init__()
        self.num_layers = len(hidden_channels)
        self.cells = nn.ModuleList()

        for i, hc in enumerate(hidden_channels):
            ic = in_channels if i == 0 else hidden_channels[i - 1]
            self.cells.append(ConvLSTMCell(ic, hc, kernel_size))

    def forward(
        self,
        x: torch.Tensor,
        states: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x: (B, T, C, H, W) 時系列フレーム
            states: 各層の初期状態リスト, or None

        Returns:
            last_h: (B, C_hidden[-1], H, W) 最終タイムステップの出力
            new_states: 各層の最終状態
        """
        b, t, c, h, w = x.shape

        if states is None:
            states = [None] * self.num_layers

        new_states: List[Tuple[torch.Tensor, torch.Tensor]] = []
        current_input = x

        for layer_idx, cell in enumerate(self.cells):
            layer_state = states[layer_idx]
            outputs = []
            for ti in range(t):
                frame = current_input[:, ti]
                h_out, layer_state = cell(frame, layer_state)
                outputs.append(h_out)

            new_states.append(layer_state)
            current_input = torch.stack(outputs, dim=1)

        # 最終タイムステップの出力
        last_h = current_input[:, -1]
        return last_h, new_states


# ---------------------------------------------------------------------------
# Decoder (CNN)
# ---------------------------------------------------------------------------


class FrameDecoder(nn.Module):
    """
    CNN Decoder: ConvLSTM の隠れ状態から次フレームを再構成する。

    Transposed convolution で解像度を復元。
    """

    def __init__(self, in_channels: int, out_channels: int = 3):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 3, padding=1),
            nn.Sigmoid(),  # 出力を [0, 1] に正規化
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


# ---------------------------------------------------------------------------
# ConvLSTM Autoencoder (Encoder-Decoder)
# ---------------------------------------------------------------------------


class ConvLSTMAutoencoder(nn.Module):
    """
    ConvLSTM Autoencoder for future frame prediction.

    入力: T フレームのシーケンス → 出力: 次の1フレームの予測
    予測と実際の差分 = 異常スコア。

    構成:
      - Spatial Encoder: CNN で空間特徴を抽出 (3ch → feature_dim)
      - Temporal Encoder: ConvLSTM で時間軸を圧縮
      - Decoder: CNN で次フレームを再構成

    Args:
        seq_len: 入力シーケンス長（デフォルト: 4）
        feature_dim: 空間特徴の次元数（デフォルト: 64）
        hidden_dims: ConvLSTM の隠れ次元リスト（デフォルト: [64, 64]）
        frame_size: 入力フレームサイズ（デフォルト: 128）
    """

    def __init__(
        self,
        seq_len: int = 4,
        feature_dim: int = 64,
        hidden_dims: Optional[List[int]] = None,
        frame_size: int = 128,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.frame_size = frame_size

        if hidden_dims is None:
            hidden_dims = [64, 64]

        # Spatial Encoder: フレーム → 空間特徴
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, feature_dim, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
        )
        # 128x128 → 64x64 → 32x32

        # Temporal Encoder: ConvLSTM
        self.temporal_encoder = ConvLSTM(
            in_channels=feature_dim,
            hidden_channels=hidden_dims,
            kernel_size=3,
        )

        # Decoder: 隠れ状態 → 次フレーム
        # まず spatial_encoder の縮小を逆転してから FrameDecoder
        self.spatial_decoder = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dims[-1], feature_dim, 4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                feature_dim, 32, 4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.frame_decoder = FrameDecoder(32, out_channels=3)

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, 3, H, W) RGB frames, normalized to [0, 1]

        Returns:
            predicted: (B, 3, H, W) predicted next frame
        """
        b, t, c, h, w = x.shape

        # Spatial encoding: 各フレームに CNN を適用
        # (B*T, 3, H, W) → (B*T, feature_dim, H/4, W/4)
        frames_flat = x.reshape(b * t, c, h, w)
        features_flat = self.spatial_encoder(frames_flat)
        _, cf, hf, wf = features_flat.shape
        features = features_flat.reshape(b, t, cf, hf, wf)

        # Temporal encoding: ConvLSTM
        last_h, _ = self.temporal_encoder(features)

        # Spatial decoding
        upsampled = self.spatial_decoder(last_h)

        # Frame reconstruction
        predicted = self.frame_decoder(upsampled)

        return predicted

    def compute_anomaly_score(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        異常スコアを計算する。

        Args:
            x: (B, T, 3, H, W) 入力シーケンス
            target: (B, 3, H, W) 実際の次フレーム

        Returns:
            scores: (B,) フレームごとの異常スコア (MSE)
        """
        predicted = self.forward(x)
        # ピクセルごとの二乗誤差を空間平均
        mse = F.mse_loss(predicted, target, reduction="none")
        scores = mse.mean(dim=(1, 2, 3))  # (B,)
        return scores


# ---------------------------------------------------------------------------
# フレーム前処理
# ---------------------------------------------------------------------------

ANOMALY_FRAME_SIZE = 128

# ImageNet normalization (SelfSupSurg ResNet50 用)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def preprocess_frame_anomaly(
    frame_bgr: np.ndarray,
    size: int = ANOMALY_FRAME_SIZE,
) -> torch.Tensor:
    """
    BGR フレームを anomaly detection モデル用に前処理する。

    Args:
        frame_bgr: (H, W, 3) BGR uint8
        size: ターゲットサイズ

    Returns:
        (3, size, size) tensor, normalized to [0, 1]
    """
    import cv2

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
    return tensor


def preprocess_frame_resnet(
    frame_bgr: np.ndarray,
    size: int = 224,
) -> torch.Tensor:
    """
    BGR フレームを ResNet50 (ImageNet正規化) 用に前処理する。

    Args:
        frame_bgr: (H, W, 3) BGR uint8
        size: ターゲットサイズ（デフォルト: 224）

    Returns:
        (3, size, size) tensor, ImageNet正規化済み
    """
    import cv2

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0

    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor


# ---------------------------------------------------------------------------
# SelfSupSurg Feature Extractor (DINO pretrained on Cholec80)
# ---------------------------------------------------------------------------

# SelfSupSurg 重みの公開URL
SELFSUP_WEIGHTS_URLS = {
    "dino": "https://s3.unistra.fr/camma_public/github/selfsupsurg/models/model_final_checkpoint_dino_surg.torch",
    "moco_v2": "https://s3.unistra.fr/camma_public/github/selfsupsurg/models/model_final_checkpoint_moco_v2_surg.torch",
    "simclr": "https://s3.unistra.fr/camma_public/github/selfsupsurg/models/model_final_checkpoint_simclr_surg.torch",
    "swav": "https://s3.unistra.fr/camma_public/github/selfsupsurg/models/model_final_checkpoint_swav_surg.torch",
}


def _load_selfsup_resnet50(
    weights_path: str,
    device: torch.device,
) -> nn.Module:
    """
    SelfSupSurg (VISSL形式) の重みを標準 PyTorch ResNet50 に読み込む。

    VISSL チェックポイントの構造:
      ckpt["classy_state_dict"]["base_model"]["model"]["trunk"]
    キー名: "_feature_blocks.conv1.weight" → "conv1.weight"

    Args:
        weights_path: .torch ファイルパス
        device: 計算デバイス

    Returns:
        ResNet50 モデル（fc = Identity, eval mode）
    """
    from torchvision.models import resnet50

    ckpt = torch.load(weights_path, map_location=device)

    # VISSL形式からtrunk重みを抽出
    trunk = ckpt["classy_state_dict"]["base_model"]["model"]["trunk"]

    # "_feature_blocks." プレフィックスを除去
    state_dict = {}
    for k, v in trunk.items():
        new_key = k.replace("_feature_blocks.", "")
        state_dict[new_key] = v

    # ResNet50 に読み込み（fc層は除外）
    model = resnet50(weights=None)
    model.load_state_dict(state_dict, strict=False)

    # fc を Identity に置換（特徴抽出用）
    model.fc = nn.Identity()
    model.to(device)
    model.eval()

    return model


class SelfSupSurgExtractor:
    """
    SelfSupSurg (DINO on Cholec80) による特徴抽出器。

    学習不要で手術動画に特化した 2048次元特徴ベクトルを抽出し、
    フレーム間の特徴距離で異常スコアを計算する。

    異常検出の原理:
      - 正常フレーム間: 特徴ベクトルが近い → cosine距離が小さい
      - 異常フレーム: 特徴ベクトルが急変 → cosine距離が大きい

    使い方:
      extractor = SelfSupSurgExtractor(device="cuda", method="dino")
      feat = extractor.extract(frame_bgr)
      score = extractor.compute_distance(prev_feat, feat)
    """

    def __init__(
        self,
        device: str = "cuda",
        weights_path: Optional[str] = None,
        method: str = "dino",
        auto_download: bool = True,
    ):
        """
        Args:
            device: 計算デバイス
            weights_path: 重みファイルパス（None で自動ダウンロード）
            method: SSL手法 ("dino", "moco_v2", "simclr", "swav")
            auto_download: 重みがなければ自動ダウンロードするか
        """
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
                print(f"SelfSupSurg ({method}) loaded: {resolved_path}")
            except Exception as e:
                print(f"SelfSupSurg load failed: {e}", file=sys.stderr)

    @staticmethod
    def _resolve_weights(
        weights_path: Optional[str],
        method: str,
        auto_download: bool,
    ) -> Optional[str]:
        """重みファイルのパスを解決する（必要に応じてダウンロード）。"""
        if weights_path and Path(weights_path).exists():
            return weights_path

        # デフォルトキャッシュパス
        cache_dir = Path.home() / "AI" / "huggingface" / "selfsupsurg"
        filename = f"model_final_checkpoint_{method}_surg.torch"
        cached_path = cache_dir / filename

        if cached_path.exists():
            return str(cached_path)

        # torch hub のキャッシュも確認（TORCH_HOME 環境変数で上書き可能）
        torch_home = Path(os.environ.get("TORCH_HOME", Path.home() / ".cache" / "torch"))
        torch_cache = torch_home / "hub" / "checkpoints" / filename
        if torch_cache.exists():
            return str(torch_cache)

        if not auto_download:
            print(f"SelfSupSurg weights not found: {cached_path}", file=sys.stderr)
            return None

        # 自動ダウンロード
        url = SELFSUP_WEIGHTS_URLS.get(method)
        if not url:
            print(f"Unknown method: {method}", file=sys.stderr)
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

    @torch.no_grad()
    def extract(self, frame_bgr: np.ndarray) -> torch.Tensor:
        """
        フレームから 2048次元特徴ベクトルを抽出する。

        Args:
            frame_bgr: (H, W, 3) BGR uint8

        Returns:
            (2048,) 特徴ベクトル (CPU tensor)
        """
        tensor = preprocess_frame_resnet(frame_bgr).unsqueeze(0).to(self.device)
        with torch.amp.autocast(self.device.type, enabled=self.device.type == "cuda"):
            feat = self.model(tensor)
        return feat.squeeze(0).cpu()

    @staticmethod
    def compute_distance(
        feat_prev: torch.Tensor,
        feat_curr: torch.Tensor,
    ) -> float:
        """
        2つの特徴ベクトル間の cosine 距離を計算する。

        Returns:
            float: 0.0 (同一) 〜 2.0 (正反対)
        """
        cos_sim = F.cosine_similarity(
            feat_prev.unsqueeze(0), feat_curr.unsqueeze(0)
        )
        return float(1.0 - cos_sim.item())

    @staticmethod
    def compute_mahalanobis_score(
        feat: torch.Tensor,
        baseline_feats: List[torch.Tensor],
    ) -> float:
        """
        ベースライン特徴群からのマハラノビス的距離を計算する。

        簡易版: ベースラインの平均ベクトルとの cosine 距離。
        ベースラインが多いほど安定する。

        Args:
            feat: (2048,) 現フレームの特徴ベクトル
            baseline_feats: ベースライン特徴ベクトルのリスト

        Returns:
            float: 異常スコア
        """
        if not baseline_feats:
            return 0.0
        baseline = torch.stack(baseline_feats).mean(dim=0)
        return SelfSupSurgExtractor.compute_distance(baseline, feat)


# ---------------------------------------------------------------------------
# モデルマネージャ（統合）
# ---------------------------------------------------------------------------


class AnomalyModelManager:
    """
    異常検出モデルの管理クラス。

    3つのモードを提供:
      1. "convlstm": ConvLSTM Autoencoder（自前学習済み重み必要）
      2. "selfsup":  SelfSupSurg 特徴距離（学習不要、推奨）
      3. "fallback": ピクセル差分（モデルなし）

    mode の自動選択:
      - selfsup_weights_path 指定 or auto_download=True → "selfsup"
      - convlstm_weights_path 指定 → "convlstm"
      - どちらもなし → "fallback"
    """

    def __init__(
        self,
        device: str = "cuda",
        weights_path: Optional[str] = None,
        seq_len: int = 4,
        feature_dim: int = 64,
        hidden_dims: Optional[List[int]] = None,
        frame_size: int = ANOMALY_FRAME_SIZE,
        selfsup_weights_path: Optional[str] = None,
        selfsup_method: str = "dino",
        auto_download_selfsup: bool = False,
    ):
        """
        Args:
            device: 計算デバイス
            weights_path: ConvLSTM Autoencoder の重みパス
            seq_len: ConvLSTM の入力シーケンス長
            feature_dim: ConvLSTM の空間特徴次元
            hidden_dims: ConvLSTM の隠れ次元リスト
            frame_size: ConvLSTM の入力フレームサイズ
            selfsup_weights_path: SelfSupSurg 重みパス
            selfsup_method: SSL手法 ("dino", "moco_v2", "simclr", "swav")
            auto_download_selfsup: SelfSupSurg 重みの自動ダウンロード
        """
        self.device = torch.device(
            device if torch.cuda.is_available() and device != "cpu" else "cpu"
        )
        self.seq_len = seq_len
        self.frame_size = frame_size
        self.model: Optional[ConvLSTMAutoencoder] = None
        self._convlstm_ready = False
        self.extractor: Optional[SelfSupSurgExtractor] = None
        self._selfsup_ready = False
        self._prev_feat: Optional[torch.Tensor] = None

        if hidden_dims is None:
            hidden_dims = [64, 64]

        # SelfSupSurg を試行
        if selfsup_weights_path or auto_download_selfsup:
            self.extractor = SelfSupSurgExtractor(
                device=device,
                weights_path=selfsup_weights_path,
                method=selfsup_method,
                auto_download=auto_download_selfsup,
            )
            self._selfsup_ready = self.extractor.is_ready

        # ConvLSTM を試行
        if weights_path and Path(weights_path).exists():
            try:
                self.model = ConvLSTMAutoencoder(
                    seq_len=seq_len,
                    feature_dim=feature_dim,
                    hidden_dims=hidden_dims,
                    frame_size=frame_size,
                )
                state = torch.load(weights_path, map_location=self.device)
                self.model.load_state_dict(state)
                self.model.to(self.device)
                self.model.eval()
                self._convlstm_ready = True
                print(f"ConvLSTM weights loaded: {weights_path}")
            except Exception as e:
                print(
                    f"ConvLSTM load failed: {e}",
                    file=sys.stderr,
                )

        # モード表示
        if self._selfsup_ready:
            print(f"Anomaly mode: SelfSupSurg ({selfsup_method})")
        elif self._convlstm_ready:
            print("Anomaly mode: ConvLSTM Autoencoder")
        else:
            print("Anomaly mode: pixel-diff fallback", file=sys.stderr)

    @property
    def mode(self) -> str:
        """現在の動作モードを返す。"""
        if self._selfsup_ready:
            return "selfsup"
        if self._convlstm_ready:
            return "convlstm"
        return "fallback"

    @property
    def is_ready(self) -> bool:
        """ConvLSTM または SelfSupSurg のいずれかが使用可能か。"""
        return self._convlstm_ready or self._selfsup_ready

    def compute_selfsup_score(
        self,
        frame_bgr: np.ndarray,
    ) -> float:
        """
        SelfSupSurg 特徴距離で異常スコアを計算する。

        前フレームとの cosine 距離を返す。
        初回フレームは 0.0 を返す。

        Args:
            frame_bgr: (H, W, 3) BGR uint8

        Returns:
            float: 異常スコア（cosine距離, 0.0〜2.0）
        """
        if not self._selfsup_ready:
            return 0.0

        feat = self.extractor.extract(frame_bgr)

        if self._prev_feat is None:
            self._prev_feat = feat
            return 0.0

        score = SelfSupSurgExtractor.compute_distance(self._prev_feat, feat)
        self._prev_feat = feat
        return score

    def reset_state(self) -> None:
        """内部状態（前フレーム特徴）をリセットする。"""
        self._prev_feat = None

    @torch.no_grad()
    def predict_and_score(
        self,
        frames: List[torch.Tensor],
        target: torch.Tensor,
    ) -> Tuple[float, Optional[torch.Tensor]]:
        """
        ConvLSTM でフレームシーケンスから次フレームを予測し、異常スコアを返す。

        SelfSupSurg モードの場合は使用しない（compute_selfsup_score を使う）。

        Args:
            frames: seq_len 個の (3, H, W) テンソルリスト
            target: (3, H, W) 実際の次フレーム

        Returns:
            (anomaly_score, predicted_frame_or_None)
        """
        if not self._convlstm_ready:
            return self._pixel_diff_fallback(frames, target), None

        # (1, T, 3, H, W)
        seq = torch.stack(frames).unsqueeze(0).to(self.device)
        tgt = target.unsqueeze(0).to(self.device)

        with torch.amp.autocast(self.device.type, enabled=self.device.type == "cuda"):
            score = self.model.compute_anomaly_score(seq, tgt)
            predicted = self.model(seq)

        return float(score.cpu().item()), predicted.squeeze(0).cpu()

    @staticmethod
    def _pixel_diff_fallback(
        frames: List[torch.Tensor],
        target: torch.Tensor,
    ) -> float:
        """
        学習済みモデルなし時のフォールバック。

        直前フレームとの差分の二乗平均を異常スコアとする。
        """
        if not frames:
            return 0.0
        prev = frames[-1]
        diff = (target - prev) ** 2
        return float(diff.mean().item())
