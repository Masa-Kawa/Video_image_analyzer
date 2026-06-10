"""
Deep-learning models for bleeding detection in laparoscopic surgery videos.

Provides two models:
  1. BleedClassifier  - ResNet-18 based frame-level bleeding probability
  2. BleedSegmenter   - Lightweight U-Net for blood region segmentation

Both models offer HSV-based fallback when trained weights are unavailable.
Mixed precision (fp16) is used by default for GPU efficiency.
"""

import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# HSV-based blood detection (fallback / baseline)
# ---------------------------------------------------------------------------

_RED_RANGES_HSV = [
    # (H_lo, H_hi, S_min, V_min)
    (0, 10, 50, 40),
    (170, 179, 50, 40),
    # Dark red / venous blood
    (0, 15, 30, 20),
    (160, 179, 30, 20),
]


def hsv_blood_mask(
    frame_bgr: np.ndarray,
    roi_mask: Optional[np.ndarray] = None,
    s_min: int = 50,
    v_min: int = 40,
) -> np.ndarray:
    """
    Enhanced HSV-based blood mask with morphological refinement.

    Returns:
        Binary mask (uint8, 0 or 255) of blood-like regions.

    Raises:
        ValueError: frame_bgr / roi_mask が不正（None・非ndarray・空・形状不正）。
    """
    # OpenCV の C++ レイヤーに不正配列を渡すとセグフォや捕捉困難な例外で
    # パイプライン全体が落ちうるため、Python 側で防御的に検証する。
    if not isinstance(frame_bgr, np.ndarray):
        raise ValueError(f"frame_bgr must be np.ndarray, got {type(frame_bgr)}")
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError(
            f"frame_bgr must be (H, W, 3) BGR, got shape {frame_bgr.shape}"
        )
    if frame_bgr.size == 0 or frame_bgr.shape[0] == 0 or frame_bgr.shape[1] == 0:
        raise ValueError("frame_bgr is empty")
    if roi_mask is not None:
        if not isinstance(roi_mask, np.ndarray):
            raise ValueError(f"roi_mask must be np.ndarray, got {type(roi_mask)}")
        if roi_mask.shape[:2] != frame_bgr.shape[:2]:
            raise ValueError(
                f"roi_mask shape {roi_mask.shape[:2]} does not match "
                f"frame {frame_bgr.shape[:2]}"
            )

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    combined = np.zeros(hsv.shape[:2], dtype=np.uint8)

    for h_lo, h_hi, s_lo, v_lo in _RED_RANGES_HSV:
        mask = cv2.inRange(
            hsv,
            np.array([h_lo, max(s_lo, s_min), max(v_lo, v_min)]),
            np.array([h_hi, 255, 255]),
        )
        combined = combined | mask

    # LAB color space supplement: high a* AND high b* = red tones
    # a* axis: green(0) - neutral(128) - red(255)
    # b* axis: blue(0) - neutral(128) - yellow(255)
    # Red blood: a* > 140 AND b* > 128 (excludes blue which has high a* but low b*)
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    lab_red = ((lab[:, :, 1] > 140) & (lab[:, :, 2] > 128)).astype(np.uint8) * 255
    combined = combined | lab_red

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)

    if roi_mask is not None:
        combined = combined & (roi_mask.astype(np.uint8) * 255)

    return combined


def hsv_blood_probability(
    frame_bgr: np.ndarray,
    roi_mask: Optional[np.ndarray] = None,
) -> float:
    """HSV-based bleeding probability (area ratio of blood mask)."""
    mask = hsv_blood_mask(frame_bgr, roi_mask)
    if roi_mask is not None:
        total = int(np.count_nonzero(roi_mask))
    else:
        total = frame_bgr.shape[0] * frame_bgr.shape[1]
    if total == 0:
        return 0.0
    return float(np.count_nonzero(mask)) / total


def hsv_blood_area_and_source(
    frame_bgr: np.ndarray,
    roi_mask: Optional[np.ndarray] = None,
) -> dict:
    """
    Compute blood area ratio and source point from HSV mask.

    Returns:
        {
            "mask": np.ndarray (binary),
            "area_ratio": float,
            "source_x": float (0-1, normalized),
            "source_y": float (0-1, normalized),
        }
    """
    mask = hsv_blood_mask(frame_bgr, roi_mask)
    h, w = mask.shape

    if roi_mask is not None:
        total = int(np.count_nonzero(roi_mask))
    else:
        total = h * w

    blood_pixels = int(np.count_nonzero(mask))
    area_ratio = blood_pixels / total if total > 0 else 0.0

    # Source localization: center of mass of blood mask
    source_x, source_y = 0.5, 0.5
    if blood_pixels > 0:
        moments = cv2.moments(mask)
        if moments["m00"] > 0:
            source_x = moments["m10"] / moments["m00"] / w
            source_y = moments["m01"] / moments["m00"] / h

    return {
        "mask": mask,
        "area_ratio": area_ratio,
        "source_x": source_x,
        "source_y": source_y,
    }


# ---------------------------------------------------------------------------
# U-Net (lightweight)
# ---------------------------------------------------------------------------

class _ConvBlock(nn.Module):
    """Double convolution block: Conv-BN-ReLU x2"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BleedSegmenter(nn.Module):
    """
    Lightweight U-Net for blood region segmentation.

    Architecture: 4-level encoder-decoder with skip connections.
    Channels: 32-64-128-256 (fits comfortably in 12GB VRAM).
    Input: (B, 3, H, W) RGB normalized to [0, 1].
    Output: (B, 1, H, W) sigmoid probability map.
    """

    def __init__(self, channels: Tuple[int, ...] = (32, 64, 128, 256)):
        super().__init__()
        ch = channels

        # Encoder
        self.enc1 = _ConvBlock(3, ch[0])
        self.enc2 = _ConvBlock(ch[0], ch[1])
        self.enc3 = _ConvBlock(ch[1], ch[2])
        self.enc4 = _ConvBlock(ch[2], ch[3])
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = _ConvBlock(ch[3], ch[3])

        # Decoder
        self.up4 = nn.ConvTranspose2d(ch[3], ch[3], 2, stride=2)
        self.dec4 = _ConvBlock(ch[3] * 2, ch[3])
        self.up3 = nn.ConvTranspose2d(ch[3], ch[2], 2, stride=2)
        self.dec3 = _ConvBlock(ch[2] * 2, ch[2])
        self.up2 = nn.ConvTranspose2d(ch[2], ch[1], 2, stride=2)
        self.dec2 = _ConvBlock(ch[1] * 2, ch[1])
        self.up1 = nn.ConvTranspose2d(ch[1], ch[0], 2, stride=2)
        self.dec1 = _ConvBlock(ch[0] * 2, ch[0])

        self.head = nn.Conv2d(ch[0], 1, 1)

    @staticmethod
    def _cat_skip(up: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Align ``up`` to ``skip``'s spatial size, then concat on channels.

        ConvTranspose2d can produce a feature map a few pixels off from the
        encoder skip when the input H/W is not a multiple of 16 (e.g. 240,
        480). Resize defensively so ``torch.cat`` never raises a size mismatch
        RuntimeError; a no-op when sizes already agree.
        """
        if up.shape[-2:] != skip.shape[-2:]:
            up = F.interpolate(
                up, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
        return torch.cat([up, skip], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder with skip connections (sizes aligned for non-16-multiple inputs)
        d4 = self.dec4(self._cat_skip(self.up4(b), e4))
        d3 = self.dec3(self._cat_skip(self.up3(d4), e3))
        d2 = self.dec2(self._cat_skip(self.up2(d3), e2))
        d1 = self.dec1(self._cat_skip(self.up1(d2), e1))

        return torch.sigmoid(self.head(d1))


# ---------------------------------------------------------------------------
# ResNet-18 Classifier
# ---------------------------------------------------------------------------

class BleedClassifier(nn.Module):
    """
    ResNet-18 based bleeding frame classifier.

    Uses ImageNet-pretrained backbone for feature extraction.
    FC head: 512 → 1 (sigmoid).
    Without fine-tuning, combine with HSV heuristic for robust detection.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        try:
            from torchvision.models import resnet18, ResNet18_Weights
            if pretrained:
                self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
            else:
                self.backbone = resnet18(weights=None)
        except ImportError:
            from torchvision.models import resnet18
            self.backbone = resnet18(pretrained=pretrained)

        # Remove the original FC layer
        self.feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # Classification head
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 224, 224) normalized with ImageNet stats.
        Returns:
            (B,) bleeding probability (sigmoid applied).
        """
        features = self.backbone(x)
        logits = self.head(features).squeeze(-1)
        return torch.sigmoid(logits)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 512-dim feature vector (for downstream use)."""
        return self.backbone(x)


# ---------------------------------------------------------------------------
# Model Manager (loading / inference)
# ---------------------------------------------------------------------------

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def preprocess_frame(
    frame_bgr: np.ndarray,
    size: int = 224,
) -> torch.Tensor:
    """
    Preprocess BGR frame for model input.

    Args:
        frame_bgr: (H, W, 3) BGR uint8
        size: target spatial size

    Returns:
        (1, 3, size, size) normalized tensor
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0

    # ImageNet normalization
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    tensor = (tensor - mean) / std

    return tensor.unsqueeze(0)


def preprocess_frame_segmentation(
    frame_bgr: np.ndarray,
    size: int = 256,
) -> torch.Tensor:
    """
    Preprocess BGR frame for segmentation model.

    Args:
        frame_bgr: (H, W, 3) BGR uint8
        size: target spatial size (must be divisible by 16)

    Returns:
        (1, 3, size, size) tensor normalized to [0, 1]
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0)


class ModelManager:
    """
    Manages model loading and inference with HSV fallback.

    Usage:
        mgr = ModelManager(device="cuda", use_deep=True)
        prob = mgr.classify(frame_bgr, roi_mask)
        seg = mgr.segment(frame_bgr, roi_mask)
    """

    def __init__(
        self,
        device: str = "cuda",
        use_deep: bool = False,
        classifier_weights: Optional[str] = None,
        segmenter_weights: Optional[str] = None,
        seg_size: int = 256,
        deep_blend_weight: float = 0.4,
    ):
        # U-Net は4回 /2 ダウンサンプルするため seg_size は16の倍数必須。
        # 不正値は forward 内の torch.cat で初めて落ちるより、構築時に弾く。
        if seg_size <= 0 or seg_size % 16 != 0:
            raise ValueError(
                f"seg_size must be a positive multiple of 16, got {seg_size}"
            )
        # deep_prob と hsv_prob のブレンド比。未ファインチューニングの分類器は
        # ノイズが大きいため、0.0 にすれば HSV のみ（deep 無効）に倒せる。
        if not 0.0 <= deep_blend_weight <= 1.0:
            raise ValueError(
                f"deep_blend_weight must be in [0, 1], got {deep_blend_weight}"
            )
        self.device = torch.device(
            device if torch.cuda.is_available() and device != "cpu" else "cpu"
        )
        self.use_deep = use_deep
        self.seg_size = seg_size
        self.deep_blend_weight = deep_blend_weight
        self.classifier: Optional[BleedClassifier] = None
        self.segmenter: Optional[BleedSegmenter] = None
        self._classifier_ready = False
        self._segmenter_ready = False

        if use_deep:
            self._load_classifier(classifier_weights)
            self._load_segmenter(segmenter_weights)

    def _load_classifier(self, weights_path: Optional[str]) -> None:
        """Load classifier model."""
        try:
            self.classifier = BleedClassifier(pretrained=True)
            if weights_path and Path(weights_path).exists():
                # weights_only=True: 悪意ある .pth 経由の pickle 任意コード実行を防ぐ
                state = torch.load(weights_path, map_location=self.device,
                                   weights_only=True)
                self.classifier.load_state_dict(state)
                print(f"Classifier weights loaded: {weights_path}")
            else:
                print("Classifier: ImageNet pretrained (no surgical fine-tuning)",
                      file=sys.stderr)
            self.classifier.to(self.device)
            self.classifier.eval()
            self._classifier_ready = True
        except Exception as e:
            print(f"Classifier load failed, using HSV fallback: {e}",
                  file=sys.stderr)
            self._classifier_ready = False

    def _load_segmenter(self, weights_path: Optional[str]) -> None:
        """Load segmenter model."""
        if weights_path and Path(weights_path).exists():
            try:
                self.segmenter = BleedSegmenter()
                # weights_only=True: 悪意ある .pth 経由の pickle 任意コード実行を防ぐ
                state = torch.load(weights_path, map_location=self.device,
                                   weights_only=True)
                self.segmenter.load_state_dict(state)
                self.segmenter.to(self.device)
                self.segmenter.eval()
                self._segmenter_ready = True
                print(f"Segmenter weights loaded: {weights_path}")
            except Exception as e:
                print(f"Segmenter load failed, using HSV fallback: {e}",
                      file=sys.stderr)
                self._segmenter_ready = False
        else:
            print("Segmenter: no weights, using HSV mask fallback",
                  file=sys.stderr)
            self._segmenter_ready = False

    @torch.no_grad()
    def classify(
        self,
        frame_bgr: np.ndarray,
        roi_mask: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute bleeding probability for a frame.

        Returns:
            float: probability 0-1. If deep model available, blends
            deep prediction with HSV score. Otherwise, HSV only.
        """
        hsv_prob = hsv_blood_probability(frame_bgr, roi_mask)

        if not self._classifier_ready:
            return hsv_prob

        inp = preprocess_frame(frame_bgr, size=224).to(self.device)
        with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
            deep_prob = float(self.classifier(inp).cpu().item())

        # Blend: deep model prediction weighted with HSV evidence.
        # deep_prob と hsv_prob はスケールの異なる量（分類確率 vs マスク面積比）
        # のため、ブレンド比は呼び出し側で調整可能にしている。未ファインチューニング
        # 時は deep_blend_weight を下げる/0 にすることで過大・過小評価を抑制できる。
        w = self.deep_blend_weight
        return w * deep_prob + (1.0 - w) * hsv_prob

    @torch.no_grad()
    def segment(
        self,
        frame_bgr: np.ndarray,
        roi_mask: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Segment blood regions and compute area + source.

        Returns:
            {"mask": np.ndarray, "area_ratio": float,
             "source_x": float, "source_y": float}
        """
        if not self._segmenter_ready:
            return hsv_blood_area_and_source(frame_bgr, roi_mask)

        h, w = frame_bgr.shape[:2]
        inp = preprocess_frame_segmentation(
            frame_bgr, size=self.seg_size,
        ).to(self.device)

        with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
            prob_map = self.segmenter(inp)

        # Resize probability map back to original size
        prob_np = prob_map.squeeze().cpu().numpy()
        prob_resized = cv2.resize(prob_np, (w, h), interpolation=cv2.INTER_LINEAR)
        mask = (prob_resized > 0.5).astype(np.uint8) * 255

        if roi_mask is not None:
            mask = mask & (roi_mask.astype(np.uint8) * 255)
            total = int(np.count_nonzero(roi_mask))
        else:
            total = h * w

        blood_pixels = int(np.count_nonzero(mask))
        area_ratio = blood_pixels / total if total > 0 else 0.0

        source_x, source_y = 0.5, 0.5
        if blood_pixels > 0:
            moments = cv2.moments(mask)
            if moments["m00"] > 0:
                source_x = moments["m10"] / moments["m00"] / w
                source_y = moments["m01"] / moments["m00"] / h

        return {
            "mask": mask,
            "area_ratio": area_ratio,
            "source_x": source_x,
            "source_y": source_y,
        }
