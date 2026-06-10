"""
Self-Supervised Video Anomaly Detection のテスト

テスト対象:
  1. ConvLSTM モデルの forward / anomaly score 計算
  2. 色変化スコア計算
  3. CSV往復テスト（record → read）
  4. アノテーションテスト（合成CSVからイベント抽出）
  5. SRT形式テスト
"""

import csv
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from src.anomaly.models import (
    ConvLSTMCell,
    ConvLSTM,
    ConvLSTMAutoencoder,
    FrameDecoder,
    preprocess_frame_anomaly,
    preprocess_frame_resnet,
    AnomalyModelManager,
    SelfSupSurgExtractor,
    ANOMALY_FRAME_SIZE,
    SELFSUP_WEIGHTS_URLS,
)
from src.anomaly.detector import (
    compute_color_change_score,
    read_anomalylog_csv,
    annotate_anomaly,
)
from src.core.time_utils import format_srt_time


# ---------------------------------------------------------------------------
# ConvLSTM Cell / Module テスト
# ---------------------------------------------------------------------------


class TestConvLSTMCell:
    def test_forward_shape(self):
        cell = ConvLSTMCell(in_channels=3, hidden_channels=16, kernel_size=3)
        x = torch.randn(2, 3, 32, 32)
        h, (h_out, c_out) = cell(x)
        assert h.shape == (2, 16, 32, 32)
        assert h_out.shape == (2, 16, 32, 32)
        assert c_out.shape == (2, 16, 32, 32)

    def test_forward_with_state(self):
        cell = ConvLSTMCell(in_channels=3, hidden_channels=16)
        x = torch.randn(2, 3, 32, 32)
        state = (
            torch.randn(2, 16, 32, 32),
            torch.randn(2, 16, 32, 32),
        )
        h, (h_out, c_out) = cell(x, state)
        assert h.shape == (2, 16, 32, 32)


class TestConvLSTM:
    def test_forward_shape(self):
        model = ConvLSTM(in_channels=3, hidden_channels=[16, 32])
        x = torch.randn(2, 4, 3, 32, 32)  # B=2, T=4
        last_h, states = model(x)
        assert last_h.shape == (2, 32, 32, 32)
        assert len(states) == 2

    def test_single_layer(self):
        model = ConvLSTM(in_channels=3, hidden_channels=[8])
        x = torch.randn(1, 3, 3, 16, 16)
        last_h, states = model(x)
        assert last_h.shape == (1, 8, 16, 16)


# ---------------------------------------------------------------------------
# ConvLSTM Autoencoder テスト
# ---------------------------------------------------------------------------


class TestConvLSTMAutoencoder:
    def test_forward_shape(self):
        model = ConvLSTMAutoencoder(
            seq_len=4,
            feature_dim=32,
            hidden_dims=[32, 32],
            frame_size=128,
        )
        x = torch.randn(2, 4, 3, 128, 128)
        out = model(x)
        assert out.shape == (2, 3, 128, 128)

    def test_output_range(self):
        """出力は sigmoid で [0, 1] 範囲になるはず"""
        model = ConvLSTMAutoencoder(
            seq_len=2,
            feature_dim=16,
            hidden_dims=[16],
            frame_size=64,
        )
        model.eval()
        x = torch.rand(1, 2, 3, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_anomaly_score(self):
        model = ConvLSTMAutoencoder(
            seq_len=2,
            feature_dim=16,
            hidden_dims=[16],
            frame_size=64,
        )
        model.eval()
        x = torch.rand(2, 2, 3, 64, 64)
        target = torch.rand(2, 3, 64, 64)
        with torch.no_grad():
            scores = model.compute_anomaly_score(x, target)
        assert scores.shape == (2,)
        assert (scores >= 0).all()


# ---------------------------------------------------------------------------
# フレーム前処理テスト
# ---------------------------------------------------------------------------


class TestPreprocess:
    def test_preprocess_frame(self):
        bgr = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        tensor = preprocess_frame_anomaly(bgr, size=128)
        assert tensor.shape == (3, 128, 128)
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0

    def test_preprocess_small_frame(self):
        bgr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        tensor = preprocess_frame_anomaly(bgr, size=128)
        assert tensor.shape == (3, 128, 128)


# ---------------------------------------------------------------------------
# 色変化スコアテスト
# ---------------------------------------------------------------------------


class TestColorChange:
    def test_identical_frames(self):
        """同一フレームの場合は変化量 0"""
        bgr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        score = compute_color_change_score(bgr, bgr)
        assert score == 0.0

    def test_different_frames(self):
        """大きく異なるフレームは正のスコア"""
        black = np.zeros((100, 100, 3), dtype=np.uint8)
        red = np.zeros((100, 100, 3), dtype=np.uint8)
        red[:, :] = [0, 0, 255]  # BGR で赤
        score = compute_color_change_score(black, red)
        assert score > 0.0

    def test_with_roi(self):
        """ROIマスク適用"""
        bgr1 = np.zeros((100, 100, 3), dtype=np.uint8)
        bgr2 = np.zeros((100, 100, 3), dtype=np.uint8)
        bgr2[:, :] = [0, 0, 255]
        roi = np.zeros((100, 100), dtype=bool)
        roi[25:75, 25:75] = True  # 中央のみ
        score = compute_color_change_score(bgr1, bgr2, roi)
        assert score > 0.0


# ---------------------------------------------------------------------------
# AnomalyModelManager (pixel-diff fallback) テスト
# ---------------------------------------------------------------------------


class TestAnomalyModelManager:
    def test_fallback_mode(self):
        """重みなしでフォールバックモードになること"""
        mgr = AnomalyModelManager(device="cpu", weights_path=None)
        assert not mgr.is_ready

    def test_fallback_score(self):
        """フォールバックでもスコアが計算できること"""
        mgr = AnomalyModelManager(device="cpu", weights_path=None)
        frames = [torch.rand(3, 64, 64) for _ in range(4)]
        target = torch.rand(3, 64, 64)
        score, predicted = mgr.predict_and_score(frames, target)
        assert isinstance(score, float)
        assert score >= 0.0
        assert predicted is None  # フォールバックでは予測フレームなし

    def test_fallback_identical_score_zero(self):
        """同一フレームの場合、フォールバックスコアは 0"""
        mgr = AnomalyModelManager(device="cpu", weights_path=None)
        frame = torch.rand(3, 64, 64)
        frames = [frame.clone() for _ in range(4)]
        target = frame.clone()
        score, _ = mgr.predict_and_score(frames, target)
        assert score == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# CSV往復テスト
# ---------------------------------------------------------------------------


class TestCSVRoundTrip:
    def test_write_and_read(self, tmp_path):
        """合成CSVを書き出して正しく読み込めることを確認"""
        csv_path = tmp_path / "test_anomalylog.csv"
        n = 20
        fps = 5.0
        dt = 1.0 / fps

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "t_sec", "t_srt",
                "anomaly_score", "color_change", "combined_score",
                "smooth_anomaly", "smooth_color", "smooth_combined",
                "reader",
            ])
            for i in range(n):
                t = i * dt
                writer.writerow([
                    f"{t:.3f}",
                    format_srt_time(t),
                    f"{i * 0.001:.6f}",
                    f"{i * 0.002:.6f}",
                    f"{i * 0.003:.6f}",
                    f"{i * 0.001:.6f}",
                    f"{i * 0.002:.6f}",
                    f"{i * 0.003:.6f}",
                    "pyav",
                ])

        data = read_anomalylog_csv(str(csv_path))
        assert len(data["times"]) == n
        assert len(data["anomaly_scores"]) == n
        assert len(data["color_changes"]) == n
        assert len(data["combined_scores"]) == n
        assert data["reader"] == "pyav"
        assert data["fps"] == pytest.approx(fps, abs=0.1)


# ---------------------------------------------------------------------------
# アノテーションテスト
# ---------------------------------------------------------------------------


class TestAnnotation:
    def _write_test_csv(self, csv_path: Path, n: int = 100, fps: float = 5.0):
        """テスト用CSVを生成。50〜70フレーム目に異常ピークを挿入。"""
        dt = 1.0 / fps
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "t_sec", "t_srt",
                "anomaly_score", "color_change", "combined_score",
                "smooth_anomaly", "smooth_color", "smooth_combined",
                "reader",
            ])
            for i in range(n):
                t = i * dt
                # 50〜70 の区間で高スコア
                if 50 <= i < 70:
                    score = 0.05
                else:
                    score = 0.001
                writer.writerow([
                    f"{t:.3f}",
                    format_srt_time(t),
                    f"{score:.6f}",
                    f"{score:.6f}",
                    f"{score:.6f}",
                    f"{score:.6f}",
                    f"{score:.6f}",
                    f"{score:.6f}",
                    "pyav",
                ])

    def test_event_extraction(self, tmp_path):
        """閾値を超える区間がイベントとして抽出されること"""
        csv_path = tmp_path / "test_anomalylog.csv"
        self._write_test_csv(csv_path)

        result = annotate_anomaly(
            csv_path=str(csv_path),
            outdir=str(tmp_path),
            thr=0.01,
            min_duration_s=1.0,
            smooth_s=5.0,
        )

        assert result["events"] >= 1
        assert Path(result["jsonl"]).exists()
        assert Path(result["srt"]).exists()

    def test_no_events_below_threshold(self, tmp_path):
        """全フレームが閾値以下の場合、イベント数が 0"""
        csv_path = tmp_path / "test_anomalylog.csv"
        n = 50
        dt = 0.2
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "t_sec", "t_srt",
                "anomaly_score", "color_change", "combined_score",
                "smooth_anomaly", "smooth_color", "smooth_combined",
                "reader",
            ])
            for i in range(n):
                t = i * dt
                writer.writerow([
                    f"{t:.3f}",
                    format_srt_time(t),
                    "0.001000", "0.001000", "0.001000",
                    "0.001000", "0.001000", "0.001000",
                    "pyav",
                ])

        result = annotate_anomaly(
            csv_path=str(csv_path),
            outdir=str(tmp_path),
            thr=0.01,
            min_duration_s=1.0,
        )
        assert result["events"] == 0

    def test_jsonl_format(self, tmp_path):
        """JSONL出力が正しいフォーマットであること"""
        csv_path = tmp_path / "test_anomalylog.csv"
        self._write_test_csv(csv_path)

        result = annotate_anomaly(
            csv_path=str(csv_path),
            outdir=str(tmp_path),
            thr=0.01,
        )

        if result["events"] > 0:
            with open(result["jsonl"], "r", encoding="utf-8") as f:
                for line in f:
                    ev = json.loads(line.strip())
                    assert "type" in ev
                    assert "start_sec" in ev
                    assert "end_sec" in ev
                    assert "start_srt" in ev
                    assert "end_srt" in ev
                    assert ev["type"] == "anomaly_candidate"
                    assert ev["start_sec"] < ev["end_sec"]

    def test_srt_format(self, tmp_path):
        """SRT出力が正しいフォーマットであること"""
        csv_path = tmp_path / "test_anomalylog.csv"
        self._write_test_csv(csv_path)

        result = annotate_anomaly(
            csv_path=str(csv_path),
            outdir=str(tmp_path),
            thr=0.01,
        )

        if result["events"] > 0:
            srt_text = Path(result["srt"]).read_text(encoding="utf-8")
            # SRT は空行区切りブロック
            blocks = [b.strip() for b in srt_text.strip().split("\n\n") if b.strip()]
            for block in blocks:
                lines = block.split("\n")
                assert len(lines) >= 3
                # 1行目: 連番
                assert lines[0].strip().isdigit()
                # 2行目: 時刻
                assert "-->" in lines[1]
                # 3行目: タグ行
                assert lines[2].startswith("[anomaly]")

    def test_max_duration_filter(self, tmp_path):
        """max_duration_s で長すぎるイベントが除外されること"""
        csv_path = tmp_path / "test_anomalylog.csv"
        # 100フレーム分（20秒間）ずっと異常スコアが高い
        n = 100
        dt = 0.2
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "t_sec", "t_srt",
                "anomaly_score", "color_change", "combined_score",
                "smooth_anomaly", "smooth_color", "smooth_combined",
                "reader",
            ])
            for i in range(n):
                t = i * dt
                writer.writerow([
                    f"{t:.3f}",
                    format_srt_time(t),
                    "0.050000", "0.050000", "0.050000",
                    "0.050000", "0.050000", "0.050000",
                    "pyav",
                ])

        # max_duration_s=10 で 19.8秒のイベントは除外される
        result = annotate_anomaly(
            csv_path=str(csv_path),
            outdir=str(tmp_path),
            thr=0.01,
            min_duration_s=1.0,
            max_duration_s=10.0,
        )
        assert result["events"] == 0


# ---------------------------------------------------------------------------
# タグテンプレート登録テスト
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# ResNet 前処理テスト
# ---------------------------------------------------------------------------


class TestPreprocessResnet:
    def test_preprocess_frame_resnet(self):
        bgr = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        tensor = preprocess_frame_resnet(bgr, size=224)
        assert tensor.shape == (3, 224, 224)
        # ImageNet正規化後は [0,1] の範囲外になる
        assert tensor.min() < 0.0 or tensor.max() > 1.0


# ---------------------------------------------------------------------------
# SelfSupSurg テスト
# ---------------------------------------------------------------------------


class TestSelfSupSurgExtractor:
    def _get_extractor(self):
        """重みがダウンロード済みであればExtractorを返す。なければスキップ。"""
        extractor = SelfSupSurgExtractor(
            device="cpu",
            method="dino",
            auto_download=False,
        )
        if not extractor.is_ready:
            pytest.skip("SelfSupSurg DINO weights not available")
        return extractor

    def test_extract_feature_shape(self):
        extractor = self._get_extractor()
        bgr = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        feat = extractor.extract(bgr)
        assert feat.shape == (2048,)

    def test_identical_frames_low_distance(self):
        extractor = self._get_extractor()
        bgr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        feat1 = extractor.extract(bgr)
        feat2 = extractor.extract(bgr)
        dist = SelfSupSurgExtractor.compute_distance(feat1, feat2)
        assert dist == pytest.approx(0.0, abs=1e-5)

    def test_different_frames_positive_distance(self):
        extractor = self._get_extractor()
        black = np.zeros((224, 224, 3), dtype=np.uint8)
        white = np.ones((224, 224, 3), dtype=np.uint8) * 255
        feat1 = extractor.extract(black)
        feat2 = extractor.extract(white)
        dist = SelfSupSurgExtractor.compute_distance(feat1, feat2)
        assert dist > 0.0

    def test_weights_url_format(self):
        """公開URLが正しい形式であること"""
        for method, url in SELFSUP_WEIGHTS_URLS.items():
            assert url.startswith("https://")
            assert method in url


class TestAnomalyModelManagerSelfsup:
    def test_selfsup_mode(self):
        """SelfSupSurg 重みをキャッシュから解決できればモードが selfsup になること。

        キャッシュが無い環境では何も検証できないため pytest.skip する
        （キャッシュ有無に依存して暗黙にトリビアルパスを通る挙動を排除）。
        """
        resolved = SelfSupSurgExtractor._resolve_weights(
            None, "dino", auto_download=False
        )
        if not resolved:
            pytest.skip("SelfSupSurg weights not cached; selfsup mode untestable")

        mgr = AnomalyModelManager(
            device="cpu",
            selfsup_weights_path=resolved,
            selfsup_method="dino",
            auto_download_selfsup=False,
        )
        assert mgr.mode == "selfsup"
        assert mgr.is_ready

    def test_fallback_without_any_weights(self):
        """重みを一切渡さなければ決定論的に fallback モードになること。

        selfsup は selfsup_weights/auto_download を渡したときのみ試行されるため、
        ここでは両方未指定 → 必ず fallback。キャッシュ状態には依存しない。
        """
        mgr = AnomalyModelManager(
            device="cpu",
            weights_path=None,
            selfsup_weights_path=None,
            auto_download_selfsup=False,
        )
        assert mgr.mode == "fallback"
        assert not mgr.is_ready


# ---------------------------------------------------------------------------
# タグテンプレート登録テスト
# ---------------------------------------------------------------------------


class TestTagRegistration:
    def test_jsonl_to_srt_template(self):
        from src.tools.jsonl_to_srt import TAG_TEMPLATES
        assert "anomaly_candidate" in TAG_TEMPLATES
        assert TAG_TEMPLATES["anomaly_candidate"] == "[anomaly] anomaly_detected"

    def test_srt_to_jsonl_pattern(self):
        from src.tools.srt_to_jsonl import TAG_PATTERNS, _parse_tag_line
        result = _parse_tag_line("[anomaly] anomaly_detected")
        assert result == "anomaly_candidate"
