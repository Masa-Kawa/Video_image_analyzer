"""
cavity_detector.py のユニットテスト

合成フレームを使い、腹腔内外判定ロジックを検証する。
"""

import csv
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.cavity.cavity_detector import (
    compute_vignette_score,
    compute_tissue_ratio,
    compute_brightness_bimodality,
    compute_hue_concentration,
    compute_cavity_score,
    read_cavitylog_csv,
    annotate_cavity,
)
from src.core.time_utils import format_srt_time


class TestVignetteScore(unittest.TestCase):
    """周辺暗域スコアのテスト"""

    def _make_vignette_frame(self, h=200, w=200):
        """腹腔鏡風ビネット画像: 中央が明るく周辺が暗い（NumPyベクトル化）"""
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        radius = min(h, w) / 2
        brightness = np.clip(200 * (1 - dist / radius), 0, 255).astype(np.uint8)
        return np.repeat(brightness[:, :, np.newaxis], 3, axis=2)

    def _make_bright_frame(self, h=200, w=200):
        """均一に明るい画像（腹腔外）"""
        return np.full((h, w, 3), 200, dtype=np.uint8)

    def test_vignette_high_score(self):
        """ビネット画像はスコアが高い"""
        frame = self._make_vignette_frame()
        score = compute_vignette_score(frame)
        self.assertGreater(score, 0.3)

    def test_bright_low_score(self):
        """均一明るい画像はスコアが低い"""
        frame = self._make_bright_frame()
        score = compute_vignette_score(frame)
        self.assertLess(score, 0.2)


class TestTissueRatio(unittest.TestCase):
    """組織色占有率のテスト"""

    def test_red_frame(self):
        """全面赤 → tissue_ratio ≈ 1.0"""
        h, w = 100, 100
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        hsv[:, :, 0] = 5    # 赤
        hsv[:, :, 1] = 200
        hsv[:, :, 2] = 200
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        ratio = compute_tissue_ratio(bgr)
        self.assertGreater(ratio, 0.8)

    def test_blue_frame(self):
        """全面青 → tissue_ratio ≈ 0.0"""
        h, w = 100, 100
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        hsv[:, :, 0] = 120  # 青
        hsv[:, :, 1] = 200
        hsv[:, :, 2] = 200
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        ratio = compute_tissue_ratio(bgr)
        self.assertLess(ratio, 0.1)

    def test_green_frame(self):
        """全面緑（ドレープ色）→ tissue_ratio ≈ 0.0"""
        h, w = 100, 100
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        hsv[:, :, 0] = 60   # 緑
        hsv[:, :, 1] = 200
        hsv[:, :, 2] = 200
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        ratio = compute_tissue_ratio(bgr)
        self.assertLess(ratio, 0.1)


class TestBrightnessBimodality(unittest.TestCase):
    """明るさ二峰性のテスト"""

    def test_bimodal(self):
        """上半分が暗く、下半分が明るい → bimodality > 0"""
        h, w = 100, 100
        gray = np.zeros((h, w), dtype=np.uint8)
        gray[:50, :] = 30    # 暗い
        gray[50:, :] = 180   # 明るい
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        score = compute_brightness_bimodality(bgr)
        self.assertGreater(score, 0.2)

    def test_uniform_bright(self):
        """均一明度 → bimodality ≈ 0"""
        bgr = np.full((100, 100, 3), 150, dtype=np.uint8)
        score = compute_brightness_bimodality(bgr)
        self.assertLess(score, 0.1)


class TestHueConcentration(unittest.TestCase):
    """色相集中度のテスト"""

    def test_single_hue(self):
        """単色 → 集中度が高い"""
        h, w = 100, 100
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        hsv[:, :, 0] = 10
        hsv[:, :, 1] = 200
        hsv[:, :, 2] = 200
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        conc = compute_hue_concentration(bgr)
        self.assertGreater(conc, 0.5)

    def test_diverse_hues(self):
        """多色 → 集中度が低い"""
        h, w = 100, 100
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        # 各行で異なる色相
        for row in range(h):
            hsv[row, :, 0] = int(row * 179 / h)
        hsv[:, :, 1] = 200
        hsv[:, :, 2] = 200
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        conc = compute_hue_concentration(bgr)
        self.assertLess(conc, 0.3)


class TestComputeCavityScore(unittest.TestCase):
    """統合スコアのテスト"""

    def test_inside_like_frame(self):
        """腹腔内風の合成画像: 中央が赤/茶、周辺が暗い"""
        h, w = 200, 200
        frame = np.zeros((h, w, 3), dtype=np.uint8)

        # 中央に赤色の円
        cv2.circle(frame, (100, 100), 70,
                   (30, 30, 180), -1)  # BGR赤
        # 周辺は暗い → vignette特徴

        inside = compute_cavity_score(frame)
        self.assertGreater(inside["cavity_score"], 0.0)
        self.assertGreater(inside["tissue_ratio"], 0.0)

        # 緩すぎる > 0.0 だけでは判別能力を保証できないため、outside風フレームの
        # cavity_score より明確に高いこと（相対比較）も検証する。
        outside = compute_cavity_score(self._make_outside_frame())
        self.assertGreater(inside["cavity_score"], outside["cavity_score"])

    @staticmethod
    def _make_outside_frame(h=200, w=200):
        """腹腔外風: 均一に明るい緑色ドレープ"""
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        hsv[:, :, 0] = 60
        hsv[:, :, 1] = 150
        hsv[:, :, 2] = 200
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def test_outside_like_frame(self):
        """腹腔外風の合成画像: 均一に明るく、青/緑が多い"""
        result = compute_cavity_score(self._make_outside_frame())
        self.assertLess(result["tissue_ratio"], 0.1)


class TestReadCavitylogCsv(unittest.TestCase):
    """CSV読み込みのテスト"""

    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test_cavitylog.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "t_sec", "t_srt",
                    "vignette_score", "tissue_ratio",
                    "brightness_bimodality", "hue_concentration",
                    "cavity_score", "smooth_cavity",
                    "reader",
                ])
                writer.writerow([
                    "0.000", "00:00:00,000",
                    "0.500000", "0.400000",
                    "0.300000", "0.600000",
                    "0.450000", "0.450000",
                    "pyav",
                ])
                writer.writerow([
                    "0.500", "00:00:00,500",
                    "0.550000", "0.420000",
                    "0.310000", "0.580000",
                    "0.465000", "0.457500",
                    "pyav",
                ])

            data = read_cavitylog_csv(str(csv_path))

            self.assertEqual(len(data["times"]), 2)
            self.assertAlmostEqual(data["times"][0], 0.0)
            self.assertAlmostEqual(data["vignette_scores"][0], 0.5)
            self.assertAlmostEqual(data["tissue_ratios"][1], 0.42)
            self.assertEqual(data["reader"], "pyav")
            self.assertAlmostEqual(data["fps"], 2.0, places=1)


class TestAnnotateCavity(unittest.TestCase):
    """区間検出のテスト"""

    def test_detect_inside_region(self):
        """閾値を超える連続区間がinsideとして検出されること"""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test_cavitylog.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "t_sec", "t_srt",
                    "vignette_score", "tissue_ratio",
                    "brightness_bimodality", "hue_concentration",
                    "cavity_score", "smooth_cavity",
                    "reader",
                ])
                # 10秒分のデータ（0.5秒間隔）
                for i in range(20):
                    t = i * 0.5
                    # 5〜15秒がinside（smooth_cavity > 0.35）
                    if 5.0 <= t <= 7.5:
                        smooth = 0.5
                    else:
                        smooth = 0.1
                    writer.writerow([
                        f"{t:.3f}", format_srt_time(t),
                        "0.0", "0.0", "0.0", "0.0",
                        f"{smooth:.6f}", f"{smooth:.6f}",
                        "pyav",
                    ])

            result = annotate_cavity(
                csv_path=str(csv_path),
                outdir=tmpdir,
                thr=0.35,
                min_duration_s=1.0,
            )

            self.assertTrue(Path(result["jsonl"]).exists())
            self.assertTrue(Path(result["srt"]).exists())

            # 検出ロジックの正確性を検証する。
            # inside は t∈[5.0, 7.5]、その前後に outside[0,5] と outside[7.5,9.5]
            # が生成されるため、全3イベント。
            events = [
                json.loads(line) for line in
                Path(result["jsonl"]).read_text(encoding="utf-8")
                .strip().splitlines()
            ]
            self.assertEqual(result["events"], 3)
            self.assertEqual(len(events), 3)

            inside = [e for e in events if e["label"] == "inside"]
            outside = [e for e in events if e["label"] == "outside"]
            self.assertEqual(len(inside), 1)
            self.assertEqual(len(outside), 2)

            # inside 区間の境界が入力どおり [5.0, 7.5] であること
            self.assertAlmostEqual(inside[0]["start_sec"], 5.0, places=3)
            self.assertAlmostEqual(inside[0]["end_sec"], 7.5, places=3)
            self.assertEqual(inside[0]["type"], "cavity_inside")

            # 時系列で start→end が単調かつ重複しないこと
            starts = [e["start_sec"] for e in events]
            self.assertEqual(starts, sorted(starts))
            for e in events:
                self.assertLessEqual(e["start_sec"], e["end_sec"])

    def test_smooth_s_recorded_in_metadata(self):
        """smooth_s が戻り値とメタJSONに記録されること（未使用引数の解消）"""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "case_cavitylog.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "t_sec", "t_srt",
                    "vignette_score", "tissue_ratio",
                    "brightness_bimodality", "hue_concentration",
                    "cavity_score", "smooth_cavity", "reader",
                ])
                for i in range(20):
                    t = i * 0.5
                    smooth = 0.8 if 2 <= i <= 12 else 0.1
                    writer.writerow([
                        f"{t:.3f}", format_srt_time(t),
                        "0.0", "0.0", "0.0", "0.0",
                        f"{smooth:.6f}", f"{smooth:.6f}", "pyav",
                    ])
            result = annotate_cavity(
                csv_path=str(csv_path), outdir=tmpdir,
                thr=0.35, min_duration_s=1.0, smooth_s=4.5,
            )
            self.assertEqual(result["smooth_s"], 4.5)
            meta_path = Path(tmpdir) / "case_cavity_meta.json"
            self.assertTrue(meta_path.exists())
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.assertEqual(meta["smooth_s"], 4.5)
            self.assertEqual(meta["events"], result["events"])


if __name__ == "__main__":
    unittest.main()
