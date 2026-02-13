"""
bleed_detector.py のユニットテスト

合成フレームペアを使い、赤色拡大検出ロジックを検証する。
"""

import csv
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.red.bleed_detector import (
    compute_red_expansion,
    make_red_mask,
    read_bleedlog_csv,
)
from src.red.redlog import make_circular_roi


class TestMakeRedMask(unittest.TestCase):
    """赤色マスク生成のテスト"""

    def test_all_red(self):
        """全画素赤 → マスク全域がTrue"""
        h, w = 50, 50
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        hsv[:, :, 0] = 0    # H=0（赤）
        hsv[:, :, 1] = 255
        hsv[:, :, 2] = 255
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        mask = make_red_mask(bgr)
        red_count = np.count_nonzero(mask)
        self.assertEqual(red_count, h * w)

    def test_all_blue(self):
        """全画素青 → マスク全域がFalse"""
        h, w = 50, 50
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        hsv[:, :, 0] = 120  # H=120（青）
        hsv[:, :, 1] = 255
        hsv[:, :, 2] = 255
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        mask = make_red_mask(bgr)
        red_count = np.count_nonzero(mask)
        self.assertEqual(red_count, 0)


class TestComputeRedExpansion(unittest.TestCase):
    """赤色拡大率計算のテスト"""

    def _make_frame(self, h, w, hue, sat=255, val=255):
        """指定HSV色の単色フレームを生成するヘルパー"""
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        hsv[:, :, 0] = hue
        hsv[:, :, 1] = sat
        hsv[:, :, 2] = val
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def test_no_change_same_red(self):
        """赤→赤（変化なし）: newly_red_ratio ≈ 0"""
        h, w = 50, 50
        frame_red = self._make_frame(h, w, 0)

        result = compute_red_expansion(frame_red, frame_red)
        self.assertAlmostEqual(result["newly_red_ratio"], 0.0, places=3)
        self.assertAlmostEqual(result["red_expansion"], 0.0, places=3)

    def test_blue_to_red_stable_bg(self):
        """青→赤（背景安定）: newly_red_ratio > 0, bg_stability ≈ 1.0"""
        h, w = 50, 50
        frame_blue = self._make_frame(h, w, 120)
        frame_red = self._make_frame(h, w, 0)

        result = compute_red_expansion(frame_blue, frame_red)
        # 全画素が新規赤化
        self.assertGreater(result["newly_red_ratio"], 0.9)
        # red_expansion も高い
        self.assertGreater(result["red_expansion"], 0.0)

    def test_partial_red_expansion(self):
        """一部領域の青→赤: newly_red_ratio > 0 だが全体ではない"""
        h, w = 100, 100
        # 前: 全面青
        prev = self._make_frame(h, w, 120)
        # 後: 上半分赤、下半分青
        curr = prev.copy()
        hsv_red = np.zeros((h // 2, w, 3), dtype=np.uint8)
        hsv_red[:, :, 0] = 0
        hsv_red[:, :, 1] = 255
        hsv_red[:, :, 2] = 255
        curr[:h // 2] = cv2.cvtColor(hsv_red, cv2.COLOR_HSV2BGR)

        result = compute_red_expansion(prev, curr)
        # 約50%が新規赤化
        self.assertGreater(result["newly_red_ratio"], 0.3)
        self.assertLess(result["newly_red_ratio"], 0.7)

    def test_camera_movement_suppression(self):
        """背景全体が大きく変化 → bg_stability が低い → red_expansion 抑制"""
        h, w = 100, 100
        # 前: 緑色
        prev = self._make_frame(h, w, 60)
        # 後: 完全に異なる画像（白色、HSVで無彩色）
        curr = np.full((h, w, 3), 255, dtype=np.uint8)

        result = compute_red_expansion(prev, curr)
        # 背景安定度が低い
        self.assertLess(result["bg_stability"], 0.5)

    def test_with_roi(self):
        """ROI指定時にROI外が無視されること"""
        h, w = 100, 100
        prev = self._make_frame(h, w, 120)  # 青
        curr = self._make_frame(h, w, 0)    # 赤
        roi = make_circular_roi(h, w, margin=0.08)

        result = compute_red_expansion(prev, curr, roi_mask=roi)
        # ROI内のみ集計されるがnewly_red_ratioは高い
        self.assertGreater(result["newly_red_ratio"], 0.5)

    def test_zero_pixels(self):
        """ROIが空（全画素除外）の場合のエッジケース"""
        h, w = 10, 10
        prev = self._make_frame(h, w, 0)
        curr = self._make_frame(h, w, 0)
        # 全画素をFalseにしたROI
        roi = np.zeros((h, w), dtype=bool)

        result = compute_red_expansion(prev, curr, roi_mask=roi)
        self.assertAlmostEqual(result["red_expansion"], 0.0)


class TestReadBleedlogCsv(unittest.TestCase):
    """CSV読み書きのテスト"""

    def test_roundtrip(self):
        """CSV読み込みが正しく動作すること"""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test_bleedlog.csv"
            # csv.writer を使い、t_srt のカンマを正しくクォートする
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "t_sec", "t_srt", "red_ratio", "newly_red_ratio",
                    "bg_stability", "red_expansion", "smooth_expansion", "reader",
                ])
                writer.writerow([
                    "0.000", "00:00:00,000", "0.500000", "0.010000",
                    "0.950000", "0.009500", "0.009500", "pyav",
                ])
                writer.writerow([
                    "0.200", "00:00:00,200", "0.520000", "0.015000",
                    "0.900000", "0.013500", "0.011500", "pyav",
                ])
                writer.writerow([
                    "0.400", "00:00:00,400", "0.510000", "0.005000",
                    "0.980000", "0.004900", "0.009300", "pyav",
                ])

            data = read_bleedlog_csv(str(csv_path))

            self.assertEqual(len(data["times"]), 3)
            self.assertAlmostEqual(data["times"][0], 0.0)
            self.assertAlmostEqual(data["times"][1], 0.2)
            self.assertAlmostEqual(data["red_ratios"][0], 0.5)
            self.assertAlmostEqual(data["newly_red_ratios"][1], 0.015)
            self.assertAlmostEqual(data["bg_stabilities"][2], 0.98)
            self.assertAlmostEqual(data["red_expansions"][0], 0.0095)
            self.assertEqual(data["reader"], "pyav")
            self.assertAlmostEqual(data["fps"], 5.0, places=1)


if __name__ == "__main__":
    unittest.main()
