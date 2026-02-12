"""
redlog.py のユニットテスト

動画ファイルなしで赤色解析ロジックを検証する。
"""

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.red.redlog import (
    compute_red_ratio,
    extract_bleed_events,
    format_srt_time,
    make_circular_roi,
    smooth_center,
)


class TestFormatSrtTime(unittest.TestCase):
    """SRT時刻フォーマットのテスト"""

    def test_zero(self):
        self.assertEqual(format_srt_time(0.0), "00:00:00,000")

    def test_simple(self):
        self.assertEqual(format_srt_time(61.5), "00:01:01,500")

    def test_hours(self):
        self.assertEqual(format_srt_time(3661.123), "01:01:01,123")

    def test_negative_clamps(self):
        self.assertEqual(format_srt_time(-5.0), "00:00:00,000")


class TestCircularRoi(unittest.TestCase):
    """円形ROI生成のテスト"""

    def test_basic_shape(self):
        mask = make_circular_roi(100, 100, margin=0.0)
        self.assertEqual(mask.shape, (100, 100))
        # 中心画素はROI内
        self.assertTrue(mask[50, 50])

    def test_corners_excluded_with_margin(self):
        mask = make_circular_roi(100, 100, margin=0.08)
        # 四隅はROI外
        self.assertFalse(mask[0, 0])
        self.assertFalse(mask[0, 99])
        self.assertFalse(mask[99, 0])
        self.assertFalse(mask[99, 99])

    def test_rectangular_frame(self):
        mask = make_circular_roi(480, 640, margin=0.08)
        self.assertEqual(mask.shape, (480, 640))
        # 中心はROI内
        self.assertTrue(mask[240, 320])


class TestComputeRedRatio(unittest.TestCase):
    """赤色率計算のテスト"""

    def test_all_red_frame(self):
        """全画素が赤色のフレーム → 赤色率≈1.0"""
        # HSV: H=0, S=255, V=255 → 赤
        import cv2
        h, w = 100, 100
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        hsv[:, :, 0] = 0    # H = 0（赤）
        hsv[:, :, 1] = 255  # S = 255
        hsv[:, :, 2] = 255  # V = 255
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        ratio = compute_red_ratio(bgr, roi_mask=None)
        self.assertGreater(ratio, 0.9)

    def test_all_blue_frame(self):
        """全画素が青色のフレーム → 赤色率≈0.0"""
        import cv2
        h, w = 100, 100
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        hsv[:, :, 0] = 120  # H = 120（青）
        hsv[:, :, 1] = 255
        hsv[:, :, 2] = 255
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        ratio = compute_red_ratio(bgr, roi_mask=None)
        self.assertAlmostEqual(ratio, 0.0, places=2)

    def test_with_roi(self):
        """ROIマスク適用時、ROI外の画素は無視されること"""
        import cv2
        h, w = 100, 100
        # 全画素赤
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        hsv[:, :, 0] = 0
        hsv[:, :, 1] = 255
        hsv[:, :, 2] = 255
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        roi = make_circular_roi(h, w, margin=0.08)
        ratio = compute_red_ratio(bgr, roi_mask=roi)
        # ROI内のみ赤なので1.0に近い
        self.assertGreater(ratio, 0.9)


class TestSmoothCenter(unittest.TestCase):
    """中心移動平均のテスト"""

    def test_identity_window_1(self):
        vals = [1.0, 2.0, 3.0]
        result = smooth_center(vals, 1)
        self.assertEqual(result, vals)

    def test_simple_average(self):
        vals = [0.0, 0.0, 1.0, 0.0, 0.0]
        result = smooth_center(vals, 3)
        # 中央のval=1.0は前後と平均: (0+1+0)/3 ≈ 0.333
        self.assertAlmostEqual(result[2], 1.0 / 3.0, places=5)

    def test_empty(self):
        self.assertEqual(smooth_center([], 5), [])

    def test_preserves_length(self):
        vals = [float(i) for i in range(20)]
        result = smooth_center(vals, 7)
        self.assertEqual(len(result), len(vals))


class TestExtractBleedEvents(unittest.TestCase):
    """出血候補イベント抽出のテスト"""

    def test_no_events_below_threshold(self):
        """閾値以下の信号 → イベントなし"""
        times = [i * 0.2 for i in range(50)]
        deltas = [0.01] * 50
        events = extract_bleed_events(times, deltas, thr=0.03, k_s=3.0, fps=5.0, smooth_s=5.0)
        self.assertEqual(len(events), 0)

    def test_single_event(self):
        """閾値超過がk_s秒以上継続 → イベント1件"""
        fps = 5.0
        # 20サンプル（4秒）の閾値超過
        times = [i * 0.2 for i in range(50)]
        deltas = [0.01] * 50
        # 10〜29（4秒間）を閾値超過に
        for i in range(10, 30):
            deltas[i] = 0.05

        events = extract_bleed_events(times, deltas, thr=0.03, k_s=3.0, fps=fps, smooth_s=5.0)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["type"], "bleed_candidate")
        self.assertGreater(events[0]["delta_max"], 0.03)

    def test_short_spike_ignored(self):
        """短すぎるスパイク（k_s未満） → イベントなし"""
        fps = 5.0
        times = [i * 0.2 for i in range(50)]
        deltas = [0.01] * 50
        # 5サンプル（1秒）のみ閾値超過 → k_s=3秒未満
        for i in range(10, 15):
            deltas[i] = 0.05

        events = extract_bleed_events(times, deltas, thr=0.03, k_s=3.0, fps=fps, smooth_s=5.0)
        self.assertEqual(len(events), 0)

    def test_event_fields(self):
        """イベント辞書が必須フィールドを持つこと"""
        fps = 5.0
        times = [i * 0.2 for i in range(50)]
        deltas = [0.05] * 50  # 全区間閾値超過

        events = extract_bleed_events(times, deltas, thr=0.03, k_s=1.0, fps=fps, smooth_s=5.0)
        self.assertGreater(len(events), 0)
        ev = events[0]
        for key in ["type", "metric", "thr", "k_s", "smooth_s", "delta_max", "start", "end"]:
            self.assertIn(key, ev, f"必須フィールド '{key}' がありません")


if __name__ == "__main__":
    unittest.main()
