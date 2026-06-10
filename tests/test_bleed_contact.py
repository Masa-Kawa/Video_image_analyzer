"""
bleed_contact.py のユニットテスト
"""

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.red.bleed_contact import compute_contact_growth, read_contactlog_csv


class TestComputeContactGrowth(unittest.TestCase):
    """Tests for contact-growth score computation."""

    def test_no_change(self):
        """Unchanged frame pair yields zero score."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:, :] = [0, 0, 128]  # dark blue (not red)
        result = compute_contact_growth(frame, frame.copy())
        self.assertAlmostEqual(result["bleed_score"], 0.0)
        self.assertEqual(result["newly_red_count"], 0)

    def test_adjacent_red_growth(self):
        """Red growing adjacent to existing red yields high contact_ratio."""
        # Previous frame: a red region in the center
        prev = np.zeros((100, 100, 3), dtype=np.uint8)
        prev[40:60, 40:60] = [0, 0, 255]  # red (BGR)

        # Current frame: red region expanded by 1 pixel (adjacent growth)
        curr = np.zeros((100, 100, 3), dtype=np.uint8)
        curr[39:61, 39:61] = [0, 0, 255]

        result = compute_contact_growth(prev, curr)
        # Newly-red pixels form the 1px ring: 22^2 - 20^2 = 84 px,
        # all adjacent to the previous red edge → full contact.
        self.assertEqual(result["newly_red_count"], 84)
        self.assertEqual(result["contact_count"], 84)
        self.assertAlmostEqual(result["contact_ratio"], 1.0)
        self.assertGreater(result["bleed_score"], 0.0)

    def test_distant_red_appearance(self):
        """Red appearing away from existing red yields low contact_ratio."""
        # Previous frame: red in the top-left
        prev = np.zeros((200, 200, 3), dtype=np.uint8)
        prev[10:30, 10:30] = [0, 0, 255]

        # Current frame: keep top-left red + add new red bottom-right
        curr = prev.copy()
        curr[170:190, 170:190] = [0, 0, 255]

        result = compute_contact_growth(prev, curr)
        # The bottom-right red is not adjacent to top-left → low contact_ratio
        self.assertLess(result["contact_ratio"], 0.1)

    def test_with_roi_mask(self):
        """ROI mask restricts pixel counting to the ROI region."""
        frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
        frame2 = np.zeros((100, 100, 3), dtype=np.uint8)
        frame2[45:55, 45:55] = [0, 0, 255]  # 10x10 = 100 red px, inside ROI

        # Only the center 40x40 = 1600 px is a valid ROI
        roi = np.zeros((100, 100), dtype=bool)
        roi[30:70, 30:70] = True

        result = compute_contact_growth(frame1, frame2, roi_mask=roi)
        # If ROI is applied: red_ratio = 100/1600 = 0.0625 (not 100/10000=0.01).
        self.assertAlmostEqual(result["red_ratio"], 0.0625, places=6)
        # All 100 red px are newly red and inside the ROI.
        self.assertEqual(result["newly_red_count"], 100)

    def test_zero_pixels(self):
        """Empty ROI does not crash and yields zero counts."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        roi = np.zeros((100, 100), dtype=bool)
        result = compute_contact_growth(frame, frame.copy(), roi_mask=roi)
        self.assertAlmostEqual(result["bleed_score"], 0.0)
        self.assertEqual(result["contact_count"], 0)
        self.assertAlmostEqual(result["contact_area"], 0.0)


class TestCsvRoundtrip(unittest.TestCase):
    """CSV読み書きの往復テスト"""

    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test_contactlog.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "t_sec", "t_srt", "red_ratio",
                    "contact_ratio", "contact_area",
                    "bleed_score", "smooth_bleed", "reader",
                ])
                writer.writerow([
                    "0.000", "00:00:00,000", "0.050000",
                    "0.800000", "0.001000", "0.000800",
                    "0.000600", "pyav",
                ])
                writer.writerow([
                    "0.200", "00:00:00,200", "0.055000",
                    "0.750000", "0.001200", "0.000900",
                    "0.000700", "pyav",
                ])

            data = read_contactlog_csv(str(csv_path))
            self.assertEqual(len(data["times"]), 2)
            self.assertAlmostEqual(data["contact_ratios"][0], 0.8)
            self.assertAlmostEqual(data["bleed_scores"][1], 0.0009)


if __name__ == "__main__":
    unittest.main()
