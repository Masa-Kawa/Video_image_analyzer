"""
bleed_trend.py のユニットテスト
"""

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.red.bleed_trend import (
    compute_trend,
    read_trendlog_csv,
)


class TestComputeTrend(unittest.TestCase):
    """トレンドスコア計算のテスト"""

    def test_increasing_values(self):
        """単調増加する値はトレンドスコアが正になること"""
        # 5秒間で 0.0 → 0.5 に線形増加 (5fps, 25サンプル)
        values = [i * 0.02 for i in range(25)]
        scores = compute_trend(values, fps=5.0, window_s=5.0)
        # 中央付近のスコアは明確に正
        mid = len(scores) // 2
        self.assertGreater(scores[mid], 0.0)

    def test_decreasing_values(self):
        """単調減少する値はトレンドスコアが0であること"""
        values = [0.5 - i * 0.02 for i in range(25)]
        scores = compute_trend(values, fps=5.0, window_s=5.0)
        # 全てのスコアが0（負の傾きは0にクリップ）
        for s in scores:
            self.assertAlmostEqual(s, 0.0, places=10)

    def test_constant_values(self):
        """一定値はトレンドスコアが0であること"""
        values = [0.3] * 25
        scores = compute_trend(values, fps=5.0, window_s=5.0)
        for s in scores:
            self.assertAlmostEqual(s, 0.0, places=10)

    def test_oscillating_values(self):
        """振動する値はR²が低く、トレンドスコアが低いこと"""
        # 上下に揺れる（カメラ移動パターン）
        values = [0.3 + 0.1 * np.sin(i * 0.5) for i in range(50)]
        scores = compute_trend(values, fps=5.0, window_s=5.0)
        # 振動のトレンドスコアは単調増加より小さいはず
        increasing = [i * 0.004 for i in range(50)]
        inc_scores = compute_trend(increasing, fps=5.0, window_s=5.0)
        mid = len(scores) // 2
        self.assertLess(scores[mid], inc_scores[mid])

    def test_short_input(self):
        """短い入力でもクラッシュしないこと"""
        scores = compute_trend([0.1, 0.2], fps=5.0, window_s=5.0)
        self.assertEqual(len(scores), 2)

    def test_output_length(self):
        """出力長が入力長と一致すること"""
        values = [float(i) for i in range(100)]
        scores = compute_trend(values, fps=5.0, window_s=3.0)
        self.assertEqual(len(scores), 100)


class TestCsvRoundtrip(unittest.TestCase):
    """CSV読み書きの往復テスト"""

    def test_roundtrip(self):
        """CSVに書いたデータが正しく読めること"""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test_trendlog.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "t_sec", "t_srt", "red_ratio",
                    "trend_score", "smooth_trend", "reader",
                ])
                writer.writerow([
                    "0.000", "00:00:00,000", "0.100000",
                    "0.005000", "0.004000", "pyav",
                ])
                writer.writerow([
                    "0.200", "00:00:00,200", "0.120000",
                    "0.008000", "0.006000", "pyav",
                ])

            data = read_trendlog_csv(str(csv_path))
            self.assertEqual(len(data["times"]), 2)
            self.assertAlmostEqual(data["times"][0], 0.0)
            self.assertAlmostEqual(data["ratios"][1], 0.12)
            self.assertAlmostEqual(data["trend_scores"][0], 0.005)
            self.assertAlmostEqual(data["smooth_trends"][1], 0.006)
            self.assertEqual(data["reader"], "pyav")


if __name__ == "__main__":
    unittest.main()
