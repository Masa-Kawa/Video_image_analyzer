"""
bleed_spread.py のユニットテスト

合成フレームで局所赤色拡散検出ロジックを検証する。
"""

import csv
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.red.bleed_spread import (
    compute_cell_ratios,
    compute_spread_score,
    read_spreadlog_csv,
)
from src.red.redlog import make_circular_roi


class TestComputeCellRatios(unittest.TestCase):
    """セル別赤色率計算のテスト"""

    def _make_frame(self, h, w, hue, sat=255, val=255):
        """指定HSV色の単色フレームを生成するヘルパー"""
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        hsv[:, :, 0] = hue
        hsv[:, :, 1] = sat
        hsv[:, :, 2] = val
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def test_all_red(self):
        """全画素赤 → 全セルが約1.0"""
        frame = self._make_frame(80, 80, 0)
        cells = compute_cell_ratios(frame, grid_size=4)
        self.assertEqual(cells.shape, (4, 4))
        # 全セルの赤色率が高い
        self.assertTrue(np.all(cells > 0.9))

    def test_all_blue(self):
        """全画素青 → 全セルが約0.0"""
        frame = self._make_frame(80, 80, 120)
        cells = compute_cell_ratios(frame, grid_size=4)
        self.assertTrue(np.all(cells < 0.01))

    def test_partial_red(self):
        """上半分赤、下半分青 → 上のセルだけ赤色率が高い"""
        h, w = 80, 80
        # 全面青
        frame = self._make_frame(h, w, 120)
        # 上半分を赤に
        hsv_red = np.zeros((h // 2, w, 3), dtype=np.uint8)
        hsv_red[:, :, 0] = 0
        hsv_red[:, :, 1] = 255
        hsv_red[:, :, 2] = 255
        frame[:h // 2] = cv2.cvtColor(hsv_red, cv2.COLOR_HSV2BGR)

        cells = compute_cell_ratios(frame, grid_size=4)
        # 上2行（行0,1）は赤色率が高い
        self.assertTrue(np.all(cells[:2, :] > 0.9))
        # 下2行（行2,3）は赤色率が低い
        self.assertTrue(np.all(cells[2:, :] < 0.01))

    def test_grid_size(self):
        """異なるグリッドサイズで正しい形状になること"""
        frame = self._make_frame(120, 120, 0)
        for gs in [2, 4, 8, 16]:
            cells = compute_cell_ratios(frame, grid_size=gs)
            self.assertEqual(cells.shape, (gs, gs))


class TestComputeSpreadScore(unittest.TestCase):
    """拡散スコア計算のテスト"""

    def test_no_change(self):
        """変化なし → spread_score = 0"""
        cells = np.full((8, 8), 0.5)
        result = compute_spread_score(cells, cells)
        self.assertAlmostEqual(result["spread_score"], 0.0)
        self.assertAlmostEqual(result["max_cell_delta"], 0.0)
        self.assertEqual(result["n_rising_cells"], 0)

    def test_uniform_change(self):
        """全セル均等変化（カメラ移動） → delta_std ≈ 0 → spread_score ≈ 0"""
        prev = np.full((8, 8), 0.3)
        curr = np.full((8, 8), 0.5)  # 全セル +0.2
        result = compute_spread_score(prev, curr)
        # 標準偏差はほぼ0
        self.assertAlmostEqual(result["delta_std"], 0.0, places=3)
        # spread_score も低い
        self.assertAlmostEqual(result["spread_score"], 0.0, places=3)

    def test_local_change(self):
        """1セルだけ変化（出血パターン） → spread_score 高い"""
        prev = np.full((8, 8), 0.3)
        curr = prev.copy()
        curr[3, 4] = 0.8  # 1セルだけ大きく赤化

        result = compute_spread_score(prev, curr)
        # 標準偏差 > 0（不均一な変化）
        self.assertGreater(result["delta_std"], 0.01)
        # max_cell_delta = 0.5
        self.assertAlmostEqual(result["max_cell_delta"], 0.5, places=2)
        # spread_score > 0
        self.assertGreater(result["spread_score"], 0.001)
        # 上昇セルは1つ
        self.assertEqual(result["n_rising_cells"], 1)

    def test_cluster_change(self):
        """隣接する数セルが変化 → spread_score が1セルより高い場合がある"""
        prev = np.full((8, 8), 0.3)
        curr = prev.copy()
        # 2×2 のクラスタが赤化
        curr[3:5, 3:5] = 0.7

        result = compute_spread_score(prev, curr)
        self.assertGreater(result["spread_score"], 0.001)
        self.assertEqual(result["n_rising_cells"], 4)

    def test_camera_vs_bleed_distinction(self):
        """カメラ移動 vs 出血: 同じ赤色増加量でも spread_score が異なることを検証"""
        prev = np.full((8, 8), 0.3)

        # カメラ移動: 全セル +0.05
        curr_camera = np.full((8, 8), 0.35)
        score_camera = compute_spread_score(prev, curr_camera)

        # 出血: 4セルだけ +0.1（他は変化なし）→ 合計変化量はほぼ同程度
        curr_bleed = prev.copy()
        curr_bleed[2:4, 2:4] = 0.4  # 4セル × 0.1 = 0.4相当の変化
        score_bleed = compute_spread_score(prev, curr_bleed)

        # 出血パターンの方が spread_score が高い
        self.assertGreater(score_bleed["spread_score"],
                           score_camera["spread_score"])


class TestReadSpreadlogCsv(unittest.TestCase):
    """CSV読み書きのテスト"""

    def test_roundtrip(self):
        """CSV読み込みが正しく動作すること"""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test_spreadlog.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "t_sec", "t_srt", "red_ratio",
                    "max_cell_delta", "delta_std",
                    "spread_score", "smooth_spread",
                    "n_rising_cells", "reader",
                ])
                writer.writerow([
                    "0.000", "00:00:00,000", "0.500000",
                    "0.050000", "0.012000",
                    "0.000600", "0.000600",
                    "3", "pyav",
                ])
                writer.writerow([
                    "0.200", "00:00:00,200", "0.520000",
                    "0.080000", "0.018000",
                    "0.001440", "0.001020",
                    "5", "pyav",
                ])

            data = read_spreadlog_csv(str(csv_path))

            self.assertEqual(len(data["times"]), 2)
            self.assertAlmostEqual(data["times"][0], 0.0)
            self.assertAlmostEqual(data["red_ratios"][0], 0.5)
            self.assertAlmostEqual(data["max_cell_deltas"][1], 0.08)
            self.assertAlmostEqual(data["delta_stds"][0], 0.012)
            self.assertAlmostEqual(data["spread_scores"][1], 0.00144)
            self.assertEqual(data["n_rising_cells"][0], 3)
            self.assertEqual(data["reader"], "pyav")


if __name__ == "__main__":
    unittest.main()
