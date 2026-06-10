"""
phase_segmenter.py のユニットテスト

合成データを使い、変化点検出・クラスタリング・フェーズマッピングを検証する。
"""

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.phase.phase_segmenter import (
    extract_heuristic_feature,
    detect_change_points,
    cluster_segments,
    _simple_kmeans,
    map_clusters_to_phases,
    read_phaselog_csv,
)
import cv2


class TestHeuristicFeature(unittest.TestCase):
    """ヒューリスティック特徴量のテスト"""

    def test_output_shape(self):
        """48次元のベクトルが返ること"""
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        feat = extract_heuristic_feature(frame)
        self.assertEqual(feat.shape, (48,))

    def test_normalized(self):
        """L2正規化されていること"""
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        feat = extract_heuristic_feature(frame)
        norm = np.linalg.norm(feat)
        self.assertAlmostEqual(norm, 1.0, places=4)

    def test_different_frames_different_features(self):
        """異なる画像で異なる特徴量が返ること"""
        red_hsv = np.zeros((100, 100, 3), dtype=np.uint8)
        red_hsv[:, :, 0] = 0
        red_hsv[:, :, 1] = 255
        red_hsv[:, :, 2] = 255
        red_bgr = cv2.cvtColor(red_hsv, cv2.COLOR_HSV2BGR)

        blue_hsv = np.zeros((100, 100, 3), dtype=np.uint8)
        blue_hsv[:, :, 0] = 120
        blue_hsv[:, :, 1] = 255
        blue_hsv[:, :, 2] = 255
        blue_bgr = cv2.cvtColor(blue_hsv, cv2.COLOR_HSV2BGR)

        feat_red = extract_heuristic_feature(red_bgr)
        feat_blue = extract_heuristic_feature(blue_bgr)

        # コサイン類似度が1ではない（異なる特徴量）
        sim = float(np.dot(feat_red, feat_blue))
        self.assertLess(sim, 0.95)


class TestDetectChangePoints(unittest.TestCase):
    """変化点検出のテスト"""

    def test_clear_change(self):
        """明確な変化がある場合に変化点が検出されること"""
        # 前半: 類似度が高い、後半: 類似度が高い、中間で急落
        sims = [0.95] * 20 + [0.3, 0.3] + [0.95] * 20

        cps = detect_change_points(sims, min_segment_length=5, sensitivity=1.0)
        self.assertGreater(len(cps), 0)

        # 変化点は中間付近にあるはず
        for cp in cps:
            self.assertGreater(cp, 10)
            self.assertLess(cp, 32)

    def test_no_change(self):
        """均一な系列では変化点がない"""
        sims = [0.95] * 40
        cps = detect_change_points(sims, min_segment_length=5, sensitivity=1.5)
        self.assertEqual(len(cps), 0)

    def test_min_segment_length(self):
        """min_segment_length未満の間隔の変化点はマージされる"""
        sims = [0.95] * 10 + [0.3] + [0.95] * 3 + [0.3] + [0.95] * 10

        cps = detect_change_points(sims, min_segment_length=5, sensitivity=1.0)
        # index10 と index14（3フレーム差）の2候補が検出され、min_segment_length=5
        # 未満なのでマージされ、ちょうど1つ（index10）に集約される。
        self.assertEqual(cps, [10])


class TestSimpleKmeans(unittest.TestCase):
    """簡易K-meansのテスト"""

    def test_two_clusters(self):
        """2つの明確なクラスタが正しく分離されること"""
        # グローバルRNG状態に依存しないよう default_rng でシード固定する
        rng = np.random.default_rng(0)
        cluster_a = rng.standard_normal((10, 5)) + np.array([5, 0, 0, 0, 0])
        cluster_b = rng.standard_normal((10, 5)) + np.array([-5, 0, 0, 0, 0])
        data = np.vstack([cluster_a, cluster_b])

        labels = _simple_kmeans(data, k=2)
        self.assertEqual(len(labels), 20)

        # 前半と後半で異なるラベル
        labels_a = set(labels[:10])
        labels_b = set(labels[10:])
        # 各クラスタは主に1つのラベル
        self.assertEqual(len(labels_a), 1)
        self.assertEqual(len(labels_b), 1)
        self.assertNotEqual(labels_a, labels_b)

    def test_single_point(self):
        """データ点がk以下の場合"""
        data = np.array([[1.0, 2.0]])
        labels = _simple_kmeans(data, k=3)
        self.assertEqual(len(labels), 1)


class TestClusterSegments(unittest.TestCase):
    """セグメントクラスタリングのテスト"""

    def test_basic(self):
        """基本的なクラスタリングが動作すること"""
        # 3つのセグメント（各10フレーム）の特徴量
        # セグメント0と2は同じ分布、セグメント1は異なる分布
        # グローバルRNGを汚染しないよう default_rng を使う（他テストへの影響回避）
        rng = np.random.default_rng(42)
        features = np.vstack([
            rng.standard_normal((10, 8)) * 0.1 + np.array([5, 0, 0, 0, 0, 0, 0, 0]),
            rng.standard_normal((10, 8)) * 0.1 + np.array([-5, 0, 0, 0, 0, 0, 0, 0]),
            rng.standard_normal((10, 8)) * 0.1 + np.array([5, 0, 0, 0, 0, 0, 0, 0]),
        ])

        change_points = [10, 20]
        labels = cluster_segments(features, change_points, n_clusters=2)

        self.assertEqual(len(labels), 3)
        # セグメント0と2は同じクラスタ（同じ分布）
        self.assertEqual(labels[0], labels[2])
        # セグメント1は異なるクラスタ
        self.assertNotEqual(labels[0], labels[1])


class TestMapClustersToPhases(unittest.TestCase):
    """フェーズマッピングのテスト"""

    def test_temporal_mapping(self):
        """時間的位置に基づくフェーズ名の割り当て"""
        times = list(range(100))
        features = np.random.randn(100, 8)
        change_points = [10, 50, 90]
        cluster_labels = [0, 1, 2, 3]

        brightness = [200.0] * 10 + [80.0] * 40 + [80.0] * 40 + [80.0] * 10
        tissue = [0.05] * 10 + [0.3] * 40 + [0.5] * 40 + [0.1] * 10

        names = map_clusters_to_phases(
            times, features, change_points, cluster_labels,
            brightness, tissue,
        )

        self.assertEqual(len(names), 4)
        # 序盤で明るい → Preparation
        self.assertEqual(names[0], "Preparation")
        # 終盤 → Extraction
        self.assertEqual(names[3], "Extraction")


class TestReadPhaselogCsv(unittest.TestCase):
    """CSV読み込みのテスト"""

    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test_phaselog.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "t_sec", "t_srt",
                    "brightness", "tissue_ratio",
                    "similarity", "smooth_similarity",
                    "reader",
                ])
                writer.writerow([
                    "0.000", "00:00:00,000",
                    "120.50", "0.350000",
                    "1.000000", "1.000000",
                    "pyav",
                ])
                writer.writerow([
                    "1.000", "00:00:01,000",
                    "115.30", "0.380000",
                    "0.950000", "0.975000",
                    "pyav",
                ])

            data = read_phaselog_csv(str(csv_path))

            self.assertEqual(len(data["times"]), 2)
            self.assertAlmostEqual(data["brightnesses"][0], 120.5)
            self.assertAlmostEqual(data["tissue_ratios"][1], 0.38)
            self.assertAlmostEqual(data["fps"], 1.0, places=1)


if __name__ == "__main__":
    unittest.main()
