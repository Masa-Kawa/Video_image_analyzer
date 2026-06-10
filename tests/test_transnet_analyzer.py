"""
transnet_analyzer.TransNetAnalyzer のユニットテスト

重い GPU 依存の SceneDetector をモックに差し替え、遅延初期化のスレッド安全性
（並行 analyze からの二重初期化防止）と、close()/コンテキストマネージャによる
リソース解放・再初期化を検証する。
"""

import threading
import time
import unittest
from unittest import mock

import numpy as np

import src.transnet.transnet_analyzer as ta
from src.transnet.transnet_analyzer import TransNetAnalyzer
from src.transnet.inference import _parse_frame_rate, DEFAULT_FPS


class _SlowFakeDetector:
    """構築回数を数える擬似 SceneDetector（構築に少し時間がかかる）。"""

    instances = 0
    lock = threading.Lock()

    def __init__(self, weights_path=None, device="cpu"):
        with _SlowFakeDetector.lock:
            _SlowFakeDetector.instances += 1
        # 競合状態を起こしやすくするための遅延
        time.sleep(0.02)
        self.weights_path = weights_path
        self.device = device

    def predict_video(self, video_path, threshold=0.5, return_scores=True):
        return [], []


class TestLazyInitThreadSafety(unittest.TestCase):
    def setUp(self):
        _SlowFakeDetector.instances = 0

    def test_concurrent_init_constructs_once(self):
        with mock.patch.object(ta, "TransNetDetector", _SlowFakeDetector):
            analyzer = TransNetAnalyzer(device="cpu")

            errors = []

            def worker():
                try:
                    analyzer._init_detector()
                except Exception as e:  # pragma: no cover
                    errors.append(e)

            threads = [threading.Thread(target=worker) for _ in range(16)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            self.assertEqual(errors, [])
            # ロックにより検出器は一度だけ構築される
            self.assertEqual(_SlowFakeDetector.instances, 1)
            self.assertIsNotNone(analyzer.detector)

    def test_close_releases_and_reinit(self):
        with mock.patch.object(ta, "TransNetDetector", _SlowFakeDetector):
            analyzer = TransNetAnalyzer(device="cpu")
            analyzer._init_detector()
            self.assertEqual(_SlowFakeDetector.instances, 1)

            analyzer.close()
            self.assertIsNone(analyzer.detector)

            # close 後は再度遅延初期化される
            analyzer._init_detector()
            self.assertEqual(_SlowFakeDetector.instances, 2)
            self.assertIsNotNone(analyzer.detector)

    def test_close_idempotent(self):
        with mock.patch.object(ta, "TransNetDetector", _SlowFakeDetector):
            analyzer = TransNetAnalyzer(device="cpu")
            analyzer._init_detector()
            analyzer.close()
            # 二重 close でも例外を出さない
            analyzer.close()
            self.assertIsNone(analyzer.detector)

    def test_context_manager_closes(self):
        with mock.patch.object(ta, "TransNetDetector", _SlowFakeDetector):
            with TransNetAnalyzer(device="cpu") as analyzer:
                analyzer._init_detector()
                self.assertIsNotNone(analyzer.detector)
            self.assertIsNone(analyzer.detector)


class TestParseFrameRate(unittest.TestCase):
    """r_frame_rate のパースとフォールバックの堅牢性。"""

    def test_normal_fraction(self):
        self.assertAlmostEqual(
            _parse_frame_rate({"r_frame_rate": "30000/1001"}), 29.97, places=2)

    def test_integer_rate_without_denominator(self):
        self.assertAlmostEqual(_parse_frame_rate({"r_frame_rate": "24"}), 24.0)

    def test_zero_over_zero_falls_back_to_avg(self):
        self.assertAlmostEqual(
            _parse_frame_rate({"r_frame_rate": "0/0", "avg_frame_rate": "25/1"}),
            25.0)

    def test_all_missing_uses_default(self):
        self.assertEqual(_parse_frame_rate({}), DEFAULT_FPS)

    def test_malformed_uses_default(self):
        self.assertEqual(_parse_frame_rate({"r_frame_rate": "abc"}), DEFAULT_FPS)


class _RecordingDetector:
    """predict_video の呼び出し引数を記録する擬似 SceneDetector。"""

    def __init__(self, weights_path=None, device="cpu"):
        self.calls = []

    def predict_video(self, video_path, threshold=0.5, return_scores=False,
                      min_scene_length=5):
        self.calls.append({
            "video_path": video_path,
            "threshold": threshold,
            "min_scene_length": min_scene_length,
        })
        return [], np.array([], dtype=np.float32)


class TestAnalyzeValidationAndForwarding(unittest.TestCase):
    def test_missing_file_raises_before_init(self):
        analyzer = TransNetAnalyzer(device="cpu")
        with self.assertRaises(FileNotFoundError):
            analyzer.analyze("/no/such/video.mp4")
        # 検出器は初期化されないまま
        self.assertIsNone(analyzer.detector)

    def test_min_scene_length_forwarded(self):
        fake = _RecordingDetector()
        with mock.patch.object(ta, "TransNetDetector", lambda **k: fake):
            analyzer = TransNetAnalyzer(device="cpu")
            with mock.patch.object(analyzer, "_get_video_info",
                                   return_value={"fps": 25.0, "duration": 1.0}), \
                 mock.patch("pathlib.Path.exists", return_value=True), \
                 mock.patch("pathlib.Path.is_file", return_value=True):
                analyzer.analyze("fake.mp4", threshold=0.7, min_scene_length=12)
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(fake.calls[0]["min_scene_length"], 12)
        self.assertEqual(fake.calls[0]["threshold"], 0.7)

    def test_predict_error_propagates(self):
        class _BoomDetector:
            def __init__(self, **k):
                pass

            def predict_video(self, *a, **k):
                raise RuntimeError("CUDA out of memory")

        with mock.patch.object(ta, "TransNetDetector", lambda **k: _BoomDetector()):
            analyzer = TransNetAnalyzer(device="cpu")
            with mock.patch.object(analyzer, "_get_video_info",
                                   return_value={"fps": 25.0, "duration": 1.0}), \
                 mock.patch("pathlib.Path.exists", return_value=True), \
                 mock.patch("pathlib.Path.is_file", return_value=True):
                with self.assertRaises(RuntimeError):
                    analyzer.analyze("fake.mp4")


if __name__ == "__main__":
    unittest.main()
