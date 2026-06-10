"""
bleed_ai モジュールのユニットテスト

合成フレーム・合成CSVを使い、AI出血検出ロジックを検証する。
"""

import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

from src.core.time_utils import format_srt_time
from src.bleed_ai.models import (
    BleedSegmenter,
    hsv_blood_area_and_source,
    hsv_blood_mask,
    hsv_blood_probability,
    preprocess_frame,
    preprocess_frame_segmentation,
)
from src.bleed_ai.detector import (
    annotate_bleed_ai,
    compute_flow_magnitude,
    compute_severity,
    read_bleedailog_csv,
    _estimate_fps,
)
from src.red.redlog import make_circular_roi


# ---------------------------------------------------------------------------
# HSV血液マスクのテスト
# ---------------------------------------------------------------------------

class TestHsvBloodMask(unittest.TestCase):
    """HSVベース血液マスク生成のテスト"""

    def _make_frame(self, h, w, hue, sat=255, val=255):
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        hsv[:, :, 0] = hue
        hsv[:, :, 1] = sat
        hsv[:, :, 2] = val
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def test_all_red_frame(self):
        """全画素赤 → マスクの大部分がTrue"""
        frame = self._make_frame(50, 50, 0)
        mask = hsv_blood_mask(frame)
        ratio = np.count_nonzero(mask) / (50 * 50)
        self.assertGreater(ratio, 0.8)

    def test_all_blue_frame(self):
        """全画素青 → マスクがほぼ空"""
        frame = self._make_frame(50, 50, 120)
        mask = hsv_blood_mask(frame)
        ratio = np.count_nonzero(mask) / (50 * 50)
        self.assertLess(ratio, 0.1)

    def test_with_roi(self):
        """ROI適用時にROI外が除外されること"""
        frame = self._make_frame(100, 100, 0)
        roi = make_circular_roi(100, 100, margin=0.08)
        mask = hsv_blood_mask(frame, roi_mask=roi)
        # ROI外は0のはず
        outside_roi = mask[~roi]
        self.assertEqual(np.count_nonzero(outside_roi), 0)


class TestHsvBloodProbability(unittest.TestCase):
    """HSV出血確率のテスト"""

    def test_red_frame_high_prob(self):
        hsv = np.zeros((50, 50, 3), dtype=np.uint8)
        hsv[:, :, 0] = 5
        hsv[:, :, 1] = 200
        hsv[:, :, 2] = 200
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        prob = hsv_blood_probability(bgr)
        self.assertGreater(prob, 0.5)

    def test_green_frame_low_prob(self):
        hsv = np.zeros((50, 50, 3), dtype=np.uint8)
        hsv[:, :, 0] = 60
        hsv[:, :, 1] = 200
        hsv[:, :, 2] = 200
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        prob = hsv_blood_probability(bgr)
        self.assertLess(prob, 0.1)


class TestHsvBloodAreaAndSource(unittest.TestCase):
    """血液領域面積と出血源のテスト"""

    def test_no_blood(self):
        """青画像 → area_ratio ≈ 0"""
        hsv = np.zeros((100, 100, 3), dtype=np.uint8)
        hsv[:, :, 0] = 120
        hsv[:, :, 1] = 200
        hsv[:, :, 2] = 200
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        result = hsv_blood_area_and_source(bgr)
        self.assertLess(result["area_ratio"], 0.1)

    def test_partial_blood_source(self):
        """上半分が赤 → source_y < 0.5（上部）"""
        h, w = 100, 100
        # 下半分: 青
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        hsv[:, :, 0] = 120
        hsv[:, :, 1] = 200
        hsv[:, :, 2] = 200
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # 上半分: 赤
        hsv_red = np.zeros((h // 2, w, 3), dtype=np.uint8)
        hsv_red[:, :, 0] = 0
        hsv_red[:, :, 1] = 255
        hsv_red[:, :, 2] = 255
        bgr[:h // 2] = cv2.cvtColor(hsv_red, cv2.COLOR_HSV2BGR)

        result = hsv_blood_area_and_source(bgr)
        self.assertGreater(result["area_ratio"], 0.2)
        self.assertLess(result["source_y"], 0.5)


# ---------------------------------------------------------------------------
# モデルのテスト
# ---------------------------------------------------------------------------

class TestBleedSegmenter(unittest.TestCase):
    """U-Netモデルの形状テスト"""

    def test_output_shape(self):
        """入力(1,3,256,256)→出力(1,1,256,256)"""
        model = BleedSegmenter(channels=(16, 32, 64, 128))
        model.eval()
        x = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            out = model(x)
        self.assertEqual(out.shape, (1, 1, 256, 256))

    def test_output_range(self):
        """Sigmoid出力は[0, 1]範囲"""
        model = BleedSegmenter(channels=(16, 32, 64, 128))
        model.eval()
        x = torch.randn(1, 3, 128, 128)
        with torch.no_grad():
            out = model(x)
        self.assertTrue(torch.all(out >= 0))
        self.assertTrue(torch.all(out <= 1))


class TestPreprocessFrame(unittest.TestCase):
    """前処理のテスト"""

    def test_shape(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        tensor = preprocess_frame(frame, size=224)
        self.assertEqual(tensor.shape, (1, 3, 224, 224))

    def test_seg_shape(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        tensor = preprocess_frame_segmentation(frame, size=256)
        self.assertEqual(tensor.shape, (1, 3, 256, 256))


# ---------------------------------------------------------------------------
# オプティカルフロー
# ---------------------------------------------------------------------------

class TestComputeFlowMagnitude(unittest.TestCase):
    """オプティカルフローのテスト"""

    def test_identical_frames_zero_flow(self):
        """同一フレーム → フロー ≈ 0"""
        gray = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        flow = compute_flow_magnitude(gray, gray)
        self.assertAlmostEqual(flow, 0.0, places=1)

    def test_shifted_frame_nonzero_flow(self):
        """シフトしたフレーム → フロー > 0"""
        gray1 = np.zeros((100, 100), dtype=np.uint8)
        gray1[20:40, 20:40] = 200
        gray2 = np.zeros((100, 100), dtype=np.uint8)
        gray2[25:45, 25:45] = 200
        flow = compute_flow_magnitude(gray1, gray2)
        self.assertGreater(flow, 0.0)

    def test_with_roi(self):
        """ROI指定時に正常動作すること"""
        gray = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        roi = make_circular_roi(100, 100, margin=0.08)
        flow = compute_flow_magnitude(gray, gray, roi_mask=roi)
        self.assertAlmostEqual(flow, 0.0, places=1)


# ---------------------------------------------------------------------------
# 重症度スコア
# ---------------------------------------------------------------------------

class TestComputeSeverity(unittest.TestCase):
    """重症度スコアのテスト"""

    def test_zero_inputs(self):
        sev = compute_severity(0.0, 0.0, 0.0)
        self.assertEqual(sev, 0.0)

    def test_high_bleeding(self):
        """高出血確率 + 高面積 + 低フロー → 高重症度"""
        sev = compute_severity(0.9, 0.4, 1.0)
        self.assertGreater(sev, 0.3)

    def test_camera_movement_suppression(self):
        """大きなカメラ動き → 重症度が抑制される"""
        sev_still = compute_severity(0.8, 0.3, 1.0, flow_suppress_thr=8.0)
        sev_moving = compute_severity(0.8, 0.3, 20.0, flow_suppress_thr=8.0)
        self.assertGreater(sev_still, sev_moving)

    def test_severity_capped_at_1(self):
        sev = compute_severity(1.0, 1.0, 0.0)
        self.assertLessEqual(sev, 1.0)

    def test_zero_suppress_threshold_no_zerodiv(self):
        """flow_suppress_thr=0/負でも ZeroDivisionError にならず抑制なし扱い"""
        sev0 = compute_severity(0.8, 0.3, 5.0, flow_suppress_thr=0.0)
        sevn = compute_severity(0.8, 0.3, 5.0, flow_suppress_thr=-1.0)
        sev_ref = compute_severity(0.8, 0.3, 0.0, flow_suppress_thr=8.0)
        # 抑制なし（flow_factor=1.0）なので通常の低フロー時と同値
        self.assertAlmostEqual(sev0, sev_ref)
        self.assertAlmostEqual(sevn, sev_ref)

    # ---- flow_factor の境界値を厳密に検証 -------------------------------
    # bleed_prob=0.5, area_ratio=0.5 → scaled_area = min(1, 1.0) = 1.0
    # なので severity == flow_factor となり、減衰式を直接検証できる。
    def test_flow_below_threshold_no_suppression(self):
        # flow_mag=0 < thr → flow_factor=1.0 → severity=0.5
        sev = compute_severity(0.5, 0.5, 0.0, flow_suppress_thr=8.0)
        self.assertAlmostEqual(sev, 0.5, places=6)

    def test_flow_exactly_at_threshold_no_suppression(self):
        # flow_mag=8.0 == thr → flow_factor = 1 - 0/8 = 1.0 → severity=0.5
        sev = compute_severity(0.5, 0.5, 8.0, flow_suppress_thr=8.0)
        self.assertAlmostEqual(sev, 0.5, places=6)

    def test_flow_half_decay(self):
        # flow_mag=12.0 (=1.5*thr) → flow_factor = 1 - 4/8 = 0.5 → severity=0.25
        sev = compute_severity(0.5, 0.5, 12.0, flow_suppress_thr=8.0)
        self.assertAlmostEqual(sev, 0.25, places=6)

    def test_flow_full_suppression_at_double_threshold(self):
        # flow_mag=16.0 (=2*thr) → flow_factor = max(0, 1-8/8) = 0.0 → severity=0
        sev = compute_severity(0.5, 0.5, 16.0, flow_suppress_thr=8.0)
        self.assertAlmostEqual(sev, 0.0, places=6)

    def test_flow_beyond_double_threshold_clamped_zero(self):
        # 2*thr を超えても負にならず 0 にクランプ
        sev = compute_severity(0.5, 0.5, 100.0, flow_suppress_thr=8.0)
        self.assertEqual(sev, 0.0)


class TestEstimateFps(unittest.TestCase):
    """FPS推定の堅牢性テスト"""

    def test_regular_intervals(self):
        self.assertAlmostEqual(_estimate_fps([0.0, 0.1, 0.2, 0.3]), 10.0)

    def test_identical_timestamps_returns_default(self):
        # 全て同一タイムスタンプ → ZeroDivisionError ではなくデフォルト
        self.assertEqual(_estimate_fps([1.0, 1.0, 1.0]), 10.0)

    def test_single_or_empty_returns_default(self):
        self.assertEqual(_estimate_fps([0.0]), 10.0)
        self.assertEqual(_estimate_fps([]), 10.0)

    def test_duplicates_use_median_of_positive(self):
        # 0間隔は無視、正の間隔(0.2)の中央値から 5fps
        self.assertAlmostEqual(_estimate_fps([0.0, 0.2, 0.2, 0.4]), 5.0)


# ---------------------------------------------------------------------------
# CSV往復テスト
# ---------------------------------------------------------------------------

class TestReadBleedailogCsv(unittest.TestCase):
    """CSV読み込みのテスト"""

    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test_bleedailog.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "t_sec", "t_srt",
                    "bleed_prob", "bleed_area", "source_x", "source_y",
                    "flow_mag", "severity",
                    "smooth_bleed_prob", "smooth_severity", "smooth_area",
                    "reader",
                ])
                writer.writerow([
                    "0.000", "00:00:00,000",
                    "0.500000", "0.100000", "0.400000", "0.300000",
                    "1.500000", "0.080000",
                    "0.480000", "0.075000", "0.095000",
                    "pyav",
                ])
                writer.writerow([
                    "0.100", "00:00:00,100",
                    "0.550000", "0.120000", "0.420000", "0.310000",
                    "1.200000", "0.090000",
                    "0.510000", "0.082000", "0.105000",
                    "pyav",
                ])
                writer.writerow([
                    "0.200", "00:00:00,200",
                    "0.480000", "0.090000", "0.410000", "0.290000",
                    "1.800000", "0.070000",
                    "0.500000", "0.078000", "0.100000",
                    "pyav",
                ])

            data = read_bleedailog_csv(str(csv_path))

            self.assertEqual(len(data["times"]), 3)
            self.assertAlmostEqual(data["times"][0], 0.0)
            self.assertAlmostEqual(data["times"][1], 0.1)
            self.assertAlmostEqual(data["bleed_probs"][0], 0.5)
            self.assertAlmostEqual(data["bleed_areas"][1], 0.12)
            self.assertAlmostEqual(data["source_xs"][0], 0.4)
            self.assertAlmostEqual(data["flow_mags"][2], 1.8)
            self.assertAlmostEqual(data["smooth_probs"][0], 0.48)
            self.assertAlmostEqual(data["smooth_sevs"][1], 0.082)
            self.assertEqual(data["reader"], "pyav")
            self.assertAlmostEqual(data["fps"], 10.0, places=1)

    # ---- 異常系 ----------------------------------------------------------
    _HEADER = ("t_sec,bleed_prob,bleed_area,source_x,source_y,flow_mag,"
               "severity,smooth_bleed_prob,smooth_severity,smooth_area,reader")
    _ROW = "{t},0.5,0.1,0.5,0.5,1.0,0.2,0.5,0.2,0.1,m"

    def _write(self, body: str) -> str:
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        p = Path(d) / "x.csv"
        p.write_text(body, encoding="utf-8")
        return str(p)

    def test_duplicate_timestamps_no_zerodiv(self):
        # 同一タイムスタンプ2行（旧実装なら ZeroDivisionError）
        body = self._HEADER + "\n" + "\n".join(
            [self._ROW.format(t="0.0"), self._ROW.format(t="0.0")])
        data = read_bleedailog_csv(self._write(body))
        self.assertEqual(len(data["times"]), 2)
        self.assertEqual(data["fps"], 10.0)  # 推定不能→デフォルト

    def test_missing_required_column_raises(self):
        with self.assertRaises(ValueError):
            read_bleedailog_csv(self._write("t_sec,bleed_prob\n0.0,0.5"))

    def test_empty_file_raises(self):
        with self.assertRaises(ValueError):
            read_bleedailog_csv(self._write(""))

    def test_non_numeric_rows_skipped(self):
        body = (self._HEADER + "\n"
                + "BAD,x,,,,,,,,,m\n"               # 非数値→スキップ
                + self._ROW.format(t="0.0") + "\n"
                + self._ROW.format(t="0.1") + "\n")
        data = read_bleedailog_csv(self._write(body))
        self.assertEqual(len(data["times"]), 2)

    def test_all_invalid_rows_raises(self):
        body = self._HEADER + "\n" + "BAD,x,,,,,,,,,m\n"
        with self.assertRaises(ValueError):
            read_bleedailog_csv(self._write(body))

    def test_header_only_raises(self):
        # ヘッダーのみ（データ行ゼロ）→ 有効データなしで ValueError
        with self.assertRaises(ValueError):
            read_bleedailog_csv(self._write(self._HEADER + "\n"))

    def test_single_row_uses_default_fps(self):
        # 1行のみ → 隣接差を取れないため fps はデフォルト(10.0)
        body = self._HEADER + "\n" + self._ROW.format(t="0.0") + "\n"
        data = read_bleedailog_csv(self._write(body))
        self.assertEqual(len(data["times"]), 1)
        self.assertEqual(data["fps"], 10.0)

    def test_uneven_intervals_use_median(self):
        # 不均一間隔 [0.1, 0.1, 0.3] → 中央値0.1 → fps=10.0（外れ値に頑健）
        ts = ["0.0", "0.1", "0.2", "0.5"]
        body = self._HEADER + "\n" + "\n".join(
            self._ROW.format(t=t) for t in ts) + "\n"
        data = read_bleedailog_csv(self._write(body))
        self.assertEqual(len(data["times"]), 4)
        self.assertAlmostEqual(data["fps"], 10.0, places=6)


# ---------------------------------------------------------------------------
# アノテーションテスト
# ---------------------------------------------------------------------------

class TestAnnotateBleedAi(unittest.TestCase):
    """アノテーション（イベント抽出）のテスト"""

    def _make_csv(self, tmpdir: str, values: list) -> str:
        """合成CSVを作成するヘルパー"""
        csv_path = Path(tmpdir) / "test_bleedailog.csv"
        fps = 10.0
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "t_sec", "t_srt",
                "bleed_prob", "bleed_area", "source_x", "source_y",
                "flow_mag", "severity",
                "smooth_bleed_prob", "smooth_severity", "smooth_area",
                "reader",
            ])
            for i, v in enumerate(values):
                t = i / fps
                writer.writerow([
                    f"{t:.3f}", format_srt_time(t),
                    f"{v:.6f}", f"{v:.6f}", "0.500000", "0.500000",
                    "1.000000", f"{v:.6f}",
                    f"{v:.6f}", f"{v:.6f}", f"{v:.6f}",
                    "pyav",
                ])
        return str(csv_path)

    def test_no_events_below_threshold(self):
        """閾値以下の信号 → イベントなし"""
        with tempfile.TemporaryDirectory() as tmpdir:
            values = [0.01] * 50
            csv_path = self._make_csv(tmpdir, values)
            result = annotate_bleed_ai(
                csv_path, tmpdir, thr=0.05,
                metric="smooth_severity", max_duration_s=0,
            )
            self.assertEqual(result["events"], 0)

    def test_single_event(self):
        """閾値超過が1秒以上継続 → イベント1件"""
        with tempfile.TemporaryDirectory() as tmpdir:
            values = [0.01] * 50
            # 10〜25（1.5秒間）を閾値超過
            for i in range(10, 25):
                values[i] = 0.10
            csv_path = self._make_csv(tmpdir, values)
            result = annotate_bleed_ai(
                csv_path, tmpdir, thr=0.05, min_duration_s=1.0,
                metric="smooth_severity", max_duration_s=0,
            )
            self.assertEqual(result["events"], 1)

    def test_short_spike_ignored(self):
        """短すぎるスパイク → イベントなし"""
        with tempfile.TemporaryDirectory() as tmpdir:
            values = [0.01] * 50
            # 5サンプル（0.5秒）のみ
            for i in range(10, 15):
                values[i] = 0.10
            csv_path = self._make_csv(tmpdir, values)
            result = annotate_bleed_ai(
                csv_path, tmpdir, thr=0.05, min_duration_s=1.0,
                metric="smooth_severity", max_duration_s=0,
            )
            self.assertEqual(result["events"], 0)

    def test_jsonl_format(self):
        """JSONLの各行が必須フィールドを含むこと"""
        with tempfile.TemporaryDirectory() as tmpdir:
            values = [0.10] * 50  # 全区間閾値超過
            csv_path = self._make_csv(tmpdir, values)
            result = annotate_bleed_ai(
                csv_path, tmpdir, thr=0.05, min_duration_s=0.5,
                metric="smooth_severity", max_duration_s=0,
            )

            self.assertGreater(result["events"], 0)

            with open(result["jsonl"], "r", encoding="utf-8") as f:
                for line in f:
                    ev = json.loads(line)
                    for key in ["type", "start_sec", "end_sec", "start_srt", "end_srt"]:
                        self.assertIn(key, ev, f"必須フィールド '{key}' がありません")
                    self.assertEqual(ev["type"], "bleed_ai_candidate")

    def test_srt_format(self):
        """SRTが正しいフォーマットであること"""
        with tempfile.TemporaryDirectory() as tmpdir:
            values = [0.10] * 50
            csv_path = self._make_csv(tmpdir, values)
            result = annotate_bleed_ai(
                csv_path, tmpdir, thr=0.05, min_duration_s=0.5,
                metric="smooth_severity", max_duration_s=0,
            )

            srt_text = Path(result["srt"]).read_text(encoding="utf-8")
            if result["events"] > 0:
                # SRTは番号行、時刻行、タグ行、空行の繰り返し
                self.assertIn("-->", srt_text)
                self.assertIn("[bleed_ai]", srt_text)

    def test_multiple_events(self):
        """複数のイベントが正しく分離されること"""
        with tempfile.TemporaryDirectory() as tmpdir:
            values = [0.01] * 100
            # イベント1: 10-25
            for i in range(10, 25):
                values[i] = 0.10
            # イベント2: 50-65
            for i in range(50, 65):
                values[i] = 0.15
            csv_path = self._make_csv(tmpdir, values)
            result = annotate_bleed_ai(
                csv_path, tmpdir, thr=0.05, min_duration_s=1.0,
                metric="smooth_severity", max_duration_s=0,
            )
            self.assertEqual(result["events"], 2)

    def test_area_delta_metric(self):
        """area_delta: ベースラインからの急増を検出すること"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 安定したベースライン（0.8）→ 急増（0.95）→ ベースライン復帰
            values = [0.80] * 200
            # 100〜130（3秒間）に面積急増
            for i in range(100, 130):
                values[i] = 0.95
            csv_path = self._make_csv(tmpdir, values)
            result = annotate_bleed_ai(
                csv_path, tmpdir, thr=0.05, min_duration_s=1.0,
                metric="area_delta", max_duration_s=0, baseline_s=10.0,
            )
            self.assertGreater(result["events"], 0)
            # JSONL中身確認
            with open(result["jsonl"], "r", encoding="utf-8") as f:
                ev = json.loads(f.readline())
                self.assertEqual(ev["metric"], "area_delta")

    def test_area_delta_no_change(self):
        """area_delta: 一定の信号 → delta=0でイベントなし"""
        with tempfile.TemporaryDirectory() as tmpdir:
            values = [0.85] * 200  # 安定
            csv_path = self._make_csv(tmpdir, values)
            result = annotate_bleed_ai(
                csv_path, tmpdir, thr=0.05, min_duration_s=1.0,
                metric="area_delta", max_duration_s=0, baseline_s=10.0,
            )
            self.assertEqual(result["events"], 0)

    def test_max_duration_filter(self):
        """max_duration_s: 長すぎるイベントが除外されること"""
        with tempfile.TemporaryDirectory() as tmpdir:
            values = [0.01] * 200
            # 全体を閾値超過（20秒間）
            for i in range(200):
                values[i] = 0.50
            csv_path = self._make_csv(tmpdir, values)
            # max_duration_s=10 → 20秒のイベントは除外
            result = annotate_bleed_ai(
                csv_path, tmpdir, thr=0.05, min_duration_s=1.0,
                metric="smooth_severity", max_duration_s=10.0,
            )
            self.assertEqual(result["events"], 0)
            # max_duration_s=0（無制限） → イベントあり
            result2 = annotate_bleed_ai(
                csv_path, tmpdir, thr=0.05, min_duration_s=1.0,
                metric="smooth_severity", max_duration_s=0,
            )
            self.assertGreater(result2["events"], 0)


# ---------------------------------------------------------------------------
# SRT→JSONL往復テスト
# ---------------------------------------------------------------------------

class TestSrtJsonlRoundtrip(unittest.TestCase):
    """SRTとJSONLの往復変換テスト"""

    def test_tag_template_exists(self):
        """bleed_ai_candidate のタグテンプレートが登録されていること"""
        from src.tools.jsonl_to_srt import TAG_TEMPLATES
        self.assertIn("bleed_ai_candidate", TAG_TEMPLATES)
        self.assertEqual(TAG_TEMPLATES["bleed_ai_candidate"],
                         "[bleed_ai] bleeding_detected")

    def test_tag_pattern_exists(self):
        """bleed_ai のタグパターンが登録されていること"""
        from src.tools.srt_to_jsonl import TAG_PATTERNS
        import re
        found = False
        for pattern, event_type in TAG_PATTERNS.items():
            if event_type == "bleed_ai_candidate":
                found = True
                # TAG_PATTERNS の値が re.compile 済みであることを前提にしている。
                # 文字列等に変更された場合は AttributeError ではなくこの assert で
                # 前提崩れを明示的に検出する。
                self.assertIsInstance(pattern, re.Pattern)
                self.assertTrue(pattern.match("[bleed_ai] bleeding_detected"))
        self.assertTrue(found, "bleed_ai_candidate パターンが未登録")


if __name__ == "__main__":
    unittest.main()
