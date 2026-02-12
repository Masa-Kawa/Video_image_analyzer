"""
jsonl_to_srt.py のユニットテスト

JSONLからSRTへの変換ロジックを検証する。
"""

import json
import tempfile
import unittest
from pathlib import Path

from src.tools.jsonl_to_srt import (
    _build_json_line,
    _build_tag_line,
    convert,
    events_to_srt,
    read_events_jsonl,
)


class TestReadEventsJsonl(unittest.TestCase):
    """JSONL読み込みのテスト"""

    def test_basic_read(self):
        """基本的なJSONL読み込み"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            f.write(json.dumps({
                "type": "bleed_candidate", "start_sec": 10.0,
                "end_sec": 15.0, "delta_max": 0.05
            }) + "\n")
            f.write(json.dumps({
                "type": "bleed_candidate", "start_sec": 20.0,
                "end_sec": 25.0, "delta_max": 0.08
            }) + "\n")
            path = f.name

        events = read_events_jsonl(path)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["start_sec"], 10.0)
        self.assertEqual(events[1]["start_sec"], 20.0)
        Path(path).unlink()

    def test_sorted_by_start_sec(self):
        """イベントがstart_secでソートされること"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            f.write(json.dumps({
                "type": "bleed_candidate", "start_sec": 30.0,
                "end_sec": 35.0
            }) + "\n")
            f.write(json.dumps({
                "type": "bleed_candidate", "start_sec": 5.0,
                "end_sec": 10.0
            }) + "\n")
            path = f.name

        events = read_events_jsonl(path)
        self.assertEqual(events[0]["start_sec"], 5.0)
        self.assertEqual(events[1]["start_sec"], 30.0)
        Path(path).unlink()

    def test_empty_lines_skipped(self):
        """空行がスキップされること"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            f.write("\n")
            f.write(json.dumps({
                "type": "bleed_candidate", "start_sec": 10.0,
                "end_sec": 15.0
            }) + "\n")
            f.write("\n")
            path = f.name

        events = read_events_jsonl(path)
        self.assertEqual(len(events), 1)
        Path(path).unlink()

    def test_transnet_format_supported(self):
        """TransNet形式（t_sec）もサポートされること"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            f.write(json.dumps({
                "t_sec": 615.2, "score": 0.93
            }) + "\n")
            path = f.name

        events = read_events_jsonl(path)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["t_sec"], 615.2)
        Path(path).unlink()


class TestBuildTagLine(unittest.TestCase):
    """タグ行生成のテスト"""

    def test_bleed_candidate(self):
        tag = _build_tag_line({"type": "bleed_candidate"})
        self.assertEqual(tag, "[bleed] delta_over_threshold")

    def test_cut(self):
        tag = _build_tag_line({"type": "cut"})
        self.assertEqual(tag, "[cut] transnet")

    def test_unknown_type(self):
        tag = _build_tag_line({"type": "unknown_type"})
        self.assertIn("unknown_type", tag)


class TestBuildJsonLine(unittest.TestCase):
    """JSON行生成のテスト"""

    def test_excludes_time_fields(self):
        """時刻情報がJSON行から除外されること"""
        event = {
            "type": "bleed_candidate",
            "metric": "red_ratio",
            "thr": 0.03,
            "start_sec": 10.0,
            "end_sec": 15.0,
            "start_srt": "00:00:10,000",
            "end_srt": "00:00:15,000",
        }
        json_str = _build_json_line(event)
        parsed = json.loads(json_str)
        self.assertNotIn("start_sec", parsed)
        self.assertNotIn("end_sec", parsed)
        self.assertNotIn("start_srt", parsed)
        self.assertNotIn("end_srt", parsed)
        self.assertEqual(parsed["type"], "bleed_candidate")
        self.assertEqual(parsed["metric"], "red_ratio")


class TestEventsToSrt(unittest.TestCase):
    """SRT文字列への変換テスト"""

    def test_basic_conversion(self):
        """基本的なSRT変換"""
        events = [{
            "type": "bleed_candidate",
            "metric": "red_ratio",
            "thr": 0.03,
            "delta_max": 0.05,
            "start_sec": 61.5,
            "end_sec": 65.0,
        }]
        srt = events_to_srt(events)
        lines = srt.split("\n")
        self.assertEqual(lines[0], "1")
        self.assertIn("-->", lines[1])
        self.assertEqual(lines[2], "[bleed] delta_over_threshold")
        # JSON行の確認
        parsed = json.loads(lines[3])
        self.assertEqual(parsed["type"], "bleed_candidate")

    def test_event_type_filter(self):
        """イベントタイプフィルタの動作"""
        events = [
            {"type": "bleed_candidate", "start_sec": 10.0, "end_sec": 15.0},
            {"type": "cut", "start_sec": 20.0, "end_sec": 20.2},
        ]
        srt = events_to_srt(events, event_type="bleed_candidate")
        # cutイベントは含まれないこと
        self.assertNotIn("[cut]", srt)
        self.assertIn("[bleed]", srt)

    def test_multiple_events(self):
        """複数イベントの連番"""
        events = [
            {"type": "bleed_candidate", "start_sec": 10.0, "end_sec": 15.0},
            {"type": "bleed_candidate", "start_sec": 30.0, "end_sec": 35.0},
        ]
        srt = events_to_srt(events)
        # 先頭は「1」で始まる
        self.assertTrue(srt.startswith("1\n"))
        self.assertIn("\n2\n", srt)


class TestConvert(unittest.TestCase):
    """エンドツーエンド変換テスト"""

    def test_end_to_end(self):
        """JSONL → SRT → ファイル出力の一貫性"""
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "events.jsonl"
            srt_path = Path(tmpdir) / "output.srt"

            # JSONLデータ作成
            with open(jsonl_path, "w", encoding="utf-8") as f:
                f.write(json.dumps({
                    "type": "bleed_candidate",
                    "metric": "red_ratio",
                    "thr": 0.03,
                    "k_s": 3.0,
                    "smooth_s": 5.0,
                    "delta_max": 0.05,
                    "start_sec": 135.6,
                    "end_sec": 138.8,
                    "start_srt": "00:02:15,600",
                    "end_srt": "00:02:18,800",
                }) + "\n")

            # 変換実行
            count = convert(str(jsonl_path), str(srt_path))
            self.assertEqual(count, 1)
            self.assertTrue(srt_path.exists())

            # SRT内容の検証
            content = srt_path.read_text(encoding="utf-8")
            self.assertIn("[bleed]", content)
            self.assertIn("00:02:15,600", content)
            self.assertIn("00:02:18,800", content)


if __name__ == "__main__":
    unittest.main()
