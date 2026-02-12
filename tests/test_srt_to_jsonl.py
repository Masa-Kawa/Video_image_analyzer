"""
srt_to_jsonl.py のユニットテスト

SRTからJSONLへの変換ロジックを検証する。
"""

import json
import tempfile
import unittest
from pathlib import Path

from src.tools.srt_to_jsonl import (
    _parse_tag_line,
    convert,
    read_srt_to_events,
)


class TestParseTagLine(unittest.TestCase):
    """タグ行パースのテスト"""

    def test_bleed_tag(self):
        result = _parse_tag_line("[bleed] delta_over_threshold")
        self.assertEqual(result, "bleed_candidate")

    def test_cut_tag(self):
        result = _parse_tag_line("[cut] transnet")
        self.assertEqual(result, "cut")

    def test_unknown_tag(self):
        result = _parse_tag_line("unknown tag line")
        self.assertIsNone(result)


class TestReadSrtToEvents(unittest.TestCase):
    """SRT読込のテスト"""

    def _write_srt(self, content: str) -> str:
        """一時SRTファイルを作成してパスを返すヘルパー"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".srt", delete=False, encoding="utf-8"
        ) as f:
            f.write(content)
            return f.name

    def test_basic_read(self):
        """基本的なSRT読み込み"""
        content = (
            "1\n"
            "00:02:15,600 --> 00:02:18,800\n"
            "[bleed] delta_over_threshold\n"
            '{"type": "bleed_candidate", "metric": "red_ratio", '
            '"thr": 0.03, "delta_max": 0.05}\n'
            "\n"
        )
        path = self._write_srt(content)
        events = read_srt_to_events(path)
        self.assertEqual(len(events), 1)

        ev = events[0]
        self.assertEqual(ev["type"], "bleed_candidate")
        self.assertAlmostEqual(ev["start_sec"], 135.6, places=1)
        self.assertAlmostEqual(ev["end_sec"], 138.8, places=1)
        self.assertEqual(ev["metric"], "red_ratio")
        self.assertEqual(ev["thr"], 0.03)
        Path(path).unlink()

    def test_time_update_from_srt(self):
        """SRTの時刻がJSONLに正しく反映されること"""
        content = (
            "1\n"
            "00:05:00,000 --> 00:05:10,000\n"
            "[bleed] delta_over_threshold\n"
            '{"type": "bleed_candidate", "thr": 0.03}\n'
            "\n"
        )
        path = self._write_srt(content)
        events = read_srt_to_events(path)
        ev = events[0]
        self.assertAlmostEqual(ev["start_sec"], 300.0, places=1)
        self.assertAlmostEqual(ev["end_sec"], 310.0, places=1)
        self.assertEqual(ev["start_srt"], "00:05:00,000")
        self.assertEqual(ev["end_srt"], "00:05:10,000")
        Path(path).unlink()

    def test_multiple_entries(self):
        """複数エントリの読み込み"""
        content = (
            "1\n"
            "00:01:00,000 --> 00:01:05,000\n"
            "[bleed] delta_over_threshold\n"
            '{"type": "bleed_candidate"}\n'
            "\n"
            "2\n"
            "00:02:00,000 --> 00:02:03,000\n"
            "[bleed] delta_over_threshold\n"
            '{"type": "bleed_candidate"}\n'
            "\n"
        )
        path = self._write_srt(content)
        events = read_srt_to_events(path)
        self.assertEqual(len(events), 2)
        Path(path).unlink()

    def test_broken_json_continues(self):
        """JSON行が壊れていても時刻・タグ情報で続行すること"""
        content = (
            "1\n"
            "00:01:00,000 --> 00:01:05,000\n"
            "[bleed] delta_over_threshold\n"
            "この行は壊れたJSON\n"
            "\n"
        )
        path = self._write_srt(content)
        events = read_srt_to_events(path)
        self.assertEqual(len(events), 1)
        ev = events[0]
        self.assertEqual(ev["type"], "bleed_candidate")
        self.assertAlmostEqual(ev["start_sec"], 60.0, places=1)
        Path(path).unlink()

    def test_cut_event(self):
        """カットイベントの読み込み"""
        content = (
            "1\n"
            "00:10:15,150 --> 00:10:15,350\n"
            "[cut] transnet\n"
            '{"type": "cut", "model": "TransNetV2", "score": 0.93}\n'
            "\n"
        )
        path = self._write_srt(content)
        events = read_srt_to_events(path)
        self.assertEqual(len(events), 1)
        ev = events[0]
        self.assertEqual(ev["type"], "cut")
        self.assertEqual(ev["model"], "TransNetV2")
        Path(path).unlink()


class TestConvert(unittest.TestCase):
    """エンドツーエンド変換テスト"""

    def test_end_to_end(self):
        """SRT → JSONL → ファイル出力の一貫性"""
        with tempfile.TemporaryDirectory() as tmpdir:
            srt_path = Path(tmpdir) / "test.srt"
            jsonl_path = Path(tmpdir) / "output.jsonl"

            # SRTデータ作成
            srt_path.write_text(
                "1\n"
                "00:02:15,600 --> 00:02:18,800\n"
                "[bleed] delta_over_threshold\n"
                '{"type": "bleed_candidate", "metric": "red_ratio", '
                '"thr": 0.03, "delta_max": 0.05}\n'
                "\n",
                encoding="utf-8",
            )

            # 変換実行
            count = convert(str(srt_path), str(jsonl_path))
            self.assertEqual(count, 1)
            self.assertTrue(jsonl_path.exists())

            # JSONL内容の検証
            content = jsonl_path.read_text(encoding="utf-8").strip()
            ev = json.loads(content)
            self.assertEqual(ev["type"], "bleed_candidate")
            self.assertAlmostEqual(ev["start_sec"], 135.6, places=1)
            self.assertAlmostEqual(ev["end_sec"], 138.8, places=1)
            self.assertEqual(ev["start_srt"], "00:02:15,600")
            self.assertEqual(ev["end_srt"], "00:02:18,800")


if __name__ == "__main__":
    unittest.main()
