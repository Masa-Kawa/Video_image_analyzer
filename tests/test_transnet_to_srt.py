"""
transnet_to_srt.py のユニットテスト
"""

import json
import tempfile
import unittest
from pathlib import Path

from src.transnet.transnet_to_srt import (
    boundaries_to_srt,
    convert,
    format_srt_time,
    read_boundaries_jsonl,
)


class TestFormatSrtTime(unittest.TestCase):
    """SRT時刻フォーマットのテスト"""

    def test_basic(self):
        self.assertEqual(format_srt_time(615.2), "00:10:15,200")

    def test_zero(self):
        self.assertEqual(format_srt_time(0.0), "00:00:00,000")


class TestReadBoundariesJsonl(unittest.TestCase):
    """JSONL読込のテスト"""

    def test_basic_read(self):
        """基本的なJSONL読込"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            f.write('{"t_sec": 10.0, "score": 0.95}\n')
            f.write('{"t_sec": 20.5, "score": 0.80}\n')
            f.write('{"t_sec": 5.0}\n')  # scoreなし
            tmp_path = f.name

        try:
            boundaries = read_boundaries_jsonl(tmp_path)
            self.assertEqual(len(boundaries), 3)
            # t_sec昇順ソートされること
            self.assertAlmostEqual(boundaries[0]["t_sec"], 5.0)
            self.assertAlmostEqual(boundaries[1]["t_sec"], 10.0)
            self.assertAlmostEqual(boundaries[2]["t_sec"], 20.5)
            # scoreなしは0.0
            self.assertAlmostEqual(boundaries[0]["score"], 0.0)
        finally:
            Path(tmp_path).unlink()

    def test_empty_lines_skipped(self):
        """空行は無視されること"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            f.write('{"t_sec": 1.0}\n')
            f.write("\n")
            f.write('{"t_sec": 2.0}\n')
            tmp_path = f.name

        try:
            boundaries = read_boundaries_jsonl(tmp_path)
            self.assertEqual(len(boundaries), 2)
        finally:
            Path(tmp_path).unlink()

    def test_missing_t_sec_skipped(self):
        """t_secがない行はスキップされること"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            f.write('{"score": 0.5}\n')
            f.write('{"t_sec": 1.0}\n')
            tmp_path = f.name

        try:
            boundaries = read_boundaries_jsonl(tmp_path)
            self.assertEqual(len(boundaries), 1)
        finally:
            Path(tmp_path).unlink()


class TestBoundariesToSrt(unittest.TestCase):
    """SRT生成のテスト"""

    def test_basic_conversion(self):
        """基本的な変換テスト"""
        boundaries = [
            {"t_sec": 10.0, "score": 0.95},
            {"t_sec": 20.0, "score": 0.80},
        ]
        srt = boundaries_to_srt(boundaries, pad_ms=100)

        # エントリが2つあること
        self.assertIn("1\n", srt)
        self.assertIn("2\n", srt)
        # ヘッダタグ
        self.assertIn("[cut] transnet", srt)
        # JSONにtype, model, scoreが含まれること
        self.assertIn('"type": "cut"', srt)
        self.assertIn('"model": "TransNetV2"', srt)

    def test_padding(self):
        """±pad_msの範囲が正しいこと"""
        boundaries = [{"t_sec": 10.0, "score": 0.9}]
        srt = boundaries_to_srt(boundaries, pad_ms=200)
        # 10.0 - 0.2 = 9.8 → 00:00:09,800
        # 10.0 + 0.2 = 10.2 → 00:00:10,200
        self.assertIn("00:00:09,800", srt)
        self.assertIn("00:00:10,200", srt)

    def test_padding_at_zero(self):
        """t_sec=0付近でも負にならないこと"""
        boundaries = [{"t_sec": 0.05, "score": 0.5}]
        srt = boundaries_to_srt(boundaries, pad_ms=100)
        # max(0, 0.05 - 0.1) = 0.0
        self.assertIn("00:00:00,000", srt)


class TestConvert(unittest.TestCase):
    """変換パイプラインのテスト"""

    def test_end_to_end(self):
        """JSONL → SRTの一気通貫テスト"""
        with tempfile.TemporaryDirectory() as tmpdir:
            in_jsonl = Path(tmpdir) / "boundaries.jsonl"
            out_srt = Path(tmpdir) / "output.srt"

            with open(in_jsonl, "w", encoding="utf-8") as f:
                f.write('{"t_sec": 615.2, "score": 0.93}\n')
                f.write('{"t_sec": 720.5, "score": 0.87}\n')

            count = convert(str(in_jsonl), str(out_srt), pad_ms=100)
            self.assertEqual(count, 2)
            self.assertTrue(out_srt.exists())

            content = out_srt.read_text(encoding="utf-8")
            self.assertIn("[cut] transnet", content)


if __name__ == "__main__":
    unittest.main()
