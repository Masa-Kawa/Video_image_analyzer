"""
merge_srt.py のユニットテスト
"""

import tempfile
import unittest
from pathlib import Path

from src.tools.merge_srt import (
    SrtEntry,
    format_srt_time,
    merge,
    merge_srts,
    parse_srt_time,
    read_srt,
    write_srt,
)


class TestSrtTimeFunctions(unittest.TestCase):
    """SRT時刻関数のテスト"""

    def test_format_basic(self):
        self.assertEqual(format_srt_time(0.0), "00:00:00,000")
        self.assertEqual(format_srt_time(61.5), "00:01:01,500")

    def test_parse_basic(self):
        self.assertAlmostEqual(parse_srt_time("00:00:00,000"), 0.0)
        self.assertAlmostEqual(parse_srt_time("00:01:01,500"), 61.5)

    def test_parse_dot_separator(self):
        """ピリオド区切りも受け付けること"""
        self.assertAlmostEqual(parse_srt_time("00:02:38.391"), 158.391)

    def test_roundtrip(self):
        original = 158.391
        formatted = format_srt_time(original)
        parsed = parse_srt_time(formatted)
        self.assertAlmostEqual(original, parsed, places=2)


class TestReadWriteSrt(unittest.TestCase):
    """SRT読み書きのテスト"""

    def _create_srt(self, tmpdir: str, name: str, entries_data: list) -> str:
        """テスト用SRTファイルを作成するヘルパー"""
        entries = [SrtEntry(start=s, end=e, text=t) for s, e, t in entries_data]
        path = str(Path(tmpdir) / name)
        write_srt(entries, path)
        return path

    def test_write_and_read(self):
        """書き込み→読み込みラウンドトリップ"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._create_srt(tmpdir, "test.srt", [
                (0.0, 10.0, "First"),
                (15.0, 25.0, "Second"),
            ])

            loaded = read_srt(path)
            self.assertEqual(len(loaded), 2)
            self.assertEqual(loaded[0].text, "First")
            self.assertEqual(loaded[1].text, "Second")
            self.assertAlmostEqual(loaded[0].start, 0.0, places=2)

    def test_multiline_text(self):
        """複数行テキストの保持"""
        with tempfile.TemporaryDirectory() as tmpdir:
            entries = [SrtEntry(start=0.0, end=5.0, text="[bleed] delta\n{\"type\": \"bleed\"}")]
            path = str(Path(tmpdir) / "multi.srt")
            write_srt(entries, path)

            loaded = read_srt(path)
            self.assertEqual(len(loaded), 1)
            self.assertIn("[bleed]", loaded[0].text)
            self.assertIn('"type"', loaded[0].text)


class TestMergeSrts(unittest.TestCase):
    """SRTマージのテスト"""

    def _create_srt(self, tmpdir: str, name: str, entries_data: list) -> str:
        """テスト用SRTファイルを作成するヘルパー"""
        entries = [SrtEntry(start=s, end=e, text=t) for s, e, t in entries_data]
        path = str(Path(tmpdir) / name)
        write_srt(entries, path)
        return path

    def test_merge_two_files(self):
        """2ファイルのマージ"""
        with tempfile.TemporaryDirectory() as tmpdir:
            srt1 = self._create_srt(tmpdir, "cut.srt", [
                (10.0, 10.2, "[cut] transnet"),
                (30.0, 30.2, "[cut] transnet"),
            ])
            srt2 = self._create_srt(tmpdir, "bleed.srt", [
                (5.0, 15.0, "[bleed] delta"),
                (25.0, 35.0, "[bleed] delta"),
            ])

            merged = merge_srts([srt1, srt2])
            self.assertEqual(len(merged), 4)
            # 時刻順にソートされていること
            starts = [e.start for e in merged]
            self.assertEqual(starts, sorted(starts))

    def test_merge_preserves_text(self):
        """マージ後もテキストが保持されること"""
        with tempfile.TemporaryDirectory() as tmpdir:
            srt1 = self._create_srt(tmpdir, "a.srt", [
                (1.0, 2.0, "Entry A"),
            ])
            srt2 = self._create_srt(tmpdir, "b.srt", [
                (0.5, 1.5, "Entry B"),
            ])

            merged = merge_srts([srt1, srt2])
            texts = [e.text for e in merged]
            self.assertIn("Entry A", texts)
            self.assertIn("Entry B", texts)
            # Bが先（0.5 < 1.0）
            self.assertEqual(merged[0].text, "Entry B")

    def test_end_to_end_merge(self):
        """マージ→ファイル出力の一気通貫テスト"""
        with tempfile.TemporaryDirectory() as tmpdir:
            srt1 = self._create_srt(tmpdir, "cut.srt", [
                (20.0, 20.2, "[cut] transnet"),
            ])
            srt2 = self._create_srt(tmpdir, "bleed.srt", [
                (10.0, 18.0, "[bleed] delta"),
            ])

            out = str(Path(tmpdir) / "merged.srt")
            count = merge(out, [srt1, srt2])
            self.assertEqual(count, 2)

            # 再読込して検証
            loaded = read_srt(out)
            self.assertEqual(len(loaded), 2)
            # bleed(10.0)が先
            self.assertIn("[bleed]", loaded[0].text)


if __name__ == "__main__":
    unittest.main()
