"""
make_proxy.py のユニットテスト

ffmpegコマンドの組み立て、ファイル検索、スキップ判定を検証する。
（実際のエンコードは行わない）
"""

import tempfile
import unittest
from pathlib import Path

from src.tools.make_proxy import (
    build_ffmpeg_command,
    find_video_files,
    make_output_path,
    VIDEO_EXTENSIONS,
)


class TestBuildFfmpegCommand(unittest.TestCase):
    """ffmpegコマンド生成のテスト"""

    def test_default_command(self):
        """デフォルト設定でのコマンド生成"""
        cmd = build_ffmpeg_command("input.mp4", "output.mp4")
        self.assertEqual(cmd[0], "ffmpeg")
        self.assertIn("-i", cmd)
        self.assertIn("input.mp4", cmd)
        self.assertIn("output.mp4", cmd)
        self.assertIn("-an", cmd)  # デフォルト: 音声除去
        self.assertIn("-vf", cmd)
        # scale フィルター
        idx = cmd.index("-vf")
        self.assertEqual(cmd[idx + 1], "scale=800:600")

    def test_custom_size(self):
        """カスタム解像度"""
        cmd = build_ffmpeg_command("in.mp4", "out.mp4", size="1280x720")
        idx = cmd.index("-vf")
        self.assertEqual(cmd[idx + 1], "scale=1280:720")

    def test_aspect_ratio_preserve(self):
        """アスペクト比維持"""
        cmd = build_ffmpeg_command("in.mp4", "out.mp4", size="800:-1")
        idx = cmd.index("-vf")
        self.assertEqual(cmd[idx + 1], "scale=800:-1")

    def test_with_audio(self):
        """音声あり"""
        cmd = build_ffmpeg_command("in.mp4", "out.mp4", no_audio=False)
        self.assertNotIn("-an", cmd)
        self.assertIn("-c:a", cmd)

    def test_crf_value(self):
        """CRF値の反映"""
        cmd = build_ffmpeg_command("in.mp4", "out.mp4", crf=18)
        idx = cmd.index("-crf")
        self.assertEqual(cmd[idx + 1], "18")

    def test_overwrite_flag(self):
        """上書きフラグ（-y）が含まれること"""
        cmd = build_ffmpeg_command("in.mp4", "out.mp4")
        self.assertIn("-y", cmd)


class TestFindVideoFiles(unittest.TestCase):
    """動画ファイル検索のテスト"""

    def test_single_file(self):
        """単一ファイル指定"""
        with tempfile.TemporaryDirectory() as tmpdir:
            mp4 = Path(tmpdir) / "test.mp4"
            mp4.touch()
            result = find_video_files(str(mp4))
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0].name, "test.mp4")

    def test_non_video_file(self):
        """動画でないファイルは無視"""
        with tempfile.TemporaryDirectory() as tmpdir:
            txt = Path(tmpdir) / "test.txt"
            txt.touch()
            result = find_video_files(str(txt))
            self.assertEqual(len(result), 0)

    def test_directory_scan(self):
        """ディレクトリスキャン"""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "a.mp4").touch()
            (Path(tmpdir) / "b.avi").touch()
            (Path(tmpdir) / "c.txt").touch()  # 非動画
            (Path(tmpdir) / "d.mkv").touch()
            result = find_video_files(tmpdir)
            self.assertEqual(len(result), 3)
            names = [f.name for f in result]
            self.assertIn("a.mp4", names)
            self.assertIn("b.avi", names)
            self.assertIn("d.mkv", names)

    def test_empty_directory(self):
        """空ディレクトリ"""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = find_video_files(tmpdir)
            self.assertEqual(len(result), 0)

    def test_sorted_output(self):
        """ファイルがソートされていること"""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "c.mp4").touch()
            (Path(tmpdir) / "a.mp4").touch()
            (Path(tmpdir) / "b.mp4").touch()
            result = find_video_files(tmpdir)
            names = [f.name for f in result]
            self.assertEqual(names, ["a.mp4", "b.mp4", "c.mp4"])

    def test_mts_extension(self):
        """MTSファイルが認識されること"""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "video.MTS").touch()
            result = find_video_files(tmpdir)
            self.assertEqual(len(result), 1)


class TestMakeOutputPath(unittest.TestCase):
    """出力パス生成のテスト"""

    def test_default_suffix(self):
        """デフォルトサフィックス"""
        result = make_output_path(Path("/tmp/case001.mp4"), Path("/tmp/out"))
        self.assertEqual(result, Path("/tmp/out/case001_proxy.mp4"))

    def test_different_extension(self):
        """入力がmp4以外でも出力はmp4"""
        result = make_output_path(Path("/tmp/video.avi"), Path("/tmp/out"))
        self.assertEqual(result, Path("/tmp/out/video_proxy.mp4"))


if __name__ == "__main__":
    unittest.main()
