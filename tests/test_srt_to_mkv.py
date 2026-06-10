"""srt_to_mkv テスト"""

import glob
import os
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from src.tools.srt_to_mkv import (
    _safe_cli_path,
    extract_chapter_name,
    filter_phase_entries,
    format_chapter_time,
    generate_ogm_chapters,
    run_mkvmerge,
    srt_to_mkv,
)
from src.tools.merge_srt import SrtEntry


class TestFilterPhaseEntries:
    def test_filters_phase_only(self):
        entries = [
            SrtEntry(0.0, 10.0, "[phase] Preparation"),
            SrtEntry(10.0, 20.0, "[bleed] delta_over_threshold"),
            SrtEntry(20.0, 30.0, "[phase] Dissection"),
            SrtEntry(30.0, 40.0, "[cut] transnet"),
        ]
        result = filter_phase_entries(entries)
        assert len(result) == 2
        assert result[0].text == "[phase] Preparation"
        assert result[1].text == "[phase] Dissection"

    def test_empty_list(self):
        assert filter_phase_entries([]) == []

    def test_no_phase_entries(self):
        entries = [
            SrtEntry(0.0, 10.0, "[bleed] something"),
            SrtEntry(10.0, 20.0, "[cut] transnet"),
        ]
        assert filter_phase_entries(entries) == []


class TestFormatChapterTime:
    def test_zero(self):
        assert format_chapter_time(0.0) == "00:00:00.000"

    def test_with_hours(self):
        assert format_chapter_time(3661.5) == "01:01:01.500"

    def test_uses_period(self):
        result = format_chapter_time(65.123)
        assert "." in result
        assert "," not in result
        assert result == "00:01:05.123"


class TestExtractChapterName:
    def test_simple(self):
        assert extract_chapter_name("[phase] Preparation") == "Preparation"

    def test_compound_name(self):
        assert extract_chapter_name("[phase] CalotTriangleDissection") == "CalotTriangleDissection"

    def test_strips_whitespace(self):
        assert extract_chapter_name("  [phase]  Observation  ") == "Observation"


class TestGenerateOgmChapters:
    def test_single_entry(self):
        entries = [SrtEntry(0.0, 227.0, "[phase] Preparation")]
        result = generate_ogm_chapters(entries)
        assert "CHAPTER01=00:00:00.000" in result
        assert "CHAPTER01NAME=Preparation" in result

    def test_multiple_entries(self):
        entries = [
            SrtEntry(0.0, 227.0, "[phase] Preparation"),
            SrtEntry(227.0, 500.0, "[phase] Dissection"),
        ]
        result = generate_ogm_chapters(entries)
        assert "CHAPTER01=00:00:00.000" in result
        assert "CHAPTER01NAME=Preparation" in result
        assert "CHAPTER02=00:03:47.000" in result
        assert "CHAPTER02NAME=Dissection" in result

    def test_roundtrip_with_srt_file(self):
        """実際のSRTファイルの読込 → フィルタ → チャプター生成の往復テスト"""
        srt_content = (
            "1\n"
            "00:00:00,000 --> 00:03:47,227\n"
            "[phase] Preparation\n\n"
            "2\n"
            "00:03:47,227 --> 00:08:20,500\n"
            "[phase] CalotTriangleDissection\n\n"
            "3\n"
            "00:08:20,500 --> 00:10:00,000\n"
            "[bleed] delta_over_threshold\n\n"
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".srt", delete=False, encoding="utf-8"
        ) as f:
            f.write(srt_content)
            srt_path = f.name

        try:
            from src.tools.merge_srt import read_srt
            entries = read_srt(srt_path)
            phase_entries = filter_phase_entries(entries)
            assert len(phase_entries) == 2

            chapters = generate_ogm_chapters(phase_entries)
            assert "CHAPTER01=00:00:00.000" in chapters
            assert "CHAPTER01NAME=Preparation" in chapters
            assert "CHAPTER02=00:03:47.227" in chapters
            assert "CHAPTER02NAME=CalotTriangleDissection" in chapters
        finally:
            Path(srt_path).unlink(missing_ok=True)


class TestSafeCliPath:
    def test_dash_relative_is_prefixed(self):
        assert _safe_cli_path("-weird.mp4") == os.path.join(".", "-weird.mp4")

    def test_normal_path_unchanged(self):
        assert _safe_cli_path("video.mp4") == "video.mp4"

    def test_absolute_path_unchanged(self):
        assert _safe_cli_path("/tmp/-x.mp4") == "/tmp/-x.mp4"

    def test_run_mkvmerge_uses_safe_path(self):
        captured = {}

        def fake_run(cmd, capture_output, text):
            captured["cmd"] = cmd
            return mock.Mock(returncode=0, stdout="", stderr="")

        with mock.patch("src.tools.srt_to_mkv.subprocess.run", side_effect=fake_run):
            run_mkvmerge("-evil.mp4", "chap.txt", "out.mkv")

        # 位置引数（最後）が './' で正規化されている
        assert captured["cmd"][-1] == os.path.join(".", "-evil.mp4")


class TestTempFileCleanup:
    def _phase_srt(self, tmpdir):
        p = Path(tmpdir) / "phase.srt"
        p.write_text(
            "1\n00:00:00,000 --> 00:03:47,227\n[phase] Preparation\n\n",
            encoding="utf-8",
        )
        return str(p)

    def test_temp_file_removed_on_mkvmerge_error(self):
        # run_mkvmerge が例外を投げても chapters_*.txt が残らないこと
        with tempfile.TemporaryDirectory() as d:
            srt_path = self._phase_srt(d)
            sys_tmp = tempfile.gettempdir()
            before = set(glob.glob(os.path.join(sys_tmp, "chapters_*.txt")))

            with mock.patch("src.tools.srt_to_mkv.shutil.which",
                            return_value="/usr/bin/mkvmerge"), \
                 mock.patch("src.tools.srt_to_mkv.run_mkvmerge",
                            side_effect=RuntimeError("boom")):
                with pytest.raises(RuntimeError):
                    srt_to_mkv(str(Path(d) / "video.mp4"), srt_path,
                               str(Path(d) / "out.mkv"))

            after = set(glob.glob(os.path.join(sys_tmp, "chapters_*.txt")))
            assert before == after  # 一時ファイルのリークなし

    def test_temp_file_removed_on_success(self):
        with tempfile.TemporaryDirectory() as d:
            srt_path = self._phase_srt(d)
            sys_tmp = tempfile.gettempdir()
            before = set(glob.glob(os.path.join(sys_tmp, "chapters_*.txt")))

            with mock.patch("src.tools.srt_to_mkv.shutil.which",
                            return_value="/usr/bin/mkvmerge"), \
                 mock.patch("src.tools.srt_to_mkv.run_mkvmerge", return_value=0):
                rc = srt_to_mkv(str(Path(d) / "video.mp4"), srt_path,
                                str(Path(d) / "out.mkv"))
            assert rc == 0

            after = set(glob.glob(os.path.join(sys_tmp, "chapters_*.txt")))
            assert before == after