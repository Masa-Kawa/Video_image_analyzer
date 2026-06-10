"""
time_utils のテスト: フォーマットの繰り上がりと fps ガード。
"""

import pytest

from src.core.time_utils import (
    format_srt_time, format_mlt_time, parse_srt_time,
    seconds_to_frames, frames_to_seconds,
)


class TestFormat:
    def test_srt_basic(self):
        assert format_srt_time(3661.5) == "01:01:01,500"

    def test_mlt_basic(self):
        assert format_mlt_time(3661.5) == "01:01:01.500"

    def test_srt_millisecond_carry(self):
        # 繰り上がりで -1秒のズレが起きないこと
        assert format_srt_time(1.9995) == "00:00:02,000"
        assert format_srt_time(59.9995) == "00:01:00,000"
        assert format_srt_time(3599.9996) == "01:00:00,000"

    def test_mlt_millisecond_carry(self):
        assert format_mlt_time(1.9995) == "00:00:02.000"

    def test_negative_clamped(self):
        assert format_srt_time(-5) == "00:00:00,000"

    def test_roundtrip(self):
        assert abs(parse_srt_time(format_srt_time(123.456)) - 123.456) < 0.001


class TestFpsGuards:
    def test_frames_to_seconds_zero_fps(self):
        with pytest.raises(ValueError):
            frames_to_seconds(10, 0)

    def test_frames_to_seconds_negative_fps(self):
        with pytest.raises(ValueError):
            frames_to_seconds(10, -1)

    def test_seconds_to_frames_zero_fps(self):
        with pytest.raises(ValueError):
            seconds_to_frames(1.0, 0)

    def test_valid_fps_ok(self):
        assert frames_to_seconds(30, 30.0) == 1.0
        assert seconds_to_frames(1.0, 30.0) == 30
