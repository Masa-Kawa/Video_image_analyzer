"""
Time formatting utilities for SRT and MLT formats.
Common time handling across all video analysis modules.
"""

import re


# SRT time format: HH:MM:SS,mmm (comma separator)
SRT_TIME_PATTERN = re.compile(r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{3})")


def format_srt_time(seconds: float) -> str:
    """
    Convert seconds to HH:MM:SS,mmm format (SRT standard).

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string

    Example:
        >>> format_srt_time(3661.5)
        '01:01:01,500'
    """
    if seconds < 0:
        seconds = 0.0
    # 総ミリ秒から再計算し、ミリ秒の繰り上がり（例: 1.9995s→2000ms）を
    # 秒・分・時へ正しく伝播させる（クランプによる -1秒のズレを防ぐ）。
    total_ms = int(round(seconds * 1000))
    h, rem = divmod(total_ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def parse_srt_time(time_str: str) -> float:
    """
    Parse HH:MM:SS,mmm format to seconds.

    Args:
        time_str: Formatted time string

    Returns:
        Time in seconds

    Raises:
        ValueError: If time format is invalid
    """
    match = SRT_TIME_PATTERN.match(time_str.strip())
    if not match:
        raise ValueError(f"Invalid SRT time format: {time_str}")
    h, m, s, ms = (int(g) for g in match.groups())
    return h * 3600 + m * 60 + s + ms / 1000.0


def format_mlt_time(seconds: float) -> str:
    """
    Convert seconds to HH:MM:SS.mmm format (MLT standard).

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string

    Example:
        >>> format_mlt_time(3661.5)
        '01:01:01.500'
    """
    if seconds < 0:
        seconds = 0.0
    # SRT 同様、総ミリ秒から再計算してミリ秒の繰り上がりを正しく伝播させる。
    total_ms = int(round(seconds * 1000))
    h, rem = divmod(total_ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def seconds_to_frames(seconds: float, fps: float) -> int:
    """
    Convert seconds to frame number.

    Args:
        seconds: Time in seconds
        fps: Frames per second (must be > 0)

    Returns:
        Frame number

    Raises:
        ValueError: If fps <= 0
    """
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    return int(seconds * fps)


def frames_to_seconds(frame: int, fps: float) -> float:
    """
    Convert frame number to seconds.

    Args:
        frame: Frame number
        fps: Frames per second (must be > 0)

    Returns:
        Time in seconds

    Raises:
        ValueError: If fps <= 0 (avoids ZeroDivisionError)
    """
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    return frame / fps
