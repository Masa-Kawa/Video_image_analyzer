"""
SRTマージツール

複数のSRTファイルを読み込み、開始時刻でソートし、
indexを振り直して統合SRTを出力する。
"""

import argparse
import re
import sys
from pathlib import Path
from typing import List, Tuple


# ---------------------------------------------------------------------------
# SRT 時間パース / フォーマット
# ---------------------------------------------------------------------------

def parse_srt_time(time_str: str) -> float:
    """HH:MM:SS,mmm 形式を秒に変換する"""
    pattern = r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{3})"
    match = re.match(pattern, time_str.strip())
    if not match:
        raise ValueError(f"無効なSRT時間形式: {time_str}")
    h, m, s, ms = (int(g) for g in match.groups())
    return h * 3600 + m * 60 + s + ms / 1000.0


def format_srt_time(seconds: float) -> str:
    """秒数を HH:MM:SS,mmm 形式に変換する"""
    if seconds < 0:
        seconds = 0.0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds % 1) * 1000))
    if ms >= 1000:
        ms = 999
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# ---------------------------------------------------------------------------
# SRT エントリ
# ---------------------------------------------------------------------------

class SrtEntry:
    """SRTファイルの1エントリ"""

    def __init__(self, start: float, end: float, text: str):
        self.start = start
        self.end = end
        self.text = text

    def __repr__(self) -> str:
        return f"SrtEntry({self.start:.3f}-{self.end:.3f}, {self.text!r})"


# ---------------------------------------------------------------------------
# SRT 読込
# ---------------------------------------------------------------------------

def read_srt(path: str) -> List[SrtEntry]:
    """
    SRTファイルを読み込んでエントリリストを返す。

    Returns:
        SrtEntryのリスト
    """
    content = Path(path).read_text(encoding="utf-8")
    entries: List[SrtEntry] = []

    # 空行で分割
    blocks = re.split(r"\n\s*\n", content.strip())

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        lines = block.split("\n")
        if len(lines) < 3:
            continue

        # 1行目: インデックス（無視、後で振り直す）
        # 2行目: 時間範囲
        time_match = re.match(r"(.+?)\s*-->\s*(.+)", lines[1].strip())
        if not time_match:
            continue

        try:
            start = parse_srt_time(time_match.group(1))
            end = parse_srt_time(time_match.group(2))
        except ValueError:
            continue

        # 3行目以降: テキスト
        text = "\n".join(lines[2:])
        entries.append(SrtEntry(start=start, end=end, text=text))

    return entries


# ---------------------------------------------------------------------------
# SRT マージ
# ---------------------------------------------------------------------------

def merge_srts(srt_paths: List[str]) -> List[SrtEntry]:
    """
    複数SRTファイルを読み込み、開始時刻でソートしてマージする。

    Args:
        srt_paths: SRTファイルパスのリスト

    Returns:
        ソート済みSrtEntryのリスト
    """
    all_entries: List[SrtEntry] = []
    for path in srt_paths:
        entries = read_srt(path)
        all_entries.extend(entries)
        print(f"読込: {path} ({len(entries)} エントリ)")

    # 開始時刻でソート
    all_entries.sort(key=lambda e: e.start)
    return all_entries


def write_srt(entries: List[SrtEntry], out_path: str) -> None:
    """
    SrtEntryリストをSRTファイルとして書き出す。

    indexは1から振り直す。
    """
    lines: List[str] = []
    for idx, entry in enumerate(entries, start=1):
        lines.append(str(idx))
        lines.append(f"{format_srt_time(entry.start)} --> {format_srt_time(entry.end)}")
        lines.append(entry.text)
        lines.append("")  # 空行区切り

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# メイン処理
# ---------------------------------------------------------------------------

def merge(out_path: str, srt_paths: List[str]) -> int:
    """マージのメインロジック。マージ後のエントリ数を返す。"""
    entries = merge_srts(srt_paths)
    write_srt(entries, out_path)
    print(f"出力: {out_path} ({len(entries)} エントリ)")
    return len(entries)


# ---------------------------------------------------------------------------
# CLI エントリポイント
# ---------------------------------------------------------------------------

def main() -> int:
    """コマンドラインエントリポイント"""
    parser = argparse.ArgumentParser(
        description="複数SRTファイルをマージ"
    )
    parser.add_argument("--out", required=True,
                        help="出力SRTファイルパス")
    parser.add_argument("srt_files", nargs="+",
                        help="マージするSRTファイル群")

    args = parser.parse_args()
    merge(args.out, args.srt_files)
    return 0


if __name__ == "__main__":
    sys.exit(main())
