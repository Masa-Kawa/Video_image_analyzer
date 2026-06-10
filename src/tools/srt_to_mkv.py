"""
SRT → チャプター付きMKV変換ツール

フェーズ認識SRT（[phase] エントリ）をOGMチャプター形式に変換し、
mkvmergeでチャプター付きMKVを生成する。
再エンコードなしでコンテナ変換のみ行う。
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List

from src.core.time_utils import format_srt_time
from src.tools.merge_srt import SrtEntry, read_srt


def filter_phase_entries(entries: List[SrtEntry]) -> List[SrtEntry]:
    """[phase]で始まるエントリのみを抽出する。"""
    return [e for e in entries if e.text.strip().startswith("[phase]")]


def format_chapter_time(seconds: float) -> str:
    """秒数をOGMチャプター時刻形式 HH:MM:SS.mmm に変換する。"""
    return format_srt_time(seconds).replace(",", ".")


def extract_chapter_name(text: str) -> str:
    """[phase] タグ行からチャプター名を抽出する。

    >>> extract_chapter_name("[phase] Preparation")
    'Preparation'
    >>> extract_chapter_name("[phase] CalotTriangleDissection")
    'CalotTriangleDissection'
    """
    line = text.strip().split("\n")[0]
    if line.startswith("[phase]"):
        return line[len("[phase]"):].strip()
    return line


def generate_ogm_chapters(entries: List[SrtEntry]) -> str:
    """フェーズエントリからOGMチャプターファイル文字列を生成する。"""
    lines: List[str] = []
    for idx, entry in enumerate(entries, start=1):
        name = extract_chapter_name(entry.text)
        timestamp = format_chapter_time(entry.start)
        lines.append(f"CHAPTER{idx:02d}={timestamp}")
        lines.append(f"CHAPTER{idx:02d}NAME={name}")
    return "\n".join(lines) + "\n"


def _safe_cli_path(path: str) -> str:
    """先頭が '-' のパスがCLIオプションと誤認されるのを防ぐ（引数インジェクション対策）。

    相対パスで '-' 始まりの場合は './' を前置する。絶対パスは '/' 始まりなので安全。
    """
    if path.startswith("-") and not os.path.isabs(path):
        return os.path.join(".", path)
    return path


def run_mkvmerge(video_path: str, chapter_path: str, output_path: str) -> int:
    """mkvmergeを実行してチャプター付きMKVを生成する。

    Returns:
        mkvmergeの終了コード (0=成功, 2=警告付き成功, 1=エラー)
    """
    # 位置引数 video_path が '-foo' のようなファイル名でもオプションと解釈
    # されないよう正規化する。-o / --chapters の値はオプション引数なので安全。
    cmd = [
        "mkvmerge",
        "-o", output_path,
        "--chapters", chapter_path,
        _safe_cli_path(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    return result.returncode


def srt_to_mkv(video_path: str, srt_path: str, output_path: str = "") -> int:
    """SRT → チャプター付きMKV変換のメインロジック。

    Args:
        video_path: 入力動画ファイルパス
        srt_path: フェーズSRTファイルパス
        output_path: 出力MKVパス（省略時: {video_stem}_chapter.mkv）

    Returns:
        終了コード (0=成功)
    """
    if not shutil.which("mkvmerge"):
        print("エラー: mkvmergeが見つかりません。", file=sys.stderr)
        print("インストール: sudo apt install mkvtoolnix", file=sys.stderr)
        return 1

    video = Path(video_path)
    if not output_path:
        output_path = str(video.with_name(video.stem + "_chapter.mkv"))

    entries = read_srt(srt_path)
    phase_entries = filter_phase_entries(entries)

    if not phase_entries:
        print(f"警告: {srt_path} に [phase] エントリが見つかりません。", file=sys.stderr)
        return 1

    print(f"[phase] エントリ: {len(phase_entries)} 件")
    for e in phase_entries:
        print(f"  {format_srt_time(e.start)} --> {format_srt_time(e.end)}  {extract_chapter_name(e.text)}")

    chapters_content = generate_ogm_chapters(phase_entries)

    # チャプターファイル名を先に確保し、書き込み・実行までを単一の try で包む。
    # こうすることで f.write() 等が例外を投げても finally で確実に削除される。
    chapter_file = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", prefix="chapters_",
            delete=False, encoding="utf-8"
        ) as f:
            chapter_file = f.name  # 書き込み前に名前を確保
            f.write(chapters_content)

        ret = run_mkvmerge(str(video), chapter_file, output_path)
    finally:
        if chapter_file:
            Path(chapter_file).unlink(missing_ok=True)

    if ret == 0:
        print(f"完了: {output_path}")
    elif ret == 2:
        print(f"完了（警告あり）: {output_path}")
    else:
        print(f"エラー: mkvmergeが終了コード {ret} で失敗", file=sys.stderr)
        return 1

    return 0


def main() -> int:
    """コマンドラインエントリポイント"""
    parser = argparse.ArgumentParser(
        description="フェーズSRTをチャプターに変換してMKVを生成"
    )
    parser.add_argument("--video", required=True,
                        help="入力動画ファイルパス")
    parser.add_argument("--srt", required=True,
                        help="フェーズSRTファイルパス")
    parser.add_argument("--output", default="",
                        help="出力MKVパス（省略時: {video_stem}_chapter.mkv）")

    args = parser.parse_args()
    return srt_to_mkv(args.video, args.srt, args.output)


if __name__ == "__main__":
    sys.exit(main())