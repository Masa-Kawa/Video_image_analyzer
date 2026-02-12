"""
SRT → JSONL変換モジュール

Shotcutで修正済みのSRTファイルを読み込み、
JSONL形式のイベント正本に変換する。
人手修正した時刻やイベントをJSONLに反映するために使用する。
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Optional

from src.tools.merge_srt import format_srt_time, parse_srt_time


# ---------------------------------------------------------------------------
# タグ行パース
# ---------------------------------------------------------------------------

# タグ行からイベントタイプを推定するパターン
TAG_PATTERNS = {
    re.compile(r"^\[bleed\]"): "bleed_candidate",
    re.compile(r"^\[cut\]"): "cut",
}


def _parse_tag_line(tag_line: str) -> Optional[str]:
    """タグ行からイベントタイプを推定する。不明ならNoneを返す。"""
    tag_line = tag_line.strip()
    for pattern, event_type in TAG_PATTERNS.items():
        if pattern.match(tag_line):
            return event_type
    return None


# ---------------------------------------------------------------------------
# SRT読込 → JSONL変換
# ---------------------------------------------------------------------------

def read_srt_to_events(srt_path: str) -> List[dict]:
    """
    SRTファイルを読み込み、イベント辞書のリストに変換する。

    各SRTエントリは以下の構造を期待する:
        インデックス
        HH:MM:SS,mmm --> HH:MM:SS,mmm
        タグ行（[bleed] ...）
        JSON行（メタデータ）

    Returns:
        イベント辞書のリスト
    """
    content = Path(srt_path).read_text(encoding="utf-8")
    events: List[dict] = []

    # 空行で分割
    blocks = re.split(r"\n\s*\n", content.strip())

    for block_num, block in enumerate(blocks, start=1):
        block = block.strip()
        if not block:
            continue

        lines = block.split("\n")
        if len(lines) < 3:
            print(f"警告: ブロック{block_num}: 行数不足（{len(lines)}行）、"
                  "スキップ", file=sys.stderr)
            continue

        # 1行目: インデックス（無視）
        # 2行目: 時間範囲
        time_match = re.match(r"(.+?)\s*-->\s*(.+)", lines[1].strip())
        if not time_match:
            print(f"警告: ブロック{block_num}: 時間範囲のパースに失敗、"
                  "スキップ", file=sys.stderr)
            continue

        try:
            start_sec = parse_srt_time(time_match.group(1))
            end_sec = parse_srt_time(time_match.group(2))
        except ValueError as e:
            print(f"警告: ブロック{block_num}: {e}、スキップ",
                  file=sys.stderr)
            continue

        # 3行目: タグ行
        tag_line = lines[2].strip()
        tag_type = _parse_tag_line(tag_line)

        # 4行目: JSON行（あれば）
        meta: dict = {}
        if len(lines) >= 4:
            json_line = lines[3].strip()
            try:
                meta = json.loads(json_line)
            except json.JSONDecodeError as e:
                print(f"警告: ブロック{block_num}: JSONパースエラー: {e}",
                      file=sys.stderr)
                # JSON行が壊れていても、時刻とタグ情報で続行

        # イベント辞書を構築
        event: dict = {}

        # メタデータを先に入れる
        event.update(meta)

        # タグ行からのイベントタイプで上書き（SRT編集で変更されている場合）
        if tag_type:
            event["type"] = tag_type

        # 時刻情報をSRTの値で更新（人手修正を反映）
        event["start_sec"] = round(start_sec, 3)
        event["end_sec"] = round(end_sec, 3)
        event["start_srt"] = format_srt_time(start_sec)
        event["end_srt"] = format_srt_time(end_sec)

        events.append(event)

    return events


# ---------------------------------------------------------------------------
# JSONL出力
# ---------------------------------------------------------------------------

def write_events_jsonl(events: List[dict], jsonl_path: str) -> None:
    """イベントリストをJSONLファイルに書き出す。"""
    out_path = Path(jsonl_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# メイン処理
# ---------------------------------------------------------------------------

def convert(in_srt: str, out_jsonl: str) -> int:
    """
    SRT → JSONL変換のメインロジック。

    Args:
        in_srt: 入力SRTファイルパス
        out_jsonl: 出力JSONLファイルパス

    Returns:
        変換されたエントリ数
    """
    events = read_srt_to_events(in_srt)
    write_events_jsonl(events, out_jsonl)

    print(f"入力 : {in_srt} ({len(events)} エントリ)")
    print(f"出力 : {out_jsonl}")
    return len(events)


# ---------------------------------------------------------------------------
# CLI エントリポイント
# ---------------------------------------------------------------------------

def main() -> int:
    """コマンドラインエントリポイント"""
    parser = argparse.ArgumentParser(
        description="SRT → JSONL変換（修正済みSRTをイベント正本に反映）"
    )
    parser.add_argument("--in-srt", required=True,
                        help="入力SRTファイル（修正済み）")
    parser.add_argument("--out-jsonl", required=True,
                        help="出力JSONLファイルパス")

    args = parser.parse_args()
    convert(args.in_srt, args.out_jsonl)
    return 0


if __name__ == "__main__":
    sys.exit(main())
