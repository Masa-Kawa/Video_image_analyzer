"""
TransNet境界JSONL → SRT変換モジュール

TransNetV2が出力した境界時刻JSONLを読み込み、
各境界を短い区間（±pad_ms）のSRTエントリに変換する。
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List


# ---------------------------------------------------------------------------
# SRT 時間フォーマット
# ---------------------------------------------------------------------------

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
# JSONL 読込
# ---------------------------------------------------------------------------

def read_boundaries_jsonl(jsonl_path: str) -> List[dict]:
    """
    TransNet境界のJSONLファイルを読み込む。

    各行に {"t_sec": float, "score": float（任意）} を期待する。

    Returns:
        境界辞書のリスト（t_sec昇順ソート済み）
    """
    boundaries = []
    path = Path(jsonl_path)
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"警告: {jsonl_path}:{line_num} JSONパースエラー: {e}",
                      file=sys.stderr)
                continue

            if "t_sec" not in obj:
                print(f"警告: {jsonl_path}:{line_num} 't_sec'フィールドなし",
                      file=sys.stderr)
                continue

            boundaries.append({
                "t_sec": float(obj["t_sec"]),
                "score": float(obj.get("score", 0.0)),
            })

    # 時刻順にソート
    boundaries.sort(key=lambda b: b["t_sec"])
    return boundaries


# ---------------------------------------------------------------------------
# SRT 生成
# ---------------------------------------------------------------------------

def boundaries_to_srt(
    boundaries: List[dict],
    pad_ms: int = 100,
) -> str:
    """
    境界リストをSRT文字列に変換する。

    各境界は t_sec ± pad_ms の短い区間として表現する。

    Args:
        boundaries: 境界辞書のリスト（t_sec, score）
        pad_ms: 表示用パディング（ミリ秒、デフォルト: 100）

    Returns:
        SRT形式の文字列
    """
    pad_sec = pad_ms / 1000.0
    lines: List[str] = []

    for idx, b in enumerate(boundaries, start=1):
        t = b["t_sec"]
        score = b.get("score", 0.0)

        start = max(0.0, t - pad_sec)
        end = t + pad_sec

        start_srt = format_srt_time(start)
        end_srt = format_srt_time(end)

        json_line = json.dumps({
            "type": "cut",
            "model": "TransNetV2",
            "score": round(score, 4),
        }, ensure_ascii=False)

        lines.append(f"{idx}")
        lines.append(f"{start_srt} --> {end_srt}")
        lines.append("[cut] transnet")
        lines.append(json_line)
        lines.append("")  # 空行区切り

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# メイン処理
# ---------------------------------------------------------------------------

def convert(in_jsonl: str, out_srt: str, pad_ms: int = 100) -> int:
    """
    JSONL → SRT変換のメインロジック。

    Returns:
        変換されたエントリ数
    """
    boundaries = read_boundaries_jsonl(in_jsonl)
    srt_text = boundaries_to_srt(boundaries, pad_ms=pad_ms)

    out_path = Path(out_srt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(srt_text, encoding="utf-8")

    print(f"入力 : {in_jsonl} ({len(boundaries)} 境界)")
    print(f"出力 : {out_srt}")
    return len(boundaries)


# ---------------------------------------------------------------------------
# CLI エントリポイント
# ---------------------------------------------------------------------------

def main() -> int:
    """コマンドラインエントリポイント"""
    parser = argparse.ArgumentParser(
        description="TransNet境界JSONL → SRT変換"
    )
    parser.add_argument("--in-jsonl", required=True,
                        help="入力JSONLファイル（TransNet境界）")
    parser.add_argument("--out-srt", required=True,
                        help="出力SRTファイルパス")
    parser.add_argument("--pad-ms", type=int, default=100,
                        help="表示用パディング（ミリ秒、デフォルト: 100）")

    args = parser.parse_args()
    convert(args.in_jsonl, args.out_srt, args.pad_ms)
    return 0


if __name__ == "__main__":
    sys.exit(main())
