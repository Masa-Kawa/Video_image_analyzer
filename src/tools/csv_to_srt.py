"""
CSV → SRT 変換ツール

赤色率ログCSVの量的データをSRT字幕に変換し、
Shotcutで動画に重ねて測定値を直接確認できるようにする。

使用例:
    python -m src.tools.csv_to_srt \\
        --in-csv out/case001_redlog.csv \\
        --out-srt out/case001_metrics.srt \\
        --columns red_ratio,smooth_delta
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Optional

from src.tools.merge_srt import format_srt_time


# ---------------------------------------------------------------------------
# 列の短縮ラベル
# ---------------------------------------------------------------------------

COLUMN_LABELS = {
    "red_ratio": "red",
    "delta": "Δ",
    "smooth_delta": "Δs",
}

# デフォルトで表示する列
DEFAULT_COLUMNS = ["red_ratio", "smooth_delta"]


# ---------------------------------------------------------------------------
# CSV → SRT 変換
# ---------------------------------------------------------------------------

def csv_to_srt(
    csv_path: str,
    columns: Optional[List[str]] = None,
) -> str:
    """
    赤色率ログCSVをSRT文字列に変換する。

    Args:
        csv_path: 入力CSVファイルパス
        columns: 表示する列名リスト（Noneならデフォルト列）

    Returns:
        SRT形式の文字列
    """
    if columns is None:
        columns = DEFAULT_COLUMNS

    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        return ""

    # 時間間隔を計算（各字幕の表示期間）
    times = [float(r["t_sec"]) for r in rows]

    blocks: List[str] = []
    for idx, row in enumerate(rows):
        t_start = times[idx]

        # 次のサンプルの時刻まで、または最後のサンプルは同じ間隔を使う
        if idx + 1 < len(times):
            t_end = times[idx + 1]
        else:
            # 最後のサンプル: 前のサンプルと同じ間隔を仮定
            if idx > 0:
                t_end = t_start + (times[idx] - times[idx - 1])
            else:
                t_end = t_start + 0.2

        # 字幕テキストを生成
        parts = []
        for col in columns:
            if col in row:
                label = COLUMN_LABELS.get(col, col)
                value = float(row[col])
                parts.append(f"{label}={value:.4f}")

        text = " ".join(parts)

        block = (
            f"{idx + 1}\n"
            f"{format_srt_time(t_start)} --> {format_srt_time(t_end)}\n"
            f"{text}\n"
        )
        blocks.append(block)

    return "\n".join(blocks) + "\n"


def convert(
    in_csv: str,
    out_srt: str,
    columns: Optional[List[str]] = None,
) -> int:
    """
    CSVファイルからSRTファイルを生成する。

    Args:
        in_csv: 入力CSVファイルパス
        out_srt: 出力SRTファイルパス
        columns: 表示する列名リスト

    Returns:
        出力した字幕エントリ数
    """
    srt_content = csv_to_srt(in_csv, columns)
    Path(out_srt).write_text(srt_content, encoding="utf-8")

    # エントリ数のカウント
    count = srt_content.count("-->")
    return count


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    """コマンドラインエントリポイント"""
    parser = argparse.ArgumentParser(
        description="赤色率ログCSVをSRT字幕に変換する"
    )
    parser.add_argument("--in-csv", required=True,
                        help="入力CSVファイル（赤色率ログ）")
    parser.add_argument("--out-srt", required=True,
                        help="出力SRTファイル")
    parser.add_argument("--columns", default=None,
                        help="表示する列名（カンマ区切り、デフォルト: red_ratio,smooth_delta）")

    args = parser.parse_args()

    columns = None
    if args.columns:
        columns = [c.strip() for c in args.columns.split(",")]

    count = convert(args.in_csv, args.out_srt, columns)
    print(f"SRT出力: {args.out_srt} （{count}エントリ）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
