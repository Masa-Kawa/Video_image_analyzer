"""
JSONL → SRT変換モジュール

JSONLイベントファイル（正本）を読み込み、SRT字幕ファイルに変換する。
Shotcutで可視化・人手修正するためのSRTを生成する。
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

from src.tools.merge_srt import format_srt_time


# ---------------------------------------------------------------------------
# タグ行テンプレート
# ---------------------------------------------------------------------------

# イベントタイプに対応するSRTタグ行
TAG_TEMPLATES: Dict[str, str] = {
    "bleed_candidate": "[bleed] delta_over_threshold",
    "bleed_ai_candidate": "[bleed_ai] bleeding_detected",
    "anomaly_candidate": "[anomaly] anomaly_detected",
    "surgical_phase": "[phase] {phase_name}",
    "cut": "[cut] transnet",
    "instrument_scene": "[scene] instrument_change",
}


# ---------------------------------------------------------------------------
# JSONL 読込
# ---------------------------------------------------------------------------

def read_events_jsonl(jsonl_path: str) -> List[dict]:
    """
    イベントJSONLファイルを読み込む。

    各行に start_sec, end_sec を持つJSONオブジェクトを期待する。

    Returns:
        イベント辞書のリスト（start_sec昇順ソート済み）
    """
    events: List[dict] = []
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

            # start_sec / end_sec が必須
            if "start_sec" not in obj or "end_sec" not in obj:
                # TransNet境界形式（t_sec）もサポート
                if "t_sec" in obj:
                    events.append(obj)
                    continue
                print(f"警告: {jsonl_path}:{line_num} "
                      "'start_sec'/'end_sec' フィールドなし",
                      file=sys.stderr)
                continue

            events.append(obj)

    # start_sec昇順でソート（t_secの場合はt_secで）
    def sort_key(ev: dict) -> float:
        return ev.get("start_sec", ev.get("t_sec", 0.0))
    events.sort(key=sort_key)
    return events


# ---------------------------------------------------------------------------
# SRT 生成
# ---------------------------------------------------------------------------

def build_tag_line(event: dict) -> str:
    """イベントからタグ行（人間向け1行目）を生成する。

    SRTの2行構造を生成する公開API。他の変換器（phase/action/bleed_model
    などの *_to_outputs）から再利用される安定インターフェース。
    """
    event_type = event.get("type", "unknown")
    template = TAG_TEMPLATES.get(event_type, f"[{event_type}] event")
    if "{" in template:
        try:
            return template.format(**event)
        except KeyError:
            return template
    return template


def build_json_line(event: dict) -> str:
    """イベントからJSON行（機械向け2行目）を生成する。

    SRTの2行構造を生成する公開API。SRT内のJSON行には時刻情報を含めない
    （SRT自体の時刻が正）。

    他の変換器（*_to_outputs）から再利用される安定インターフェース。
    """
    # 時刻情報を除外したメタデータのみ
    exclude_keys = {"start_sec", "end_sec", "start_srt", "end_srt", "t_sec"}
    meta = {k: v for k, v in event.items() if k not in exclude_keys}
    return json.dumps(meta, ensure_ascii=False)


# 後方互換エイリアス（旧プライベート名。新規コードは公開名を使うこと）
_build_tag_line = build_tag_line
_build_json_line = build_json_line


def events_to_srt(
    events: List[dict],
    event_type: Optional[str] = None,
    include_meta: bool = False,
) -> str:
    """
    イベントリストをSRT文字列に変換する。

    Args:
        events: イベント辞書のリスト
        event_type: フィルタするイベントタイプ（Noneなら全イベント）
        include_meta: Trueの場合、タグ行の後にJSONメタ行を出力する。
            メタ行には時刻以外の全フィールド（id等）が含まれ、
            srt_to_jsonl で読み戻せる（round-trip用）。
            デフォルトFalseで従来の3行ブロック出力を維持する。

    Returns:
        SRT形式の文字列
    """
    # イベントタイプでフィルタ
    if event_type:
        events = [ev for ev in events if ev.get("type") == event_type]

    lines: List[str] = []

    for idx, ev in enumerate(events, start=1):
        # 時刻の取得
        start_sec = ev.get("start_sec", ev.get("t_sec", 0.0))
        end_sec = ev.get("end_sec", start_sec)

        start_srt = format_srt_time(start_sec)
        end_srt = format_srt_time(end_sec)

        tag_line = build_tag_line(ev)

        lines.append(f"{idx}")
        lines.append(f"{start_srt} --> {end_srt}")
        lines.append(tag_line)
        if include_meta:
            lines.append(build_json_line(ev))
        lines.append("")  # 空行区切り

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# メイン処理
# ---------------------------------------------------------------------------

def convert(
    in_jsonl: str,
    out_srt: str,
    event_type: Optional[str] = None,
) -> int:
    """
    JSONL → SRT変換のメインロジック。

    Args:
        in_jsonl: 入力JSONLファイルパス
        out_srt: 出力SRTファイルパス
        event_type: フィルタするイベントタイプ（Noneなら全イベント）

    Returns:
        変換されたエントリ数
    """
    events = read_events_jsonl(in_jsonl)

    if event_type:
        filtered = [ev for ev in events if ev.get("type") == event_type]
    else:
        filtered = events

    srt_text = events_to_srt(events, event_type=event_type)

    out_path = Path(out_srt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(srt_text, encoding="utf-8")

    print(f"入力 : {in_jsonl} ({len(events)} イベント)")
    if event_type:
        print(f"フィルタ: type={event_type} ({len(filtered)} 件)")
    print(f"出力 : {out_srt}")
    return len(filtered)


# ---------------------------------------------------------------------------
# CLI エントリポイント
# ---------------------------------------------------------------------------

def main() -> int:
    """コマンドラインエントリポイント"""
    parser = argparse.ArgumentParser(
        description="JSONL → SRT変換（イベント正本から字幕を生成）"
    )
    parser.add_argument("--in-jsonl", required=True,
                        help="入力JSONLファイル（イベント正本）")
    parser.add_argument("--out-srt", required=True,
                        help="出力SRTファイルパス")
    parser.add_argument("--event-type", default=None,
                        help="フィルタするイベントタイプ"
                             "（例: bleed_candidate, cut）")

    args = parser.parse_args()
    convert(args.in_jsonl, args.out_srt, args.event_type)
    return 0


if __name__ == "__main__":
    sys.exit(main())
