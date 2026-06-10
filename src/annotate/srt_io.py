"""
安定ID付きのSRT読み込み/保存。

既存の SRT⇄JSONL 変換（src.tools.srt_to_jsonl / src.tools.jsonl_to_srt）の
薄いラッパ。各セグメントに一意な ``id`` を持たせることで、元SRTと
人手修正SRTを ID で突き合わせ（ペアリング）できるようにする。

- 読み込み時: メタJSON行に ``id`` が無いセグメントへ採番する
  （自動生成SRTを初めて開いたときにIDが付く）。
- 保存時: ``events_to_srt(include_meta=True)`` でメタJSON行（id含む）を
  必ず出力する。Shotcut 等の従来ツールでもタグ行はそのまま表示できる。

セグメントは ``{id, type, phase_name, start_sec, end_sec, ...}`` の
dict として扱う（専用クラスは設けない）。
"""

import uuid
from pathlib import Path
from typing import List

from src.core.time_utils import format_srt_time
from src.tools.jsonl_to_srt import events_to_srt
from src.tools.srt_to_jsonl import read_srt_to_events

Segment = dict


def new_id() -> str:
    """新しい安定IDを生成する（8桁のhex）。"""
    return uuid.uuid4().hex[:8]


def _ensure_id(seg: Segment) -> Segment:
    """セグメントに id が無ければ採番する（破壊的に付与）。"""
    if not seg.get("id"):
        seg["id"] = new_id()
    return seg


def load_segments(srt_path) -> List[Segment]:
    """
    SRTファイルを読み込み、ID付きセグメントのリストを返す。

    存在しない / 空のパスの場合は空リストを返す（新規作成モード）。
    メタJSON行に id が無いセグメントには読み込み時に採番する。

    Args:
        srt_path: 入力SRTパス（None や未存在も許容）

    Returns:
        start_sec昇順のセグメントdictリスト
    """
    if not srt_path:
        return []
    path = Path(srt_path)
    if not path.exists() or path.stat().st_size == 0:
        return []

    events = read_srt_to_events(str(path))
    for ev in events:
        _ensure_id(ev)
    events.sort(key=lambda e: e.get("start_sec", 0.0))
    return events


def save_segments(segments: List[Segment], srt_path) -> int:
    """
    セグメントを安定ID付きSRT（メタJSON行あり）として保存する。

    各セグメントに id を保証し、start_sec昇順に並べて書き出す。
    start_srt / end_srt はメタ行には含まれない（時刻はSRT本体が正）が、
    JSONL等の下流互換のため start_sec / end_sec から補完しておく。

    Args:
        segments: セグメントdictのリスト
        srt_path: 出力SRTパス

    Returns:
        書き出したセグメント数
    """
    prepared: List[Segment] = []
    for seg in segments:
        seg = _ensure_id(dict(seg))
        start_sec = float(seg.get("start_sec", 0.0))
        end_sec = float(seg.get("end_sec", start_sec))
        seg["start_sec"] = round(start_sec, 3)
        seg["end_sec"] = round(end_sec, 3)
        seg["start_srt"] = format_srt_time(start_sec)
        seg["end_srt"] = format_srt_time(end_sec)
        seg.setdefault("type", "surgical_phase")
        prepared.append(seg)

    prepared.sort(key=lambda e: e.get("start_sec", 0.0))

    srt_text = events_to_srt(prepared, include_meta=True)

    out_path = Path(srt_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # 末尾に改行を付けて一般的なSRTツールとの互換性を保つ
    out_path.write_text(srt_text + "\n", encoding="utf-8")
    return len(prepared)
