"""
DPO的な追加学習用のペア生成。

元SRT（自動生成 = rejected）と人手修正SRT（chosen）を安定IDで突き合わせ、
変更のあったセグメントだけを ``{id, procedure, video, change, rejected, chosen}``
の形でJSONL素材として出力する。

change の種別:
  - "edited"   : id一致かつ ラベル or 時刻が変化
  - "inserted" : 修正後にのみ存在（新規追加） → rejected: null
  - "deleted"  : 元にのみ存在（人手で削除）   → chosen:   null

元SRTが空（新規作成モード）の場合、修正後の全セグメントが "inserted"
（rejected: null = 純ゴールドラベル）として出力される。
"""

from typing import Dict, List, Optional

Segment = dict


def _safe_sec(value) -> float:
    """時刻を float に安全変換する。None・数値変換不能な値は 0.0 に丸める。

    DPO素材生成を止めないため、欠損/不正な時刻でも例外を投げずに既定値へ倒す。
    """
    if value is None:
        return 0.0
    try:
        return round(float(value), 3)
    except (TypeError, ValueError):
        return 0.0


def _view(seg: Optional[Segment]) -> Optional[dict]:
    """ペアに載せるセグメントの正規化ビュー（時刻・ラベルの要点のみ）。"""
    if seg is None:
        return None
    return {
        "phase_name": seg.get("phase_name", seg.get("label")),
        "start_sec": _safe_sec(seg.get("start_sec")),
        "end_sec": _safe_sec(seg.get("end_sec")),
    }


def _changed(orig: Segment, corr: Segment) -> bool:
    """ラベル名 or 時刻（start/end）が変化していれば True。"""
    o, c = _view(orig), _view(corr)
    return o != c


def _index_by_id(segments: List[Segment]) -> Dict[str, Segment]:
    """id -> セグメント の辞書。id 欠落分は無視（呼び出し側で採番済み前提）。"""
    return {seg["id"]: seg for seg in segments if seg.get("id")}


def make_pairs(
    original_segments: List[Segment],
    corrected_segments: List[Segment],
    procedure: str = "",
    video: str = "",
) -> List[dict]:
    """
    元/修正セグメントから変更ペアのリストを生成する。

    Args:
        original_segments: 元（自動生成）セグメント。新規作成時は空でよい。
        corrected_segments: 人手修正後セグメント。
        procedure: 術式識別子（出力に付与）。
        video: 動画識別子/パス（出力に付与）。

    Returns:
        変更のあったセグメントのペアdictリスト
        （start_sec昇順、欠落側は None）。
    """
    orig_by_id = _index_by_id(original_segments)
    corr_by_id = _index_by_id(corrected_segments)

    pairs: List[dict] = []

    # 修正後を基準に edited / inserted を判定
    for seg_id, corr in corr_by_id.items():
        orig = orig_by_id.get(seg_id)
        if orig is None:
            change = "inserted"
        elif _changed(orig, corr):
            change = "edited"
        else:
            continue  # 変更なしは出力しない
        pairs.append({
            "id": seg_id,
            "procedure": procedure,
            "video": video,
            "change": change,
            "rejected": _view(orig),
            "chosen": _view(corr),
        })

    # 元にのみ存在＝削除
    for seg_id, orig in orig_by_id.items():
        if seg_id not in corr_by_id:
            pairs.append({
                "id": seg_id,
                "procedure": procedure,
                "video": video,
                "change": "deleted",
                "rejected": _view(orig),
                "chosen": None,
            })

    def sort_key(p: dict) -> float:
        seg = p["chosen"] or p["rejected"] or {}
        return seg.get("start_sec", 0.0)

    pairs.sort(key=sort_key)
    return pairs
