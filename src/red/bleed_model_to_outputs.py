"""
出血検出「変換器」（bleed_model_to_outputs）

先行研究の出血検出結果（深層学習ベースの出血確率出力、SurgBlood 風の
region+point 検出、MultiBypass140 の術中有害事象(IAE)ラベル＝出血イベント＋
5段階 severity 等）を読み込み、リポジトリ標準の二層構造（SRT＋JSONL/CSV）へ
変換するツール。

推論モデル本体は実装しない。あくまで「フレーム予測 / 区間ラベル → 標準出血
イベント」の整形・時刻解決・フォーマット変換に徹する補完的な変換器である。
既存の src/red/（redlog, bleed_detector, bleed_spread）はヒューリスティックな
一次検出であり、本モジュールはそれらと衝突しない別サフィックス
（_bleed_model）で追加出力する。

入力契約（複数形式をサポート）:
  1) per-frame 出血スコア CSV
     列: frame_idx, timestamp_sec(任意), bleed_prob
         [, x,y,w,h（bbox 任意） | point_x,point_y（出血点 任意）]
  2) 区間ラベル形式（MultiBypass140 IAE 風）
     出血イベントの開始/終了（秒またはフレーム）＋ severity(1..5) を持つ
     JSON または CSV。
  3) 汎用イベント JSON（区間＋種別）
     種別が出血系のレコードのみを変換対象とする（--include-all で全件）。
  時刻解決は ① と同様に --video の PTS（VFR 耐性）優先、無ければ --fps。

変換ロジック:
  - 確率系列の場合: ヒステリシス閾値（--thr-on 既定 0.5 / --thr-off 既定 0.3）で
    出血イベント区間を生成し、最小継続長（--min-duration 既定 1.0 秒）で
    短すぎる区間を除去する。confidence は区間内ピーク確率を採用。
  - severity が入力にあれば保持。region(bbox)/point は payload に pass-through
    （SRT 2行目 JSON と JSONL に格納、SRT 時刻計算には使わない）。
  - 既存 red/ の出血イベントと同じイベント型（type="bleed_candidate"）に揃える。
    冪等（同入力→同出力）。

出力（二層構造、別サフィックスで既存と非衝突）:
  - {stem}_bleed_model.jsonl … 正本。1区間=1行。start/end 時刻, type,
        severity(任意), region/point(任意), confidence(任意),
        source="bleed_model_converter"。
  - {stem}_bleed_model.srt   … 既存の2行構造。1行目=「[bleed] bleeding」
        （severity があれば「[bleed] bleeding(sev=3)」）、
        2行目=機械向けJSON（時間フィールドは含めない＝SRTの時刻が正）。
  - {stem}_bleed_model.csv   … per-frame の bleed_prob（あれば）＋
        イベント該当フラグ/区間IDの時系列。区間入力ではイベント単位の時系列。

CLI:
  python -m src.red.bleed_model_to_outputs --in <pred.csv|iae.json> \\
      [--video case.mp4 | --fps 25] --outdir out/ \\
      [--thr-on 0.5] [--thr-off 0.3] [--min-duration 1.0]
"""

import argparse
import csv
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional, Tuple

from src.core.time_utils import format_srt_time, frames_to_seconds
from src.tools.jsonl_to_srt import build_json_line


logger = logging.getLogger(__name__)

VERSION = "1.0.0"

# 既存 red/ の出血イベントと同じイベント型に揃える（redlog.py 準拠）。
# これにより jsonl_to_srt / srt_to_jsonl / merge_srt と無改変で往復・統合できる。
EVENT_TYPE = "bleed_candidate"
SOURCE = "bleed_model_converter"

# 出血系イベントと判定するキーワード（汎用イベントJSONのフィルタ用）
BLEED_KEYWORDS = ("bleed", "bleeding", "hemorrh", "haemorrh", "出血")

# 入力カラム名のエイリアス
_FRAME_COLS = ("frame_idx", "frame", "Frame", "frame_id")
_TIME_COLS = ("timestamp_sec", "t_sec", "time_sec", "timestamp", "time")
_PROB_COLS = ("bleed_prob", "prob", "probability", "score", "bleed_probability")
_START_SEC_COLS = ("start_sec", "start", "begin", "start_time", "t_start", "onset")
_END_SEC_COLS = ("end_sec", "end", "stop", "end_time", "t_end", "offset")
_START_FRAME_COLS = ("start_frame", "frame_start", "begin_frame")
_END_FRAME_COLS = ("end_frame", "frame_end", "stop_frame")
_SEVERITY_COLS = ("severity", "sev", "grade", "iae_severity")
_LABEL_COLS = ("type", "label", "event", "category", "class", "name")
_CONF_COLS = ("confidence", "conf", "score", "prob", "probability")


# ---------------------------------------------------------------------------
# フレーム予測レコード
# ---------------------------------------------------------------------------

@dataclass
class FrameScore:
    """1フレームの出血確率（と任意の region/point）。"""

    frame_idx: int
    t_sec: Optional[float]
    bleed_prob: float
    region: Optional[Dict[str, float]] = None
    point: Optional[Dict[str, float]] = None


# ---------------------------------------------------------------------------
# ユーティリティ
# ---------------------------------------------------------------------------

def _pick(d: Dict[str, Any], names: Tuple[str, ...]) -> Optional[Any]:
    """辞書から最初に見つかったキーの値を返す（空文字は無視）。

    d が dict でない（None 等）場合は安全に None を返す（呼び出し元の
    isinstance チェック漏れによる TypeError を防ぐ）。
    """
    if not isinstance(d, dict):
        return None
    for n in names:
        if n in d and d[n] is not None and str(d[n]).strip() != "":
            return d[n]
    return None


def _col(fields: Dict[str, str], names: Tuple[str, ...]) -> Optional[str]:
    """CSVヘッダ（正規化名→実名）から最初に一致した実カラム名を返す。"""
    for n in names:
        if n in fields:
            return fields[n]
    return None


def _is_bleeding(label: Optional[str]) -> bool:
    """ラベルが出血系か判定する。ラベル無し（None）は出血として扱う。"""
    if label is None:
        return True
    low = str(label).lower()
    return any(kw in low for kw in BLEED_KEYWORDS)


def _region_from(d: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """辞書から region(bbox) を抽出する。x,y,w,h または bbox=[x,y,w,h]。"""
    bbox = d.get("bbox") if isinstance(d, dict) else None
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        return {"x": float(bbox[0]), "y": float(bbox[1]),
                "w": float(bbox[2]), "h": float(bbox[3])}
    region = d.get("region") if isinstance(d, dict) else None
    if isinstance(region, dict) and all(k in region for k in ("x", "y", "w", "h")):
        return {k: float(region[k]) for k in ("x", "y", "w", "h")}
    if all(_pick(d, (k,)) is not None for k in ("x", "y", "w", "h")):
        return {k: float(_pick(d, (k,))) for k in ("x", "y", "w", "h")}
    return None


def _point_from(d: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """辞書から point(出血点) を抽出する。point_x/point_y または point=[x,y]。"""
    pt = d.get("point") if isinstance(d, dict) else None
    if isinstance(pt, (list, tuple)) and len(pt) == 2:
        return {"x": float(pt[0]), "y": float(pt[1])}
    if isinstance(pt, dict) and "x" in pt and "y" in pt:
        return {"x": float(pt["x"]), "y": float(pt["y"])}
    px = _pick(d, ("point_x", "px", "source_x"))
    py = _pick(d, ("point_y", "py", "source_y"))
    if px is not None and py is not None:
        return {"x": float(px), "y": float(py)}
    return None


# ---------------------------------------------------------------------------
# 入力形式の判定
# ---------------------------------------------------------------------------

def detect_input_kind(in_path: str) -> str:
    """
    入力ファイル形式を推定する。

    Returns:
        "perframe_csv" / "interval_csv" / "interval_json"
    """
    ext = Path(in_path).suffix.lower()
    if ext == ".json":
        return "interval_json"

    # CSV: ヘッダから per-frame 確率系列か区間ラベルかを判定
    with open(in_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header: List[str] = next(reader, [])
    fields = {h.strip(): h.strip() for h in header}

    has_prob = _col(fields, _PROB_COLS) is not None
    has_start = _col(fields, _START_SEC_COLS + _START_FRAME_COLS) is not None
    has_end = _col(fields, _END_SEC_COLS + _END_FRAME_COLS) is not None

    if has_start and has_end:
        return "interval_csv"
    if has_prob:
        return "perframe_csv"
    raise ValueError(
        f"CSVの形式を判定できません: {in_path} "
        f"（bleed_prob 列も start/end 列も見つかりません。検出列: {list(fields)}）"
    )


# ---------------------------------------------------------------------------
# 入力パーサ
# ---------------------------------------------------------------------------

def parse_perframe_csv(in_path: str) -> List[FrameScore]:
    """
    per-frame 出血スコア CSV を読み込む。

    認識する列:
      frame_idx（任意。無ければ行順を採番）
      timestamp_sec（任意。あれば時刻として優先採用）
      bleed_prob（必須）
      x,y,w,h（任意 bbox） / point_x,point_y（任意 出血点）
    """
    frames: List[FrameScore] = []
    with open(in_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields = {h.strip(): h for h in (reader.fieldnames or [])}

        c_frame = _col(fields, _FRAME_COLS)
        c_time = _col(fields, _TIME_COLS)
        c_prob = _col(fields, _PROB_COLS)
        if c_prob is None:
            raise ValueError(
                f"per-frame CSV に bleed_prob 列がありません: {in_path}")

        for auto_idx, row in enumerate(reader):
            row = {(k.strip() if k else k): v for k, v in row.items()}
            prob_raw = _pick(row, (c_prob,))
            if prob_raw is None:
                continue
            if c_frame is not None and str(row.get(c_frame, "")).strip():
                frame_idx = int(float(row[c_frame]))
            else:
                frame_idx = auto_idx

            t_sec: Optional[float] = None
            if c_time is not None and str(row.get(c_time, "")).strip():
                t_sec = float(row[c_time])

            frames.append(FrameScore(
                frame_idx=frame_idx,
                t_sec=t_sec,
                bleed_prob=float(prob_raw),
                region=_region_from(row),
                point=_point_from(row),
            ))
    return frames


def _normalize_interval(rec: Dict[str, Any]) -> Dict[str, Any]:
    """区間レコード（JSON/CSV の生データ）を共通スキーマへ正規化する。"""
    out: Dict[str, Any] = {}

    start_sec = _pick(rec, _START_SEC_COLS)
    end_sec = _pick(rec, _END_SEC_COLS)
    start_frame = _pick(rec, _START_FRAME_COLS)
    end_frame = _pick(rec, _END_FRAME_COLS)

    out["start_sec"] = float(start_sec) if start_sec is not None else None
    out["end_sec"] = float(end_sec) if end_sec is not None else None
    out["start_frame"] = int(float(start_frame)) if start_frame is not None else None
    out["end_frame"] = int(float(end_frame)) if end_frame is not None else None

    sev = _pick(rec, _SEVERITY_COLS)
    out["severity"] = int(float(sev)) if sev is not None else None

    label = _pick(rec, _LABEL_COLS)
    out["label"] = str(label) if label is not None else None

    conf = _pick(rec, _CONF_COLS)
    out["confidence"] = float(conf) if conf is not None else None

    out["region"] = _region_from(rec)
    out["point"] = _point_from(rec)
    return out


def parse_interval_csv(in_path: str) -> List[Dict[str, Any]]:
    """区間ラベル CSV（IAE 風）を読み込み、正規化レコードのリストを返す。"""
    records: List[Dict[str, Any]] = []
    with open(in_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row = {(k.strip() if k else k): v for k, v in row.items()}
            records.append(_normalize_interval(row))
    return records


def parse_interval_json(in_path: str) -> List[Dict[str, Any]]:
    """
    区間ラベル / 汎用イベント JSON を読み込み、正規化レコードのリストを返す。

    受理する構造:
      - イベントオブジェクトの配列
      - {"events"|"annotations"|"intervals"|"labels": [...]} を持つ辞書
      - start/end を持つ単一オブジェクト
    """
    data = json.loads(Path(in_path).read_text(encoding="utf-8"))

    raw: List[Dict[str, Any]]
    if isinstance(data, list):
        raw = [r for r in data if isinstance(r, dict)]
    elif isinstance(data, dict):
        for key in ("events", "annotations", "intervals", "labels"):
            if isinstance(data.get(key), list):
                raw = [r for r in data[key] if isinstance(r, dict)]
                break
        else:
            raw = [data]
    else:
        raise ValueError(f"未対応のJSON構造です: {in_path}")

    return [_normalize_interval(r) for r in raw]


# ---------------------------------------------------------------------------
# 時刻解決（PTS / fps）
# ---------------------------------------------------------------------------

def build_pts_index(video_path: str) -> List[float]:
    """PyAVで全フレームのPTS（秒）をデコード順に取得する（VFR耐性）。"""
    import av

    container = av.open(video_path)
    try:
        stream = container.streams.video[0]
        time_base = float(stream.time_base)
        pts_list: List[float] = []
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            pts_list.append(frame.pts * time_base)
    finally:
        container.close()
    return pts_list


def resolve_frame_times(
    frames: List[FrameScore],
    video_path: Optional[str],
    fps: float,
) -> None:
    """フレームの時刻未解決（t_sec is None）を埋める（in-place）。"""
    pts_list: Optional[List[float]] = None
    if video_path:
        pts_list = build_pts_index(video_path)
    for fr in frames:
        if fr.t_sec is not None:
            continue
        if pts_list is not None and 0 <= fr.frame_idx < len(pts_list):
            fr.t_sec = pts_list[fr.frame_idx]
        else:
            fr.t_sec = frames_to_seconds(fr.frame_idx, fps)


def resolve_interval_times(
    records: List[Dict[str, Any]],
    video_path: Optional[str],
    fps: float,
) -> None:
    """区間レコードの start_sec/end_sec を、フレーム指定からも解決する（in-place）。"""
    pts_list: Optional[List[float]] = None
    needs_pts = any(
        r.get("start_sec") is None or r.get("end_sec") is None for r in records
    )
    if video_path and needs_pts:
        pts_list = build_pts_index(video_path)

    def frame_to_sec(frame: int) -> float:
        if pts_list is not None and 0 <= frame < len(pts_list):
            return pts_list[frame]
        return frames_to_seconds(frame, fps)

    for r in records:
        if r.get("start_sec") is None and r.get("start_frame") is not None:
            r["start_sec"] = frame_to_sec(r["start_frame"])
        if r.get("end_sec") is None and r.get("end_frame") is not None:
            r["end_sec"] = frame_to_sec(r["end_frame"])


# ---------------------------------------------------------------------------
# ヒステリシスによる区間化（確率系列）
# ---------------------------------------------------------------------------

def hysteresis_segments(
    probs: List[float],
    thr_on: float,
    thr_off: float,
) -> List[Tuple[int, int]]:
    """
    ヒステリシス閾値で ON 区間（半開区間 [start, end)）のリストを返す。

    prob >= thr_on で ON、prob < thr_off で OFF。決定的・冪等。
    """
    segments: List[Tuple[int, int]] = []
    in_event = False
    start = 0
    for i, p in enumerate(probs):
        if not in_event:
            if p >= thr_on:
                in_event = True
                start = i
        else:
            if p < thr_off:
                segments.append((start, i))
                in_event = False
    if in_event:
        segments.append((start, len(probs)))
    return segments


def _estimate_period(frames: List[FrameScore], fps: float) -> float:
    """フレーム間隔（秒）を時刻列の中央値から推定する。"""
    times = [f.t_sec for f in frames if f.t_sec is not None]
    diffs = [b - a for a, b in zip(times, times[1:]) if b > a]
    if diffs:
        return median(diffs)
    return 1.0 / fps if fps > 0 else 1.0


def frames_to_events(
    frames: List[FrameScore],
    thr_on: float,
    thr_off: float,
    min_duration: float,
    fps: float,
) -> Tuple[List[Dict[str, Any]], List[Optional[int]]]:
    """
    確率系列のフレーム列を出血イベント区間へ変換する。

    Returns:
        (events, frame_event_ids)
        frame_event_ids[i] = フレーム i が属する 1-based イベントID（無ければ None）。
        min_duration フィルタで除去された区間のフレームは None。
    """
    n = len(frames)
    frame_event_ids: List[Optional[int]] = [None] * n
    if n == 0:
        return [], frame_event_ids

    period = _estimate_period(frames, fps)
    probs = [f.bleed_prob for f in frames]
    segments = hysteresis_segments(probs, thr_on, thr_off)

    events: List[Dict[str, Any]] = []
    for (s, e) in segments:
        start_t = frames[s].t_sec
        end_t = frames[e].t_sec if e < n else frames[e - 1].t_sec + period
        duration = end_t - start_t
        if duration < min_duration - 1e-9:
            continue

        # 区間内ピーク確率のフレームを代表として region/point を pass-through
        peak_i = max(range(s, e), key=lambda i: frames[i].bleed_prob)
        peak = frames[peak_i]

        ev = _make_event(
            start=start_t,
            end=end_t,
            confidence=peak.bleed_prob,
            severity=None,
            region=peak.region,
            point=peak.point,
            label=None,
        )
        events.append(ev)
        event_id = len(events)
        for i in range(s, e):
            frame_event_ids[i] = event_id

    return events, frame_event_ids


# ---------------------------------------------------------------------------
# 区間レコード → イベント
# ---------------------------------------------------------------------------

def intervals_to_events(
    records: List[Dict[str, Any]],
    min_duration: float,
    include_all: bool,
) -> List[Dict[str, Any]]:
    """
    正規化済み区間レコードを出血イベントへ変換する。

    - 出血系（ラベルが出血キーワードを含む or ラベル無し）のみ採用。
      include_all=True で全件採用。
    - start_sec/end_sec が解決済みであることを前提とする。
    - min_duration 未満の区間は除去（既定 0 では何もしない／呼び出し側で制御）。
    - 開始時刻昇順で安定ソート。冪等。
    """
    events: List[Dict[str, Any]] = []
    for r in records:
        if not include_all and not _is_bleeding(r.get("label")):
            continue
        start = r.get("start_sec")
        end = r.get("end_sec")
        if start is None or end is None:
            continue
        if end < start:
            start, end = end, start
        if (end - start) < min_duration - 1e-9:
            continue
        events.append(_make_event(
            start=start,
            end=end,
            confidence=r.get("confidence"),
            severity=r.get("severity"),
            region=r.get("region"),
            point=r.get("point"),
            label=r.get("label"),
        ))

    events.sort(key=lambda ev: (ev["start_sec"], ev["end_sec"]))
    return events


def _make_event(
    start: float,
    end: float,
    confidence: Optional[float],
    severity: Optional[int],
    region: Optional[Dict[str, float]],
    point: Optional[Dict[str, float]],
    label: Optional[str],
) -> Dict[str, Any]:
    """JSONL正本レコード（dict）を構築する。キー命名は既存スキーマ準拠。"""
    ev: Dict[str, Any] = {
        "type": EVENT_TYPE,
        "source": SOURCE,
        "start_sec": round(float(start), 3),
        "end_sec": round(float(end), 3),
        "start_srt": format_srt_time(start),
        "end_srt": format_srt_time(end),
        "duration_sec": round(float(end) - float(start), 3),
    }
    if severity is not None:
        ev["severity"] = int(severity)
    if confidence is not None:
        ev["confidence"] = round(float(confidence), 4)
    if region is not None:
        ev["region"] = region
    if point is not None:
        ev["point"] = point
    if label is not None and label.lower() != EVENT_TYPE:
        ev["label"] = label
    return ev


# ---------------------------------------------------------------------------
# SRT タグ行
# ---------------------------------------------------------------------------

def _bleed_tag_line(event: Dict[str, Any]) -> str:
    """人間向けタグ行を生成する。severity があれば併記する。"""
    sev = event.get("severity")
    if sev is not None:
        return f"[bleed] bleeding(sev={sev})"
    return "[bleed] bleeding"


# ---------------------------------------------------------------------------
# 出力ライタ
# ---------------------------------------------------------------------------

def write_jsonl(events: List[Dict[str, Any]], path: Path) -> None:
    """JSONL正本を書き出す。1区間=1行。"""
    with open(path, "w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")


def write_srt(events: List[Dict[str, Any]], path: Path) -> None:
    """
    SRTを書き出す（既存の2行構造）。

    1行目: 「[bleed] bleeding」（severity があれば「[bleed] bleeding(sev=N)」）
    2行目: 機械向けJSON（時間フィールドは含めない＝SRTの時刻が正）
    """
    lines: List[str] = []
    for idx, ev in enumerate(events, start=1):
        lines.append(str(idx))
        lines.append(f"{ev['start_srt']} --> {ev['end_srt']}")
        lines.append(_bleed_tag_line(ev))
        lines.append(build_json_line(ev))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_frame_csv(
    frames: List[FrameScore],
    frame_event_ids: List[Optional[int]],
    path: Path,
) -> None:
    """per-frame の bleed_prob ＋ イベント該当フラグ/区間IDの時系列CSV。"""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame_idx", "t_sec", "t_srt", "bleed_prob",
            "in_event", "event_id",
        ])
        for fr, eid in zip(frames, frame_event_ids):
            t = fr.t_sec if fr.t_sec is not None else 0.0
            writer.writerow([
                fr.frame_idx,
                f"{t:.3f}",
                format_srt_time(t),
                f"{fr.bleed_prob:.6f}",
                1 if eid is not None else 0,
                "" if eid is None else eid,
            ])


def write_interval_csv(events: List[Dict[str, Any]], path: Path) -> None:
    """区間入力（per-frame 確率なし）向けの、イベント単位の時系列CSV。"""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "event_id", "start_sec", "start_srt", "end_sec", "end_srt",
            "duration_sec", "severity", "confidence",
        ])
        for i, ev in enumerate(events, start=1):
            writer.writerow([
                i,
                f"{ev['start_sec']:.3f}",
                ev["start_srt"],
                f"{ev['end_sec']:.3f}",
                ev["end_srt"],
                f"{ev['duration_sec']:.3f}",
                "" if ev.get("severity") is None else ev["severity"],
                "" if ev.get("confidence") is None else f"{ev['confidence']:.4f}",
            ])


# ---------------------------------------------------------------------------
# 変換コア
# ---------------------------------------------------------------------------

@dataclass
class BuildResult:
    """build_events の結果（ファイルI/Oなし）。"""

    events: List[Dict[str, Any]]
    kind: str
    frames: List[FrameScore] = field(default_factory=list)
    frame_event_ids: List[Optional[int]] = field(default_factory=list)


def build_events(
    in_path: str,
    video_path: Optional[str] = None,
    fps: float = 25.0,
    thr_on: float = 0.5,
    thr_off: float = 0.3,
    min_duration: float = 1.0,
    include_all: bool = False,
) -> BuildResult:
    """
    入力ファイルを読み込み、整形済みイベント列を返す純粋関数。

    テスト・BaseAnalyzer 連携に利用する（ファイルI/Oを伴わない）。
    """
    kind = detect_input_kind(in_path)

    if kind == "perframe_csv":
        frames = parse_perframe_csv(in_path)
        frames.sort(key=lambda fr: fr.frame_idx)
        resolve_frame_times(frames, video_path, fps)
        events, frame_event_ids = frames_to_events(
            frames, thr_on, thr_off, min_duration, fps,
        )
        return BuildResult(events=events, kind=kind, frames=frames,
                           frame_event_ids=frame_event_ids)

    # 区間入力（CSV / JSON）
    if kind == "interval_csv":
        records = parse_interval_csv(in_path)
    else:
        records = parse_interval_json(in_path)
    resolve_interval_times(records, video_path, fps)
    # 区間ラベルはモデル/データセットの確定ラベルとして忠実に変換する。
    # 短区間でも severity 付きの有害事象は保持したいので min_duration は適用しない。
    events = intervals_to_events(records, min_duration=0.0, include_all=include_all)
    return BuildResult(events=events, kind=kind)


def convert(
    in_path: str,
    outdir: str,
    video_path: Optional[str] = None,
    fps: float = 25.0,
    thr_on: float = 0.5,
    thr_off: float = 0.3,
    min_duration: float = 1.0,
    include_all: bool = False,
    stem: Optional[str] = None,
) -> Dict[str, Any]:
    """
    先行研究の出血検出結果を SRT/JSONL/CSV へ変換するメインロジック。

    Args:
        in_path: 入力（per-frame CSV / 区間ラベル JSON・CSV）
        outdir: 出力ディレクトリ
        video_path: 時刻解決に使う動画（PyAV PTS、VFR耐性）
        fps: 動画が無い場合の換算fps（既定 25）
        thr_on: ヒステリシス ON 閾値（確率系列のみ）
        thr_off: ヒステリシス OFF 閾値（確率系列のみ）
        min_duration: 最小継続長（秒、確率系列のみ）
        include_all: 汎用JSONで非出血ラベルも含める
        stem: 出力ファイル名の語幹（Noneなら入力ファイル名から）

    Returns:
        生成したファイルパス等の辞書
    """
    result = build_events(
        in_path=in_path,
        video_path=video_path,
        fps=fps,
        thr_on=thr_on,
        thr_off=thr_off,
        min_duration=min_duration,
        include_all=include_all,
    )

    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    if stem is None:
        stem = Path(in_path).stem

    jsonl_path = out_path / f"{stem}_bleed_model.jsonl"
    srt_path = out_path / f"{stem}_bleed_model.srt"
    csv_path = out_path / f"{stem}_bleed_model.csv"

    write_jsonl(result.events, jsonl_path)
    write_srt(result.events, srt_path)

    if result.kind == "perframe_csv":
        write_frame_csv(result.frames, result.frame_event_ids, csv_path)
    else:
        write_interval_csv(result.events, csv_path)

    logger.info("入力 : %s（形式: %s）", in_path, result.kind)
    logger.info("JSONL: %s （正本）", jsonl_path)
    logger.info("SRT  : %s", srt_path)
    logger.info("CSV  : %s", csv_path)
    logger.info("出血イベント数: %d", len(result.events))

    return {
        "jsonl": str(jsonl_path),
        "srt": str(srt_path),
        "csv": str(csv_path),
        "events": len(result.events),
        "kind": result.kind,
    }


# ---------------------------------------------------------------------------
# BaseAnalyzer 連携（pipeline.py から呼べる形）
# ---------------------------------------------------------------------------

def make_analyzer():
    """BaseAnalyzer 互換インスタンスを返す公開ファクトリ。"""
    from src.analyzers.base import AnalysisResult, BaseAnalyzer

    class BleedModelConverterAnalyzer(BaseAnalyzer):
        """先行研究の出血検出結果を標準フォーマットへ変換するアナライザ。"""

        def __init__(self):
            super().__init__(name="bleed_model_converter", version=VERSION)

        def analyze(self, video_path: str, **params) -> AnalysisResult:
            in_path = params.get("in_path")
            if not in_path:
                raise ValueError(
                    "params['in_path'] が必要です（出血検出の予測/ラベルファイル）")

            fps = params.get("fps", 25.0)
            thr_on = params.get("thr_on", 0.5)
            thr_off = params.get("thr_off", 0.3)
            min_duration = params.get("min_duration", 1.0)
            include_all = params.get("include_all", False)

            result = build_events(
                in_path=in_path,
                video_path=video_path,
                fps=fps,
                thr_on=thr_on,
                thr_off=thr_off,
                min_duration=min_duration,
                include_all=include_all,
            )

            return AnalysisResult(
                analyzer_type="bleeding",
                analyzer_version=self.version,
                parameters={
                    "in_path": in_path,
                    "fps": fps,
                    "thr_on": thr_on,
                    "thr_off": thr_off,
                    "min_duration": min_duration,
                    "source": SOURCE,
                },
                video_info={"path": video_path} if video_path else {},
                results=result.events,
                metadata={"kind": result.kind, "events": len(result.events)},
            )

    return BleedModelConverterAnalyzer()


# ---------------------------------------------------------------------------
# CLI エントリポイント
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="先行研究の出血検出結果を標準SRT/JSONL/CSVへ変換する変換器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
  # per-frame 出血確率CSV をヒステリシスで区間化（動画PTSで時刻解決）
  python -m src.red.bleed_model_to_outputs \\
      --in pred.csv --video case.mp4 --outdir out/ \\
      --thr-on 0.5 --thr-off 0.3 --min-duration 1.0

  # MultiBypass140 IAE 風の区間ラベル（severity 付き）を変換
  python -m src.red.bleed_model_to_outputs \\
      --in iae.json --fps 25 --outdir out/

  # 生成した _bleed_model.srt を既存の _bleed.srt 等と統合
  python -m src.tools.merge_srt --out out/merged.srt \\
      out/case_bleed.srt out/pred_bleed_model.srt
        """,
    )
    parser.add_argument("--in", dest="in_path", required=True,
                        help="入力ファイル（per-frame CSV / 区間ラベル JSON・CSV）")
    parser.add_argument("--video", default=None,
                        help="時刻解決用の動画（PyAV PTS、VFR耐性）")
    parser.add_argument("--fps", type=float, default=25.0,
                        help="動画が無い場合の換算fps（既定: 25）")
    parser.add_argument("--outdir", required=True, help="出力ディレクトリ")
    parser.add_argument("--thr-on", type=float, default=0.5,
                        help="ヒステリシス ON 閾値（確率系列、既定: 0.5）")
    parser.add_argument("--thr-off", type=float, default=0.3,
                        help="ヒステリシス OFF 閾値（確率系列、既定: 0.3）")
    parser.add_argument("--min-duration", type=float, default=1.0,
                        help="最小継続長フィルタ（秒、確率系列、既定: 1.0）")
    parser.add_argument("--include-all", action="store_true",
                        help="汎用JSONで非出血ラベルも変換対象に含める")
    parser.add_argument("--stem", default=None,
                        help="出力ファイル名の語幹（既定: 入力ファイル名）")
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """数値引数の妥当性を検証する。不正なら parser.error で終了する。"""
    if args.fps <= 0:
        parser.error(f"--fps は正の値である必要があります（指定値: {args.fps}）")
    if not (0.0 <= args.thr_off <= args.thr_on <= 1.0):
        parser.error(
            "閾値は 0 <= --thr-off <= --thr-on <= 1 を満たす必要があります "
            f"（thr_on={args.thr_on}, thr_off={args.thr_off}）。"
            "thr_on <= thr_off だと区間が細切れになります。"
        )
    if args.min_duration < 0:
        parser.error(
            f"--min-duration は 0 以上である必要があります（指定値: {args.min_duration}）")


def main(argv: Optional[List[str]] = None) -> int:
    """コマンドラインエントリポイント"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)

    convert(
        in_path=args.in_path,
        outdir=args.outdir,
        video_path=args.video,
        fps=args.fps,
        thr_on=args.thr_on,
        thr_off=args.thr_off,
        min_duration=args.min_duration,
        include_all=args.include_all,
        stem=args.stem,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
