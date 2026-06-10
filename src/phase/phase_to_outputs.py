"""
フェーズ認識「変換器」（phase_to_outputs）

先行研究（Cholec80系の手術フェーズ認識。例: TeCNO, Trans-SVNet, LoViT,
Surgformer 等のモデル出力、または Cholec80 公式アノテーション）の
**フレーム単位のフェーズ予測**を読み込み、リポジトリ標準の二層構造
（SRT＋JSONL/CSV）へ変換するツール。

推論モデル本体は実装しない。あくまで「フレーム予測 → 区間（セグメント）」の
整形・時刻解決・フォーマット変換に徹する変換器である。

入力契約（複数形式をサポート）:
  1) Cholec80 公式アノテーション形式
     タブ区切り「Frame<TAB>Phase」（ヘッダ行あり、既定 25fps）。
     7フェーズ taxonomy:
       Preparation / CalotTriangleDissection / ClippingCutting /
       GallbladderDissection / GallbladderPackaging /
       CleaningCoagulation / GallbladderRetraction
  2) 汎用 per-frame 予測 CSV
     列: frame_idx, timestamp_sec(任意), phase_id または phase_name,
         confidence(任意)
  3) 時刻解決
     --video があれば PyAV の PTS から絶対時間を得る（VFR耐性）。
     --video が無ければ --fps で換算する。
  phase_id ↔ phase_name のマッピングは JSON 設定で差し替え可能
  （Cholec80 既定マップ maps/cholec80_phases.json を同梱）。

変換ロジック:
  - 連続する同一フェーズのフレームを 1 区間に統合。
  - フリッカ除去:
      * 最小継続長フィルタ（--min-duration 秒、既定 2.0）
      * 任意の多数決スムージング（--smooth-window フレーム、既定 0=無効）
  - 区間境界・時刻は src.core.time_utils を介して算出。冪等（同入力→同出力）。

出力（二層構造、既存スキーマ準拠）:
  - {stem}_phase.jsonl … 正本。1区間=1行。type="surgical_phase",
        label=phase_name, confidence(任意), source="phase_converter" を含む。
  - {stem}_phase.srt   … 既存の2行構造。1行目=「[phase] <phase_name>」、
        2行目=機械向けJSON（時間フィールドは含めない＝SRTの時刻が正）。
  - {stem}_phase.csv   … 時系列。--level segment/frame/both。

CLI:
  python -m src.phase.phase_to_outputs --in <cholec80.txt|pred.csv> \
      [--video case.mp4 | --fps 25] --outdir out/ \
      [--min-duration 2.0] [--smooth-window 0] \
      [--phase-map maps/cholec80_phases.json] [--level segment]
"""

import argparse
import csv
import json
import logging
import sys
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from statistics import median
from typing import Dict, List, Optional, Tuple

from src.core.time_utils import format_srt_time, frames_to_seconds
from src.tools.jsonl_to_srt import build_json_line, build_tag_line


logger = logging.getLogger(__name__)

VERSION = "1.0.0"

# このイベントタイプは [phase] タグへ対応する（jsonl_to_srt / srt_to_jsonl 準拠）
EVENT_TYPE = "surgical_phase"
SOURCE = "phase_converter"

# 同梱の既定フェーズマップ
DEFAULT_PHASE_MAP = Path(__file__).parent / "maps" / "cholec80_phases.json"


# ---------------------------------------------------------------------------
# フェーズマップ
# ---------------------------------------------------------------------------

@dataclass
class PhaseMap:
    """phase_id ↔ phase_name の双方向マッピングと既定fps。"""

    id_to_name: Dict[int, str]
    name_to_id: Dict[str, int]
    fps: float = 25.0
    name: str = "custom"

    @classmethod
    def load(cls, path: Optional[str]) -> "PhaseMap":
        """JSON設定からマッピングを読み込む。Noneなら同梱の既定マップ。"""
        p = Path(path) if path else DEFAULT_PHASE_MAP
        data = json.loads(Path(p).read_text(encoding="utf-8"))
        phases = data.get("phases", {})
        id_to_name = {int(k): str(v) for k, v in phases.items()}
        # name → id は大小無視で引けるよう正規化キーも保持
        name_to_id = {v: k for k, v in id_to_name.items()}
        return cls(
            id_to_name=id_to_name,
            name_to_id=name_to_id,
            fps=float(data.get("fps", 25.0)),
            name=str(data.get("name", "custom")),
        )

    def resolve_id(self, phase_name: str) -> Optional[int]:
        """フェーズ名からIDを引く（大小無視のフォールバックあり）。"""
        if phase_name in self.name_to_id:
            return self.name_to_id[phase_name]
        lower = {k.lower(): v for k, v in self.name_to_id.items()}
        return lower.get(phase_name.lower())

    def resolve_name(self, phase_id: int) -> str:
        """IDからフェーズ名を引く。未知なら Phase_<id>。"""
        return self.id_to_name.get(phase_id, f"Phase_{phase_id}")


# ---------------------------------------------------------------------------
# フレーム予測レコード
# ---------------------------------------------------------------------------

@dataclass
class FrameRec:
    """1フレームのフェーズ予測。"""

    frame_idx: int
    t_sec: Optional[float]
    phase_name: str
    phase_id: Optional[int]
    confidence: Optional[float] = None


# ---------------------------------------------------------------------------
# 入力パーサ
# ---------------------------------------------------------------------------

def _looks_like_cholec80(first_line: str) -> bool:
    """先頭行が Cholec80 形式（Frame<TAB>Phase ヘッダ）かを判定する。"""
    if "\t" not in first_line:
        return False
    cols = [c.strip().lower() for c in first_line.split("\t")]
    return len(cols) >= 2 and cols[0] == "frame" and cols[1] == "phase"


def detect_format(in_path: str) -> str:
    """入力ファイル形式を 'cholec80' / 'csv' として推定する。

    先頭の非空行のみで判定するため、ファイル全体は読み込まず行単位で
    ストリーミングし、判定でき次第打ち切る（大きなCSVのメモリ消費を回避）。
    """
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                return "cholec80" if _looks_like_cholec80(line) else "csv"
    return "csv"


def parse_cholec80(in_path: str, phase_map: PhaseMap) -> List[FrameRec]:
    """
    Cholec80 公式アノテーション（Frame<TAB>Phase）を読み込む。

    ヘッダ行（Frame/Phase）はスキップする。時刻は未解決（None）のまま返し、
    呼び出し側で --video / --fps により解決する。
    """
    records: List[FrameRec] = []
    text = Path(in_path).read_text(encoding="utf-8")
    for line_num, line in enumerate(text.splitlines(), start=1):
        line = line.rstrip("\n")
        if not line.strip():
            continue
        cols = line.split("\t")
        if len(cols) < 2:
            cols = line.split()  # 空白区切りフォールバック
        if len(cols) < 2:
            continue
        # ヘッダ行はスキップ
        if not cols[0].strip().lstrip("-").isdigit():
            continue
        frame_idx = int(cols[0].strip())
        phase_name = cols[1].strip()
        records.append(FrameRec(
            frame_idx=frame_idx,
            t_sec=None,
            phase_name=phase_name,
            phase_id=phase_map.resolve_id(phase_name),
        ))
    return records


def parse_pred_csv(in_path: str, phase_map: PhaseMap) -> List[FrameRec]:
    """
    汎用 per-frame 予測 CSV を読み込む。

    認識する列:
      frame_idx           （frame_idx か timestamp_sec の少なくとも一方が必須）
      timestamp_sec       （任意。あれば時刻として優先採用。frame_idx が
                            無い場合は行順を frame_idx として用いる）
      phase_name / phase_id（少なくとも一方が必須）
      confidence          （任意）
    """
    records: List[FrameRec] = []
    with open(in_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields = {h.strip(): h for h in (reader.fieldnames or [])}

        def col(*names: str) -> Optional[str]:
            for n in names:
                if n in fields:
                    return fields[n]
            return None

        c_frame = col("frame_idx", "frame", "Frame")
        c_time = col("timestamp_sec", "t_sec", "time_sec")
        c_name = col("phase_name", "Phase", "phase")
        c_id = col("phase_id", "label", "class_id")
        c_conf = col("confidence", "conf", "score")

        if c_frame is None and c_time is None:
            raise ValueError(
                f"CSVに frame_idx / timestamp_sec のどちらもありません: "
                f"{in_path} (検出列: {list(fields)})"
            )
        if c_name is None and c_id is None:
            raise ValueError(
                f"CSVに phase_name / phase_id 列のどちらもありません: {in_path}"
            )

        for auto_idx, row in enumerate(reader):
            # frame_idx: 明示列があれば使用、無ければ行順
            if c_frame is not None and (row.get(c_frame) or "").strip():
                frame_idx = int(float(row[c_frame].strip()))
            else:
                frame_idx = auto_idx

            phase_id: Optional[int] = None
            phase_name: Optional[str] = None
            if c_name is not None and (row.get(c_name) or "").strip():
                phase_name = row[c_name].strip()
                phase_id = phase_map.resolve_id(phase_name)
            elif c_id is not None and (row.get(c_id) or "").strip():
                phase_id = int(float(row[c_id].strip()))
                phase_name = phase_map.resolve_name(phase_id)
            else:
                continue  # フェーズ情報なしの行はスキップ

            t_sec: Optional[float] = None
            if c_time is not None and (row.get(c_time) or "").strip():
                t_sec = float(row[c_time].strip())

            confidence: Optional[float] = None
            if c_conf is not None and (row.get(c_conf) or "").strip():
                confidence = float(row[c_conf].strip())

            records.append(FrameRec(
                frame_idx=frame_idx,
                t_sec=t_sec,
                phase_name=phase_name,
                phase_id=phase_id,
                confidence=confidence,
            ))
    return records


# ---------------------------------------------------------------------------
# 時刻解決
# ---------------------------------------------------------------------------

def build_pts_index(video_path: str) -> List[float]:
    """
    動画の各フレームのPTS（秒）を昇順で返す（VFR耐性）。

    パケットレベルで demux して PTS を収集するため、全フレームをデコード
    する必要がなく高速（長時間・高解像度動画でのオーバーヘッドを回避）。
    Bフレーム等で PTS が demux 順と前後しうるため、presentation order に
    なるよう昇順ソートして返す。

    Returns:
        フレーム位置（presentation order）→ 絶対時刻（秒）のリスト
    """
    import av

    container = av.open(video_path)
    try:
        stream = container.streams.video[0]
        time_base = float(stream.time_base)
        # demux はストリーム末尾で pts=None のフラッシュパケットを返すため除外。
        pts_list: List[float] = [
            packet.pts * time_base
            for packet in container.demux(stream)
            if packet.pts is not None
        ]
    finally:
        container.close()
    pts_list.sort()
    return pts_list


def resolve_times(
    records: List[FrameRec],
    video_path: Optional[str],
    fps: float,
) -> None:
    """
    時刻未解決（t_sec is None）のフレームに絶対時刻を割り当てる（in-place）。

    優先順位:
      1) レコードが既に timestamp_sec を持つ → そのまま
      2) --video あり → PyAV PTS（範囲外フレームは fps で換算）
      3) それ以外 → frame_idx / fps
    """
    pts_list: Optional[List[float]] = None
    if video_path:
        pts_list = build_pts_index(video_path)

    fallback_count = 0
    for rec in records:
        if rec.t_sec is not None:
            continue
        if pts_list is not None and 0 <= rec.frame_idx < len(pts_list):
            rec.t_sec = pts_list[rec.frame_idx]
        else:
            # --video 指定時に PTS インデックス範囲外だった場合は fps 換算へ
            # フォールバックする。ユーザーが PTS 解決を期待している可能性が
            # あるため件数をまとめて警告する。
            if pts_list is not None:
                fallback_count += 1
            rec.t_sec = frames_to_seconds(rec.frame_idx, fps)

    if fallback_count:
        logger.warning(
            "PTS解決: %d 件のフレームが PTS インデックス範囲外（PTS数=%d）のため "
            "fps=%g での換算にフォールバックしました。frame_idx が1始まり、または "
            "動画と予測ファイルの不一致の可能性があります。",
            fallback_count, len(pts_list), fps,
        )


# ---------------------------------------------------------------------------
# 区間化（セグメンテーション）
# ---------------------------------------------------------------------------

@dataclass
class Segment:
    """同一フェーズの連続区間。"""

    phase_name: str
    phase_id: Optional[int]
    start: float
    end: float
    confidences: List[float] = field(default_factory=list)
    frame_count: int = 0

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def mean_confidence(self) -> Optional[float]:
        if not self.confidences:
            return None
        return sum(self.confidences) / len(self.confidences)


def smooth_frames(
    records: List[FrameRec],
    window: int,
    phase_map: PhaseMap,
) -> List[FrameRec]:
    """
    多数決スムージングでフリッカ（孤立した誤分類）を除去する。

    各フレームを、中心 window フレームの多数決ラベルで置き換える。
    タイは「自分のラベルが候補に含まれればそれを維持、なければ
    アルファベット順で最小」で決定的に解決する。

    ラベルが置き換えられたフレームの confidence は、元の値が新ラベルに
    対する確信度を表さなくなるため None にリセットする（ラベル不変の
    フレームは元の confidence を保持する）。

    window <= 1 なら何もしない。
    """
    if window <= 1 or not records:
        return records

    half = window // 2
    phases = [r.phase_name for r in records]
    out: List[FrameRec] = []
    for idx, rec in enumerate(records):
        lo = max(0, idx - half)
        hi = min(len(records), idx + half + 1)
        counts = Counter(phases[lo:hi])
        max_count = max(counts.values())
        candidates = [p for p, c in counts.items() if c == max_count]
        if rec.phase_name in candidates:
            chosen = rec.phase_name
        else:
            chosen = sorted(candidates)[0]
        if chosen == rec.phase_name:
            out.append(rec)
        else:
            # ラベルが変わったので元の confidence は無効。None にリセットする。
            out.append(replace(
                rec,
                phase_name=chosen,
                phase_id=phase_map.resolve_id(chosen),
                confidence=None,
            ))
    return out


def _estimate_period(records: List[FrameRec], fps: float) -> float:
    """フレーム間隔（秒）を時刻列の中央値から推定する。"""
    times = [r.t_sec for r in records if r.t_sec is not None]
    diffs = [b - a for a, b in zip(times, times[1:]) if b > a]
    if diffs:
        return median(diffs)
    return 1.0 / fps if fps > 0 else 1.0


def build_segments(records: List[FrameRec], period: float) -> List[Segment]:
    """
    連続する同一フェーズのフレームを区間へ統合する。

    区間は連続（seg[i].end == seg[i+1].start）になるように、各区間の終端を
    次区間の開始時刻に合わせる。最終区間のみ最後のフレーム時刻 + period。
    """
    segments: List[Segment] = []
    n = len(records)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and records[j + 1].phase_name == records[i].phase_name:
            j += 1

        start_t = records[i].t_sec
        if j + 1 < n:
            end_t = records[j + 1].t_sec
        else:
            end_t = records[j].t_sec + period

        confidences = [
            r.confidence for r in records[i:j + 1] if r.confidence is not None
        ]
        segments.append(Segment(
            phase_name=records[i].phase_name,
            phase_id=records[i].phase_id,
            start=start_t,
            end=end_t,
            confidences=confidences,
            frame_count=j - i + 1,
        ))
        i = j + 1
    return segments


def _coalesce_adjacent(segments: List[Segment]) -> List[Segment]:
    """隣接する同一フェーズの区間をマージする（時刻は連続前提）。"""
    if not segments:
        return []
    merged: List[Segment] = [segments[0]]
    for seg in segments[1:]:
        last = merged[-1]
        if seg.phase_name == last.phase_name:
            last.end = seg.end
            last.confidences = last.confidences + seg.confidences
            last.frame_count += seg.frame_count
        else:
            merged.append(seg)
    return merged


def enforce_min_duration(
    segments: List[Segment],
    min_duration: float,
) -> List[Segment]:
    """
    最小継続長フィルタ。min_duration 秒未満の区間を隣接区間へ吸収する。

    最短の短区間を選び、直前区間（無ければ直後区間）のフェーズに吸収させ、
    同一フェーズになった隣接区間を再統合する。決定的・冪等。
    """
    if min_duration <= 0 or len(segments) <= 1:
        return segments

    segs = [replace(s, confidences=list(s.confidences)) for s in segments]

    while len(segs) > 1:
        # 最短かつ閾値未満の区間を決定的に選ぶ（duration→index 昇順）
        target = None
        target_dur = None
        for i, s in enumerate(segs):
            if s.duration < min_duration - 1e-9:
                if target is None or s.duration < target_dur:
                    target = i
                    target_dur = s.duration
        if target is None:
            break

        i = target
        if i > 0:
            prev = segs[i - 1]
            prev.end = segs[i].end
            prev.confidences = prev.confidences + segs[i].confidences
            prev.frame_count += segs[i].frame_count
            del segs[i]
        else:
            nxt = segs[i + 1]
            nxt.start = segs[i].start
            nxt.confidences = segs[i].confidences + nxt.confidences
            nxt.frame_count += segs[i].frame_count
            del segs[i]

        segs = _coalesce_adjacent(segs)

    return segs


# ---------------------------------------------------------------------------
# イベント（JSONL正本レコード）構築
# ---------------------------------------------------------------------------

def segments_to_events(segments: List[Segment]) -> List[dict]:
    """区間リストをJSONL正本レコード（dict）のリストへ変換する。"""
    events: List[dict] = []
    for seg in segments:
        ev: dict = {
            "type": EVENT_TYPE,
            "phase_id": seg.phase_id,
            "phase_name": seg.phase_name,
            "label": seg.phase_name,
            "source": SOURCE,
            "start_sec": round(seg.start, 3),
            "end_sec": round(seg.end, 3),
            "start_srt": format_srt_time(seg.start),
            "end_srt": format_srt_time(seg.end),
            "duration_sec": round(seg.duration, 3),
        }
        conf = seg.mean_confidence
        if conf is not None:
            ev["confidence"] = round(conf, 4)
        events.append(ev)
    return events


# ---------------------------------------------------------------------------
# 出力ライタ
# ---------------------------------------------------------------------------

def write_jsonl(events: List[dict], path: Path) -> None:
    """JSONL正本を書き出す。1区間=1行。"""
    with open(path, "w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")


def write_srt(events: List[dict], path: Path) -> None:
    """
    SRTを書き出す（既存の2行構造）。

    1行目: 人間向けタグ「[phase] <phase_name>」
    2行目: 機械向けJSON（時間フィールドは含めない＝SRTの時刻が正）

    jsonl_to_srt のヘルパを再利用してタグ行/JSON行を生成する。
    """
    lines: List[str] = []
    for idx, ev in enumerate(events, start=1):
        lines.append(str(idx))
        lines.append(f"{ev['start_srt']} --> {ev['end_srt']}")
        lines.append(build_tag_line(ev))
        lines.append(build_json_line(ev))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_segment_csv(events: List[dict], path: Path) -> None:
    """区間（セグメント）単位の時系列CSVを書き出す。"""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "segment_id", "phase_id", "phase_name",
            "start_sec", "start_srt", "end_sec", "end_srt",
            "duration_sec", "confidence",
        ])
        for i, ev in enumerate(events, start=1):
            writer.writerow([
                i,
                ev.get("phase_id", ""),
                ev["phase_name"],
                f"{ev['start_sec']:.3f}",
                ev["start_srt"],
                f"{ev['end_sec']:.3f}",
                ev["end_srt"],
                f"{ev['duration_sec']:.3f}",
                "" if ev.get("confidence") is None else f"{ev['confidence']:.4f}",
            ])


def write_frame_csv(records: List[FrameRec], path: Path) -> None:
    """フレーム単位の時系列CSVを書き出す。"""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame_idx", "t_sec", "t_srt",
            "phase_id", "phase_name", "confidence",
        ])
        for rec in records:
            t = rec.t_sec if rec.t_sec is not None else 0.0
            writer.writerow([
                rec.frame_idx,
                f"{t:.3f}",
                format_srt_time(t),
                "" if rec.phase_id is None else rec.phase_id,
                rec.phase_name,
                "" if rec.confidence is None else f"{rec.confidence:.4f}",
            ])


# ---------------------------------------------------------------------------
# 変換コア
# ---------------------------------------------------------------------------

def build_events(
    in_path: str,
    phase_map: PhaseMap,
    video_path: Optional[str] = None,
    fps: Optional[float] = None,
    min_duration: float = 2.0,
    smooth_window: int = 0,
) -> Tuple[List[dict], List[FrameRec]]:
    """
    入力ファイルを読み込み、整形済みのイベント列とフレーム列を返す。

    ファイルI/Oを伴わない純粋関数（テスト・BaseAnalyzer連携に利用）。

    Returns:
        (events, frames)
    """
    if fps is None:
        fps = phase_map.fps

    fmt = detect_format(in_path)
    if fmt == "cholec80":
        records = parse_cholec80(in_path, phase_map)
    else:
        records = parse_pred_csv(in_path, phase_map)

    # frame_idx 昇順（安定ソート）
    records.sort(key=lambda r: r.frame_idx)

    # 時刻解決
    resolve_times(records, video_path, fps)

    if not records:
        return [], []

    # スムージング（フレーム単位の多数決）
    records = smooth_frames(records, smooth_window, phase_map)

    # 区間化 → 最小継続長フィルタ
    period = _estimate_period(records, fps)
    segments = build_segments(records, period)
    segments = enforce_min_duration(segments, min_duration)

    events = segments_to_events(segments)
    return events, records


def convert(
    in_path: str,
    outdir: str,
    video_path: Optional[str] = None,
    fps: Optional[float] = None,
    min_duration: float = 2.0,
    smooth_window: int = 0,
    phase_map_path: Optional[str] = None,
    level: str = "segment",
    stem: Optional[str] = None,
) -> dict:
    """
    フェーズ予測ファイルを SRT/JSONL/CSV へ変換するメインロジック。

    Args:
        in_path: 入力（Cholec80 txt または per-frame CSV）
        outdir: 出力ディレクトリ
        video_path: 時刻解決に使う動画（PyAV PTS、VFR耐性）
        fps: 動画が無い場合の換算fps（Noneならフェーズマップのfps）
        min_duration: 最小継続長（秒）
        smooth_window: 多数決スムージング窓（フレーム、0=無効）
        phase_map_path: フェーズマップJSON（Noneなら同梱の既定マップ）
        level: CSVの粒度（segment / frame / both）
        stem: 出力ファイル名の語幹（Noneなら入力ファイル名から）

    Returns:
        生成したファイルパスの辞書
    """
    phase_map = PhaseMap.load(phase_map_path)
    events, records = build_events(
        in_path=in_path,
        phase_map=phase_map,
        video_path=video_path,
        fps=fps,
        min_duration=min_duration,
        smooth_window=smooth_window,
    )

    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    if stem is None:
        stem = Path(in_path).stem

    jsonl_path = out_path / f"{stem}_phase.jsonl"
    srt_path = out_path / f"{stem}_phase.srt"
    csv_path = out_path / f"{stem}_phase.csv"

    write_jsonl(events, jsonl_path)
    write_srt(events, srt_path)

    result = {
        "jsonl": str(jsonl_path),
        "srt": str(srt_path),
        "segments": len(events),
        "frames": len(records),
    }

    if level == "frame":
        write_frame_csv(records, csv_path)
        result["csv"] = str(csv_path)
    elif level == "both":
        write_segment_csv(events, csv_path)
        frame_csv_path = out_path / f"{stem}_phase_frames.csv"
        write_frame_csv(records, frame_csv_path)
        result["csv"] = str(csv_path)
        result["csv_frames"] = str(frame_csv_path)
    else:  # segment（既定）
        write_segment_csv(events, csv_path)
        result["csv"] = str(csv_path)

    print(f"JSONL: {jsonl_path} （正本）")
    print(f"SRT  : {srt_path}")
    print(f"CSV  : {csv_path} （level={level}）")
    print(f"区間数: {len(events)} / フレーム数: {len(records)}")
    return result


# ---------------------------------------------------------------------------
# BaseAnalyzer 連携（pipeline.py から呼べる形）
# ---------------------------------------------------------------------------

def _make_analyzer():
    """
    BaseAnalyzer 互換の変換器を生成する（遅延importで base への依存を局所化）。

    analyze(video_path, in_path=..., ...) でフレーム予測を読み込み、
    区間化した結果を AnalysisResult.results に格納して返す。
    video_path は時刻解決（PTS）にのみ使用する。
    """
    from src.analyzers.base import AnalysisResult, BaseAnalyzer

    class PhaseConverterAnalyzer(BaseAnalyzer):
        """先行研究のフェーズ予測を標準フォーマットへ変換するアナライザ。"""

        def __init__(self):
            super().__init__(name="phase_converter", version=VERSION)

        def analyze(self, video_path: str, **params) -> AnalysisResult:
            in_path = params.get("in_path")
            if not in_path:
                raise ValueError("params['in_path'] が必要です（フェーズ予測ファイル）")

            phase_map = PhaseMap.load(params.get("phase_map"))
            min_duration = params.get("min_duration", 2.0)
            smooth_window = params.get("smooth_window", 0)
            fps = params.get("fps")

            events, records = build_events(
                in_path=in_path,
                phase_map=phase_map,
                video_path=video_path,
                fps=fps,
                min_duration=min_duration,
                smooth_window=smooth_window,
            )

            return AnalysisResult(
                analyzer_type="surgical_phase",
                analyzer_version=self.version,
                parameters={
                    "in_path": in_path,
                    "fps": fps if fps is not None else phase_map.fps,
                    "min_duration": min_duration,
                    "smooth_window": smooth_window,
                    "phase_map": phase_map.name,
                    "source": SOURCE,
                },
                video_info={"path": video_path} if video_path else {},
                results=events,
                metadata={"frames": len(records), "segments": len(events)},
            )

    return PhaseConverterAnalyzer()


def make_analyzer():
    """BaseAnalyzer 互換インスタンスを返す公開ファクトリ。"""
    return _make_analyzer()


# ---------------------------------------------------------------------------
# CLI エントリポイント
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="先行研究のフェーズ認識結果を標準SRT/JSONL/CSVへ変換する変換器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
  # Cholec80 公式アノテーション（25fps、最小2秒）
  python -m src.phase.phase_to_outputs \\
      --in video01-phase.txt --fps 25 --outdir out/

  # per-frame 予測CSV を動画PTSで時刻解決（VFR耐性）
  python -m src.phase.phase_to_outputs \\
      --in pred.csv --video case.mp4 --outdir out/ \\
      --min-duration 2.0 --smooth-window 5 --level both
        """,
    )
    parser.add_argument("--in", dest="in_path", required=True,
                        help="入力ファイル（Cholec80 txt または per-frame CSV）")
    parser.add_argument("--video", default=None,
                        help="時刻解決用の動画（PyAV PTS、VFR耐性）")
    parser.add_argument("--fps", type=float, default=None,
                        help="動画が無い場合の換算fps（既定: フェーズマップのfps=25）")
    parser.add_argument("--outdir", required=True, help="出力ディレクトリ")
    parser.add_argument("--min-duration", type=float, default=2.0,
                        help="最小継続長フィルタ（秒、既定: 2.0）")
    parser.add_argument("--smooth-window", type=int, default=0,
                        help="多数決スムージング窓（フレーム、既定: 0=無効）")
    parser.add_argument("--phase-map", default=None,
                        help="phase_id↔phase_name マップJSON（既定: 同梱Cholec80）")
    parser.add_argument("--level", default="segment",
                        choices=["segment", "frame", "both"],
                        help="CSVの粒度（既定: segment）")
    parser.add_argument("--stem", default=None,
                        help="出力ファイル名の語幹（既定: 入力ファイル名）")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """コマンドラインエントリポイント"""
    parser = build_parser()
    args = parser.parse_args(argv)

    convert(
        in_path=args.in_path,
        outdir=args.outdir,
        video_path=args.video,
        fps=args.fps,
        min_duration=args.min_duration,
        smooth_window=args.smooth_window,
        phase_map_path=args.phase_map,
        level=args.level,
        stem=args.stem,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
