"""
手技・動作認識「変換器」（action_to_outputs）

先行研究の動作/手技認識結果（フレーム単位またはクリップ単位の予測）を読み込み、
リポジトリ標準の二層構造（SRT＋JSONL/CSV）へ変換するツール。

主対象は CholecT50 の triplet〈instrument, verb, target〉認識（Rendezvous 系の
出力等）。加えて、SLAM の action、JIGSAWS gesture、SAR-RARP50 action のような
汎用の per-frame / per-clip ラベルもサポートする。

推論モデル本体は実装しない。あくまで「フレーム/クリップ予測 → 区間（セグメント）」の
整形・時刻解決・フォーマット変換に徹する変換器である。phase_to_outputs /
bleed_model_to_outputs と同じ流儀（時刻解決は PTS 優先、二層出力、冪等）に従う。

入力契約（複数形式をサポート）:
  1) CholecT50 triplet 予測（per-frame）
     a. triplet id（argmax）: 列 frame_idx(任意), timestamp_sec(任意),
        triplet_id, confidence(任意)。同一 frame_idx の複数行で多ラベルも可。
     b. 確率ベクトル（100クラス）: 列 triplet_0 .. triplet_99（または ivt_*）。
        --triplet-thr 以上の列を多ラベル区間として採用する。
     triplet id → (instrument, verb, target) の分解は JSON 設定で差し替え可能で、
     既定は同梱の CholecT50 マップ（6/10/15, 100 triplets）。
  2) 汎用 per-frame / per-clip action
     列: frame_idx または clip_idx, timestamp_sec(任意),
         action_id または action_name, confidence(任意)。
         action_0 .. action_K の確率ベクトル形式も可。
  3) clip 単位ラベル（SLAM 風 7アクション等）
     clip_idx ＋ action ラベル。clip 長（--clip-len / timestamp 間隔 / start,end 列）
     から時刻区間を復元する。
  時刻解決は --video の PTS 優先（VFR 耐性）、無ければ --fps。

変換ロジック:
  - per-frame/clip 予測を、同一ラベルの連続で区間化（ラベルごとに独立）。
  - 多ラベル同時成立（複数 triplet/action の重なり）を許容する区間表現。
  - 最小継続長（--min-duration 既定 0.5 秒）で短区間を除去。
  - 任意スムージング（--smooth-window 既定 0）でフリッカを除去。
  - --decompose 指定時は triplet を instrument / verb / target の別トラック区間
    （type="action", role=...）へ分解して追加出力する。
  - 冪等（同入力→同出力）。

出力（二層構造、既存スキーマ準拠）:
  - {stem}_action.jsonl … 正本。1区間=1行。type（"triplet" / "action"）, label,
        triplet_id / action_id(任意), components{instrument, verb, target}（triplet時。
        任意）, role（分解時）, confidence(任意), 時刻フィールド, source="action_converter"。
  - {stem}_action.srt   … 2行構造。1行目「[action] grasper,retract,gallbladder」
        （triplet は instrument,verb,target をカンマ連結）または「[action] <action_name>」、
        2行目=機械向けJSON（時間フィールドは除外＝SRTの時刻が正）。
  - {stem}_action.csv   … per-frame と per-segment（--level frame|segment|both、既定 segment）。

CLI:
  python -m src.action.action_to_outputs --in <triplet_pred.csv|action.csv> \\
      [--video case.mp4 | --fps 25] --outdir out/ [--decompose] \\
      [--min-duration 0.5] [--smooth-window 0] \\
      [--triplet-map maps/cholect50_triplets.json] [--triplet-thr 0.5] \\
      [--clip-len 1.0] [--level segment]
"""

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional, Tuple

from src.core.time_utils import format_srt_time, frames_to_seconds
from src.tools.jsonl_to_srt import build_json_line


VERSION = "1.0.0"

# JSONL 正本の type 値（SRT は常に [action] タグへ対応する）。
TYPE_TRIPLET = "triplet"
TYPE_ACTION = "action"
SOURCE = "action_converter"

# 同梱の既定 triplet マップ（CholecT50）
DEFAULT_TRIPLET_MAP = Path(__file__).parent / "maps" / "cholect50_triplets.json"

# ---- 入力カラム名のエイリアス ----------------------------------------------
_FRAME_COLS = ("frame_idx", "frame", "Frame", "frame_id")
_CLIP_COLS = ("clip_idx", "clip", "Clip", "clip_id", "segment_idx")
_TIME_COLS = ("timestamp_sec", "t_sec", "time_sec", "timestamp", "time")
_START_COLS = ("start_sec", "start", "t_start", "begin", "onset")
_END_COLS = ("end_sec", "end", "t_end", "stop", "offset")
_TRIPLET_ID_COLS = ("triplet_id", "triplet", "ivt_id", "ivt")
_ACTION_ID_COLS = ("action_id", "label_id", "class_id", "gesture_id")
_ACTION_NAME_COLS = ("action_name", "action", "label", "gesture", "class", "name")
_CONF_COLS = ("confidence", "conf", "score", "prob", "probability")

# 確率ベクトル列（triplet_0.. / ivt_3 / action_2 ..）を検出する正規表現
_TRIPLET_VEC_RE = re.compile(r"^(?:triplet|tri|ivt)[_-]?(\d+)$", re.IGNORECASE)
_ACTION_VEC_RE = re.compile(r"^(?:action|act|class|gesture)[_-]?(\d+)$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# triplet マップ
# ---------------------------------------------------------------------------

@dataclass
class TripletMap:
    """triplet id → (instrument, verb, target) の分解と既定fps。"""

    instruments: List[str]
    verbs: List[str]
    targets: List[str]
    triplets: Dict[int, Tuple[int, int, int]]
    fps: float = 25.0
    name: str = "custom"

    @classmethod
    def load(cls, path: Optional[str]) -> "TripletMap":
        """JSON設定から triplet マップを読み込む。Noneなら同梱の既定（CholecT50）。"""
        p = Path(path) if path else DEFAULT_TRIPLET_MAP
        data = json.loads(Path(p).read_text(encoding="utf-8"))
        triplets: Dict[int, Tuple[int, int, int]] = {}
        for k, v in data.get("triplets", {}).items():
            ivt = tuple(int(x) for x in v)
            if len(ivt) != 3:
                raise ValueError(f"triplet {k} は [i, v, t] の3要素である必要があります: {v}")
            triplets[int(k)] = ivt  # type: ignore[assignment]
        return cls(
            instruments=[str(x) for x in data.get("instruments", [])],
            verbs=[str(x) for x in data.get("verbs", [])],
            targets=[str(x) for x in data.get("targets", [])],
            triplets=triplets,
            fps=float(data.get("fps", 25.0)),
            name=str(data.get("name", "custom")),
        )

    def decompose(self, triplet_id: int) -> Optional[Tuple[str, str, str]]:
        """triplet id を (instrument, verb, target) の名前へ分解する。未知ならNone。"""
        ivt = self.triplets.get(triplet_id)
        if ivt is None:
            return None
        i, v, t = ivt

        def _name(lst: List[str], idx: int, prefix: str) -> str:
            return lst[idx] if 0 <= idx < len(lst) else f"{prefix}_{idx}"

        return (
            _name(self.instruments, i, "instrument"),
            _name(self.verbs, v, "verb"),
            _name(self.targets, t, "target"),
        )

    def label(self, triplet_id: int) -> str:
        """triplet id を表示ラベル（instrument,verb,target）にする。未知なら triplet_<id>。"""
        comp = self.decompose(triplet_id)
        if comp is None:
            return f"triplet_{triplet_id}"
        return ",".join(comp)


# ---------------------------------------------------------------------------
# ラベルキー / 入力ユニット
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LabelKey:
    """区間トラックを一意に識別するキー（ハッシュ可能）。"""

    type: str                                   # "triplet" / "action"
    label: str                                  # 表示ラベル
    label_id: Optional[int] = None              # triplet_id / action_id
    components: Optional[Tuple[str, str, str]] = None  # triplet 分解（任意）
    role: Optional[str] = None                  # 分解トラック: instrument/verb/target

    def sort_key(self) -> Tuple[str, str, str, int]:
        return (self.type, self.role or "", self.label,
                self.label_id if self.label_id is not None else -1)


@dataclass
class Unit:
    """1フレーム/1クリップの予測（多ラベル可）。"""

    idx: int
    t_sec: Optional[float] = None          # 明示時刻（クリップは開始）
    t_end_in: Optional[float] = None       # 明示終了（クリップのみ）
    active: List[Tuple[LabelKey, Optional[float]]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 小道具
# ---------------------------------------------------------------------------

def _col(fields: Dict[str, str], names: Tuple[str, ...]) -> Optional[str]:
    """CSVヘッダ（正規化名→実名）から最初に一致した実カラム名を返す。"""
    for n in names:
        if n in fields:
            return fields[n]
    return None


def _vector_columns(fields: Dict[str, str], pattern: re.Pattern) -> Dict[int, str]:
    """確率ベクトル列（prefix+番号）を {クラスid: 実カラム名} で返す。"""
    out: Dict[int, str] = {}
    for norm, real in fields.items():
        m = pattern.match(norm)
        if m:
            out[int(m.group(1))] = real
    return out


def _to_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    s = str(val).strip()
    if s == "":
        return None
    return float(s)


# ---------------------------------------------------------------------------
# 入力パーサ
# ---------------------------------------------------------------------------

def parse_input(
    in_path: str,
    triplet_map: TripletMap,
    triplet_thr: float,
) -> Tuple[List[Unit], str, str]:
    """
    入力 CSV を読み込み、ユニット列とラベル種別・ユニット種別を返す。

    Returns:
        (units, label_kind, unit_kind)
          label_kind: "triplet" / "action"
          unit_kind:  "frame" / "clip"
    多ラベルは ① 確率ベクトル列を閾値処理、または ② 同一 frame_idx/clip_idx の
    複数行、で表現できる。冪等。
    """
    with open(in_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        fields = {(h.strip() if h else h): h for h in fieldnames}

        c_frame = _col(fields, _FRAME_COLS)
        c_clip = _col(fields, _CLIP_COLS)
        c_time = _col(fields, _TIME_COLS)
        c_start = _col(fields, _START_COLS)
        c_end = _col(fields, _END_COLS)
        c_conf = _col(fields, _CONF_COLS)

        tri_vec = _vector_columns(fields, _TRIPLET_VEC_RE)
        act_vec = _vector_columns(fields, _ACTION_VEC_RE)
        c_tri_id = _col(fields, _TRIPLET_ID_COLS)
        c_act_id = _col(fields, _ACTION_ID_COLS)
        c_act_name = _col(fields, _ACTION_NAME_COLS)

        # --- ラベル種別の判定（triplet を優先） -------------------------
        if tri_vec or c_tri_id is not None:
            label_kind = "triplet"
        elif act_vec or c_act_id is not None or c_act_name is not None:
            label_kind = "action"
        else:
            raise ValueError(
                f"入力の動作ラベル列を判定できません: {in_path} "
                f"（triplet_id / triplet_* / action_id / action_name / action_* "
                f"のいずれも見つかりません。検出列: {list(fields)}）"
            )

        # --- ユニット種別の判定（clip 列 or 明示 start/end → clip） -----
        if c_clip is not None or (c_frame is None and c_start is not None and c_end is not None):
            unit_kind = "clip"
        else:
            unit_kind = "frame"

        c_idx = c_clip if unit_kind == "clip" else c_frame
        # clip の開始時刻は start 列も時刻として受理する
        c_unit_time = c_time or (c_start if unit_kind == "clip" else None)

        # --- 行を読み、ユニット（idx でグルーピング）に集約 -------------
        by_idx: Dict[int, Unit] = {}
        order: List[int] = []
        for auto_idx, row in enumerate(reader):
            row = {(k.strip() if k else k): v for k, v in row.items()}

            if c_idx is not None and str(row.get(c_idx, "")).strip() != "":
                idx = int(float(row[c_idx]))
            else:
                idx = auto_idx  # 明示の index 列が無ければ行順

            unit = by_idx.get(idx)
            if unit is None:
                t_sec = _to_float(row.get(c_unit_time)) if c_unit_time else None
                t_end_in = _to_float(row.get(c_end)) if (c_end and unit_kind == "clip") else None
                unit = Unit(idx=idx, t_sec=t_sec, t_end_in=t_end_in)
                by_idx[idx] = unit
                order.append(idx)

            conf = _to_float(row.get(c_conf)) if c_conf else None

            if label_kind == "triplet":
                _collect_triplet(unit, row, tri_vec, c_tri_id, conf,
                                 triplet_thr, triplet_map)
            else:
                _collect_action(unit, row, act_vec, c_act_id, c_act_name,
                                conf, triplet_thr)

    units = [by_idx[i] for i in sorted(order)]
    return units, label_kind, unit_kind


def _add_active(unit: Unit, key: LabelKey, conf: Optional[float]) -> None:
    """ユニットへアクティブラベルを追加（同一キーは confidence の最大を保持）。"""
    for i, (k, c) in enumerate(unit.active):
        if k == key:
            if conf is not None and (c is None or conf > c):
                unit.active[i] = (k, conf)
            return
    unit.active.append((key, conf))


def _collect_triplet(
    unit: Unit,
    row: Dict[str, Any],
    tri_vec: Dict[int, str],
    c_tri_id: Optional[str],
    conf: Optional[float],
    thr: float,
    tmap: TripletMap,
) -> None:
    """1行から triplet のアクティブラベルを抽出して unit に足す。"""
    def _key(tid: int) -> LabelKey:
        return LabelKey(
            type=TYPE_TRIPLET,
            label=tmap.label(tid),
            label_id=tid,
            components=tmap.decompose(tid),
        )

    if tri_vec:
        for tid, real in tri_vec.items():
            p = _to_float(row.get(real))
            if p is not None and p >= thr:
                _add_active(unit, _key(tid), p)
        return

    if c_tri_id is not None and str(row.get(c_tri_id, "")).strip() != "":
        tid = int(float(row[c_tri_id]))
        # confidence があり閾値未満なら採用しない（ハードラベルは常に採用）
        if conf is not None and conf < thr:
            return
        _add_active(unit, _key(tid), conf)


def _collect_action(
    unit: Unit,
    row: Dict[str, Any],
    act_vec: Dict[int, str],
    c_act_id: Optional[str],
    c_act_name: Optional[str],
    conf: Optional[float],
    thr: float,
) -> None:
    """1行から汎用 action のアクティブラベルを抽出して unit に足す。"""
    if act_vec:
        for aid, real in act_vec.items():
            p = _to_float(row.get(real))
            if p is not None and p >= thr:
                _add_active(unit, LabelKey(TYPE_ACTION, f"action_{aid}", aid), p)
        return

    name: Optional[str] = None
    aid: Optional[int] = None
    if c_act_name is not None and str(row.get(c_act_name, "")).strip() != "":
        name = str(row[c_act_name]).strip()
    if c_act_id is not None and str(row.get(c_act_id, "")).strip() != "":
        aid = int(float(row[c_act_id]))
    if name is None and aid is None:
        return
    if name is None:
        name = f"action_{aid}"
    if conf is not None and conf < thr:
        return
    _add_active(unit, LabelKey(TYPE_ACTION, name, aid), conf)


# ---------------------------------------------------------------------------
# 時刻解決（PTS / fps / clip 長）
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


def _median_period(values: List[float], fps: float) -> float:
    """昇順値列の隣接差の中央値。差が無ければ 1/fps。"""
    vals = sorted(v for v in values if v is not None)
    diffs = [b - a for a, b in zip(vals, vals[1:]) if b > a]
    if diffs:
        return median(diffs)
    return 1.0 / fps if fps > 0 else 1.0


def resolve_frame_times(
    units: List[Unit],
    video_path: Optional[str],
    fps: float,
) -> List[Tuple[float, float]]:
    """
    フレーム単位ユニットへ (t_start, t_end) を割り当てる。

    各フレームは [t_i, t_{i+1}) を占有する（最終フレームのみ t_last + period）。
    phase_to_outputs.build_segments と同じ境界規約。
    """
    n = len(units)
    pts: Optional[List[float]] = build_pts_index(video_path) if video_path else None

    starts: List[float] = []
    for u in units:
        if u.t_sec is not None:
            starts.append(u.t_sec)
        elif pts is not None and 0 <= u.idx < len(pts):
            starts.append(pts[u.idx])
        else:
            starts.append(frames_to_seconds(u.idx, fps))

    period = _median_period(starts, fps)
    times: List[Tuple[float, float]] = []
    for i in range(n):
        s = starts[i]
        e = starts[i + 1] if i + 1 < n else s + period
        if e <= s:
            e = s + period
        times.append((s, e))
    return times


def resolve_clip_times(
    units: List[Unit],
    fps: float,
    clip_len: Optional[float],
) -> List[Tuple[float, float]]:
    """
    クリップ単位ユニットへ (t_start, t_end) を割り当てる。

    優先順位: 明示 start/end → --clip-len → timestamp 間隔の中央値。
    """
    starts = [u.t_sec for u in units]
    ends = [u.t_end_in for u in units]

    length = clip_len
    if length is None:
        known = [s for s in starts if s is not None]
        if len(known) >= 2:
            length = _median_period(known, fps)

    times: List[Tuple[float, float]] = []
    for i, u in enumerate(units):
        s = starts[i]
        e = ends[i]
        if s is None:
            if length is None:
                raise ValueError(
                    "clip 入力の時刻を解決できません: --clip-len か "
                    "timestamp/start 列を指定してください")
            s = u.idx * length
        if e is None:
            if length is None:
                raise ValueError(
                    "clip 入力の時刻を解決できません: --clip-len か end 列が必要です")
            e = s + length
        if e <= s:
            e = s + (length if length else (1.0 / fps if fps > 0 else 1.0))
        times.append((s, e))
    return times


# ---------------------------------------------------------------------------
# 区間化（多ラベル・ラベルごと独立）
# ---------------------------------------------------------------------------

def _smooth_presence(presence: List[bool], window: int) -> List[bool]:
    """ブール在/不在系列を窓幅 window の多数決で平滑化する。

    cnt*2 > total で在。タイ（cnt*2 == total）は元の値を維持し決定的に解決する。
    window <= 1 は何もしない。
    """
    if window <= 1:
        return presence
    n = len(presence)
    half = window // 2
    out: List[bool] = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        cnt = sum(1 for p in presence[lo:hi] if p)
        total = hi - lo
        if cnt * 2 > total:
            out.append(True)
        elif cnt * 2 == total:
            out.append(presence[i])
        else:
            out.append(False)
    return out


def build_presence(
    units: List[Unit],
    decompose: bool,
) -> Tuple[Dict[LabelKey, List[bool]], Dict[LabelKey, List[Optional[float]]]]:
    """
    ユニット列から、ラベルごとの在/不在系列と confidence 系列を構築する。

    decompose=True の場合、triplet を instrument/verb/target の別トラック
    （type="action", role=...）にも展開して同時に在/不在を立てる。
    """
    n = len(units)
    presence: Dict[LabelKey, List[bool]] = {}
    conf: Dict[LabelKey, List[Optional[float]]] = {}

    def _mark(key: LabelKey, pos: int, c: Optional[float]) -> None:
        if key not in presence:
            presence[key] = [False] * n
            conf[key] = [None] * n
        presence[key][pos] = True
        if c is not None and (conf[key][pos] is None or c > conf[key][pos]):
            conf[key][pos] = c

    for pos, u in enumerate(units):
        for key, c in u.active:
            _mark(key, pos, c)
            if decompose and key.type == TYPE_TRIPLET and key.components:
                inst, verb, targ = key.components
                for role, val in (("instrument", inst), ("verb", verb), ("target", targ)):
                    _mark(LabelKey(TYPE_ACTION, val, None, None, role), pos, c)

    return presence, conf


def _segments_for_label(
    key: LabelKey,
    presence: List[bool],
    conf: List[Optional[float]],
    times: List[Tuple[float, float]],
    min_duration: float,
) -> List[dict]:
    """1ラベルの在/不在系列を区間イベント（dict）のリストへ変換する。"""
    events: List[dict] = []
    n = len(presence)
    i = 0
    while i < n:
        if not presence[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and presence[j + 1]:
            j += 1
        start = times[i][0]
        end = times[j][1]
        if (end - start) >= min_duration - 1e-9:
            confs = [conf[k] for k in range(i, j + 1) if conf[k] is not None]
            mean_conf = sum(confs) / len(confs) if confs else None
            events.append(_make_event(key, start, end, mean_conf))
        i = j + 1
    return events


def _make_event(
    key: LabelKey,
    start: float,
    end: float,
    confidence: Optional[float],
) -> dict:
    """JSONL正本レコード（dict）を構築する。キー命名は既存スキーマ準拠。"""
    ev: dict = {
        "type": key.type,
        "label": key.label,
        "source": SOURCE,
        "start_sec": round(float(start), 3),
        "end_sec": round(float(end), 3),
        "start_srt": format_srt_time(start),
        "end_srt": format_srt_time(end),
        "duration_sec": round(float(end) - float(start), 3),
    }
    if key.label_id is not None:
        ev["triplet_id" if key.type == TYPE_TRIPLET else "action_id"] = key.label_id
    if key.type == TYPE_TRIPLET and key.components is not None:
        inst, verb, targ = key.components
        ev["components"] = {"instrument": inst, "verb": verb, "target": targ}
    if key.role is not None:
        ev["role"] = key.role
    if confidence is not None:
        ev["confidence"] = round(float(confidence), 4)
    return ev


# ---------------------------------------------------------------------------
# 変換コア（ファイルI/Oなし）
# ---------------------------------------------------------------------------

@dataclass
class BuildResult:
    """build_events の結果（ファイルI/Oを伴わない）。"""

    events: List[dict]
    units: List[Unit]
    times: List[Tuple[float, float]]
    presence: Dict[LabelKey, List[bool]]
    conf: Dict[LabelKey, List[Optional[float]]]
    label_kind: str
    unit_kind: str


def build_events(
    in_path: str,
    triplet_map: TripletMap,
    video_path: Optional[str] = None,
    fps: Optional[float] = None,
    min_duration: float = 0.5,
    smooth_window: int = 0,
    triplet_thr: float = 0.5,
    clip_len: Optional[float] = None,
    decompose: bool = False,
) -> BuildResult:
    """
    入力ファイルを読み込み、整形済みイベント列（区間）を返す純粋関数。

    テスト・BaseAnalyzer 連携に利用する。返す events は (start_sec, end_sec,
    type, role, label) で安定ソート済み。冪等。
    """
    if fps is None:
        fps = triplet_map.fps

    units, label_kind, unit_kind = parse_input(in_path, triplet_map, triplet_thr)
    if not units:
        return BuildResult([], [], [], {}, {}, label_kind, unit_kind)

    if unit_kind == "clip":
        times = resolve_clip_times(units, fps, clip_len)
    else:
        times = resolve_frame_times(units, video_path, fps)

    presence, conf = build_presence(units, decompose)

    # スムージング（ラベルごとの多数決）
    if smooth_window > 1:
        presence = {k: _smooth_presence(v, smooth_window) for k, v in presence.items()}

    events: List[dict] = []
    for key in sorted(presence.keys(), key=lambda k: k.sort_key()):
        events.extend(_segments_for_label(
            key, presence[key], conf[key], times, min_duration))

    events.sort(key=lambda ev: (
        ev["start_sec"], ev["end_sec"], ev["type"],
        ev.get("role", ""), ev["label"],
    ))
    return BuildResult(events, units, times, presence, conf, label_kind, unit_kind)


# ---------------------------------------------------------------------------
# SRT タグ行
# ---------------------------------------------------------------------------

def _action_tag_line(event: dict) -> str:
    """人間向けタグ行を生成する。triplet/action とも「[action] <label>」。"""
    return f"[action] {event['label']}"


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

    1行目: 「[action] <label>」（triplet は instrument,verb,target のカンマ連結）
    2行目: 機械向けJSON（時間フィールドは含めない＝SRTの時刻が正）
    """
    lines: List[str] = []
    for idx, ev in enumerate(events, start=1):
        lines.append(str(idx))
        lines.append(f"{ev['start_srt']} --> {ev['end_srt']}")
        lines.append(_action_tag_line(ev))
        lines.append(build_json_line(ev))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_segment_csv(events: List[dict], path: Path) -> None:
    """区間（セグメント）単位の時系列CSVを書き出す。"""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "segment_id", "type", "label", "label_id", "role",
            "instrument", "verb", "target",
            "start_sec", "start_srt", "end_sec", "end_srt",
            "duration_sec", "confidence",
        ])
        for i, ev in enumerate(events, start=1):
            comp = ev.get("components") or {}
            label_id = ev.get("triplet_id", ev.get("action_id", ""))
            writer.writerow([
                i,
                ev["type"],
                ev["label"],
                "" if label_id == "" or label_id is None else label_id,
                ev.get("role", ""),
                comp.get("instrument", ""),
                comp.get("verb", ""),
                comp.get("target", ""),
                f"{ev['start_sec']:.3f}",
                ev["start_srt"],
                f"{ev['end_sec']:.3f}",
                ev["end_srt"],
                f"{ev['duration_sec']:.3f}",
                "" if ev.get("confidence") is None else f"{ev['confidence']:.4f}",
            ])


def write_frame_csv(result: BuildResult, path: Path) -> None:
    """
    per-frame / per-clip（ユニット）単位の時系列CSVを書き出す（long形式）。

    多ラベルを表現するため、1行 = (ユニット, アクティブな基本ラベル)。
    分解トラック（role 付き）は含めない（入力ビュー＝基本ラベルのみ）。
    アクティブラベルが無いユニットは出力しない。
    """
    base_keys = sorted(
        (k for k in result.presence if k.role is None),
        key=lambda k: k.sort_key(),
    )
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "unit_idx", "t_start_sec", "t_end_sec", "t_srt",
            "type", "label", "label_id", "instrument", "verb", "target",
            "confidence",
        ])
        for pos, u in enumerate(result.units):
            s, e = result.times[pos]
            for key in base_keys:
                if not result.presence[key][pos]:
                    continue
                c = result.conf[key][pos]
                comp = key.components or (None, None, None)
                writer.writerow([
                    u.idx,
                    f"{s:.3f}",
                    f"{e:.3f}",
                    format_srt_time(s),
                    key.type,
                    key.label,
                    "" if key.label_id is None else key.label_id,
                    comp[0] or "",
                    comp[1] or "",
                    comp[2] or "",
                    "" if c is None else f"{c:.4f}",
                ])


# ---------------------------------------------------------------------------
# 変換（ファイル出力あり）
# ---------------------------------------------------------------------------

def convert(
    in_path: str,
    outdir: str,
    video_path: Optional[str] = None,
    fps: Optional[float] = None,
    min_duration: float = 0.5,
    smooth_window: int = 0,
    triplet_map_path: Optional[str] = None,
    triplet_thr: float = 0.5,
    clip_len: Optional[float] = None,
    decompose: bool = False,
    level: str = "segment",
    stem: Optional[str] = None,
) -> dict:
    """
    動作/手技認識結果を SRT/JSONL/CSV へ変換するメインロジック。

    Args:
        in_path: 入力（triplet 予測 CSV / 汎用 action CSV）
        outdir: 出力ディレクトリ
        video_path: 時刻解決に使う動画（PyAV PTS、VFR耐性）
        fps: 動画が無い場合の換算fps（Noneなら triplet マップの fps）
        min_duration: 最小継続長（秒）
        smooth_window: 多数決スムージング窓（フレーム/クリップ、0=無効）
        triplet_map_path: triplet マップJSON（Noneなら同梱の既定 CholecT50）
        triplet_thr: 確率ベクトル入力の多ラベル閾値
        clip_len: クリップ長（秒、clip 入力の時刻復元用）
        decompose: triplet を instrument/verb/target トラックへ分解して追加出力
        level: CSVの粒度（segment / frame / both）
        stem: 出力ファイル名の語幹（Noneなら入力ファイル名から）

    Returns:
        生成したファイルパス等の辞書
    """
    triplet_map = TripletMap.load(triplet_map_path)
    result = build_events(
        in_path=in_path,
        triplet_map=triplet_map,
        video_path=video_path,
        fps=fps,
        min_duration=min_duration,
        smooth_window=smooth_window,
        triplet_thr=triplet_thr,
        clip_len=clip_len,
        decompose=decompose,
    )

    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    if stem is None:
        stem = Path(in_path).stem

    jsonl_path = out_path / f"{stem}_action.jsonl"
    srt_path = out_path / f"{stem}_action.srt"
    csv_path = out_path / f"{stem}_action.csv"

    write_jsonl(result.events, jsonl_path)
    write_srt(result.events, srt_path)

    out: dict = {
        "jsonl": str(jsonl_path),
        "srt": str(srt_path),
        "segments": len(result.events),
        "units": len(result.units),
        "label_kind": result.label_kind,
        "unit_kind": result.unit_kind,
    }

    if level == "frame":
        write_frame_csv(result, csv_path)
        out["csv"] = str(csv_path)
    elif level == "both":
        write_segment_csv(result.events, csv_path)
        frame_csv_path = out_path / f"{stem}_action_frames.csv"
        write_frame_csv(result, frame_csv_path)
        out["csv"] = str(csv_path)
        out["csv_frames"] = str(frame_csv_path)
    else:  # segment（既定）
        write_segment_csv(result.events, csv_path)
        out["csv"] = str(csv_path)

    print(f"入力 : {in_path}（{result.label_kind} / {result.unit_kind}）")
    print(f"JSONL: {jsonl_path} （正本）")
    print(f"SRT  : {srt_path}")
    print(f"CSV  : {csv_path} （level={level}）")
    print(f"区間数: {len(result.events)} / ユニット数: {len(result.units)}")
    return out


# ---------------------------------------------------------------------------
# BaseAnalyzer 連携（pipeline.py から呼べる形）
# ---------------------------------------------------------------------------

def make_analyzer():
    """BaseAnalyzer 互換インスタンスを返す公開ファクトリ。"""
    from src.analyzers.base import AnalysisResult, BaseAnalyzer

    class ActionConverterAnalyzer(BaseAnalyzer):
        """先行研究の動作/手技認識結果を標準フォーマットへ変換するアナライザ。"""

        def __init__(self):
            super().__init__(name="action_converter", version=VERSION)

        def analyze(self, video_path: str, **params) -> AnalysisResult:
            in_path = params.get("in_path")
            if not in_path:
                raise ValueError("params['in_path'] が必要です（動作予測ファイル）")

            triplet_map = TripletMap.load(params.get("triplet_map"))
            fps = params.get("fps")
            result = build_events(
                in_path=in_path,
                triplet_map=triplet_map,
                video_path=video_path,
                fps=fps,
                min_duration=params.get("min_duration", 0.5),
                smooth_window=params.get("smooth_window", 0),
                triplet_thr=params.get("triplet_thr", 0.5),
                clip_len=params.get("clip_len"),
                decompose=params.get("decompose", False),
            )

            return AnalysisResult(
                analyzer_type="surgical_action",
                analyzer_version=self.version,
                parameters={
                    "in_path": in_path,
                    "fps": fps if fps is not None else triplet_map.fps,
                    "min_duration": params.get("min_duration", 0.5),
                    "smooth_window": params.get("smooth_window", 0),
                    "triplet_thr": params.get("triplet_thr", 0.5),
                    "decompose": params.get("decompose", False),
                    "triplet_map": triplet_map.name,
                    "source": SOURCE,
                },
                video_info={"path": video_path} if video_path else {},
                results=result.events,
                metadata={
                    "label_kind": result.label_kind,
                    "unit_kind": result.unit_kind,
                    "units": len(result.units),
                    "segments": len(result.events),
                },
            )

    return ActionConverterAnalyzer()


# ---------------------------------------------------------------------------
# CLI エントリポイント
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="先行研究の動作/手技認識結果（CholecT50 triplet / 汎用 action）を"
                    "標準SRT/JSONL/CSVへ変換する変換器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
  # CholecT50 triplet 確率ベクトルCSV（triplet_0..99）を動画PTSで時刻解決
  python -m src.action.action_to_outputs \\
      --in triplet_pred.csv --video case.mp4 --outdir out/ \\
      --triplet-thr 0.5 --decompose --level both

  # triplet id（argmax）の per-frame CSV を 25fps で区間化
  python -m src.action.action_to_outputs \\
      --in triplet_ids.csv --fps 25 --outdir out/ --min-duration 0.5

  # SLAM 風 clip 単位 action（7アクション、clip 長 1 秒）
  python -m src.action.action_to_outputs \\
      --in clip_actions.csv --clip-len 1.0 --outdir out/

  # 生成した _action.srt を既存の _bleed.srt / _cut.srt 等と統合
  python -m src.tools.merge_srt --out out/merged.srt \\
      out/case_bleed.srt out/triplet_pred_action.srt
        """,
    )
    parser.add_argument("--in", dest="in_path", required=True,
                        help="入力ファイル（triplet 予測 CSV / 汎用 action CSV）")
    parser.add_argument("--video", default=None,
                        help="時刻解決用の動画（PyAV PTS、VFR耐性）")
    parser.add_argument("--fps", type=float, default=None,
                        help="動画が無い場合の換算fps（既定: triplet マップの fps=25）")
    parser.add_argument("--outdir", required=True, help="出力ディレクトリ")
    parser.add_argument("--decompose", action="store_true",
                        help="triplet を instrument/verb/target トラックへ分解して追加出力")
    parser.add_argument("--min-duration", type=float, default=0.5,
                        help="最小継続長フィルタ（秒、既定: 0.5）")
    parser.add_argument("--smooth-window", type=int, default=0,
                        help="多数決スムージング窓（フレーム/クリップ、既定: 0=無効）")
    parser.add_argument("--triplet-map", default=None,
                        help="triplet 分解マップJSON（既定: 同梱 CholecT50）")
    parser.add_argument("--triplet-thr", type=float, default=0.5,
                        help="確率ベクトル入力の多ラベル閾値（既定: 0.5）")
    parser.add_argument("--clip-len", type=float, default=None,
                        help="クリップ長（秒、clip 入力の時刻復元用）")
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
        triplet_map_path=args.triplet_map,
        triplet_thr=args.triplet_thr,
        clip_len=args.clip_len,
        decompose=args.decompose,
        level=args.level,
        stem=args.stem,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
