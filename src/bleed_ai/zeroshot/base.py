"""Zero-shot scorer 共通インターフェース。

各モデルアダプタ（BiomedCLIP, SurgVLP, ...）はこのクラスを継承し、
`score_frame()` を実装するだけでよい。動画走査・CSV書き出し・SRT変換は
共通実装で行う。
"""

from __future__ import annotations

import csv
import json
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from src.core.time_utils import format_srt_time
from src.red.redlog import iter_frames, make_circular_roi, smooth_center


def validate_input_file(path: str, kind: str = "ファイル") -> str:
    """入力ファイルの存在を検証し、解決済み絶対パスを返す。"""
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"{kind}が見つかりません: {path}")
    return str(p.resolve())


def validated_outdir(outdir: str) -> str:
    """出力ディレクトリを解決・検証して作成し、絶対パスを返す。

    `..` はここで解決され、ファイルシステムのルート直下への書き込みは拒否する
    （意図しない上書き・パストラバーサルの抑止）。
    """
    if not outdir or not str(outdir).strip():
        raise ValueError("出力ディレクトリが空です")
    p = Path(outdir).expanduser().resolve()
    if str(p) == p.anchor:  # "/" や "C:\\" など FS ルート
        raise ValueError(f"出力先がルート直下のため拒否します: {outdir}")
    if p.exists() and not p.is_dir():
        raise NotADirectoryError(f"出力先がディレクトリではありません: {outdir}")
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


class ZeroShotScorer(ABC):
    """フレーム→出血スコア [0, 1] を返すモデルアダプタの基底クラス。"""

    name: str = "zeroshot"

    def __init__(self, device: str = "cuda"):
        self.device = device
        # load() の check-then-act を保護し、複数スレッドからの重複初期化
        # （モデルの二重 GPU 配置）を防ぐ。
        self._load_lock = threading.Lock()

    @abstractmethod
    def load(self) -> None:
        """重みのロード。__init__ では行わず明示的に呼ぶ。"""

    @abstractmethod
    def score_frames(self, bgr_frames: List[np.ndarray]) -> List[float]:
        """BGRフレームのバッチを受け取り、出血スコア [0, 1] のリストを返す。"""

    # -----------------------------------------------------------------
    # 動画走査
    # -----------------------------------------------------------------
    def score_video(
        self,
        video_path: str,
        outdir: str,
        fps: float = 1.0,
        batch_size: int = 16,
        roi_margin: float = 0.08,
        no_roi: bool = False,
        smooth_s: float = 5.0,
    ) -> str:
        """動画を走査して `<stem>_<name>.csv` を出力。CSVパスを返す。"""
        out_path = Path(outdir)
        out_path.mkdir(parents=True, exist_ok=True)
        stem = Path(video_path).stem
        csv_path = out_path / f"{stem}_{self.name}.csv"

        self.load()

        times: List[float] = []
        scores: List[float] = []
        buf_t: List[float] = []
        buf_f: List[np.ndarray] = []
        roi_mask: Optional[np.ndarray] = None
        roi_initialized = False

        for t_sec, bgr, _reader in iter_frames(video_path, fps):
            if not roi_initialized:
                h, w = bgr.shape[:2]
                if not no_roi:
                    roi_mask = make_circular_roi(h, w, margin=roi_margin)
                roi_initialized = True

            frame = bgr
            if roi_mask is not None:
                frame = bgr.copy()
                frame[~roi_mask] = 0

            buf_t.append(t_sec)
            buf_f.append(frame)
            if len(buf_f) >= batch_size:
                scores.extend(self.score_frames(buf_f))
                times.extend(buf_t)
                buf_t.clear()
                buf_f.clear()

        if buf_f:
            scores.extend(self.score_frames(buf_f))
            times.extend(buf_t)

        if not times:
            raise RuntimeError(f"フレーム取得に失敗: {video_path}")

        sample_fps = 1.0 / (times[1] - times[0]) if len(times) >= 2 else fps
        window = max(1, int(round(smooth_s * sample_fps)))
        smooth = smooth_center(scores, window)

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w_ = csv.writer(f)
            w_.writerow(["t_sec", "t_srt", "score", "smooth_score"])
            for t, s, ss in zip(times, scores, smooth):
                w_.writerow([
                    f"{t:.3f}", format_srt_time(t),
                    f"{s:.6f}", f"{ss:.6f}",
                ])
        print(f"CSV: {csv_path} ({len(times)} frames)")
        return str(csv_path)


# ---------------------------------------------------------------------------
# CSV → SRT （しきい値ベース）
# ---------------------------------------------------------------------------

def csv_to_events(
    csv_path: str,
    outdir: str,
    thr: float = 0.5,
    min_duration_s: float = 2.0,
    use_smooth: bool = True,
    suffix: str = "",
) -> dict:
    """スコアCSVをしきい値で区切ってJSONL+SRTを出力。"""
    times: List[float] = []
    scores: List[float] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            times.append(float(row["t_sec"]))
            key = "smooth_score" if use_smooth else "score"
            scores.append(float(row[key]))

    if len(times) < 2:
        raise RuntimeError("CSV frames < 2")
    fps = 1.0 / (times[1] - times[0])
    min_samples = max(1, int(round(min_duration_s * fps)))

    raw: List[tuple] = []
    in_ev = False
    s = 0
    for i, v in enumerate(scores):
        if v >= thr and not in_ev:
            in_ev, s = True, i
        elif v < thr and in_ev:
            in_ev = False
            if i - s >= min_samples:
                raw.append((s, i))
    if in_ev and len(scores) - s >= min_samples:
        raw.append((s, len(scores)))

    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    stem = Path(csv_path).stem + suffix

    jsonl_path = out_path / f"{stem}_events.jsonl"
    srt_path = out_path / f"{stem}.srt"

    events = []
    with open(jsonl_path, "w", encoding="utf-8") as fj, \
         open(srt_path, "w", encoding="utf-8") as fs:
        for idx, (a, b) in enumerate(raw, 1):
            seg = scores[a:b]
            ev = {
                "type": "bleed_zeroshot",
                "thr": thr,
                "peak_score": round(max(seg), 6),
                "mean_score": round(sum(seg) / len(seg), 6),
                "duration_s": round(times[b - 1] - times[a], 3),
                "start_sec": times[a],
                "end_sec": times[b - 1],
                "start_srt": format_srt_time(times[a]),
                "end_srt": format_srt_time(times[b - 1]),
            }
            events.append(ev)
            fj.write(json.dumps(ev, ensure_ascii=False) + "\n")
            fs.write(
                f"{idx}\n{ev['start_srt']} --> {ev['end_srt']}\n"
                f"[bleed_zs] peak={ev['peak_score']:.2f}\n\n"
            )

    print(f"JSONL: {jsonl_path}  events={len(events)}")
    print(f"SRT  : {srt_path}")
    return {"jsonl": str(jsonl_path), "srt": str(srt_path),
            "events": len(events)}
