"""Zero-shot 意味スコア × 既存赤色物理スコア の時系列融合。

zero-shot (BC×HV) は大出血のみ検出、redlog 系は短時間の赤色変化に敏感。
両者を組み合わせて短時間出血(区間1)も拾えるか試す。

入力:
  - zero-shot CSV (BiomedCLIP, HecVL): smooth_score
  - redlog/bleedlog CSV: red_ratio, smooth_delta, smooth_expansion など

出力:
  - 融合スコア CSV
"""

from __future__ import annotations

import argparse
import csv as _csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.core.time_utils import format_srt_time
from src.red.redlog import smooth_center

from .base import csv_to_events
from .evaluate import load_csv_scores


def _normalize(v: np.ndarray) -> np.ndarray:
    if v.size == 0:
        return v  # 空配列はそのまま返す（max/min で ValueError を出さない）
    rng = v.max() - v.min()
    if rng < 1e-9:
        return np.zeros_like(v)
    return (v - v.min()) / rng


def _load_redlog(
    csv_path: str, required: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    """redlog/bleedlog CSV を辞書 {col: ndarray} で返す。

    required を指定すると、その列が欠落 or 非数値の場合に明示的に ValueError を
    送出する（後続の KeyError による原因不明のクラッシュを防ぐ）。非必須列の
    非数値はスキップする。
    """
    cols: Dict[str, List] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV にヘッダーがありません: {csv_path}")
        for row in reader:
            for k, v in row.items():
                if k in ("t_srt", "reader"):
                    continue
                cols.setdefault(k, []).append(v)
    out = {}
    for k, vs in cols.items():
        try:
            out[k] = np.array([float(x) for x in vs])
        except (ValueError, TypeError):
            if required and k in required:
                raise ValueError(
                    f"必須列 '{k}' に非数値/空値が含まれます: {csv_path}"
                )
            # 非必須の非数値列はスキップ
    if required:
        missing = [c for c in required if c not in out]
        if missing:
            raise ValueError(
                f"CSV に必須列がありません: {', '.join(missing)} ({csv_path})"
            )
    return out


def _resample(
    src_t: np.ndarray, src_v: np.ndarray, dst_t: np.ndarray,
) -> np.ndarray:
    """src を dst_t の各時刻に線形補間する。

    np.interp は src_t が単調増加であることを前提とするため、未ソート/重複
    タイムスタンプの CSV を誤投入しても不正補間にならないよう、昇順ソートと
    重複除去を行う。dst_t がソース範囲を大きく外れる場合は端点クランプを警告。
    """
    src_t = np.asarray(src_t, dtype=float)
    src_v = np.asarray(src_v, dtype=float)
    if src_t.size == 0:
        raise ValueError("リサンプル元の時刻配列が空です")

    # 昇順ソート → 重複タイムスタンプを除去（先勝ち）
    order = np.argsort(src_t, kind="stable")
    st, sv = src_t[order], src_v[order]
    keep = np.concatenate(([True], np.diff(st) > 0))
    st, sv = st[keep], sv[keep]

    if dst_t.size and (dst_t.min() < st.min() - 1e-6
                       or dst_t.max() > st.max() + 1e-6):
        print(
            "警告: 補間先がソース時刻範囲外のため端点にクランプします "
            f"(src [{st.min():.1f}, {st.max():.1f}]s, "
            f"dst [{float(dst_t.min()):.1f}, {float(dst_t.max()):.1f}]s)",
            file=sys.stderr,
        )
    return np.interp(dst_t, st, sv)


def _compute_area_delta(
    smooth_areas: np.ndarray, fps: float, baseline_s: float = 60.0,
) -> np.ndarray:
    """detector.py 互換: 短期-長期ベースラインの差分（正部分のみ）。"""
    window = max(1, int(round(baseline_s * fps)))
    baseline = np.array(smooth_center(smooth_areas.tolist(), window))
    return np.maximum(0.0, smooth_areas - baseline)


def fuse(
    bc_csv: str,
    hv_csv: str,
    redlog_csv: str,
    bleedlog_csv: str,
    outdir: str,
    out_name: str = "fuse",
) -> Tuple[str, Dict[str, str]]:
    """主要シグナルを揃えて融合候補を出力。

    Returns:
        (出力ディレクトリ, {融合名: CSVパス}) — 値はファイルパス文字列。
    """
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    # zero-shot (1fps) を基準時刻に
    t_base, bc = load_csv_scores(bc_csv, key="smooth_score")
    _, hv = load_csv_scores(hv_csv, key="smooth_score")

    # redlog 5fps を 1fps にリサンプル（必須列を検証）
    rd = _load_redlog(redlog_csv, required=["t_sec", "red_ratio", "smooth_delta"])
    bl = _load_redlog(
        bleedlog_csv,
        required=["t_sec", "red_expansion", "smooth_expansion", "newly_red_ratio"],
    )
    rd_t = rd["t_sec"]
    bl_t = bl["t_sec"]

    red_ratio = _resample(rd_t, rd["red_ratio"], t_base)
    smooth_delta = _resample(rd_t, rd["smooth_delta"], t_base)
    red_expansion = _resample(bl_t, bl["red_expansion"], t_base)
    smooth_expansion = _resample(bl_t, bl["smooth_expansion"], t_base)
    newly_red = _resample(bl_t, bl["newly_red_ratio"], t_base)

    # area_delta 風指標 (1fps grid 上で再計算)
    area_delta = _compute_area_delta(red_ratio, fps=1.0, baseline_s=60.0)

    sigs = {
        "BC": bc,
        "HV": hv,
        "BCxHV": bc * hv,
        "red_ratio": red_ratio,
        "area_delta": area_delta,
        "smooth_delta": smooth_delta,
        "red_expansion": red_expansion,
        "smooth_expansion": smooth_expansion,
        "newly_red": newly_red,
    }

    # 各シグナルを 0-1 正規化
    norm = {k: _normalize(v) for k, v in sigs.items()}

    # 融合候補
    fusions = {
        "BCxHV":                    norm["BCxHV"],
        "BC+HV+area_delta":         (norm["BC"] + norm["HV"] + norm["area_delta"]) / 3,
        "BCxHV * area_delta":       norm["BCxHV"] * norm["area_delta"],
        "max(BC,HV) + area_delta":  np.maximum(norm["BC"], norm["HV"]) + norm["area_delta"],
        "BCxHV + 0.5*expansion":    norm["BCxHV"] + 0.5 * norm["smooth_expansion"],
        "BCxHV + 0.5*newly_red":    norm["BCxHV"] + 0.5 * norm["newly_red"],
        "max(BCxHV, area_delta)":   np.maximum(norm["BCxHV"], norm["area_delta"]),
        "max(BCxHV, expansion)":    np.maximum(norm["BCxHV"], norm["smooth_expansion"]),
        "BC*expansion":             norm["BC"] * norm["smooth_expansion"],
        "HV*expansion":             norm["HV"] * norm["smooth_expansion"],
    }

    # 個別CSV出力（各融合）
    csv_paths = {}
    stem = Path(bc_csv).stem.replace("_biomedclip", "")
    for name, vals in fusions.items():
        scores = _normalize(vals)
        smooth = np.array(smooth_center(scores.tolist(), 5))
        safe = name.replace(" ", "").replace(",", "_").replace("(", "").replace(")", "").replace("*", "x").replace("+", "p")
        p = out_path / f"{stem}_{out_name}_{safe}.csv"
        with open(p, "w", newline="", encoding="utf-8") as f:
            w_ = _csv.writer(f)
            w_.writerow(["t_sec", "t_srt", "score", "smooth_score"])
            for t, s, ss in zip(t_base, scores, smooth):
                w_.writerow([
                    f"{t:.3f}", format_srt_time(float(t)),
                    f"{s:.6f}", f"{ss:.6f}",
                ])
        csv_paths[name] = str(p)
    print(f"融合CSV: {len(csv_paths)} 件出力")
    return str(out_path), csv_paths


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--bc", required=True, help="BiomedCLIP score CSV")
    p.add_argument("--hv", required=True, help="HecVL score CSV")
    p.add_argument("--redlog", required=True)
    p.add_argument("--bleedlog", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--name", default="fuse",
                   help="出力CSV名のサフィックス（デフォルト: fuse）")
    args = p.parse_args()

    fuse(
        bc_csv=args.bc, hv_csv=args.hv,
        redlog_csv=args.redlog, bleedlog_csv=args.bleedlog,
        outdir=args.outdir, out_name=args.name,
    )
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
