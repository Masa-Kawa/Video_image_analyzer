"""DINOv2 特徴 + zero-shot 合意ラベルによる bootstrap 出血検出。

戦略:
  1. BiomedCLIP × HecVL のスコア積を「擬似教師」として使う
  2. 上位 K% のフレーム = positive アンカー（出血らしい）
  3. 「正常窓内 + 擬似教師スコア下位」フレーム = negative アンカー
  4. 各フレームを DINOv2 特徴空間で
        score(i) = mean_sim(i, pos_anchors) - mean_sim(i, neg_anchors)
     と定義し、min-max 正規化

教師ラベル不使用（True_Bleed.srt は使わない）なので zero-shot 系統に留まる。

入力:
  - DINOv2 特徴 .npz: features(N,D), times(N) ※anomaly_dinov2 の副産物
  - BiomedCLIP CSV (smooth_score)
  - HecVL CSV       (smooth_score)
"""

from __future__ import annotations

import argparse
import csv as _csv
from pathlib import Path
from typing import Tuple

import numpy as np

from src.core.time_utils import format_srt_time
from src.red.redlog import smooth_center

from .base import csv_to_events
from .evaluate import load_csv_scores


# アンカー選定の既定下限・既定窓（短尺動画等で調整できるよう外部化）
DEFAULT_NEG_WINDOW_S: Tuple[float, float] = (0.0, 360.0)
MIN_POS_ANCHORS = 5      # positive アンカー数の下限
MIN_NEG_ANCHORS = 20     # negative アンカー数の下限
MIN_NORMAL_FRAMES = 20   # 正常窓内フレーム数の下限（これ未満で RuntimeError）
NEG_FRACTION = 0.5       # 正常窓内のうち negative に回す割合


def _normalize(v: np.ndarray) -> np.ndarray:
    rng = v.max() - v.min()
    if rng < 1e-9:
        return np.zeros_like(v)
    return (v - v.min()) / rng


def bootstrap_score(
    features_npz: str,
    biomedclip_csv: str,
    hecvl_csv: str,
    outdir: str,
    pos_top_pct: float = 5.0,
    neg_window_s: Tuple[float, float] = DEFAULT_NEG_WINDOW_S,
    smooth_s: float = 5.0,
    out_name: str = "bootstrap",
    min_pos_anchors: int = MIN_POS_ANCHORS,
    min_neg_anchors: int = MIN_NEG_ANCHORS,
    min_normal_frames: int = MIN_NORMAL_FRAMES,
    neg_fraction: float = NEG_FRACTION,
) -> str:
    """Bootstrap スコアを計算して CSV を出力。

    Args:
        features_npz: anomaly_dinov2 が出力した npz
        biomedclip_csv: BiomedCLIP の smooth_score 入り CSV
        hecvl_csv: HecVL の smooth_score 入り CSV
        pos_top_pct: 擬似教師スコア上位N%を positive アンカーに採用
        neg_window_s: 正常参照窓（秒）。この範囲内のフレームを neg 候補に
        smooth_s: スコア平滑化窓（秒）
        out_name: 出力 CSV のサフィックス
        min_pos_anchors: positive アンカー数の下限
        min_neg_anchors: negative アンカー数の下限
        min_normal_frames: 正常窓内フレーム数の下限（未満なら RuntimeError）
        neg_fraction: 正常窓内のうち negative に回す割合

    Returns:
        出力 CSV パス
    """
    # with でラップし、NpzFile の FD / メモリマップを確実に解放する
    with np.load(features_npz) as data:
        F = data["features"].astype(np.float32)  # (N, D)
        T = data["times"].astype(np.float64)     # (N,)

    # 行ごとに L2 正規化。ノルムが極小（ゼロ）の行はゼロベクトルのまま残し、
    # 1e-9 で割って歪んだ単位ベクトルを作らないようにする（安全な除算）。
    norms = np.linalg.norm(F, axis=1, keepdims=True)
    F = np.divide(F, norms, out=np.zeros_like(F), where=norms > 1e-9)

    # zero-shot スコアを揃える（時系列が一致している前提）
    t_bc, s_bc = load_csv_scores(biomedclip_csv, key="smooth_score")
    t_hv, s_hv = load_csv_scores(hecvl_csv, key="smooth_score")
    if len(t_bc) != len(T) or len(t_hv) != len(T):
        raise RuntimeError(
            f"フレーム数不一致: features={len(T)} BC={len(t_bc)} HV={len(t_hv)}"
        )

    # 擬似教師: 各モデルのスコアを min-max 正規化して積
    bc = _normalize(s_bc)
    hv = _normalize(s_hv)
    pseudo = bc * hv  # 両方が高いフレームを強く positive にする

    # Positive アンカー: 擬似教師上位N%
    n = len(T)
    n_pos = max(min_pos_anchors, int(n * pos_top_pct / 100.0))
    pos_idx = np.argsort(-pseudo)[:n_pos]

    # Negative アンカー: 正常窓内 ∧ 擬似教師スコア下位
    a, b = neg_window_s
    in_normal = (T >= a) & (T <= b)
    if in_normal.sum() < min_normal_frames:
        raise RuntimeError(
            f"正常窓内フレームが不足: {int(in_normal.sum())} "
            f"(必要 >= {min_normal_frames}, 窓 {a:.0f}-{b:.0f}s)。"
            "--normal-start/--normal-end や --min-normal-frames を調整してください。"
        )
    # 正常窓内で擬似教師の小さい順
    normal_idx = np.where(in_normal)[0]
    normal_pseudo = pseudo[normal_idx]
    n_neg = max(min_neg_anchors, int(len(normal_idx) * neg_fraction))
    neg_idx = normal_idx[np.argsort(normal_pseudo)[:n_neg]]

    print(f"擬似教師スコア range: [{pseudo.min():.3f}, {pseudo.max():.3f}]")
    print(f"Positive アンカー: {len(pos_idx)} フレーム")
    print(f"  代表時刻: {sorted(T[pos_idx][:10].astype(int).tolist())}...")
    print(f"Negative アンカー: {len(neg_idx)} フレーム ({a:.0f}-{b:.0f}s 内)")

    # 各フレームの positive / negative 平均類似度（コサイン）。
    # mean_j(F_i · A_j) == F_i · mean_j(A_j) なので、(N, n_anchor) の密行列を
    # 作らずアンカー重心との内積で等価に計算する（長尺動画での OOM を回避）。
    pos_centroid = F[pos_idx].mean(axis=0)  # (D,)
    neg_centroid = F[neg_idx].mean(axis=0)  # (D,)
    sim_pos = F @ pos_centroid  # (N,)
    sim_neg = F @ neg_centroid  # (N,)
    raw = sim_pos - sim_neg
    scores = _normalize(raw)

    # 平滑化
    sample_fps = 1.0 / (T[1] - T[0]) if len(T) >= 2 else 1.0
    window = max(1, int(round(smooth_s * sample_fps)))
    smooth = smooth_center(scores.tolist(), window)

    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    stem = Path(features_npz).stem.replace("_dinov2_features", "")
    csv_path = out_path / f"{stem}_{out_name}.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w_ = _csv.writer(f)
        w_.writerow(["t_sec", "t_srt", "score", "smooth_score"])
        for t, s, ss in zip(T, scores, smooth):
            w_.writerow([
                f"{t:.3f}", format_srt_time(float(t)),
                f"{s:.6f}", f"{ss:.6f}",
            ])
    print(f"CSV: {csv_path}")
    return str(csv_path)


def main() -> int:
    p = argparse.ArgumentParser(
        description="DINOv2 + zero-shot bootstrap 出血検出"
    )
    p.add_argument("--features", required=True, help="DINOv2 特徴 .npz")
    p.add_argument("--biomedclip-csv", required=True)
    p.add_argument("--hecvl-csv", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--pos-top-pct", type=float, default=5.0)
    p.add_argument("--normal-start", type=float, default=DEFAULT_NEG_WINDOW_S[0])
    p.add_argument("--normal-end", type=float, default=DEFAULT_NEG_WINDOW_S[1])
    p.add_argument("--name", default="bootstrap")
    p.add_argument("--smooth-s", type=float, default=5.0,
                   help="スコア平滑化窓（秒、デフォルト: 5）")
    p.add_argument("--min-pos-anchors", type=int, default=MIN_POS_ANCHORS,
                   help=f"positive アンカー下限（デフォルト: {MIN_POS_ANCHORS}）")
    p.add_argument("--min-neg-anchors", type=int, default=MIN_NEG_ANCHORS,
                   help=f"negative アンカー下限（デフォルト: {MIN_NEG_ANCHORS}）")
    p.add_argument("--min-normal-frames", type=int, default=MIN_NORMAL_FRAMES,
                   help=f"正常窓内フレーム下限（デフォルト: {MIN_NORMAL_FRAMES}）")
    p.add_argument("--neg-fraction", type=float, default=NEG_FRACTION,
                   help=f"正常窓内のうち negative 割合（デフォルト: {NEG_FRACTION}）")
    p.add_argument("--thr", type=float, default=0.5)
    p.add_argument("--min-duration-s", type=float, default=2.0)
    args = p.parse_args()

    csv_path = bootstrap_score(
        features_npz=args.features,
        biomedclip_csv=args.biomedclip_csv,
        hecvl_csv=args.hecvl_csv,
        outdir=args.outdir,
        pos_top_pct=args.pos_top_pct,
        neg_window_s=(args.normal_start, args.normal_end),
        smooth_s=args.smooth_s,
        out_name=args.name,
        min_pos_anchors=args.min_pos_anchors,
        min_neg_anchors=args.min_neg_anchors,
        min_normal_frames=args.min_normal_frames,
        neg_fraction=args.neg_fraction,
    )
    csv_to_events(
        csv_path=csv_path, outdir=args.outdir,
        thr=args.thr, min_duration_s=args.min_duration_s,
    )
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
