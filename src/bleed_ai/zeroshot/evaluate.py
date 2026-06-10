"""Zero-shot スコアCSV を True_Bleed.srt と比較評価。

指標:
  - フレームレベル: PR-AUC, ROC-AUC（参考）
  - 区間レベル: tIoU >= 閾値で hit と判定する F1, Precision, Recall

使用例:
    python -m src.bleed_ai.zeroshot.evaluate \\
        --truth out_lapc_eval/True_Bleed.srt \\
        --csv out_lapc_eval/LapC_EvalDemo_480p_biomedclip.csv \\
        --label biomedclip
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from src.core.time_utils import parse_srt_time


# ---------------------------------------------------------------------------
# SRT読み込み（True_Bleed.srt のような単純なSRT用）
# ---------------------------------------------------------------------------

_TIME_RE = re.compile(
    r"(\d{1,2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}[,.]\d{3})"
)


def load_truth_intervals(srt_path: str) -> List[Tuple[float, float]]:
    """SRT → [(start_sec, end_sec), ...]。"""
    text = Path(srt_path).read_text(encoding="utf-8")
    intervals: List[Tuple[float, float]] = []
    for m in _TIME_RE.finditer(text):
        a = parse_srt_time(m.group(1))
        b = parse_srt_time(m.group(2))
        if b > a:
            intervals.append((a, b))
    return intervals


def load_csv_scores(
    csv_path: str, key: str = "smooth_score",
) -> Tuple[np.ndarray, np.ndarray]:
    times: List[float] = []
    scores: List[float] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            times.append(float(row["t_sec"]))
            scores.append(float(row[key]))
    return np.array(times), np.array(scores)


# ---------------------------------------------------------------------------
# フレームラベル
# ---------------------------------------------------------------------------

def frames_to_labels(
    times: np.ndarray, intervals: List[Tuple[float, float]],
) -> np.ndarray:
    y = np.zeros(len(times), dtype=np.int8)
    for a, b in intervals:
        y |= ((times >= a) & (times <= b)).astype(np.int8)
    return y


# ---------------------------------------------------------------------------
# PR / ROC （sklearn なしで実装）
# ---------------------------------------------------------------------------

def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    return float(np.trapz(y, x))


def pr_auc(y: np.ndarray, s: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """Average precision の近似（PR曲線の台形積分、x昇順）。"""
    order = np.argsort(-s)
    y_s = y[order]
    tp = np.cumsum(y_s)
    fp = np.cumsum(1 - y_s)
    P = y.sum()
    if P == 0:
        return float("nan"), np.array([]), np.array([])
    recall = tp / P
    precision = tp / np.maximum(tp + fp, 1)
    # 先頭に (recall=0, precision=1) を付与して安定化
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])
    return _trapz(precision, recall), recall, precision


def roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    P = y.sum()
    N = len(y) - P
    if P == 0 or N == 0:
        return float("nan")
    order = np.argsort(-s)
    y_s = y[order]
    tp = np.cumsum(y_s)
    fp = np.cumsum(1 - y_s)
    tpr = tp / P
    fpr = fp / N
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])
    return _trapz(tpr, fpr)


# ---------------------------------------------------------------------------
# 区間F1 (tIoU)
# ---------------------------------------------------------------------------

def _extract_pred_intervals(
    times: np.ndarray, scores: np.ndarray,
    thr: float, min_duration_s: float,
    merge_gap_s: float = 0.0,
) -> List[Tuple[float, float]]:
    """しきい値超過の連続区間を抽出し、オプションで近接区間をマージする。

    Args:
        merge_gap_s: この秒数以内のギャップは同一区間として結合する（0で無効）。
    """
    if len(times) < 2:
        return []
    # 重複/非単調タイムスタンプでも 0除算しないよう、正の間隔の中央値から fps を推定
    diffs = np.diff(np.asarray(times, dtype=float))
    positive = diffs[diffs > 0]
    fps = 1.0 / float(np.median(positive)) if positive.size else 1.0
    min_samples = max(1, int(round(min_duration_s * fps)))
    raw: List[Tuple[float, float]] = []
    in_ev = False
    s = 0
    for i, v in enumerate(scores):
        if v >= thr and not in_ev:
            in_ev, s = True, i
        elif v < thr and in_ev:
            in_ev = False
            if i - s >= min_samples:
                raw.append((float(times[s]), float(times[i - 1])))
    if in_ev and len(scores) - s >= min_samples:
        raw.append((float(times[s]), float(times[-1])))

    if merge_gap_s <= 0 or not raw:
        return raw

    merged: List[List[float]] = [list(raw[0])]
    for a, b in raw[1:]:
        if a - merged[-1][1] <= merge_gap_s:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [tuple(x) for x in merged]  # type: ignore


def _tiou(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def interval_f1(
    pred: List[Tuple[float, float]],
    truth: List[Tuple[float, float]],
    iou_thr: float = 0.3,
) -> Dict[str, float]:
    if not pred and not truth:
        return {"precision": float("nan"), "recall": float("nan"),
                "f1": float("nan"), "tp": 0, "fp": 0, "fn": 0}
    matched_truth = set()
    tp = 0
    for p in pred:
        best_i, best_iou = -1, 0.0
        for i, t in enumerate(truth):
            if i in matched_truth:
                continue
            iou = _tiou(p, t)
            if iou > best_iou:
                best_iou, best_i = iou, i
        if best_i >= 0 and best_iou >= iou_thr:
            matched_truth.add(best_i)
            tp += 1
    fp = len(pred) - tp
    fn = len(truth) - len(matched_truth)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"precision": prec, "recall": rec, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn}


# ---------------------------------------------------------------------------
# しきい値スイープ
# ---------------------------------------------------------------------------

def best_threshold(
    times: np.ndarray, scores: np.ndarray,
    truth: List[Tuple[float, float]],
    min_duration_s: float = 2.0,
    iou_thr: float = 0.3,
    merge_gap_s: float = 0.0,
    n_thr: int = 41,
) -> Dict:
    qs = np.linspace(0.0, 1.0, n_thr)
    score_qs = np.quantile(scores, np.linspace(0.5, 0.99, n_thr))
    candidates = sorted(set(np.concatenate([qs, score_qs]).round(4).tolist()))

    best = {"f1": -1.0}
    rows = []
    for thr in candidates:
        pred = _extract_pred_intervals(
            times, scores, thr, min_duration_s, merge_gap_s=merge_gap_s,
        )
        m = interval_f1(pred, truth, iou_thr=iou_thr)
        rows.append({"thr": thr, **m, "n_pred": len(pred)})
        if m["f1"] > best["f1"]:
            best = {"thr": thr, **m, "n_pred": len(pred)}
    return {"best": best, "sweep": rows}


# ---------------------------------------------------------------------------
# メイン評価
# ---------------------------------------------------------------------------

def evaluate(
    truth_srt: str,
    csv_paths: List[str],
    labels: List[str],
    outdir: str,
    iou_thr: float = 0.3,
    min_duration_s: float = 2.0,
    merge_gap_s: float = 0.0,
    score_key: str = "smooth_score",
) -> Dict:
    truth = load_truth_intervals(truth_srt)
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Truth intervals: {len(truth)}")
    for a, b in truth:
        print(f"  {a:.1f} - {b:.1f}  ({b-a:.1f}s)")

    summary = []
    for csv_path, label in zip(csv_paths, labels):
        times, scores = load_csv_scores(csv_path, key=score_key)
        y = frames_to_labels(times, truth)
        ap, _, _ = pr_auc(y, scores)
        au_roc = roc_auc(y, scores)
        sweep = best_threshold(
            times, scores, truth,
            min_duration_s=min_duration_s, iou_thr=iou_thr,
            merge_gap_s=merge_gap_s,
        )
        best = sweep["best"]
        row = {
            "label": label,
            "csv": csv_path,
            "n_frames": int(len(times)),
            "pos_frames": int(y.sum()),
            "pr_auc": ap,
            "roc_auc": au_roc,
            "best_thr": best.get("thr"),
            "best_f1": best.get("f1"),
            "best_precision": best.get("precision"),
            "best_recall": best.get("recall"),
            "tp": best.get("tp"),
            "fp": best.get("fp"),
            "fn": best.get("fn"),
            "n_pred": best.get("n_pred"),
        }
        summary.append(row)
        print(
            f"\n[{label}]  PR-AUC={ap:.3f}  ROC-AUC={au_roc:.3f}  "
            f"best_F1={best.get('f1'):.3f} @ thr={best.get('thr'):.3f}  "
            f"P/R={best.get('precision'):.2f}/{best.get('recall'):.2f}  "
            f"TP/FP/FN={best.get('tp')}/{best.get('fp')}/{best.get('fn')}"
        )

    out_json = out / "zeroshot_summary.json"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSummary: {out_json}")

    # 簡易プロット（matplotlibが使える場合）
    try:
        _plot_timelines(truth, csv_paths, labels, out, score_key)
    except Exception as e:
        print(f"プロットスキップ: {e}")

    return {"summary": summary, "json": str(out_json)}


def _plot_timelines(
    truth, csv_paths, labels, out: Path, score_key: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(csv_paths)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.2 * n + 1), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, csv_path, label in zip(axes, csv_paths, labels):
        times, scores = load_csv_scores(csv_path, key=score_key)
        ax.plot(times, scores, lw=0.8, label=label)
        for a, b in truth:
            ax.axvspan(a, b, color="red", alpha=0.18, lw=0)
        ax.set_ylabel(label, fontsize=9)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("Zero-shot bleed score vs True_Bleed (red span)")
    fig.tight_layout()
    p = out / "zeroshot_timeline.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    print(f"Plot: {p}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--truth", required=True, help="True_Bleed.srt")
    p.add_argument("--csv", action="append", required=True,
                   help="スコアCSV（複数可）")
    p.add_argument("--label", action="append", required=True,
                   help="CSVに対応するラベル（同じ数）")
    p.add_argument("--outdir", required=True)
    p.add_argument("--iou-thr", type=float, default=0.3)
    p.add_argument("--min-duration-s", type=float, default=2.0)
    p.add_argument("--merge-gap-s", type=float, default=0.0,
                   help="この秒数以内の近接予測区間をマージ（デフォルト: 無効）")
    p.add_argument("--score-key", default="smooth_score",
                   choices=["smooth_score", "score"])
    args = p.parse_args()

    if len(args.csv) != len(args.label):
        raise SystemExit("--csv と --label の数が一致しません")

    evaluate(
        truth_srt=args.truth,
        csv_paths=args.csv,
        labels=args.label,
        outdir=args.outdir,
        iou_thr=args.iou_thr,
        min_duration_s=args.min_duration_s,
        merge_gap_s=args.merge_gap_s,
        score_key=args.score_key,
    )
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
