"""src.bleed_ai.zeroshot.evaluate のスモークテスト。"""

from __future__ import annotations

import csv
import textwrap
from pathlib import Path

import numpy as np
import pytest

from src.bleed_ai.zeroshot.evaluate import (
    best_threshold,
    frames_to_labels,
    interval_f1,
    load_csv_scores,
    load_truth_intervals,
    pr_auc,
    roc_auc,
)


def test_load_truth_intervals(tmp_path: Path) -> None:
    srt = tmp_path / "truth.srt"
    srt.write_text(textwrap.dedent("""
    1
    00:00:10,000 --> 00:00:20,000
    bleed

    2
    00:01:00,000 --> 00:01:30,500
    bleed
    """).strip(), encoding="utf-8")

    iv = load_truth_intervals(str(srt))
    assert iv == [(10.0, 20.0), (60.0, 90.5)]


def test_frames_to_labels() -> None:
    times = np.arange(0, 100, dtype=float)
    iv = [(10.0, 20.0), (50.0, 55.0)]
    y = frames_to_labels(times, iv)
    # 失敗時にどのインデックスが想定外か特定できるよう個別に検証する
    assert y[0] == 0
    assert y[15] == 1
    assert y[30] == 0
    assert y[52] == 1


def test_pr_roc_auc_perfect() -> None:
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    ap, _, _ = pr_auc(y, s)
    assert ap == pytest.approx(1.0, abs=1e-6)
    assert roc_auc(y, s) == pytest.approx(1.0, abs=1e-6)


def test_pr_roc_auc_no_positives_returns_nan() -> None:
    # 正例が無い（y 全 0）→ pr_auc/roc_auc は nan を返すガード節
    y = np.array([0, 0, 0, 0])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    ap, _, _ = pr_auc(y, s)
    assert np.isnan(ap)
    assert np.isnan(roc_auc(y, s))


def test_roc_auc_no_negatives_returns_nan() -> None:
    # 負例が無い（y 全 1）→ roc_auc は nan
    y = np.array([1, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    assert np.isnan(roc_auc(y, s))


def test_interval_f1_match() -> None:
    truth = [(10.0, 20.0), (50.0, 60.0)]
    pred_perfect = [(10.5, 19.5), (50.0, 60.0)]
    m = interval_f1(pred_perfect, truth, iou_thr=0.3)
    assert m["tp"] == 2
    assert m["f1"] == pytest.approx(1.0)

    pred_miss = [(0.0, 5.0)]
    m2 = interval_f1(pred_miss, truth, iou_thr=0.3)
    assert m2["tp"] == 0
    assert m2["fp"] == 1
    assert m2["fn"] == 2


def test_interval_f1_both_empty_returns_nan() -> None:
    # pred も truth も空 → 計算不能なので nan（tp/fp/fn は 0）
    m = interval_f1([], [], iou_thr=0.3)
    assert np.isnan(m["f1"])
    assert np.isnan(m["precision"])
    assert np.isnan(m["recall"])
    assert m["tp"] == 0
    assert m["fp"] == 0
    assert m["fn"] == 0


def test_best_threshold_synthetic(tmp_path: Path) -> None:
    # 0-100s, 1Hz; 出血: 30-50s
    times = np.arange(0, 100, dtype=float)
    scores = np.where((times >= 30) & (times <= 50), 0.8, 0.1)
    truth = [(30.0, 50.0)]
    res = best_threshold(times, scores, truth, min_duration_s=2.0, iou_thr=0.3)
    assert res["best"]["f1"] == pytest.approx(1.0)


def test_load_csv_scores(tmp_path: Path) -> None:
    p = tmp_path / "s.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t_sec", "t_srt", "score", "smooth_score"])
        w.writerow(["0.0", "00:00:00,000", "0.1", "0.2"])
        w.writerow(["1.0", "00:00:01,000", "0.7", "0.6"])
    t, s = load_csv_scores(str(p))
    # 時刻列とスコア列を個別に検証（どちらが失敗したか明確化）
    assert list(t) == [0.0, 1.0]
    assert list(s) == [0.2, 0.6]
