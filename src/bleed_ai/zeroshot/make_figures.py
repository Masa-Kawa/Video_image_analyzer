"""発表用図表の一括生成。

出力: out_lapc_eval/zs/figures/
  fig01_overview.png         アプローチ全体フロー
  fig02_phase1_models.png    Phase 1 各 VLP モデルの性能
  fig03_interval_heatmap.png 区間別 mean スコアのヒートマップ
  fig04_timeline.png         タイムライン比較（truth + 主要モデル）
  fig05_complementary.png    BC×HV と expansion の補完性
  fig06_progression.png      Phase 0→1→2→3 の PR-AUC 推移
  fig07_alpha_sweep.png      融合重み α スイープ
  fig08_summary_table.png    最終比較サマリー表
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

from src.bleed_ai.zeroshot.evaluate import (
    _extract_pred_intervals, frames_to_labels, interval_f1,
    load_csv_scores, load_truth_intervals, pr_auc, roc_auc,
)
from src.bleed_ai.zeroshot.fusion import _load_redlog, _normalize, _resample
from src.red.redlog import smooth_center


# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------

import os


def _setup_japanese_font() -> None:
    """利用可能な日本語フォントを検索して設定する（OS非依存）。

    既知の絶対パス（Linux/Noto）が存在すれば登録を試み、最終的には
    fontManager に登録済みのフォント名から優先順で選ぶ。見つからなければ
    sans-serif のまま続行する（tofu になるが処理は止めない）。
    """
    # OS により所在が異なるため、存在するものだけ登録する（任意）
    for p in (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    ):
        if Path(p).exists():
            try:
                font_manager.fontManager.addfont(p)
            except Exception:
                pass
    preferred = ["Noto Sans CJK JP", "IPAexGothic", "TakaoPGothic",
                 "VL PGothic", "Hiragino Sans", "Yu Gothic", "MS Gothic"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in preferred:
        if name in available:
            plt.rcParams["font.family"] = [name, "DejaVu Sans"]
            return
    print("警告: 日本語フォントが見つかりません。ラベルが □ になる可能性があります。",
          file=sys.stderr)


def _safe_dpi(default: int = 150) -> int:
    """FIG_DPI を安全に解釈する。無効値はデフォルトにフォールバック。"""
    raw = os.environ.get("FIG_DPI")
    if raw is None:
        return default
    try:
        dpi = int(raw)
        return dpi if dpi > 0 else default
    except (ValueError, TypeError):
        print(f"警告: FIG_DPI='{raw}' は無効です。{default} を使用します。",
              file=sys.stderr)
        return default


_setup_japanese_font()
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 11
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["savefig.dpi"] = _safe_dpi()


OUT_DIR = Path(os.environ.get("FIG_OUT_DIR", "out_lapc_eval/zs/figures"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRUTH_SRT = "out_lapc_eval/True_Bleed.srt"
ZS = "out_lapc_eval/zs"
BLEEDLOG = "out_lapc_eval/LapC_EvalDemo_bleedlog.csv"

# 主要モデルのCSVパス
CSVS = {
    "BiomedCLIP":   f"{ZS}/LapC_EvalDemo_480p_biomedclip.csv",
    "PeskaVLP":     f"{ZS}/LapC_EvalDemo_480p_surgvlp_peskavlp.csv",
    "SurgVLP":      f"{ZS}/LapC_EvalDemo_480p_surgvlp_surgvlp.csv",
    "HecVL":        f"{ZS}/LapC_EvalDemo_480p_surgvlp_hecvl.csv",
    "BC×HV":        f"{ZS}/LapC_EvalDemo_480p_fuse_BCxHV.csv",
    "DINOv2_anom":  f"{ZS}/LapC_EvalDemo_480p_dinov2_anomaly.csv",
    "Bootstrap":    f"{ZS}/LapC_EvalDemo_480p_bootstrap.csv",
    "BC×HV+0.5×exp": f"{ZS}/LapC_EvalDemo_480p_fuse_BCxHVp0.5xexpansion.csv",
}

# 模型カラー（カテゴリで色分け）
MODEL_COLORS = {
    "BiomedCLIP":     "#2b8cbe",  # 青系（汎用医療VLP）
    "PeskaVLP":       "#9ecae1",  # 薄青
    "SurgVLP":        "#fd8d3c",  # 橙（手術VLP）
    "HecVL":          "#e6550d",  # 濃橙
    "BC×HV":          "#31a354",  # 緑（アンサンブル）
    "DINOv2_anom":    "#bcbcbc",  # 灰（失敗）
    "Bootstrap":      "#969696",  # 灰
    "BC×HV+0.5×exp":  "#d62728",  # 赤（最終ベスト）
}

TRUTH_LABELS = ["区間1\n(11.8s 軽度)", "区間2\n(36.9s 中規模)", "区間3\n(137.7s 大出血)"]

# 図に表示する代表指標の単一情報源（ハードコード値）。実データから計算した値と
# 乖離していないか _verify_reported_metrics() で起動時に検証し、ドリフトを警告する。
REPORTED = {
    "biomedclip_pr_auc": 0.408,
    "bcxhv_pr_auc": 0.443,
    "bcxhv_f1": 0.500,
    "final_pr_auc": 0.473,
    "final_f1": 0.500,
}
# REPORTED キー → 実測に使う CSV キー
_REPORTED_CSV = {
    "biomedclip_pr_auc": "BiomedCLIP",
    "bcxhv_pr_auc": "BC×HV",
    "bcxhv_f1": "BC×HV",
    "final_pr_auc": "BC×HV+0.5×exp",
    "final_f1": "BC×HV+0.5×exp",
}


# ---------------------------------------------------------------------------
# 共通ロード
# ---------------------------------------------------------------------------

def _compute_metrics(times, scores, truth) -> Dict:
    y = frames_to_labels(times, truth)
    ap, _, _ = pr_auc(y, scores)
    au = roc_auc(y, scores)
    best = (0, 0, 0, None)
    for gap in [0, 30, 60]:
        for thr in np.arange(0.05, 0.95, 0.025):
            pred = _extract_pred_intervals(
                times, scores, thr, 5.0, merge_gap_s=gap)
            mm = interval_f1(pred, truth, iou_thr=0.3)
            if mm["f1"] > best[2]:
                best = (gap, thr, mm["f1"], mm)
    bg = best[3] or {"tp": 0, "fp": 0, "fn": 3,
                     "precision": 0, "recall": 0}
    return {
        "pr_auc": ap, "roc_auc": au,
        "f1": best[2], "thr": best[1], "gap": best[0],
        "tp": bg["tp"], "fp": bg["fp"], "fn": bg["fn"],
        "precision": bg["precision"], "recall": bg["recall"],
    }


def _interval_means(times, scores, truth) -> List[float]:
    return [float(scores[(times >= a) & (times <= b)].mean())
            for a, b in truth]


# ===========================================================================
# Figure 1: アプローチ全体フロー
# ===========================================================================

def fig_overview() -> None:
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7)
    ax.axis("off")

    def box(x, y, w, h, text, color, fontsize=10, fontweight="normal"):
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.05",
            linewidth=1.2, edgecolor="#333", facecolor=color,
        )
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=fontsize, fontweight=fontweight,
                wrap=True)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#444", lw=1.4))

    # Title
    ax.text(6.5, 6.7, "Zero-shot 出血検出パイプライン",
            ha="center", fontsize=16, fontweight="bold")

    # 入力
    box(0.5, 5.0, 2.3, 1.0,
        "腹腔鏡動画\n(854×480, 60fps, 34分)\nLapC_EvalDemo", "#fff7bc")
    box(0.5, 3.6, 2.3, 0.8, "1fps サンプリング\n2038 フレーム", "#fff7bc")
    arrow(1.65, 5.0, 1.65, 4.4)

    # 4経路
    box(3.4, 5.4, 2.4, 0.9, "Phase 1\nVLP × プロンプト類似度", "#deebf7", 10, "bold")
    box(6.1, 5.4, 2.4, 0.9, "Phase 2\nDINOv2 異常検知", "#f0f0f0", 10, "bold")
    box(8.8, 5.4, 2.4, 0.9, "Phase 2'\n自己ブートストラップ", "#f0f0f0", 10, "bold")
    box(11.5, 5.4, 1.3, 0.9, "Phase 3\n物理融合", "#fee0d2", 10, "bold")

    # 中継
    arrow(2.8, 4.0, 4.6, 5.4)
    arrow(2.8, 4.0, 7.3, 5.4)
    arrow(2.8, 4.0, 10.0, 5.4)
    arrow(2.8, 4.0, 12.15, 5.4)

    # 各 phase の中身
    box(3.4, 4.3, 2.4, 0.9,
        "BiomedCLIP\nPeskaVLP / SurgVLP\nHecVL", "#deebf7", 9)
    box(6.1, 4.3, 2.4, 0.9,
        "DINOv2-base 特徴\n+ kNN 距離\n(0–360s 正常参照)", "#f0f0f0", 9)
    box(8.8, 4.3, 2.4, 0.9,
        "BC×HV 上位5%\n→ positive anchor\nDINOv2 類似度", "#f0f0f0", 9)
    box(11.5, 4.3, 1.3, 0.9,
        "BC×HV +\nα×expansion", "#fee0d2", 9)
    arrow(4.6, 5.4, 4.6, 5.2)
    arrow(7.3, 5.4, 7.3, 5.2)
    arrow(10.0, 5.4, 10.0, 5.2)
    arrow(12.15, 5.4, 12.15, 5.2)

    # 結果
    box(3.4, 2.9, 2.4, 0.7,
        f"F1={REPORTED['bcxhv_f1']:.3f} (BC×HV)\nPR-AUC={REPORTED['bcxhv_pr_auc']:.3f}",
        "#deebf7", 9)
    box(6.1, 2.9, 2.4, 0.7, "失敗\nROC-AUC=0.426", "#f0f0f0", 9)
    box(8.8, 2.9, 2.4, 0.7, "F1=0.400\nPR-AUC=0.151", "#f0f0f0", 9)
    box(11.5, 2.9, 1.3, 0.7,
        f"F1={REPORTED['final_f1']:.3f}\nPR-AUC={REPORTED['final_pr_auc']:.3f}",
        "#fee0d2", 9, "bold")

    arrow(4.6, 4.3, 4.6, 3.6)
    arrow(7.3, 4.3, 7.3, 3.6)
    arrow(10.0, 4.3, 10.0, 3.6)
    arrow(12.15, 4.3, 12.15, 3.6)

    # 評価
    box(3.5, 1.2, 6.0, 1.4,
        "評価 vs True_Bleed.srt (3区間)\n"
        "・フレーム PR-AUC / ROC-AUC\n"
        "・区間 F1 (時間 IoU ≥ 0.3)",
        "#f7f7f7", 11)
    arrow(6.5, 2.9, 6.5, 2.6)

    box(10.0, 1.2, 2.8, 1.4,
        "最終形\nBC×HV + 0.5×expansion\n"
        f"PR-AUC = {REPORTED['final_pr_auc']:.3f}\nF1 = {REPORTED['final_f1']:.3f}",
        "#fee0d2", 11, "bold")
    arrow(12.1, 2.9, 11.4, 2.6)

    # フッター
    ax.text(6.5, 0.4, "教師データ: 単一動画3区間 / 制約: RTX 4070 12GB / 教師あり学習なし",
            ha="center", fontsize=9, style="italic", color="#555")

    plt.savefig(OUT_DIR / "fig01_overview.png")
    plt.close()
    print(f"  fig01_overview.png")


# ===========================================================================
# Figure 2: Phase 1 各モデルの性能比較
# ===========================================================================

def fig_phase1_models() -> None:
    truth = load_truth_intervals(TRUTH_SRT)
    models = ["BiomedCLIP", "PeskaVLP", "SurgVLP", "HecVL", "BC×HV"]
    metrics = {}
    for m in models:
        t, s = load_csv_scores(CSVS[m])
        metrics[m] = _compute_metrics(t, s, truth)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    keys = ["pr_auc", "roc_auc", "f1"]
    titles = ["PR-AUC", "ROC-AUC", "区間 F1 (best)"]

    for ax, key, title in zip(axes, keys, titles):
        vals = [metrics[m][key] for m in models]
        colors = [MODEL_COLORS[m] for m in models]
        bars = ax.bar(models, vals, color=colors, edgecolor="#333", linewidth=0.7)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylim(0, max(0.95, max(vals) * 1.2))
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.02,
                    f"{v:.3f}", ha="center", fontsize=10)
        ax.tick_params(axis="x", rotation=20)

    fig.suptitle("Phase 1: 各 zero-shot VLP モデルの出血検出性能",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.savefig(OUT_DIR / "fig02_phase1_models.png")
    plt.close()
    print(f"  fig02_phase1_models.png")


# ===========================================================================
# Figure 3: 区間別スコアヒートマップ
# ===========================================================================

def fig_interval_heatmap() -> None:
    truth = load_truth_intervals(TRUTH_SRT)
    models = ["BiomedCLIP", "PeskaVLP", "SurgVLP", "HecVL", "BC×HV",
              "DINOv2_anom", "Bootstrap", "BC×HV+0.5×exp"]
    matrix = []
    for m in models:
        t, s = load_csv_scores(CSVS[m])
        matrix.append(_interval_means(t, s, truth))

    matrix = np.array(matrix)  # (models, 3 intervals)

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(3))
    ax.set_xticklabels(TRUTH_LABELS, fontsize=10)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=10)
    for i in range(len(models)):
        for j in range(3):
            color = "white" if matrix[i, j] < 0.3 or matrix[i, j] > 0.7 else "#222"
            ax.text(j, i, f"{matrix[i, j]:.2f}",
                    ha="center", va="center", fontsize=10,
                    color=color, fontweight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("区間内 平均 smooth_score", fontsize=10)
    ax.set_title("各モデルの区間別検出強度（平均スコア）",
                 fontsize=13, fontweight="bold")

    # 補足注記
    ax.text(1, len(models) + 0.4,
            "緑=高スコア（よく検出） / 赤=低スコア（検出弱）",
            ha="center", fontsize=9, color="#555", style="italic")

    plt.savefig(OUT_DIR / "fig03_interval_heatmap.png")
    plt.close()
    print(f"  fig03_interval_heatmap.png")


# ===========================================================================
# Figure 4: タイムライン比較
# ===========================================================================

def fig_timeline() -> None:
    truth = load_truth_intervals(TRUTH_SRT)
    show = ["BiomedCLIP", "HecVL", "BC×HV", "BC×HV+0.5×exp"]

    fig, axes = plt.subplots(len(show), 1, figsize=(14, 8), sharex=True)
    for ax, m in zip(axes, show):
        t, s = load_csv_scores(CSVS[m])
        ax.plot(t, s, color=MODEL_COLORS[m], lw=0.9)
        ax.fill_between(t, 0, s, color=MODEL_COLORS[m], alpha=0.15)
        for i, (a, b) in enumerate(truth):
            ax.axvspan(a, b, color="red", alpha=0.18, lw=0)
        ax.set_ylim(0, 1)
        ax.set_ylabel(m, fontsize=10, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.3)

    # 教師ラベル注記（最上段）
    for i, ((a, b), lbl) in enumerate(zip(truth, ["区間1", "区間2", "区間3"])):
        axes[0].text((a + b) / 2, 1.05, lbl, ha="center",
                     fontsize=9, color="#a00", fontweight="bold")

    axes[-1].set_xlabel("time (s)", fontsize=11)
    fig.suptitle("zero-shot スコアのタイムライン（赤帯 = True_Bleed 区間）",
                 fontsize=13, fontweight="bold", y=0.995)
    plt.subplots_adjust(hspace=0.18)
    plt.savefig(OUT_DIR / "fig04_timeline.png")
    plt.close()
    print(f"  fig04_timeline.png")


# ===========================================================================
# Figure 5: BC×HV と expansion の補完性
# ===========================================================================

def fig_complementary() -> None:
    truth = load_truth_intervals(TRUTH_SRT)
    t_base, bc = load_csv_scores(CSVS["BiomedCLIP"])
    _, hv = load_csv_scores(CSVS["HecVL"])
    bl = _load_redlog(BLEEDLOG)
    expansion = _resample(bl["t_sec"], bl["smooth_expansion"], t_base)

    bcxhv = _normalize(bc * hv)
    expn = _normalize(expansion)
    fused = _normalize(bcxhv + 0.5 * expn)
    fused = np.array(smooth_center(fused.tolist(), 5))

    # 上段: 2 シグナル時系列、下段: 区間平均バー
    fig = plt.figure(figsize=(14, 7.5))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.2],
                          hspace=0.45, wspace=0.25)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, :], sharex=ax1)
    ax3 = fig.add_subplot(gs[2, 0])
    ax4 = fig.add_subplot(gs[2, 1])

    # 時系列
    for ax, sig, name, color in [
        (ax1, bcxhv, "BC×HV (意味シグナル)", "#31a354"),
        (ax2, expn, "expansion (赤色急拡大の物理シグナル)", "#d62728"),
    ]:
        ax.plot(t_base, sig, color=color, lw=0.9)
        ax.fill_between(t_base, 0, sig, color=color, alpha=0.2)
        for a, b in truth:
            ax.axvspan(a, b, color="red", alpha=0.15, lw=0)
        ax.set_ylim(0, 1)
        ax.set_ylabel(name, fontsize=10, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.3)

    for i, ((a, b), lbl) in enumerate(zip(truth, ["区間1", "区間2", "区間3"])):
        ax1.text((a + b) / 2, 1.06, lbl, ha="center",
                 fontsize=9, color="#a00", fontweight="bold")

    ax2.set_xlabel("time (s)", fontsize=11)

    # 区間平均バー（補完性の数字証拠）
    bcxhv_means = [bcxhv[(t_base >= a) & (t_base <= b)].mean() for a, b in truth]
    exp_means = [expn[(t_base >= a) & (t_base <= b)].mean() for a, b in truth]

    x = np.arange(3)
    w = 0.35
    ax3.bar(x - w/2, bcxhv_means, w, label="BC×HV",
            color="#31a354", edgecolor="#333")
    ax3.bar(x + w/2, exp_means, w, label="expansion",
            color="#d62728", edgecolor="#333")
    ax3.set_xticks(x)
    ax3.set_xticklabels(["区間1", "区間2", "区間3"])
    ax3.set_ylabel("区間内 平均スコア")
    ax3.set_title("シグナルは互いに補完的", fontsize=11, fontweight="bold")
    ax3.legend(fontsize=9)
    ax3.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax3.set_axisbelow(True)
    for i, (b, e) in enumerate(zip(bcxhv_means, exp_means)):
        ax3.text(i - w/2, b + 0.02, f"{b:.2f}", ha="center", fontsize=9)
        ax3.text(i + w/2, e + 0.02, f"{e:.2f}", ha="center", fontsize=9)

    # 融合スコアタイムライン
    ax4.plot(t_base, fused, color="#762a83", lw=1.0)
    ax4.fill_between(t_base, 0, fused, color="#762a83", alpha=0.25)
    for a, b in truth:
        ax4.axvspan(a, b, color="red", alpha=0.18, lw=0)
    ax4.set_ylim(0, 1)
    ax4.set_xlabel("time (s)")
    ax4.set_ylabel("融合スコア")
    ax4.set_title("BC×HV + 0.5×expansion (融合)", fontsize=11, fontweight="bold")
    ax4.grid(True, linestyle="--", alpha=0.3)

    fig.suptitle("意味シグナル × 物理シグナルの補完融合 (Phase 3)",
                 fontsize=13, fontweight="bold", y=0.995)
    plt.savefig(OUT_DIR / "fig05_complementary.png")
    plt.close()
    print(f"  fig05_complementary.png")


# ===========================================================================
# Figure 6: PR-AUC 推移
# ===========================================================================

def fig_progression() -> None:
    """各 Phase の代表値で PR-AUC 推移を示す。"""
    phases = [
        ("Phase 1\nBiomedCLIP\n単独", REPORTED["biomedclip_pr_auc"], "#9ecae1"),
        ("Phase 1\nBC×HV\nensemble", REPORTED["bcxhv_pr_auc"], "#31a354"),
        ("Phase 2\nDINOv2\nanomaly", 0.071, "#bcbcbc"),
        ("Phase 2'\nBootstrap", 0.151, "#969696"),
        ("Phase 3\nBC×HV+0.5×exp\n(最終形)", REPORTED["final_pr_auc"], "#d62728"),
    ]
    base = REPORTED["biomedclip_pr_auc"]
    final = REPORTED["final_pr_auc"]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    names = [p[0] for p in phases]
    vals = [p[1] for p in phases]
    colors = [p[2] for p in phases]

    bars = ax.bar(names, vals, color=colors, edgecolor="#333", linewidth=0.8)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.012,
                f"{v:.3f}", ha="center", fontsize=11, fontweight="bold")

    # 改善線（最良の進行）
    ax.axhline(base, color="#888", linestyle=":", lw=1, alpha=0.7)
    ax.text(4.45, base + 0.010, "Phase 1 ベースライン", color="#666", fontsize=9)
    ax.annotate("",
                xy=(4, final), xytext=(0, base),
                arrowprops=dict(arrowstyle="->", color="#d62728",
                                lw=1.8, alpha=0.6, connectionstyle="arc3,rad=-0.3"))
    ax.text(2, 0.50, f"PR-AUC を {base:.3f} → {final:.3f} に改善",
            color="#d62728", fontsize=11, fontweight="bold", ha="center")

    ax.set_ylabel("PR-AUC", fontsize=12)
    ax.set_ylim(0, 0.6)
    ax.set_title("各 Phase の代表モデルにおける PR-AUC 推移",
                 fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    plt.savefig(OUT_DIR / "fig06_progression.png")
    plt.close()
    print(f"  fig06_progression.png")


# ===========================================================================
# Figure 7: 融合重み α スイープ
# ===========================================================================

def fig_alpha_sweep() -> None:
    truth = load_truth_intervals(TRUTH_SRT)
    t_base, bc = load_csv_scores(CSVS["BiomedCLIP"])
    _, hv = load_csv_scores(CSVS["HecVL"])
    bl = _load_redlog(BLEEDLOG)
    expansion = _resample(bl["t_sec"], bl["smooth_expansion"], t_base)
    n_bcxhv = _normalize(bc * hv)
    n_exp = _normalize(expansion)

    alphas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
    pr_aucs, f1s, tps, fps_, region1_means = [], [], [], [], []
    for a in alphas:
        fused = _normalize(n_bcxhv + a * n_exp)
        fused = np.array(smooth_center(fused.tolist(), 5))
        m = _compute_metrics(t_base, fused, truth)
        pr_aucs.append(m["pr_auc"])
        f1s.append(m["f1"])
        tps.append(m["tp"])
        fps_.append(m["fp"])
        region1_means.append(
            float(fused[(t_base >= truth[0][0]) & (t_base <= truth[0][1])].mean())
        )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.0))

    # 左: PR-AUC, F1, 区間1スコア
    ax = axes[0]
    ax.plot(alphas, pr_aucs, "o-", label="PR-AUC", color="#31a354", lw=2)
    ax.plot(alphas, f1s, "s-", label="best F1", color="#d62728", lw=2)
    ax.plot(alphas, region1_means, "^--",
            label="区間1 mean score", color="#fd8d3c", lw=1.6)
    ax.set_xlabel("α (expansion 重み)", fontsize=11)
    ax.set_ylabel("score", fontsize=11)
    ax.set_title("BC×HV + α×expansion の性能曲線", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_ylim(0, 0.65)
    ax.axvline(0.5, color="#666", lw=1, linestyle=":")
    ax.annotate("最良 (α=0.5)",
                xy=(0.5, REPORTED["final_pr_auc"]), xytext=(0.9, 0.55),
                fontsize=10, color="#333",
                arrowprops=dict(arrowstyle="->", color="#333", lw=1))

    # 右: TP, FP の変化
    ax = axes[1]
    ax2 = ax.twinx()
    line1 = ax.plot(alphas, tps, "o-", color="#1b9e77", lw=2,
                    label="TP (検出区間数)")
    line2 = ax2.plot(alphas, fps_, "s-", color="#888", lw=2, alpha=0.8,
                     label="FP (誤検出数)")
    ax.set_xlabel("α (expansion 重み)", fontsize=11)
    ax.set_ylabel("TP", fontsize=11, color="#1b9e77")
    ax2.set_ylabel("FP", fontsize=11, color="#888")
    ax.set_title("α 増加で全区間検出可能 vs 誤検出爆発",
                 fontsize=12, fontweight="bold")
    ax.set_yticks([0, 1, 2, 3])
    ax.set_ylim(-0.2, 3.3)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    # アノテーション
    ax.annotate("α=2.0: TP=2",
                xy=(2.0, tps[alphas.index(2.0)]),
                xytext=(2.1, 1.3), fontsize=9, color="#1b9e77",
                arrowprops=dict(arrowstyle="->", color="#1b9e77", lw=1))
    ax.annotate("α=3.0: TP=3 達成\n(全区間検出)",
                xy=(3.0, tps[alphas.index(3.0)]),
                xytext=(2.0, 2.3), fontsize=9, color="#1b9e77", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#1b9e77", lw=1))

    lines = line1 + line2
    ax.legend(lines, [l.get_label() for l in lines], loc="upper left", fontsize=10)

    fig.suptitle("Phase 3: 融合重み α のスイープ — F1の天井と全区間検出のトレードオフ",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.savefig(OUT_DIR / "fig07_alpha_sweep.png")
    plt.close()
    print(f"  fig07_alpha_sweep.png")


# ===========================================================================
# Figure 8: 最終比較サマリー（表）
# ===========================================================================

def fig_summary_table() -> None:
    truth = load_truth_intervals(TRUTH_SRT)

    rows = []
    for m in ["BiomedCLIP", "PeskaVLP", "SurgVLP", "HecVL", "BC×HV",
              "DINOv2_anom", "Bootstrap", "BC×HV+0.5×exp"]:
        t, s = load_csv_scores(CSVS[m])
        metric = _compute_metrics(t, s, truth)
        means = _interval_means(t, s, truth)
        rows.append([
            m,
            f"{metric['pr_auc']:.3f}",
            f"{metric['roc_auc']:.3f}",
            f"{metric['f1']:.3f}",
            f"{metric['tp']}/{metric['fp']}/{metric['fn']}",
            f"{means[0]:.2f}", f"{means[1]:.2f}", f"{means[2]:.2f}",
        ])

    headers = ["モデル", "PR-AUC", "ROC-AUC", "F1", "TP/FP/FN",
               "区間1\nmean", "区間2\nmean", "区間3\nmean"]

    fig, ax = plt.subplots(figsize=(14, 5.2))
    ax.axis("off")
    table = ax.table(
        cellText=rows, colLabels=headers,
        cellLoc="center", loc="center",
        colWidths=[0.20] + [0.10] * 7,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)

    # ヘッダ装飾
    for j, _ in enumerate(headers):
        cell = table[0, j]
        cell.set_facecolor("#444")
        cell.set_text_props(color="white", fontweight="bold")

    # 最終ベスト行を強調
    final_row_idx = len(rows)  # 最後の行
    for j in range(len(headers)):
        cell = table[final_row_idx, j]
        cell.set_facecolor("#fee0d2")
        cell.set_text_props(fontweight="bold")

    # 行帯（縞）
    for i in range(1, len(rows) + 1):
        if i == final_row_idx:
            continue
        if i % 2 == 0:
            for j in range(len(headers)):
                table[i, j].set_facecolor("#f7f7f7")

    fig.suptitle("全 zero-shot 手法の比較サマリー",
                 fontsize=14, fontweight="bold", y=0.97)
    fig.text(0.5, 0.04,
             "赤帯 = 最終ベスト (Phase 3 融合)。区間mean は smooth_score の区間内平均。",
             ha="center", fontsize=9, color="#555", style="italic")

    plt.savefig(OUT_DIR / "fig08_summary_table.png")
    plt.close()
    print(f"  fig08_summary_table.png")


# ===========================================================================
# main
# ===========================================================================

def _check_inputs() -> List[str]:
    """図生成に必要な入力ファイルの不足を返す（空なら全て存在）。"""
    required = [TRUTH_SRT, BLEEDLOG, *CSVS.values()]
    return [p for p in required if not Path(p).exists()]


def _verify_reported_metrics(tol: float = 0.02) -> List[str]:
    """図に焼き込んだ REPORTED 値を実データから再計算し、乖離を警告として返す。

    データ・前処理・モデルが変わっても図の数値が古いまま残るのを検知する。
    計算に失敗（列欠落等）したキーはスキップする。
    """
    truth = load_truth_intervals(TRUTH_SRT)
    cache: Dict[str, Dict] = {}
    warnings: List[str] = []
    for key, reported in REPORTED.items():
        csv_key = _REPORTED_CSV[key]
        try:
            if csv_key not in cache:
                t, s = load_csv_scores(CSVS[csv_key])
                cache[csv_key] = _compute_metrics(t, s, truth)
            metric = "f1" if key.endswith("_f1") else "pr_auc"
            actual = cache[csv_key][metric]
        except Exception as e:  # 計算不能な場合は検証スキップ
            warnings.append(f"{key}: 検証スキップ（{e}）")
            continue
        if abs(actual - reported) > tol:
            warnings.append(
                f"{key}: 図の表示値 {reported:.3f} と実測 {actual:.3f} が乖離 "
                f"(|Δ|={abs(actual - reported):.3f} > {tol})"
            )
    return warnings


def main() -> int:
    # 事前検証: 1枚でも入力が欠けていれば、部分的な不完全出力を残さず即失敗する
    missing = _check_inputs()
    if missing:
        import sys
        print("エラー: 必要な入力ファイルが見つかりません:", file=sys.stderr)
        for p in missing:
            print(f"  - {p}", file=sys.stderr)
        print("先に評価/融合パイプライン（score_video → fuse 等）を実行してください。",
              file=sys.stderr)
        return 1

    # 図に焼き込んだ代表値が実データとずれていないかを検証（警告のみ、生成は継続）
    for w in _verify_reported_metrics():
        print(f"警告[指標ドリフト]: {w}", file=sys.stderr)

    print(f"出力先: {OUT_DIR}")
    fig_overview()
    fig_phase1_models()
    fig_interval_heatmap()
    fig_timeline()
    fig_complementary()
    fig_progression()
    fig_alpha_sweep()
    fig_summary_table()
    print(f"\n全 8 図を生成完了 → {OUT_DIR}/")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
