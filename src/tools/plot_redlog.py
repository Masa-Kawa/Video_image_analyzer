"""
赤色率ログCSV → PNGグラフ 変換ツール

赤色率ログCSV（_redlog.csv / _bleedlog.csv）の時系列データをグラフ化し、PNG画像に出力する。

_redlog.csv の場合（2パネル）:
  - 上パネル: red_ratio（赤色率）の時系列
  - 下パネル: smooth_delta（平滑化変化量）の時系列 + 閾値ライン

_bleedlog.csv の場合（3パネル）:
  - 上パネル: red_ratio（赤色率）の時系列
  - 中パネル: newly_red_ratio（新規赤化画素率）と bg_stability（背景安定度）
  - 下パネル: smooth_expansion（平滑化赤色拡大率）+ 閾値ライン

使用例:
    python -m src.tools.plot_redlog \\
        --in-csv out/case001_redlog.csv \\
        --out-png out/case001_plot.png \\
        --thr 0.03

    python -m src.tools.plot_redlog \\
        --in-csv out/case001_bleedlog.csv \\
        --out-png out/case001_bleed_plot.png \\
        --thr 0.005
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # GUIバックエンドを使用しない（ヘッドレス対応）
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


# ---------------------------------------------------------------------------
# 日本語フォント設定
# ---------------------------------------------------------------------------

def _setup_japanese_font() -> None:
    """利用可能な日本語フォントを検索して設定する。"""
    # 優先順位で日本語フォントを検索
    preferred = ["IPAexGothic", "Noto Sans CJK JP", "TakaoPGothic", "VL PGothic"]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in preferred:
        if name in available:
            plt.rcParams["font.family"] = name
            return
    # 見つからない場合は sans-serif のまま（警告は出る）


_setup_japanese_font()


# ---------------------------------------------------------------------------
# CSVタイプ判定
# ---------------------------------------------------------------------------

def _detect_csv_type(csv_path: str) -> str:
    """
    CSVのヘッダからタイプを判定する。

    Returns:
        "redlog", "bleedlog", or "spreadlog"
    """
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, [])
    if "smooth_spread" in header:
        return "spreadlog"
    if "smooth_expansion" in header:
        return "bleedlog"
    return "redlog"


# ---------------------------------------------------------------------------
# redlog グラフ生成（2パネル）
# ---------------------------------------------------------------------------

def plot_redlog(
    csv_path: str,
    out_png: str,
    thr: Optional[float] = None,
    title: Optional[str] = None,
    figsize: tuple = (14, 7),
) -> str:
    """
    赤色率ログCSVから2パネルの時系列グラフをPNG出力する。

    Args:
        csv_path: 入力CSVファイルパス（赤色率ログ）
        out_png: 出力PNGファイルパス
        thr: 閾値ライン（Noneなら描画しない）
        title: グラフタイトル（Noneならファイル名から自動生成）
        figsize: 図のサイズ（幅, 高さ）インチ

    Returns:
        出力PNGファイルパス
    """
    from src.red.redlog import read_redlog_csv

    data = read_redlog_csv(csv_path)
    times = data["times"]
    ratios = data["ratios"]
    smooth_deltas = data["smooth_deltas"]

    if len(times) == 0:
        print("警告: CSVにデータがありません。空のグラフを出力します。", file=sys.stderr)

    # タイトルの自動生成
    if title is None:
        stem = Path(csv_path).stem.replace("_redlog", "")
        title = f"赤色率解析: {stem}"

    # --- グラフ作成 ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # 上パネル: red_ratio
    ax1.plot(times, ratios, color="#e74c3c", linewidth=0.8, alpha=0.9)
    ax1.fill_between(times, ratios, alpha=0.15, color="#e74c3c")
    ax1.set_ylabel("赤色率 (red_ratio)")
    ax1.set_ylim(bottom=0)
    ax1.grid(True, alpha=0.3)
    ax1.legend(["red_ratio"], loc="upper right")

    # 下パネル: smooth_delta
    ax2.plot(times, smooth_deltas, color="#3498db", linewidth=0.8, alpha=0.9)
    ax2.fill_between(times, smooth_deltas, alpha=0.15, color="#3498db")
    ax2.set_ylabel("平滑化変化量 (smooth_delta)")
    ax2.set_xlabel("時間 (秒)")
    ax2.grid(True, alpha=0.3)

    # 閾値ラインの描画
    if thr is not None:
        ax2.axhline(y=thr, color="#e67e22", linestyle="--", linewidth=1.2,
                     label=f"閾値 (thr={thr})")
        ax2.legend(loc="upper right")
    else:
        ax2.legend(["smooth_delta"], loc="upper right")

    plt.tight_layout()

    # --- 出力 ---
    out_path = Path(out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    return str(out_path)


# ---------------------------------------------------------------------------
# bleedlog グラフ生成（3パネル）
# ---------------------------------------------------------------------------

def plot_bleedlog(
    csv_path: str,
    out_png: str,
    thr: Optional[float] = None,
    title: Optional[str] = None,
    figsize: tuple = (14, 10),
) -> str:
    """
    赤色拡大ログCSVから3パネルの時系列グラフをPNG出力する。

    Args:
        csv_path: 入力CSVファイルパス（赤色拡大ログ）
        out_png: 出力PNGファイルパス
        thr: 閾値ライン（Noneなら描画しない）
        title: グラフタイトル（Noneならファイル名から自動生成）
        figsize: 図のサイズ（幅, 高さ）インチ

    Returns:
        出力PNGファイルパス
    """
    from src.red.bleed_detector import read_bleedlog_csv

    data = read_bleedlog_csv(csv_path)
    times = data["times"]
    red_ratios = data["red_ratios"]
    newly_red_ratios = data["newly_red_ratios"]
    bg_stabilities = data["bg_stabilities"]
    smooth_expansions = data["smooth_expansions"]

    if len(times) == 0:
        print("警告: CSVにデータがありません。空のグラフを出力します。", file=sys.stderr)

    # タイトルの自動生成
    if title is None:
        stem = Path(csv_path).stem.replace("_bleedlog", "")
        title = f"赤色拡大解析: {stem}"

    # --- グラフ作成（3パネル） ---
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize, sharex=True)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # 上パネル: red_ratio
    ax1.plot(times, red_ratios, color="#e74c3c", linewidth=0.8, alpha=0.9)
    ax1.fill_between(times, red_ratios, alpha=0.15, color="#e74c3c")
    ax1.set_ylabel("赤色率 (red_ratio)")
    ax1.set_ylim(bottom=0)
    ax1.grid(True, alpha=0.3)
    ax1.legend(["red_ratio"], loc="upper right")

    # 中パネル: newly_red_ratio + bg_stability
    ax2.plot(times, newly_red_ratios, color="#9b59b6", linewidth=0.8, alpha=0.9,
             label="newly_red_ratio")
    ax2_twin = ax2.twinx()
    ax2_twin.plot(times, bg_stabilities, color="#2ecc71", linewidth=0.8, alpha=0.7,
                  label="bg_stability")
    ax2.set_ylabel("新規赤化率 (newly_red_ratio)")
    ax2_twin.set_ylabel("背景安定度 (bg_stability)")
    ax2_twin.set_ylim(0, 1.1)
    ax2.grid(True, alpha=0.3)
    # 凡例を統合
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    # 下パネル: smooth_expansion
    ax3.plot(times, smooth_expansions, color="#3498db", linewidth=0.8, alpha=0.9)
    ax3.fill_between(times, smooth_expansions, alpha=0.15, color="#3498db")
    ax3.set_ylabel("平滑化拡大率 (smooth_expansion)")
    ax3.set_xlabel("時間 (秒)")
    ax3.grid(True, alpha=0.3)

    # 閾値ラインの描画
    if thr is not None:
        ax3.axhline(y=thr, color="#e67e22", linestyle="--", linewidth=1.2,
                     label=f"閾値 (thr={thr})")
        ax3.legend(loc="upper right")
    else:
        ax3.legend(["smooth_expansion"], loc="upper right")

    plt.tight_layout()

    # --- 出力 ---
    out_path = Path(out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    return str(out_path)


# ---------------------------------------------------------------------------
# spreadlog グラフ生成（3パネル）
# ---------------------------------------------------------------------------

def plot_spreadlog(
    csv_path: str,
    out_png: str,
    thr: Optional[float] = None,
    title: Optional[str] = None,
    figsize: tuple = (14, 10),
) -> str:
    """
    局所拡散スコアログCSVから3パネルの時系列グラフをPNG出力する。

    Args:
        csv_path: 入力CSVファイルパス（拡散スコアログ）
        out_png: 出力PNGファイルパス
        thr: 閾値ライン（Noneなら描画しない）
        title: グラフタイトル（Noneならファイル名から自動生成）
        figsize: 図のサイズ（幅, 高さ）インチ

    Returns:
        出力PNGファイルパス
    """
    from src.red.bleed_spread import read_spreadlog_csv

    data = read_spreadlog_csv(csv_path)
    times = data["times"]
    red_ratios = data["red_ratios"]
    max_cell_deltas = data["max_cell_deltas"]
    delta_stds = data["delta_stds"]
    smooth_spreads = data["smooth_spreads"]

    if len(times) == 0:
        print("警告: CSVにデータがありません。空のグラフを出力します。", file=sys.stderr)

    # タイトルの自動生成
    if title is None:
        stem = Path(csv_path).stem.replace("_spreadlog", "")
        title = f"局所拡散解析: {stem}"

    # --- グラフ作成（3パネル） ---
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize, sharex=True)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # 上パネル: red_ratio
    ax1.plot(times, red_ratios, color="#e74c3c", linewidth=0.8, alpha=0.9)
    ax1.fill_between(times, red_ratios, alpha=0.15, color="#e74c3c")
    ax1.set_ylabel("赤色率 (red_ratio)")
    ax1.set_ylim(bottom=0)
    ax1.grid(True, alpha=0.3)
    ax1.legend(["red_ratio"], loc="upper right")

    # 中パネル: max_cell_delta + delta_std
    ax2.plot(times, max_cell_deltas, color="#e67e22", linewidth=0.8, alpha=0.9,
             label="max_cell_delta")
    ax2_twin = ax2.twinx()
    ax2_twin.plot(times, delta_stds, color="#9b59b6", linewidth=0.8, alpha=0.7,
                  label="delta_std")
    ax2.set_ylabel("最大セル差分 (max_cell_delta)")
    ax2_twin.set_ylabel("差分標準偏差 (delta_std)")
    ax2.grid(True, alpha=0.3)
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    # 下パネル: smooth_spread
    ax3.plot(times, smooth_spreads, color="#3498db", linewidth=0.8, alpha=0.9)
    ax3.fill_between(times, smooth_spreads, alpha=0.15, color="#3498db")
    ax3.set_ylabel("平滑化拡散スコア (smooth_spread)")
    ax3.set_xlabel("時間 (秒)")
    ax3.grid(True, alpha=0.3)

    # 閾値ラインの描画
    if thr is not None:
        ax3.axhline(y=thr, color="#e67e22", linestyle="--", linewidth=1.2,
                     label=f"閾値 (thr={thr})")
        ax3.legend(loc="upper right")
    else:
        ax3.legend(["smooth_spread"], loc="upper right")

    plt.tight_layout()

    # --- 出力 ---
    out_path = Path(out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    return str(out_path)


# ---------------------------------------------------------------------------
# 統合プロット関数（自動判定）
# ---------------------------------------------------------------------------

def plot_auto(
    csv_path: str,
    out_png: str,
    thr: Optional[float] = None,
    title: Optional[str] = None,
) -> str:
    """
    CSVのタイプを自動判定してグラフを出力する。

    Args:
        csv_path: 入力CSVファイルパス
        out_png: 出力PNGファイルパス
        thr: 閾値ライン
        title: グラフタイトル

    Returns:
        出力PNGファイルパス
    """
    csv_type = _detect_csv_type(csv_path)
    if csv_type == "spreadlog":
        return plot_spreadlog(csv_path, out_png, thr=thr, title=title)
    elif csv_type == "bleedlog":
        return plot_bleedlog(csv_path, out_png, thr=thr, title=title)
    else:
        return plot_redlog(csv_path, out_png, thr=thr, title=title)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    """コマンドラインエントリポイント"""
    parser = argparse.ArgumentParser(
        description="赤色率ログCSVを時系列グラフ（PNG）に変換する（redlog/bleedlog/spreadlog自動判定）"
    )
    parser.add_argument("--in-csv", required=True,
                        help="入力CSVファイル（_redlog.csv / _bleedlog.csv / _spreadlog.csv）")
    parser.add_argument("--out-png", required=True,
                        help="出力PNGファイル")
    parser.add_argument("--thr", type=float, default=None,
                        help="閾値ライン（下パネルに描画）")
    parser.add_argument("--title", default=None,
                        help="グラフタイトル（デフォルト: ファイル名から自動生成）")

    args = parser.parse_args()

    out = plot_auto(
        csv_path=args.in_csv,
        out_png=args.out_png,
        thr=args.thr,
        title=args.title,
    )
    print(f"PNG出力: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
