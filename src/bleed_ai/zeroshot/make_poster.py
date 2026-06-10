"""縦長ポスター（870×1850mm, 150dpi）の生成。

レイアウト:
  上段: 演題番号 (180×180mm) + タイトル帯 (670×180mm)
  本体: 2 列 × 4 段 = 8 パネル
  各パネル: 約 400×380mm

出力:
  out_lapc_eval/zs/poster/poster.png   ── PosteRazor で分割印刷する元画像
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib import font_manager


# ---------------------------------------------------------------------------
# フォント
# ---------------------------------------------------------------------------

for p in [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
]:
    if Path(p).exists():
        font_manager.fontManager.addfont(p)
plt.rcParams["font.family"] = ["Noto Sans CJK JP"]
plt.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------------------
# レイアウト定数（mm）
# ---------------------------------------------------------------------------

W_MM = 870
H_MM = 1850
DPI = 150

MARGIN = 25
GUTTER = 30
TITLE_H = 180
TITLE_PNL_W = 180  # 演題番号エリア
N_ROWS = 4
N_COLS = 2

BODY_TOP = H_MM - MARGIN - TITLE_H - GUTTER
BODY_BOTTOM = MARGIN
BODY_H = BODY_TOP - BODY_BOTTOM
PANEL_W = (W_MM - 2 * MARGIN - GUTTER) / N_COLS
PANEL_H = (BODY_H - (N_ROWS - 1) * GUTTER) / N_ROWS

# カラーテーマ
COLOR_TITLE = "#1f4e79"
COLOR_PANEL_BG = "#666"
COLOR_VLP = "#2b8cbe"        # 青系（VLP）
COLOR_PHASE = "#31a354"      # 緑系（フェーズ認識）
COLOR_PHYSICAL = "#fd8d3c"   # 橙系（物理シグナル）
COLOR_FINAL = "#762a83"      # 紫系（最終形）
COLOR_NEUTRAL = "#666"       # 灰系（背景・結論）

OUT_PATH = Path("out_lapc_eval/zs/poster")
OUT_PATH.mkdir(parents=True, exist_ok=True)
FIG_DIR = Path("out_lapc_eval/zs/figures_print")  # 300dpi figures


# ---------------------------------------------------------------------------
# 描画ヘルパ
# ---------------------------------------------------------------------------

def add_box(fig, left_mm, bottom_mm, w_mm, h_mm):
    """mm 単位で axes を追加。"""
    ax = fig.add_axes([
        left_mm / W_MM, bottom_mm / H_MM,
        w_mm / W_MM, h_mm / H_MM,
    ])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return ax


def embed_image(ax, path, inset_rect):
    """図PNGを inset として埋め込む。不在時はプレースホルダを描いて続行する。

    研究成果出力（ポスター）の最終段で1枚の図が無いだけで全体が落ちないよう、
    存在チェックして欠落を可視化する（FileNotFoundError でクラッシュさせない）。
    """
    img_ax = ax.inset_axes(inset_rect)
    img_ax.axis("off")
    p = Path(path)
    if not p.exists():
        print(f"警告: 図が見つかりません（プレースホルダ表示）: {p}", file=sys.stderr)
        img_ax.add_patch(mpatches.Rectangle(
            (0, 0), 1, 1, facecolor="#eeeeee", edgecolor="#bbbbbb"))
        img_ax.text(0.5, 0.5, f"[図なし]\n{p.name}", ha="center", va="center",
                    fontsize=11, color="#888888", family="Noto Sans CJK JP")
        img_ax.set_xlim(0, 1)
        img_ax.set_ylim(0, 1)
        return img_ax
    img_ax.imshow(mpimg.imread(str(p)))
    return img_ax


def panel_position(col, row):
    """col 0=左, 1=右; row 0=最上段, 3=最下段。"""
    left = MARGIN + col * (PANEL_W + GUTTER)
    bottom = BODY_BOTTOM + (N_ROWS - 1 - row) * (PANEL_H + GUTTER)
    return left, bottom, PANEL_W, PANEL_H


def draw_panel(fig, col, row, title, color, body_render):
    """1 パネルを描く。body_render(ax) でパネル内描画を委譲。"""
    l, b, w, h = panel_position(col, row)
    ax = add_box(fig, l, b, w, h)

    # 外枠
    ax.add_patch(mpatches.Rectangle(
        (0, 0), 1, 1, fill=False,
        edgecolor=color, linewidth=2.5, transform=ax.transAxes,
    ))
    # ヘッダ
    header_h = 0.10
    ax.add_patch(mpatches.Rectangle(
        (0, 1 - header_h), 1, header_h,
        facecolor=color, edgecolor="none", transform=ax.transAxes,
    ))
    ax.text(0.025, 1 - header_h / 2, title,
            fontsize=22, fontweight="bold", va="center", color="white",
            transform=ax.transAxes)

    # 本文領域（ヘッダの下）
    body_ax = add_box(
        fig,
        l + 8, b + 8,
        w - 16, h - 16 - (PANEL_H * header_h),
    )
    body_render(body_ax)


# ---------------------------------------------------------------------------
# パネル本文の描画関数
# ---------------------------------------------------------------------------

def panel1_intro(ax):
    txt = (
        "■ 本研究の目的\n"
        "  (1) SRT/JSONL を中心としたアノテーション基盤\n"
        "  (2) 基盤的な手術フェーズ認識\n"
        "  (3) 出血検出本体（教師なし抽出）\n\n"
        "■ 評価動画\n"
        "  LapC_EvalDemo_480p.MP4\n"
        "  854×480, 60fps, 34分, 122,225 frames\n\n"
        "■ 教師アノテーション True_Bleed.srt\n"
        "  区間1: 6:32 – 6:44   ( 11.8 秒・軽度 )\n"
        "  区間2: 7:35 – 8:11   ( 36.9 秒・中規模 )\n"
        "  区間3: 8:28 – 10:46  (137.7 秒・大出血 )\n\n"
        "■ 制約\n"
        "  RTX 4070 / VRAM 12 GB\n"
        "  単一動画のみ・教師あり学習なし\n"
        "  既存事前学習モデルでどこまで届くかを検証"
    )
    ax.text(0.02, 0.98, txt, fontsize=15, va="top", family="Noto Sans CJK JP",
            transform=ax.transAxes)


def panel2_annotation(ax):
    txt = (
        "■ 設計思想\n"
        "  正本: JSONL（機械可読1行1イベント）\n"
        "  ビュー: SRT（Shotcut で人手修正）\n\n"
        "■ 2 ステップ規約 (ANALYZER_SPEC.md)\n"
        "  Step 1  動画 → CSV (時系列指標)\n"
        "  Step 2  CSV → JSONL/SRT (区間化)\n"
        "  → 閾値再探索が動画再処理なしで可能\n\n"
        "■ Human-in-the-loop\n"
        "    SRT を Shotcut で修正\n"
        "       ↓ srt_to_jsonl\n"
        "    JSONL に反映 → 学習データ化\n\n"
        "■ 実装規模\n"
        "  ・主要モジュール 7 種\n"
        "    (jsonl_to_srt, srt_to_jsonl,\n"
        "     merge_srt, csv_to_srt, mlt_generator…)\n"
        "  ・単体テスト 102 件 PASS\n"
        "  ・全解析器（red, bleed_ai, cholec_phase,\n"
        "     transnet, yolo, motion）が遵守"
    )
    ax.text(0.02, 0.98, txt, fontsize=15, va="top", family="Noto Sans CJK JP",
            transform=ax.transAxes)


def panel3_phase(ax):
    txt = (
        "■ モデル構成\n"
        "  ResNet50 (SelfSupSurg DINO, 凍結)\n"
        "      ↓ 2,048 dim 特徴\n"
        "  Bidirectional LSTM (h=512, layers=2)\n"
        "      ↓ FC(256) → ReLU → FC(7)\n"
        "  → 7 フェーズ確率\n\n"
        "■ 2 段階学習で効率化\n"
        "  Stage 1 特徴抽出: 80分（80動画一括）\n"
        "  Stage 2 BiLSTM学習: 80秒（50epoch）\n\n"
        "■ Cholec80 テスト精度 (video41-80)\n"
    )
    ax.text(0.02, 0.98, txt, fontsize=15, va="top", family="Noto Sans CJK JP",
            transform=ax.transAxes)

    # 表
    rows = [
        ("Preparation",            "77.4%"),
        ("CalotTriangleDissection","90.8%"),
        ("ClippingCutting",        "67.1%"),
        ("GallbladderDissection",  "90.7%"),
        ("GallbladderPackaging",   "83.1%"),
        ("CleaningCoagulation",    "76.9%"),
        ("GallbladderRetraction",  "89.1%"),
        ("全体（87.1%）",          "★"),
    ]
    y0 = 0.36
    for i, (k, v) in enumerate(rows):
        bold = "全体" in k
        ax.text(0.04, y0 - i * 0.04, k, fontsize=14,
                fontweight=("bold" if bold else "normal"),
                family="Noto Sans CJK JP",
                color=("#31a354" if bold else "#222"),
                transform=ax.transAxes)
        ax.text(0.78, y0 - i * 0.04, v, fontsize=14,
                fontweight=("bold" if bold else "normal"),
                color=("#31a354" if bold else "#222"),
                transform=ax.transAxes)

    ax.text(0.02, 0.025,
            "■ 評価動画 LapC_EvalDemo への適用\n"
            "  CalotTriangleDissection: 信頼度 0.92\n"
            "  GallbladderDissection: 信頼度 0.92\n"
            "  → 主要術式部分は実用域に到達",
            fontsize=14, va="bottom", family="Noto Sans CJK JP",
            transform=ax.transAxes)


def panel4_initial(ax):
    txt = (
        "■ 試行した 3 アプローチ\n\n"
        "① 赤色解析 (HSV)\n"
        "    redlog, bleed_detector, bleed_spread\n"
        "    + bleed_contact, bleed_flow, bleed_trend\n"
        "    → 偽陽性多。電気メス・灌流・ズーム\n"
        "      でも誤反応\n\n"
        "② RAFT オプティカルフロー\n"
        "    「液体的な動き」を検出\n"
        "    → カメラ移動と分離不能\n\n"
        "③ ResNet-18 + HSV ハイブリッド\n"
        "    severity = bleed_prob × area_ratio\n"
        "             × flow_suppression\n"
        "    → 出血特化の事前学習重みが入手困難\n\n"
        "■ 根本原因\n"
        "  教師動画がほぼ無い状態で従来手法を\n"
        "  組み合わせても精度が出ない\n\n"
        "→ ゼロショット研究へ方針転換\n\n"
        "■ 副産物\n"
        "  bleedlog.csv の smooth_expansion ほか\n"
        "  → Phase 3 物理融合で再利用される\n"
        "    重要な低位特徴"
    )
    ax.text(0.02, 0.98, txt, fontsize=15, va="top", family="Noto Sans CJK JP",
            transform=ax.transAxes)


def panel5_zeroshot(ax):
    txt = (
        "■ 4 種の VLP × プロンプト類似度\n"
    )
    ax.text(0.02, 0.98, txt, fontsize=15, va="top", family="Noto Sans CJK JP",
            transform=ax.transAxes)

    # 図3を埋め込む（ヒートマップが情報密度高い）
    embed_image(ax, FIG_DIR / "fig03_interval_heatmap.png",
                [0.0, 0.18, 1.0, 0.78])

    ax.text(0.02, 0.13,
            "■ 主な発見\n"
            "  BiomedCLIP は大出血特化（区間3=0.52）\n"
            "  HecVL/SurgVLP は中規模出血で強い\n"
            "  BC × HV ensemble が PR-AUC 最良 (0.443)\n"
            "  → ただし 区間1 (11.8秒) は全モデル不可",
            fontsize=14, va="bottom", family="Noto Sans CJK JP",
            transform=ax.transAxes)


def panel6_fusion(ax):
    txt = (
        "■ 意味 × 物理 ─ 完全に補完的"
    )
    ax.text(0.02, 0.98, txt, fontsize=15, va="top", fontweight="bold",
            family="Noto Sans CJK JP", transform=ax.transAxes)

    # 図5を埋め込む
    embed_image(ax, FIG_DIR / "fig05_complementary.png",
                [0.0, 0.32, 1.0, 0.62])

    ax.text(0.02, 0.27,
            "■ シグナル単体の特性（区間平均スコア）\n",
            fontsize=14, va="top", fontweight="bold",
            family="Noto Sans CJK JP", transform=ax.transAxes)

    # 表
    headers = ["シグナル", "区間1", "区間2", "区間3"]
    rows = [
        ("BC × HV (意味)",   "0.001", "0.054", "0.456"),
        ("expansion (物理)", "0.532", "0.449", "0.283"),
    ]
    xs = [0.02, 0.40, 0.60, 0.80]
    y_h = 0.20
    for i, (k, x) in enumerate(zip(headers, xs)):
        ax.text(x, y_h, k, fontsize=13, fontweight="bold",
                family="Noto Sans CJK JP", transform=ax.transAxes)
    for r, row in enumerate(rows):
        y = y_h - 0.05 * (r + 1)
        for v, x in zip(row, xs):
            color = "#fd8d3c" if r == 1 else "#2b8cbe"
            ax.text(x, y, v, fontsize=13, color=color,
                    family="Noto Sans CJK JP", transform=ax.transAxes)

    ax.text(0.02, 0.025,
            "→ BC×HV は大出血、expansion は短時間出血で強い\n"
            "→ 加重和: fused = BC×HV + α × expansion",
            fontsize=14, va="bottom", fontweight="bold",
            color=COLOR_FINAL, family="Noto Sans CJK JP",
            transform=ax.transAxes)


def panel7_final(ax):
    # 図6を埋め込む（PR-AUC推移）
    embed_image(ax, FIG_DIR / "fig06_progression.png",
                [0.0, 0.62, 1.0, 0.36])

    ax.text(0.02, 0.59,
            "■ 表 10: 10 種の融合候補（抜粋）",
            fontsize=14, va="top", fontweight="bold",
            family="Noto Sans CJK JP", transform=ax.transAxes)

    rows = [
        ("BC × HV (Phase1)",        "0.443", "1/0/2", False),
        ("BC + HV + area_delta",    "0.369", "1/0/2", False),
        ("BC × HV × area_delta",    "0.191", "1/3/2", False),
        ("max(BC,HV) + area_delta", "0.180", "1/4/2", False),
        ("BC × HV + 0.5×expansion", "0.473", "1/0/2", True),
        ("BC × HV + 0.5×newly_red", "0.422", "1/0/2", False),
        ("max(BC×HV, area_delta)",  "0.396", "1/0/2", False),
        ("max(BC×HV, expansion)",   "0.239", "1/3/2", False),
        ("BC × expansion",          "0.258", "2/5/1", False),
        ("HV × expansion",          "0.208", "1/4/2", False),
    ]
    # 表ヘッダ
    y0 = 0.55
    ax.text(0.02, y0, "融合式", fontsize=12, fontweight="bold",
            family="Noto Sans CJK JP", transform=ax.transAxes)
    ax.text(0.62, y0, "PR-AUC", fontsize=12, fontweight="bold",
            transform=ax.transAxes)
    ax.text(0.83, y0, "TP/FP/FN", fontsize=12, fontweight="bold",
            transform=ax.transAxes)
    for i, (label, prauc, tp, hl) in enumerate(rows):
        y = y0 - 0.04 * (i + 1)
        if hl:
            ax.add_patch(mpatches.Rectangle(
                (0.0, y - 0.018), 1.0, 0.038,
                facecolor="#fee0d2", edgecolor=COLOR_FINAL,
                linewidth=1.2, transform=ax.transAxes,
            ))
        c = COLOR_FINAL if hl else "#222"
        fw = "bold" if hl else "normal"
        ax.text(0.02, y, ("★ " if hl else "") + label,
                fontsize=12, color=c, fontweight=fw,
                family="Noto Sans CJK JP", transform=ax.transAxes)
        ax.text(0.62, y, prauc, fontsize=12, color=c, fontweight=fw,
                transform=ax.transAxes)
        ax.text(0.83, y, tp, fontsize=12, color=c, fontweight=fw,
                transform=ax.transAxes)

    ax.text(0.02, 0.05,
            "★ 最終形 BC×HV + 0.5×expansion ─\n"
            "  PR-AUC 0.473 / F1 0.500 / Precision 1.00",
            fontsize=14, va="bottom", fontweight="bold",
            color=COLOR_FINAL, family="Noto Sans CJK JP",
            transform=ax.transAxes)


def panel8_conclusion(ax):
    txt = (
        "■ 達成事項\n"
        "  ① アノテーション基盤 (102 単体テスト)\n"
        "  ② フェーズ認識 87.1% (Cholec80)\n"
        "  ③ ゼロショット出血検出\n"
        "      PR-AUC 0.408 → 0.473 改善\n"
        "      Precision 1.00 で大出血を確実検出\n\n"
        "■ Zero-shot 系統の限界（実証）\n"
        "  ・軽度出血の確実検出は不可能\n"
        "  ・CLIP系は fine-grained に弱い\n"
        "    （Marzullo 2024, Sharma 2025 と整合）\n"
        "  ・「出血⊂異常」は不成立\n"
        "    （DINOv2 anomaly ROC-AUC = 0.426）\n\n"
        "■ 今後の方向\n"
        "  (a) 多動画・教師あり学習へ移行\n"
        "  (b) SAM2 系 segmentation との統合\n"
        "      (BlooDet, Pei 2026)\n"
        "  (c) fine-CLIP / few-shot 適応\n"
        "  (d) フェーズ認識との文脈統合\n\n"
        "■ 推奨運用\n"
        "  「大出血アラート」として実用化\n"
        "  thr=0.725, merge_gap=60s\n"
        "  → SRT 出力 → Shotcut で人手確認\n"
        "  → Active learning へ"
    )
    ax.text(0.02, 0.98, txt, fontsize=15, va="top",
            family="Noto Sans CJK JP", transform=ax.transAxes)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    fig = plt.figure(
        figsize=(W_MM / 25.4, H_MM / 25.4), dpi=DPI, facecolor="white"
    )

    # === 演題番号エリア ===
    ax_pn = add_box(fig, MARGIN, H_MM - MARGIN - TITLE_H, TITLE_PNL_W, TITLE_H)
    ax_pn.add_patch(mpatches.Rectangle(
        (0, 0), 1, 1, facecolor="#fff7e6",
        edgecolor="#bbb", linewidth=2, linestyle="--",
        transform=ax_pn.transAxes,
    ))
    ax_pn.text(0.5, 0.5,
               "演題\n番号\n(会場貼付)",
               ha="center", va="center",
               fontsize=22, color="#999",
               family="Noto Sans CJK JP",
               transform=ax_pn.transAxes)

    # === タイトル帯 ===
    title_left = MARGIN + TITLE_PNL_W + GUTTER
    title_w = W_MM - 2 * MARGIN - TITLE_PNL_W - GUTTER
    ax_t = add_box(fig, title_left, H_MM - MARGIN - TITLE_H, title_w, TITLE_H)
    ax_t.add_patch(mpatches.Rectangle(
        (0, 0), 1, 1, facecolor=COLOR_TITLE,
        edgecolor="none", transform=ax_t.transAxes,
    ))
    ax_t.text(0.5, 0.72,
              "腹腔鏡手術動画解析プラットフォームの開発",
              ha="center", va="center",
              fontsize=42, fontweight="bold", color="white",
              family="Noto Sans CJK JP",
              transform=ax_t.transAxes)
    ax_t.text(0.5, 0.45,
              "── アノテーション基盤・フェーズ認識・ゼロショット出血検出 ──",
              ha="center", va="center",
              fontsize=22, color="#cfe2f3",
              family="Noto Sans CJK JP",
              transform=ax_t.transAxes)
    ax_t.text(0.5, 0.18,
              "川口 雅彦   /   横浜栄共済病院   /   surgkw@gmail.com",
              ha="center", va="center",
              fontsize=22, color="white",
              family="Noto Sans CJK JP",
              transform=ax_t.transAxes)

    # === 8 パネル ===
    draw_panel(fig, 0, 0, "①  背景・目的",
               COLOR_NEUTRAL, panel1_intro)
    draw_panel(fig, 1, 0, "②  SRT/JSONL アノテーション基盤",
               COLOR_NEUTRAL, panel2_annotation)
    draw_panel(fig, 0, 1, "③  フェーズ認識  (Cholec80)",
               COLOR_PHASE, panel3_phase)
    draw_panel(fig, 1, 1, "④  初期出血検出 ── 試行と限界",
               "#999", panel4_initial)
    draw_panel(fig, 0, 2, "⑤  ゼロショット VLP 評価  (Phase 1)",
               COLOR_VLP, panel5_zeroshot)
    draw_panel(fig, 1, 2, "⑥  物理シグナルとの融合  (Phase 3)",
               COLOR_PHYSICAL, panel6_fusion)
    draw_panel(fig, 0, 3, "⑦  最終結果 ── BC×HV + 0.5×expansion",
               COLOR_FINAL, panel7_final)
    draw_panel(fig, 1, 3, "⑧  結論と今後",
               COLOR_NEUTRAL, panel8_conclusion)

    # 出力
    out = OUT_PATH / "poster.png"
    fig.savefig(out, dpi=DPI, facecolor="white",
                bbox_inches=None, pad_inches=0)
    plt.close(fig)
    print(f"出力: {out}")
    print(f"  サイズ: {W_MM} × {H_MM} mm  /  DPI: {DPI}")
    print(f"  ピクセル: {int(W_MM/25.4*DPI)} × {int(H_MM/25.4*DPI)}")
    print(f"  PosteRazor で A3 タイル 2列×5段に分割推奨")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
