"""LibreOffice Draw 形式 (.odg) でポスターを生成。

レイアウト: 870 × 1850 mm 縦長、8 パネル + タイトル帯
出力: out_lapc_eval/zs/poster/poster.odg

LibreOffice Draw で開いてテキストや図を直接編集できる。
"""

from __future__ import annotations

import sys
from pathlib import Path

from odf.draw import Frame, Image, Page, Rect, TextBox
from odf.opendocument import OpenDocumentDrawing
from odf.style import (
    DrawingPageProperties, GraphicProperties, MasterPage, PageLayout,
    PageLayoutProperties, ParagraphProperties, Style, TextProperties,
)
from odf.text import P, Span


# ---------------------------------------------------------------------------
# 寸法定数 (mm)
# ---------------------------------------------------------------------------

W_MM = 870
H_MM = 1850
MARGIN = 25
GUTTER = 30
TITLE_H = 180
TITLE_PNL_W = 180
N_ROWS = 4
N_COLS = 2

BODY_TOP = MARGIN + TITLE_H + GUTTER  # 上端からの距離
BODY_BOTTOM_FROM_TOP = H_MM - MARGIN
PANEL_W = (W_MM - 2 * MARGIN - GUTTER) / N_COLS
BODY_H = (H_MM - MARGIN) - BODY_TOP
PANEL_H = (BODY_H - (N_ROWS - 1) * GUTTER) / N_ROWS

# パネル内部レイアウト定数（mm）。マジックナンバーを意味づけして一元管理する。
PANEL_HEADER_H = 18    # パネル上部ヘッダ帯の高さ
PANEL_BODY_PAD = 4     # パネル枠と本文の左右/下パディング
SECTION_TITLE_H = 13   # 本文中の「■見出し」1行ぶんの高さ
TABLE_ROW_H = 7        # 表の1行の高さ

# 配色
COLOR_TITLE = "#1f4e79"
COLOR_TITLE_TEXT = "#ffffff"
COLOR_TITLE_SUB = "#cfe2f3"
COLOR_VLP = "#2b8cbe"
COLOR_PHASE = "#31a354"
COLOR_PHYSICAL = "#fd8d3c"
COLOR_FINAL = "#762a83"
COLOR_NEUTRAL = "#666666"
COLOR_HIGHLIGHT_BG = "#fee0d2"

OUT_DIR = Path("out_lapc_eval/zs/poster")
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = Path("out_lapc_eval/zs/figures_print")  # 300dpi 用


def mm(x: float) -> str:
    return f"{x}mm"


# ---------------------------------------------------------------------------
# スタイル定義
# ---------------------------------------------------------------------------

def make_styles(doc: OpenDocumentDrawing):
    """テキスト・矩形スタイルを定義。"""
    styles = {}

    def gp_style(name, fillcolor, strokecolor=None, strokewidth="0mm"):
        s = Style(name=name, family="graphic")
        attrs = {
            "fill": "solid",
            "fillcolor": fillcolor,
            "stroke": "none" if strokecolor is None else "solid",
            "textareaverticalalign": "top",
            "padding": "3mm",
        }
        if strokecolor is not None:
            attrs["strokecolor"] = strokecolor
            attrs["strokewidth"] = strokewidth
        s.addElement(GraphicProperties(**attrs))
        doc.automaticstyles.addElement(s)
        styles[name] = s
        return s

    def text_only_style(name):
        s = Style(name=name, family="graphic")
        s.addElement(GraphicProperties(
            fill="none", stroke="none",
            textareaverticalalign="top",
            padding="2mm",
        ))
        doc.automaticstyles.addElement(s)
        styles[name] = s
        return s

    def para_style(name, align="start"):
        s = Style(name=name, family="paragraph")
        s.addElement(ParagraphProperties(textalign=align))
        doc.automaticstyles.addElement(s)
        styles[name] = s
        return s

    def text_style(name, size_pt, color="#000000", bold=False):
        s = Style(name=name, family="text")
        attrs = {
            "fontsize": f"{size_pt}pt",
            "color": color,
            "fontfamily": "Noto Sans CJK JP",
        }
        if bold:
            attrs["fontweight"] = "bold"
        s.addElement(TextProperties(**attrs))
        doc.automaticstyles.addElement(s)
        styles[name] = s
        return s

    # 矩形スタイル
    gp_style("title_bar", COLOR_TITLE)
    gp_style("panel_num", "#fff7e6", strokecolor="#bbbbbb", strokewidth="0.5mm")
    gp_style("panel_neutral", "#ffffff",
             strokecolor=COLOR_NEUTRAL, strokewidth="0.7mm")
    gp_style("panel_vlp", "#ffffff",
             strokecolor=COLOR_VLP, strokewidth="0.7mm")
    gp_style("panel_phase", "#ffffff",
             strokecolor=COLOR_PHASE, strokewidth="0.7mm")
    gp_style("panel_physical", "#ffffff",
             strokecolor=COLOR_PHYSICAL, strokewidth="0.7mm")
    gp_style("panel_final", "#ffffff",
             strokecolor=COLOR_FINAL, strokewidth="0.7mm")
    gp_style("header_neutral", COLOR_NEUTRAL)
    gp_style("header_vlp", COLOR_VLP)
    gp_style("header_phase", COLOR_PHASE)
    gp_style("header_physical", COLOR_PHYSICAL)
    gp_style("header_final", COLOR_FINAL)
    gp_style("highlight_row", COLOR_HIGHLIGHT_BG,
             strokecolor=COLOR_FINAL, strokewidth="0.4mm")
    text_only_style("text_only")
    gp_style("img_frame", "#ffffff")

    # 段落スタイル
    para_style("p_left", "start")
    para_style("p_center", "center")

    # 文字スタイル
    text_style("t_title_main", 42, COLOR_TITLE_TEXT, bold=True)
    text_style("t_title_sub", 22, COLOR_TITLE_SUB, bold=False)
    text_style("t_title_author", 22, COLOR_TITLE_TEXT, bold=False)
    text_style("t_panel_num", 22, "#999999", bold=False)
    text_style("t_header", 22, "#ffffff", bold=True)
    text_style("t_body", 15, "#222222", bold=False)
    text_style("t_body_bold", 15, "#222222", bold=True)
    text_style("t_table", 13, "#222222", bold=False)
    text_style("t_table_h", 13, "#000000", bold=True)
    text_style("t_final_emph", 14, COLOR_FINAL, bold=True)
    text_style("t_phase_emph", 14, COLOR_PHASE, bold=True)
    text_style("t_vlp_emph", 14, COLOR_VLP, bold=True)
    text_style("t_physical_emph", 14, COLOR_PHYSICAL, bold=True)

    return styles


# ---------------------------------------------------------------------------
# ページレイアウト
# ---------------------------------------------------------------------------

def setup_page(doc: OpenDocumentDrawing):
    """A0 サイズに近い 870×1850mm のページを設定。"""
    pl = PageLayout(name="PosterLayout")
    pl.addElement(PageLayoutProperties(
        pagewidth=mm(W_MM),
        pageheight=mm(H_MM),
        printorientation="portrait",
        margin="0mm",
    ))
    doc.automaticstyles.addElement(pl)

    dps = Style(name="dp1", family="drawing-page")
    dps.addElement(DrawingPageProperties(backgroundsize="border", fill="none"))
    doc.automaticstyles.addElement(dps)

    mp = MasterPage(name="Default", pagelayoutname="PosterLayout")
    doc.masterstyles.addElement(mp)


# ---------------------------------------------------------------------------
# 描画ヘルパ
# ---------------------------------------------------------------------------

_STYLES = {}


def init_styles(doc: OpenDocumentDrawing) -> dict:
    """スタイルを生成してモジュールに登録し、辞書を返す。

    描画ヘルパ（add_rect / add_textbox / draw_panel_frame …）は _STYLES に
    依存するため、main() 以外（テスト・再利用）からヘルパを使う前に必ず本関数を
    呼ぶ。これにより「未初期化の _STYLES 参照」による KeyError を防ぐ。
    """
    global _STYLES
    _STYLES = make_styles(doc)
    return _STYLES


def _resolve(name):
    """文字列ならスタイル辞書から取得、それ以外はそのまま返す。"""
    if isinstance(name, str):
        if not _STYLES:
            raise RuntimeError(
                "スタイルが未初期化です。先に init_styles(doc) を呼んでください。"
            )
        return _STYLES[name]
    return name


def add_rect(page, x, y, w, h, style_name):
    r = Rect(
        x=mm(x), y=mm(y), width=mm(w), height=mm(h),
        stylename=_resolve(style_name),
    )
    page.addElement(r)
    return r


def add_textbox(page, x, y, w, h, lines, style="text_only"):
    """lines: [(text, charstyle), ...] のリスト、または文字列。"""
    frame = Frame(
        x=mm(x), y=mm(y), width=mm(w), height=mm(h),
        stylename=_resolve(style),
    )
    tb = TextBox()
    frame.addElement(tb)

    if isinstance(lines, str):
        lines = [lines]

    for ln in lines:
        if isinstance(ln, tuple):
            text, charstyle = ln
        else:
            text, charstyle = ln, "t_body"
        p = P(stylename=_resolve("p_left"))
        sp = Span(stylename=_resolve(charstyle), text=text)
        p.addElement(sp)
        tb.addElement(p)

    page.addElement(frame)
    return frame


def add_centered_text(page, x, y, w, h, text, charstyle, style="text_only"):
    frame = Frame(
        x=mm(x), y=mm(y), width=mm(w), height=mm(h),
        stylename=_resolve(style),
    )
    tb = TextBox()
    frame.addElement(tb)
    p = P(stylename=_resolve("p_center"))
    sp = Span(stylename=_resolve(charstyle), text=text)
    p.addElement(sp)
    tb.addElement(p)
    page.addElement(frame)
    return frame


def add_image(doc, page, x, y, w, h, image_path: Path):
    """画像を埋め込む。ファイル不在時はプレースホルダ文を置いて続行する。

    make_figures.py で図を事前生成していない／カレントディレクトリが異なる場合に
    1枚の欠落でポスター全体がクラッシュしないよう、存在を検証する。
    """
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"警告: 図が見つかりません（プレースホルダ表示）: {image_path}",
              file=sys.stderr)
        add_textbox(page, x, y + h / 2 - 4, w, 12,
                    [(f"[図なし: {image_path.name}]", "t_body")])
        return None
    href = doc.addPicture(str(image_path))
    frame = Frame(
        x=mm(x), y=mm(y), width=mm(w), height=mm(h),
    )
    img = Image(href=href, type="simple", show="embed", actuate="onLoad")
    frame.addElement(img)
    page.addElement(frame)
    return frame


# ---------------------------------------------------------------------------
# パネル枠
# ---------------------------------------------------------------------------

def panel_pos(col: int, row: int) -> tuple:
    """col 0=左, 1=右; row 0=最上段, 3=最下段。"""
    x = MARGIN + col * (PANEL_W + GUTTER)
    y = BODY_TOP + row * (PANEL_H + GUTTER)
    return x, y, PANEL_W, PANEL_H


def draw_panel_frame(page, col, row, title, panel_style, header_style):
    """パネルの枠とヘッダを描画し、本文領域の (x, y, w, h) を返す。"""
    x, y, w, h = panel_pos(col, row)
    add_rect(page, x, y, w, h, panel_style)
    header_h = PANEL_HEADER_H
    add_rect(page, x, y, w, header_h, header_style)
    add_textbox(page, x + 5, y + 4, w - 10, header_h - 4,
                [(title, "t_header")])
    body_y = y + header_h + 3
    body_h = h - header_h - 6
    return x + PANEL_BODY_PAD, body_y, w - 2 * PANEL_BODY_PAD, body_h


# ---------------------------------------------------------------------------
# 各パネル本文
# ---------------------------------------------------------------------------

def panel1(page, doc, col, row):
    bx, by, bw, bh = draw_panel_frame(
        page, col, row, "①  背景・目的",
        "panel_neutral", "header_neutral",
    )
    text = (
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
    lines = [(line, "t_body") for line in text.split("\n")]
    add_textbox(page, bx, by, bw, bh, lines)


def panel2(page, doc, col, row):
    bx, by, bw, bh = draw_panel_frame(
        page, col, row, "②  SRT/JSONL アノテーション基盤",
        "panel_neutral", "header_neutral",
    )
    text = (
        "■ 設計思想\n"
        "  正本: JSONL（機械可読 1 行 1 イベント）\n"
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
        "    (jsonl_to_srt, srt_to_jsonl, merge_srt,\n"
        "     csv_to_srt, mlt_generator …)\n"
        "  ・単体テスト 102 件 PASS\n"
        "  ・全解析器（red, bleed_ai, cholec_phase,\n"
        "    transnet, yolo, motion）が遵守"
    )
    lines = [(line, "t_body") for line in text.split("\n")]
    add_textbox(page, bx, by, bw, bh, lines)


def panel3(page, doc, col, row):
    bx, by, bw, bh = draw_panel_frame(
        page, col, row, "③  フェーズ認識  (Cholec80)",
        "panel_phase", "header_phase",
    )
    upper = (
        "■ モデル構成\n"
        "  ResNet50 (SelfSupSurg DINO, 凍結)\n"
        "      ↓ 2,048 dim 特徴\n"
        "  Bidirectional LSTM (h=512, layers=2)\n"
        "      ↓ FC(256) → ReLU → FC(7)\n"
        "  → 7 フェーズ確率\n\n"
        "■ 2 段階学習で効率化\n"
        "  Stage 1 特徴抽出: 80 分（80 動画一括）\n"
        "  Stage 2 BiLSTM 学習: 80 秒（50 epoch）\n\n"
        "■ Cholec80 テスト精度 (video41-80)"
    )
    lines = [(line, "t_body") for line in upper.split("\n")]
    add_textbox(page, bx, by, bw, 130, lines)

    # 表
    table_y = by + 130
    table_lines = [
        "  Preparation                          77.4%",
        "  CalotTriangleDissection         90.8%",
        "  ClippingCutting                       67.1%",
        "  GallbladderDissection            90.7%",
        "  GallbladderPackaging             83.1%",
        "  CleaningCoagulation              76.9%",
        "  GallbladderRetraction            89.1%",
    ]
    tlines = [(line, "t_table") for line in table_lines]
    add_textbox(page, bx, table_y, bw, 65, tlines)

    # 全体 87.1% 強調
    add_textbox(page, bx, table_y + 65, bw, 12,
                [("  ★ 全体精度 87.1%", "t_phase_emph")])

    # 評価動画への適用
    bottom_y = table_y + 80
    bottom_text = (
        "■ 評価動画 LapC_EvalDemo への適用\n"
        "  CalotTriangleDissection: 信頼度 0.92\n"
        "  GallbladderDissection: 信頼度 0.92\n"
        "  → 主要術式部分は実用域に到達"
    )
    lines = [(line, "t_body") for line in bottom_text.split("\n")]
    add_textbox(page, bx, bottom_y, bw, bh - (bottom_y - by), lines)


def panel4(page, doc, col, row):
    bx, by, bw, bh = draw_panel_frame(
        page, col, row, "④  初期出血検出 ── 試行と限界",
        "panel_neutral", "header_neutral",
    )
    text = (
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
        "  → Phase 3 物理融合で再利用される"
    )
    lines = [(line, "t_body") for line in text.split("\n")]
    add_textbox(page, bx, by, bw, bh, lines)


def panel5(page, doc, col, row):
    bx, by, bw, bh = draw_panel_frame(
        page, col, row, "⑤  ゼロショット VLP 評価  (Phase 1)",
        "panel_vlp", "header_vlp",
    )
    add_textbox(page, bx, by, bw, 12,
                [("■ 4 種の VLP × プロンプト類似度", "t_body_bold")])
    # 図3（見出し1行ぶん下げ、下部100mmは「主な発見」テキスト用に確保）
    fig_y = by + SECTION_TITLE_H
    fig_h = bh - 100
    add_image(doc, page, bx, fig_y, bw, fig_h,
              FIG_DIR / "fig03_interval_heatmap.png")
    # 主な発見
    bottom_y = fig_y + fig_h + 4
    bottom_text = (
        "■ 主な発見\n"
        "  BiomedCLIP: 大出血特化（区間3 mean=0.52）\n"
        "  HecVL/SurgVLP: 中規模出血で強い\n"
        "  ★ BC × HV ensemble PR-AUC 最良 (0.443)\n"
        "  → 区間1 (11.8秒) は全モデル不可"
    )
    lines = []
    for line in bottom_text.split("\n"):
        style = "t_vlp_emph" if line.startswith("  ★") else "t_body"
        lines.append((line, style))
    add_textbox(page, bx, bottom_y, bw, bh - (bottom_y - by), lines)


def panel6(page, doc, col, row):
    bx, by, bw, bh = draw_panel_frame(
        page, col, row, "⑥  物理シグナルとの融合  (Phase 3)",
        "panel_physical", "header_physical",
    )
    add_textbox(page, bx, by, bw, 12,
                [("■ 意味 × 物理 ── 完全に補完的", "t_body_bold")])
    # 図5（見出し1行ぶん下げ、下部130mmは表＋まとめ用に確保）
    fig_y = by + SECTION_TITLE_H
    fig_h = bh - 130
    add_image(doc, page, bx, fig_y, bw, fig_h,
              FIG_DIR / "fig05_complementary.png")
    # 表 + まとめ
    bottom_y = fig_y + fig_h + 4
    add_textbox(page, bx, bottom_y, bw, 12,
                [("■ シグナル単体（区間平均スコア）", "t_body_bold")])
    table_lines = [
        "                       区間1     区間2     区間3",
        "  BC × HV (意味)       0.001     0.054     0.456",
        "  expansion (物理)    0.532     0.449     0.283",
    ]
    tlines = [(line, "t_table") for line in table_lines]
    add_textbox(page, bx, bottom_y + 12, bw, 36, tlines)
    add_textbox(page, bx, bottom_y + 50, bw, 30, [
        ("→ BC×HV は大出血、expansion は短時間出血で強い", "t_body"),
        ("→ 加重和: fused = BC×HV + α × expansion",
         "t_physical_emph"),
    ])


def panel7(page, doc, col, row):
    bx, by, bw, bh = draw_panel_frame(
        page, col, row, "⑦  最終結果 ── BC×HV + 0.5×expansion",
        "panel_final", "header_final",
    )
    # 図6 (PR-AUC progression)
    fig_h = 90
    add_image(doc, page, bx, by, bw, fig_h,
              FIG_DIR / "fig06_progression.png")

    # 表 10
    table_y = by + fig_h + 5
    add_textbox(page, bx, table_y, bw, 10,
                [("■ 表10  10 種の融合候補（抜粋）", "t_body_bold")])

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
    tbl_top = table_y + 11
    row_h = TABLE_ROW_H

    # ヘッダ
    add_textbox(page, bx, tbl_top, bw, row_h, [
        ("  融合式                                            PR-AUC    TP/FP/FN",
         "t_table_h"),
    ])

    for i, (label, prauc, tp, hl) in enumerate(rows):
        row_y = tbl_top + (i + 1) * row_h
        if hl:
            add_rect(page, bx, row_y - 0.5, bw, row_h, "highlight_row")
            charstyle = "t_final_emph"
            prefix = "  ★ "
        else:
            charstyle = "t_table"
            prefix = "       "
        # ラベル列を固定幅で揃える: 簡易にスペースでパディング
        lbl_padded = (prefix + label).ljust(36)
        add_textbox(page, bx, row_y, bw, row_h,
                    [(lbl_padded + prauc + "      " + tp, charstyle)])

    # 最終強調
    bottom_y = tbl_top + (len(rows) + 1) * row_h + 4
    add_textbox(page, bx, bottom_y, bw, bh - (bottom_y - by), [
        ("★ 最終形 BC×HV + 0.5×expansion", "t_final_emph"),
        ("    PR-AUC 0.473 / F1 0.500 / Precision 1.00",
         "t_final_emph"),
    ])


def panel8(page, doc, col, row):
    bx, by, bw, bh = draw_panel_frame(
        page, col, row, "⑧  結論と今後",
        "panel_neutral", "header_neutral",
    )
    text = (
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
    lines = [(line, "t_body") for line in text.split("\n")]
    add_textbox(page, bx, by, bw, bh, lines)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    doc = OpenDocumentDrawing()
    setup_page(doc)
    init_styles(doc)

    page = Page(name="poster", masterpagename="Default")
    doc.drawing.addElement(page)

    # === 演題番号エリア（左上）===
    add_rect(page, MARGIN, MARGIN, TITLE_PNL_W, TITLE_H, "panel_num")
    add_centered_text(
        page, MARGIN, MARGIN + TITLE_H / 2 - 25,
        TITLE_PNL_W, 50,
        "演題番号", "t_panel_num",
    )
    add_centered_text(
        page, MARGIN, MARGIN + TITLE_H / 2 + 5,
        TITLE_PNL_W, 30,
        "(会場貼付)", "t_panel_num",
    )

    # === タイトル帯（演題番号の右）===
    title_x = MARGIN + TITLE_PNL_W + GUTTER
    title_w = W_MM - 2 * MARGIN - TITLE_PNL_W - GUTTER
    add_rect(page, title_x, MARGIN, title_w, TITLE_H, "title_bar")

    add_centered_text(
        page, title_x, MARGIN + 30,
        title_w, 60,
        "腹腔鏡手術動画解析プラットフォームの開発",
        "t_title_main",
    )
    add_centered_text(
        page, title_x, MARGIN + 90,
        title_w, 35,
        "── アノテーション基盤・フェーズ認識・ゼロショット出血検出 ──",
        "t_title_sub",
    )
    add_centered_text(
        page, title_x, MARGIN + 130,
        title_w, 35,
        "川口 雅彦   /   横浜栄共済病院   /   surgkw@gmail.com",
        "t_title_author",
    )

    # === 8 パネル ===
    panel1(page, doc, 0, 0)
    panel2(page, doc, 1, 0)
    panel3(page, doc, 0, 1)
    panel4(page, doc, 1, 1)
    panel5(page, doc, 0, 2)
    panel6(page, doc, 1, 2)
    panel7(page, doc, 0, 3)
    panel8(page, doc, 1, 3)

    out = OUT_DIR / "poster.odg"
    doc.save(str(out))
    print(f"出力: {out}")
    print(f"  サイズ: {W_MM} × {H_MM} mm")
    print(f"  パネル: {N_COLS}列 × {N_ROWS}段 = 8個")
    print(f"  各パネル: 約 {PANEL_W:.0f} × {PANEL_H:.0f} mm")
    print()
    print("LibreOffice Draw で開いて編集できます:")
    print(f"  libreoffice --draw {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
