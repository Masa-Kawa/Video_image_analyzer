"""
ODG ポスター生成（make_poster_odg.py）の単体テスト。

レイアウト計算・スタイル解決・描画ヘルパ・画像埋め込み（不在時プレースホルダ）
の回帰防止。ODF ドキュメントを実際に組み立てて要素が追加されることを確認する。
"""

import tempfile
from pathlib import Path

import pytest

from odf.opendocument import OpenDocumentDrawing
from odf.draw import Page

import src.bleed_ai.zeroshot.make_poster_odg as M


@pytest.fixture
def page_doc():
    """スタイル初期化済みの (doc, page) を返す。"""
    doc = OpenDocumentDrawing()
    M.setup_page(doc)
    M.init_styles(doc)
    page = Page(name="test", masterpagename="Default")
    doc.drawing.addElement(page)
    return doc, page


# ---------------------------------------------------------------------------
# レイアウト計算
# ---------------------------------------------------------------------------

class TestPanelPos:
    def test_origin_panel(self):
        x, y, w, h = M.panel_pos(0, 0)
        assert x == M.MARGIN
        assert y == M.BODY_TOP
        assert w == M.PANEL_W and h == M.PANEL_H

    def test_second_column_offset(self):
        x0, _, _, _ = M.panel_pos(0, 0)
        x1, _, _, _ = M.panel_pos(1, 0)
        assert abs(x1 - (x0 + M.PANEL_W + M.GUTTER)) < 1e-9

    def test_row_offset(self):
        _, y0, _, _ = M.panel_pos(0, 0)
        _, y1, _, _ = M.panel_pos(0, 1)
        assert abs(y1 - (y0 + M.PANEL_H + M.GUTTER)) < 1e-9

    def test_layout_constants_defined(self):
        # マジックナンバーの定数化（回帰防止）
        assert M.PANEL_HEADER_H == 18
        assert M.TABLE_ROW_H == 7


# ---------------------------------------------------------------------------
# スタイル解決
# ---------------------------------------------------------------------------

class TestStyles:
    def test_resolve_requires_init(self, monkeypatch):
        monkeypatch.setattr(M, "_STYLES", {})
        with pytest.raises(RuntimeError):
            M._resolve("t_body")

    def test_init_styles_populates(self):
        doc = OpenDocumentDrawing()
        M.setup_page(doc)
        styles = M.init_styles(doc)
        assert isinstance(styles, dict) and styles
        # 代表的なスタイル名が解決できる
        assert M._resolve("t_body") is not None

    def test_resolve_passthrough_non_string(self):
        # 文字列以外はそのまま返す（スタイルオブジェクト直渡し対応）
        sentinel = object()
        assert M._resolve(sentinel) is sentinel


# ---------------------------------------------------------------------------
# 描画ヘルパ
# ---------------------------------------------------------------------------

class TestDrawHelpers:
    def test_add_rect_adds_element(self, page_doc):
        _, page = page_doc
        before = len(page.childNodes)
        M.add_rect(page, 0, 0, 10, 10, "panel_neutral")
        assert len(page.childNodes) == before + 1

    def test_add_textbox_multiline(self, page_doc):
        _, page = page_doc
        before = len(page.childNodes)
        M.add_textbox(page, 0, 0, 50, 20,
                      [("行1", "t_body"), ("行2", "t_body_bold")])
        assert len(page.childNodes) == before + 1

    def test_draw_panel_frame_returns_body_box(self, page_doc):
        _, page = page_doc
        bx, by, bw, bh = M.draw_panel_frame(
            page, 0, 0, "見出し", "panel_neutral", "header_neutral")
        # 本文領域はパネルより内側
        px, py, pw, ph = M.panel_pos(0, 0)
        assert bx == px + M.PANEL_BODY_PAD
        assert by > py  # ヘッダ分下がる
        assert bw == pw - 2 * M.PANEL_BODY_PAD


# ---------------------------------------------------------------------------
# 画像埋め込み
# ---------------------------------------------------------------------------

class TestAddImage:
    def test_missing_image_placeholder(self, page_doc):
        doc, page = page_doc
        before = len(page.childNodes)
        # 不在パス → 例外なくプレースホルダ（textbox）を追加し None を返す
        r = M.add_image(doc, page, 0, 0, 50, 50, Path("/no/such/fig.png"))
        assert r is None
        assert len(page.childNodes) == before + 1

    def test_existing_image_embedded(self, page_doc):
        doc, page = page_doc
        # 1x1 PNG を生成して埋め込み
        png = (b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00"
               b"\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9c"
               b"c\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82")
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.png"
            p.write_bytes(png)
            before = len(page.childNodes)
            r = M.add_image(doc, page, 0, 0, 50, 50, p)
            assert r is not None
            assert len(page.childNodes) == before + 1
