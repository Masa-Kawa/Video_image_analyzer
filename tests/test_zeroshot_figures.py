"""
発表用図表生成スクリプト（make_figures.py）のスモークテスト。

データ非依存のユニット（FIG_DPI 解釈・入力検証）に加え、評価入力が
揃っている環境では「8図が生成される」エンドツーエンドを確認する。
入力が無い環境（CI 等）では該当テストを skip する。
"""

import os
import tempfile
from pathlib import Path

import pytest

import src.bleed_ai.zeroshot.make_figures as M


# ---------------------------------------------------------------------------
# データ非依存ユニット
# ---------------------------------------------------------------------------

class TestSafeDpi:
    def test_invalid_falls_back(self, monkeypatch):
        monkeypatch.setenv("FIG_DPI", "abc")
        assert M._safe_dpi(150) == 150

    def test_valid_used(self, monkeypatch):
        monkeypatch.setenv("FIG_DPI", "300")
        assert M._safe_dpi() == 300

    def test_nonpositive_falls_back(self, monkeypatch):
        monkeypatch.setenv("FIG_DPI", "-5")
        assert M._safe_dpi(150) == 150

    def test_unset_uses_default(self, monkeypatch):
        monkeypatch.delenv("FIG_DPI", raising=False)
        assert M._safe_dpi(123) == 123


class TestCheckInputs:
    def test_detects_missing(self, monkeypatch):
        monkeypatch.setattr(M, "TRUTH_SRT", "/no/such/truth.srt")
        monkeypatch.setattr(M, "BLEEDLOG", "/no/such/bleedlog.csv")
        missing = M._check_inputs()
        assert "/no/such/truth.srt" in missing
        assert "/no/such/bleedlog.csv" in missing

    def test_main_aborts_when_missing(self, monkeypatch):
        monkeypatch.setattr(M, "TRUTH_SRT", "/no/such/truth.srt")
        # 入力欠落なら何も生成せず非ゼロ終了
        assert M.main() == 1


# ---------------------------------------------------------------------------
# データ依存スモーク（入力が無ければ skip）
# ---------------------------------------------------------------------------

_INPUTS_PRESENT = len(M._check_inputs()) == 0
_needs_data = pytest.mark.skipif(
    not _INPUTS_PRESENT, reason="評価入力（CSV/SRT）が無いため skip"
)


@_needs_data
class TestSmoke:
    def test_generates_eight_figures(self, monkeypatch):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            monkeypatch.setattr(M, "OUT_DIR", out)
            rc = M.main()
            assert rc == 0
            pngs = sorted(out.glob("*.png"))
            assert len(pngs) == 8, [p.name for p in pngs]
            # 各ファイルが空でない
            for p in pngs:
                assert p.stat().st_size > 0

    def test_reported_metrics_match_live(self):
        # 図に焼き込んだ代表値が実測と乖離していない（ドリフト無し）
        assert M._verify_reported_metrics() == []
