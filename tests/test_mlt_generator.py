"""
MLTGenerator のテスト: マーカー/アノテーション/テキストの出力とフレームレート。
"""

import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from src.mlt.mlt_generator import MLTGenerator


def _gen(fps=30.0, cuts=None, anns=None, texts=None, markers=None):
    g = MLTGenerator("/tmp/v.mp4", fps=fps)
    if cuts:
        g.add_cuts(cuts)
    if anns:
        g.add_annotation_track(anns, "Motion")
    if texts:
        g.add_text_track(texts)
    if markers:
        g.add_markers(markers)
    d = tempfile.mkdtemp()
    out = Path(d) / "x.mlt"
    g.generate(str(out))
    return ET.parse(out).getroot(), out


class TestFrameRate:
    def test_integer(self):
        root, _ = _gen(fps=30.0)
        prof = root.find("profile")
        assert (prof.get("frame_rate_num"), prof.get("frame_rate_den")) == ("30", "1")

    def test_ntsc_not_truncated(self):
        root, _ = _gen(fps=29.97)
        prof = root.find("profile")
        num, den = int(prof.get("frame_rate_num")), int(prof.get("frame_rate_den"))
        assert den != 1
        assert abs(num / den - 29.97) < 0.001


class TestElementEmission:
    def test_annotations_emitted_to_v2(self):
        root, _ = _gen(anns=[{"start_time_sec": 1.0, "end_time_sec": 2.0,
                              "label": "m"}])
        pl2 = root.find("playlist[@id='playlist2']")
        assert len(pl2.findall("entry")) == 1

    def test_text_track_emitted(self):
        root, _ = _gen(texts=[{"start_time_sec": 3.0, "end_time_sec": 4.0,
                               "text": "hi"}])
        pl3 = root.find("playlist[@id='playlist3']")
        assert pl3 is not None and len(pl3.findall("entry")) == 1
        # tractor に V3 トラックと transition が追加される
        tractor = root.find("tractor")
        producers = [t.get("producer") for t in tractor.findall("track")]
        assert "playlist3" in producers

    def test_markers_emitted_as_properties(self):
        root, _ = _gen(markers=[{"time_sec": 2.5, "label": "a"},
                                {"time_sec": 10.0}])
        tractor = root.find("tractor")
        props = [p for p in tractor.findall("property")
                 if p.get("name", "").startswith("shotcut:marker")]
        assert len(props) >= 2

    def test_no_text_track_when_absent(self):
        root, _ = _gen(anns=[{"start_time_sec": 0.0, "end_time_sec": 1.0}])
        assert root.find("playlist[@id='playlist3']") is None

    def test_invalid_entries_skipped(self):
        # キー欠落・非dict はスキップされ、例外なく生成完了
        root, _ = _gen(anns=[{"start_time_sec": 0.0},  # 欠落
                             "bad",                      # 非dict
                             {"start_time_sec": 1.0, "end_time_sec": 2.0}])
        pl2 = root.find("playlist[@id='playlist2']")
        assert len(pl2.findall("entry")) == 1

    def test_output_is_valid_pretty_xml(self):
        root, out = _gen(cuts=[{"start_time_sec": 0.0, "end_time_sec": 5.0}])
        text = out.read_text()
        assert text.startswith("<?xml")
        ET.parse(out)  # 妥当な XML
