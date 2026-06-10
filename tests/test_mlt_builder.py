"""
MLTBuilder のテスト（トラック/トランジション整合性）。
"""

import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from src.mlt.mlt_builder import MLTBuilder


def _build(tracks):
    d = tempfile.mkdtemp()
    b = MLTBuilder("proj", ["/tmp/v.mp4"], fps=30.0)
    for tid, ttype, name in tracks:
        b.add_track(tid, ttype, name)
    out = Path(d) / "out.mlt"
    b.generate(str(out))
    return ET.parse(out).getroot()


def _b_tracks(tractor):
    return [int(p.text) for tr in tractor.findall("transition")
            for p in tr.findall("property") if p.get("name") == "b_track"]


class TestTrackTransitions:
    def test_no_transition_exceeds_track_count(self):
        # V1/A1（スキップ）+ 追加 V2,V3 → tractor index 最大 4
        root = _build([("V1", "video", "V1"), ("A1", "audio", "A1"),
                       ("V2", "video", "V2"), ("V3", "video", "V3")])
        tractor = root.find("tractor")
        n_tracks = len(tractor.findall("track"))  # background+V1+A1+V2+V3 = 5
        assert n_tracks == 5
        assert max(_b_tracks(tractor)) <= n_tracks - 1

    def test_no_extra_tracks(self):
        # V1/A1 のみ → 追加トラックなし、b_track は 1,2 のみ
        root = _build([("V1", "video", "V1"), ("A1", "audio", "A1")])
        tractor = root.find("tractor")
        assert len(tractor.findall("track")) == 3  # background+V1+A1
        assert max(_b_tracks(tractor)) <= 2

    def test_noop_marker_api_removed(self):
        b = MLTBuilder("p", ["/tmp/v.mp4"])
        # 誤用防止のため未実装の no-op API は削除済み
        assert not hasattr(b, "add_markers")
        assert not hasattr(b, "add_text_track")


class TestInputValidation:
    def test_apply_cuts_filters_invalid(self):
        b = MLTBuilder("p", ["/tmp/v.mp4"])
        b.apply_cuts({"results": [
            {"start_time_sec": 0.0, "end_time_sec": 1.0},  # OK
            {"start_time_sec": 1.0},                         # 欠落
            "not a dict",                                    # 非dict
            {"foo": 1},                                      # キー無し
        ]})
        assert len(b.cuts) == 1

    def test_apply_cuts_non_container(self):
        b = MLTBuilder("p", ["/tmp/v.mp4"])
        b.apply_cuts(12345)  # dict/list 以外 → 空
        assert b.cuts == []

    def test_apply_cuts_single_scene_dict(self):
        b = MLTBuilder("p", ["/tmp/v.mp4"])
        b.apply_cuts({"start_time_sec": 2.0, "end_time_sec": 3.0})
        assert len(b.cuts) == 1

    def test_add_annotations_invalid_input(self):
        b = MLTBuilder("p", ["/tmp/v.mp4"])
        b.add_annotations("garbage", track="V2")
        assert b.annotations_by_track["V2"] == []

    def test_generate_completes_with_validated_data(self):
        b = MLTBuilder("p", ["/tmp/v.mp4"], fps=30.0)
        b.apply_cuts({"results": [{"start_time_sec": 0.0, "end_time_sec": 1.0}]})
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "x.mlt"
            b.generate(str(out))
            assert out.exists()


class TestFrameRate:
    def _frame_rate(self, fps):
        b = MLTBuilder("p", ["/tmp/v.mp4"], fps=fps)
        b.apply_cuts({"results": [{"start_time_sec": 0.0, "end_time_sec": 1.0}]})
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "x.mlt"
            b.generate(str(out))
            prof = ET.parse(out).getroot().find("profile")
            return int(prof.get("frame_rate_num")), int(prof.get("frame_rate_den"))

    def test_integer_fps(self):
        assert self._frame_rate(30.0) == (30, 1)
        assert self._frame_rate(25.0) == (25, 1)

    def test_ntsc_fps_not_truncated(self):
        # 29.97 が int 切り捨てで 29/1 にならず、分数で精度維持されること
        num, den = self._frame_rate(29.97)
        assert abs(num / den - 29.97) < 0.001
        assert den != 1  # 整数化されていない

    def test_output_is_valid_xml(self):
        b = MLTBuilder("p", ["/tmp/v.mp4"], fps=30.0)
        b.apply_cuts({"results": [{"start_time_sec": 0.0, "end_time_sec": 1.0}]})
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "x.mlt"
            b.generate(str(out))
            text = out.read_text()
            assert text.startswith("<?xml")
            ET.parse(out)  # 整形済み＆妥当な XML
