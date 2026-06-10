"""
アノテーション編集ツールのユニットテスト。

検証対象:
  1. label_sets   : cholecystectomy 語彙が CHOLEC80_PHASES と一致
  2. srt_io        : 既存SRT読込→ID採番→保存→再読込で id/時刻が安定（round-trip）、
                     保存SRTにメタJSON行（id含む）が出力されること
  3. pairing       : 編集/挿入/削除/変更なし、元なし（新規作成）の各ケース
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from src.cholec_phase import CHOLEC80_PHASES
from src.annotate.label_sets import get_label_set, available_procedures
from src.annotate.srt_io import load_segments, save_segments
from src.annotate.pairing import make_pairs


def _seg(id_, name, start, end):
    return {"id": id_, "type": "surgical_phase", "phase_name": name,
            "start_sec": start, "end_sec": end}


class TestLabelSets(unittest.TestCase):
    def test_cholecystectomy_matches_cholec80(self):
        self.assertEqual(get_label_set("cholecystectomy"), list(CHOLEC80_PHASES))

    def test_unknown_procedure_raises(self):
        with self.assertRaises(KeyError):
            get_label_set("does_not_exist")

    def test_available_lists_cholecystectomy(self):
        self.assertIn("cholecystectomy", available_procedures())


class TestSrtIo(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        # mkdtemp は自動削除されないため、CI での一時ディレクトリ蓄積を防ぐ
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _path(self, name):
        return str(Path(self.tmp) / name)

    def test_save_writes_meta_line_with_id(self):
        segs = [_seg("aaaa1111", "Preparation", 0.0, 5.0)]
        p = self._path("out.srt")
        save_segments(segs, p)
        text = Path(p).read_text(encoding="utf-8")
        # タグ行 + メタJSON行（id を含む）。JSON出力の体裁（空白・キー順）に
        # 依存しないよう、メタ行を json.loads でパースして検証する。
        self.assertIn("[phase] Preparation", text)
        meta = self._extract_meta(text)
        self.assertEqual(meta["id"], "aaaa1111")

    @staticmethod
    def _extract_meta(srt_text: str) -> dict:
        """SRT本文から meta JSON 行を1件抽出して dict で返す。"""
        for line in srt_text.splitlines():
            line = line.strip()
            if line.startswith("{") and '"id"' in line:
                return json.loads(line)
        raise AssertionError("meta JSON行が見つかりません")

    def test_roundtrip_stable_id_and_time(self):
        segs = [
            _seg("id000001", "Preparation", 0.0, 12.5),
            _seg("id000002", "CalotTriangleDissection", 12.5, 30.0),
        ]
        p = self._path("rt.srt")
        save_segments(segs, p)
        loaded = load_segments(p)
        self.assertEqual([s["id"] for s in loaded], ["id000001", "id000002"])
        self.assertEqual(loaded[0]["phase_name"], "Preparation")
        self.assertAlmostEqual(loaded[1]["start_sec"], 12.5, places=3)
        self.assertAlmostEqual(loaded[1]["end_sec"], 30.0, places=3)

    def test_load_assigns_id_when_missing(self):
        # メタ行の無い（id無し）従来SRTを手で用意
        raw = (
            "1\n00:00:00,000 --> 00:00:05,000\n[phase] Preparation\n\n"
            "2\n00:00:05,000 --> 00:00:10,000\n[phase] ClippingCutting\n\n"
        )
        p = self._path("legacy.srt")
        Path(p).write_text(raw, encoding="utf-8")
        loaded = load_segments(p)
        self.assertEqual(len(loaded), 2)
        # 採番IDは uuid4().hex[:8] = 8桁16進。フォーマットと一意性を検証する。
        for s in loaded:
            self.assertRegex(s["id"], r"^[0-9a-f]{8}$")
        self.assertEqual(len({s["id"] for s in loaded}), 2)
        # 採番後に保存→再読込しても id が安定
        save_segments(loaded, p)
        again = load_segments(p)
        self.assertEqual([s["id"] for s in loaded], [s["id"] for s in again])

    def test_missing_path_returns_empty(self):
        self.assertEqual(load_segments(None), [])
        self.assertEqual(load_segments(self._path("nope.srt")), [])


class TestPairing(unittest.TestCase):
    def test_label_change_is_edited(self):
        orig = [_seg("x1", "Preparation", 0.0, 5.0)]
        corr = [_seg("x1", "CalotTriangleDissection", 0.0, 5.0)]
        pairs = make_pairs(orig, corr)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["change"], "edited")
        self.assertEqual(pairs[0]["rejected"]["phase_name"], "Preparation")
        self.assertEqual(pairs[0]["chosen"]["phase_name"], "CalotTriangleDissection")

    def test_time_change_is_edited(self):
        orig = [_seg("x1", "Preparation", 0.0, 5.0)]
        corr = [_seg("x1", "Preparation", 0.0, 7.5)]
        pairs = make_pairs(orig, corr)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["change"], "edited")

    def test_no_change_emits_nothing(self):
        orig = [_seg("x1", "Preparation", 0.0, 5.0)]
        corr = [_seg("x1", "Preparation", 0.0, 5.0)]
        self.assertEqual(make_pairs(orig, corr), [])

    def test_insert_has_null_rejected(self):
        orig = []
        corr = [_seg("new1", "Preparation", 0.0, 5.0)]
        pairs = make_pairs(orig, corr)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["change"], "inserted")
        self.assertIsNone(pairs[0]["rejected"])
        self.assertIsNotNone(pairs[0]["chosen"])

    def test_delete_has_null_chosen(self):
        orig = [_seg("d1", "Preparation", 0.0, 5.0)]
        corr = []
        pairs = make_pairs(orig, corr)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["change"], "deleted")
        self.assertIsNone(pairs[0]["chosen"])

    def test_empty_original_all_inserted(self):
        corr = [_seg("a", "Preparation", 0.0, 5.0),
                _seg("b", "ClippingCutting", 5.0, 10.0)]
        pairs = make_pairs([], corr, procedure="cholecystectomy", video="case.mp4")
        self.assertEqual(len(pairs), 2)
        self.assertTrue(all(p["change"] == "inserted" for p in pairs))
        self.assertTrue(all(p["rejected"] is None for p in pairs))
        # procedure/video は全ペアに伝播していること（最初の要素だけの実装を弾く）
        self.assertTrue(all(p["procedure"] == "cholecystectomy" for p in pairs))
        self.assertTrue(all(p["video"] == "case.mp4" for p in pairs))
        # start_sec昇順
        self.assertEqual(pairs[0]["chosen"]["start_sec"], 0.0)

    def test_pairs_are_json_serializable(self):
        pairs = make_pairs([_seg("x1", "Preparation", 0.0, 5.0)],
                           [_seg("x1", "ClippingCutting", 0.0, 5.0)])
        # JSONL化できること
        line = json.dumps(pairs[0], ensure_ascii=False)
        self.assertIn("ClippingCutting", line)

    # ---- 異常系 / 境界条件 -------------------------------------------------
    def test_none_times_default_to_zero(self):
        # start_sec/end_sec が None でも例外を投げず 0.0 に倒す
        orig = [_seg("x1", "Preparation", None, None)]
        corr = [_seg("x1", "ClippingCutting", None, None)]
        pairs = make_pairs(orig, corr)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["change"], "edited")
        self.assertEqual(pairs[0]["rejected"]["start_sec"], 0.0)
        self.assertEqual(pairs[0]["rejected"]["end_sec"], 0.0)

    def test_non_numeric_time_string_defaults_to_zero(self):
        # 数値変換不能な時刻文字列でもクラッシュせず 0.0 に倒す
        orig = [_seg("x1", "Preparation", "abc", 5.0)]
        corr = [_seg("x1", "Preparation", "abc", 7.5)]
        pairs = make_pairs(orig, corr)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["rejected"]["start_sec"], 0.0)
        self.assertEqual(pairs[0]["chosen"]["end_sec"], 7.5)

    def test_numeric_time_string_is_parsed(self):
        # 数値として解釈可能な文字列は float 化される
        orig = [_seg("x1", "Preparation", "0.0", "5.0")]
        corr = [_seg("x1", "Preparation", "0.0", "8.0")]
        pairs = make_pairs(orig, corr)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["change"], "edited")
        self.assertEqual(pairs[0]["chosen"]["end_sec"], 8.0)

    def test_missing_id_segment_is_ignored(self):
        # id 欠落セグメントは突き合わせ対象から除外される
        orig = [_seg(None, "Preparation", 0.0, 5.0),
                _seg("x1", "ClippingCutting", 5.0, 10.0)]
        corr = [{"phase_name": "Preparation", "start_sec": 0.0, "end_sec": 5.0},
                _seg("x1", "ClippingCutting", 5.0, 10.0)]
        pairs = make_pairs(orig, corr)
        # id無しは両側で無視され、id一致かつ無変更の x1 のみ → 出力ゼロ
        self.assertEqual(pairs, [])

    def test_duplicate_id_keeps_last_occurrence(self):
        # 同一 id が複数ある場合は後勝ち（dict採番）で1件に集約される
        orig = [_seg("dup", "Preparation", 0.0, 5.0),
                _seg("dup", "ClippingCutting", 5.0, 10.0)]
        corr = [_seg("dup", "ClippingCutting", 5.0, 10.0)]
        pairs = make_pairs(orig, corr)
        # 後勝ちした orig（ClippingCutting/5-10）と corr が一致 → 変更なし
        self.assertEqual(pairs, [])

    def test_empty_both_emits_nothing(self):
        self.assertEqual(make_pairs([], []), [])


if __name__ == "__main__":
    unittest.main()
