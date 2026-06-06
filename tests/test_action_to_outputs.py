"""
action_to_outputs.py のユニットテスト

合成入力（triplet 確率ベクトルCSV、triplet id CSV、汎用 action CSV、clip 単位
ラベルCSV）を使い、triplet マップの読み込みと id→(i,v,t) 分解、区間化・最小継続長・
スムージング・triplet 分解(--decompose)・多ラベル重なり・3出力フォーマット・
時刻整合、および JSONL→SRT→JSONL の往復（情報保持）を検証する。
"""

import json
import tempfile
import unittest
from pathlib import Path

from src.action.action_to_outputs import (
    BuildResult,
    LabelKey,
    TripletMap,
    Unit,
    build_events,
    convert,
    main,
    parse_input,
    resolve_clip_times,
    resolve_frame_times,
    _smooth_presence,
)
from src.tools.srt_to_jsonl import convert as srt_to_jsonl


def _write(path: Path, text: str) -> str:
    path.write_text(text, encoding="utf-8")
    return str(path)


def _csv(tmpdir: str, name: str, header: str, rows) -> str:
    lines = [header] + [",".join(str(c) for c in r) for r in rows]
    return _write(Path(tmpdir) / name, "\n".join(lines) + "\n")


# ===========================================================================
# triplet マップ
# ===========================================================================

class TestTripletMap(unittest.TestCase):
    def test_default_map_counts(self):
        tm = TripletMap.load(None)
        self.assertEqual(tm.name, "cholect50")
        self.assertEqual(tm.fps, 25.0)
        self.assertEqual(len(tm.triplets), 100)
        self.assertEqual(len(tm.instruments), 6)
        self.assertEqual(len(tm.verbs), 10)
        self.assertEqual(len(tm.targets), 15)

    def test_decompose_known_ids(self):
        tm = TripletMap.load(None)
        # 公式 ivtmetrics maps.txt と Rendezvous taxonomy に基づく検証
        self.assertEqual(tm.decompose(7), ("grasper", "grasp", "gallbladder"))
        self.assertEqual(tm.decompose(1), ("grasper", "dissect", "gallbladder"))
        self.assertEqual(tm.decompose(77), ("clipper", "clip", "blood_vessel"))
        self.assertEqual(tm.decompose(82), ("irrigator", "aspirate", "fluid"))
        self.assertEqual(tm.decompose(94), ("grasper", "null_verb", "null_target"))

    def test_label_string(self):
        tm = TripletMap.load(None)
        self.assertEqual(tm.label(7), "grasper,grasp,gallbladder")

    def test_unknown_id(self):
        tm = TripletMap.load(None)
        self.assertIsNone(tm.decompose(999))
        self.assertEqual(tm.label(999), "triplet_999")

    def test_custom_map(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "m.json"
            p.write_text(json.dumps({
                "name": "tiny", "fps": 10,
                "instruments": ["a"], "verbs": ["x"], "targets": ["m", "n"],
                "triplets": {"0": [0, 0, 0], "1": [0, 0, 1]},
            }), encoding="utf-8")
            tm = TripletMap.load(str(p))
            self.assertEqual(tm.fps, 10.0)
            self.assertEqual(tm.decompose(1), ("a", "x", "n"))


# ===========================================================================
# 入力パース
# ===========================================================================

class TestParseInput(unittest.TestCase):
    def test_triplet_id_long_multilabel(self):
        # 同一 frame_idx の複数行 = 多ラベル
        with tempfile.TemporaryDirectory() as d:
            path = _csv(d, "t.csv", "frame_idx,triplet_id", [
                (0, 7), (0, 1), (1, 7),
            ])
            tm = TripletMap.load(None)
            units, lk, uk = parse_input(path, tm, triplet_thr=0.5)
            self.assertEqual(lk, "triplet")
            self.assertEqual(uk, "frame")
            self.assertEqual(len(units), 2)
            self.assertEqual(len(units[0].active), 2)  # frame 0 は 2 triplet
            self.assertEqual(len(units[1].active), 1)

    def test_triplet_vector_threshold(self):
        with tempfile.TemporaryDirectory() as d:
            path = _csv(d, "t.csv", "frame_idx,triplet_7,triplet_1", [
                (0, 0.9, 0.1), (1, 0.9, 0.8),
            ])
            tm = TripletMap.load(None)
            units, lk, uk = parse_input(path, tm, triplet_thr=0.5)
            self.assertEqual(lk, "triplet")
            self.assertEqual(len(units[0].active), 1)  # triplet_1 は閾値未満
            self.assertEqual(len(units[1].active), 2)

    def test_action_name(self):
        with tempfile.TemporaryDirectory() as d:
            path = _csv(d, "a.csv", "frame_idx,action_name,confidence", [
                (0, "navigation", 0.9), (1, "navigation", 0.8),
            ])
            tm = TripletMap.load(None)
            units, lk, uk = parse_input(path, tm, triplet_thr=0.5)
            self.assertEqual(lk, "action")
            self.assertEqual(uk, "frame")
            self.assertEqual(units[0].active[0][0].label, "navigation")

    def test_clip_detection(self):
        with tempfile.TemporaryDirectory() as d:
            path = _csv(d, "c.csv", "clip_idx,action_name", [
                (0, "A"), (1, "A"), (2, "B"),
            ])
            tm = TripletMap.load(None)
            units, lk, uk = parse_input(path, tm, triplet_thr=0.5)
            self.assertEqual(uk, "clip")
            self.assertEqual(len(units), 3)

    def test_no_label_column_raises(self):
        with tempfile.TemporaryDirectory() as d:
            path = _csv(d, "x.csv", "frame_idx,foo", [(0, 1)])
            tm = TripletMap.load(None)
            with self.assertRaises(ValueError):
                parse_input(path, tm, triplet_thr=0.5)


# ===========================================================================
# 時刻解決
# ===========================================================================

class TestTimeResolution(unittest.TestCase):
    def test_frame_fps(self):
        units = [Unit(idx=i) for i in range(4)]
        times = resolve_frame_times(units, video_path=None, fps=2.0)
        self.assertAlmostEqual(times[0][0], 0.0)
        self.assertAlmostEqual(times[0][1], 0.5)
        self.assertAlmostEqual(times[1][0], 0.5)
        # 最終フレームは +period
        self.assertAlmostEqual(times[3][1], 2.0)

    def test_clip_len(self):
        units = [Unit(idx=i) for i in range(3)]
        times = resolve_clip_times(units, fps=25.0, clip_len=1.0)
        self.assertEqual(times[0], (0.0, 1.0))
        self.assertEqual(times[2], (2.0, 3.0))

    def test_clip_explicit_start_end(self):
        units = [Unit(idx=0, t_sec=0.0, t_end_in=2.0),
                 Unit(idx=1, t_sec=2.0, t_end_in=5.0)]
        times = resolve_clip_times(units, fps=25.0, clip_len=None)
        self.assertEqual(times[0], (0.0, 2.0))
        self.assertEqual(times[1], (2.0, 5.0))

    def test_existing_timestamp_preserved(self):
        units = [Unit(idx=0, t_sec=3.14), Unit(idx=1, t_sec=3.64)]
        times = resolve_frame_times(units, video_path=None, fps=25.0)
        self.assertAlmostEqual(times[0][0], 3.14)


# ===========================================================================
# スムージング
# ===========================================================================

class TestSmoothing(unittest.TestCase):
    def test_flicker_filled_and_removed(self):
        # 孤立した False（在の中の不在）を多数決で埋める
        pres = [True, True, False, True, True]
        out = _smooth_presence(pres, window=3)
        self.assertEqual(out, [True, True, True, True, True])

    def test_isolated_true_removed(self):
        pres = [False, False, True, False, False]
        out = _smooth_presence(pres, window=3)
        self.assertEqual(out, [False, False, False, False, False])

    def test_window_zero_noop(self):
        pres = [True, False, True]
        self.assertIs(_smooth_presence(pres, 0), pres)


# ===========================================================================
# 区間化・多ラベル重なり
# ===========================================================================

class TestSegmentationMultiLabel(unittest.TestCase):
    def test_overlapping_triplets(self):
        # triplet_7: frame 0..5 / triplet_1: frame 2..3（重なり）
        with tempfile.TemporaryDirectory() as d:
            rows = []
            for i in range(6):
                rows.append((i, 0.9, 0.9 if i in (2, 3) else 0.1))
            path = _csv(d, "t.csv", "frame_idx,triplet_7,triplet_1", rows)
            tm = TripletMap.load(None)
            res = build_events(path, tm, fps=2.0, min_duration=0.4)
            # 2 区間（triplet 7 と triplet 1）、時間が重なる
            self.assertEqual(len(res.events), 2)
            by_label = {e["label"]: e for e in res.events}
            t7 = by_label["grasper,grasp,gallbladder"]
            t1 = by_label["grasper,dissect,gallbladder"]
            self.assertAlmostEqual(t7["start_sec"], 0.0)
            self.assertAlmostEqual(t7["end_sec"], 3.0)
            self.assertAlmostEqual(t1["start_sec"], 1.0)
            self.assertAlmostEqual(t1["end_sec"], 2.0)
            # 重なり区間 [1,2) が t7 の内側にある（多ラベル同時成立）
            self.assertLess(t7["start_sec"], t1["start_sec"])
            self.assertGreater(t7["end_sec"], t1["end_sec"])

    def test_min_duration_drops_short(self):
        with tempfile.TemporaryDirectory() as d:
            rows = [(i, 0.9, 0.9 if i == 2 else 0.1) for i in range(6)]
            path = _csv(d, "t.csv", "frame_idx,triplet_7,triplet_1", rows)
            tm = TripletMap.load(None)
            # triplet_1 は 1 フレーム（0.5秒）→ min_duration 1.0 で除去
            res = build_events(path, tm, fps=2.0, min_duration=1.0)
            labels = {e["label"] for e in res.events}
            self.assertIn("grasper,grasp,gallbladder", labels)
            self.assertNotIn("grasper,dissect,gallbladder", labels)

    def test_idempotent(self):
        with tempfile.TemporaryDirectory() as d:
            rows = [(i, 0.9, 0.9 if i in (2, 3) else 0.1) for i in range(6)]
            path = _csv(d, "t.csv", "frame_idx,triplet_7,triplet_1", rows)
            tm = TripletMap.load(None)
            a = build_events(path, tm, fps=2.0, min_duration=0.4)
            b = build_events(path, tm, fps=2.0, min_duration=0.4)
            self.assertEqual(a.events, b.events)


# ===========================================================================
# triplet 分解（--decompose）
# ===========================================================================

class TestDecompose(unittest.TestCase):
    def test_decompose_adds_component_tracks(self):
        with tempfile.TemporaryDirectory() as d:
            rows = [(i, 7) for i in range(4)]
            path = _csv(d, "t.csv", "frame_idx,triplet_id", rows)
            tm = TripletMap.load(None)
            res = build_events(path, tm, fps=2.0, min_duration=0.4,
                               decompose=True)
            types = [(e["type"], e.get("role"), e["label"]) for e in res.events]
            # triplet トラック 1 + 分解 3（instrument/verb/target）
            self.assertIn((("triplet"), None, "grasper,grasp,gallbladder"), types)
            self.assertIn(("action", "instrument", "grasper"), types)
            self.assertIn(("action", "verb", "grasp"), types)
            self.assertIn(("action", "target", "gallbladder"), types)
            self.assertEqual(len(res.events), 4)

    def test_decompose_merges_shared_components(self):
        # triplet 7 と 1 は instrument=grasper, target=gallbladder を共有
        with tempfile.TemporaryDirectory() as d:
            rows = []
            for i in range(4):
                rows.append((i, 7))
                rows.append((i, 1))
            path = _csv(d, "t.csv", "frame_idx,triplet_id", rows)
            tm = TripletMap.load(None)
            res = build_events(path, tm, fps=2.0, min_duration=0.4,
                               decompose=True)
            inst = [e for e in res.events
                    if e.get("role") == "instrument" and e["label"] == "grasper"]
            # grasper は 1 区間に統合（両 triplet で連続して在）
            self.assertEqual(len(inst), 1)
            verbs = {e["label"] for e in res.events if e.get("role") == "verb"}
            self.assertEqual(verbs, {"grasp", "dissect"})

    def test_no_decompose_by_default(self):
        with tempfile.TemporaryDirectory() as d:
            rows = [(i, 7) for i in range(4)]
            path = _csv(d, "t.csv", "frame_idx,triplet_id", rows)
            tm = TripletMap.load(None)
            res = build_events(path, tm, fps=2.0, min_duration=0.4)
            self.assertTrue(all(e.get("role") is None for e in res.events))


# ===========================================================================
# イベントスキーマ
# ===========================================================================

class TestEventSchema(unittest.TestCase):
    def test_triplet_event_keys(self):
        with tempfile.TemporaryDirectory() as d:
            rows = [(i, 0.8) for i in range(4)]
            path = _csv(d, "t.csv", "frame_idx,triplet_7", rows)
            tm = TripletMap.load(None)
            res = build_events(path, tm, fps=2.0, min_duration=0.4)
            ev = res.events[0]
            for key in ("type", "label", "source", "start_sec", "end_sec",
                        "start_srt", "end_srt", "duration_sec", "triplet_id",
                        "components", "confidence"):
                self.assertIn(key, ev)
            self.assertEqual(ev["type"], "triplet")
            self.assertEqual(ev["source"], "action_converter")
            self.assertEqual(ev["triplet_id"], 7)
            self.assertEqual(ev["components"],
                             {"instrument": "grasper", "verb": "grasp",
                              "target": "gallbladder"})
            self.assertAlmostEqual(ev["confidence"], 0.8)

    def test_action_event_keys(self):
        with tempfile.TemporaryDirectory() as d:
            rows = [(i, "navigation") for i in range(4)]
            path = _csv(d, "a.csv", "frame_idx,action_name", rows)
            tm = TripletMap.load(None)
            res = build_events(path, tm, fps=2.0, min_duration=0.4)
            ev = res.events[0]
            self.assertEqual(ev["type"], "action")
            self.assertEqual(ev["label"], "navigation")
            self.assertNotIn("components", ev)


# ===========================================================================
# convert() の3出力
# ===========================================================================

class TestConvertOutputs(unittest.TestCase):
    def _run(self, level="segment", decompose=False):
        d = tempfile.mkdtemp()
        rows = [(i, 7) for i in range(4)] + [(i, 1) for i in range(4, 8)]
        path = _csv(d, "triplet_pred.csv", "frame_idx,triplet_id", rows)
        outdir = Path(d) / "out"
        result = convert(in_path=path, outdir=str(outdir), fps=2.0,
                         min_duration=0.4, level=level, decompose=decompose)
        return outdir, result

    def test_three_files(self):
        _, result = self._run()
        self.assertTrue(Path(result["jsonl"]).exists())
        self.assertTrue(Path(result["srt"]).exists())
        self.assertTrue(Path(result["csv"]).exists())
        self.assertTrue(result["jsonl"].endswith("triplet_pred_action.jsonl"))
        self.assertTrue(result["srt"].endswith("triplet_pred_action.srt"))
        self.assertTrue(result["csv"].endswith("triplet_pred_action.csv"))

    def test_jsonl_valid(self):
        _, result = self._run()
        lines = Path(result["jsonl"]).read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 2)
        for line in lines:
            obj = json.loads(line)
            self.assertEqual(obj["type"], "triplet")

    def test_srt_two_line_structure(self):
        _, result = self._run()
        content = Path(result["srt"]).read_text(encoding="utf-8")
        blocks = [b for b in content.strip().split("\n\n") if b.strip()]
        self.assertEqual(len(blocks), 2)
        first = blocks[0].split("\n")
        self.assertEqual(len(first), 4)
        self.assertIn("-->", first[1])
        self.assertTrue(first[2].startswith("[action] "))
        meta = json.loads(first[3])
        self.assertEqual(meta["type"], "triplet")
        self.assertNotIn("start_sec", meta)
        self.assertNotIn("start_srt", meta)

    def test_segment_csv_header(self):
        _, result = self._run(level="segment")
        head = Path(result["csv"]).read_text(encoding="utf-8").splitlines()[0]
        for col in ("segment_id", "type", "label", "instrument", "verb",
                    "target", "role"):
            self.assertIn(col, head)

    def test_frame_level_csv(self):
        _, result = self._run(level="frame")
        head = Path(result["csv"]).read_text(encoding="utf-8").splitlines()[0]
        self.assertIn("unit_idx", head)
        rows = Path(result["csv"]).read_text(encoding="utf-8").strip().splitlines()
        # 8 フレーム × 各1ラベル
        self.assertEqual(len(rows) - 1, 8)

    def test_both_level_creates_frames_csv(self):
        _, result = self._run(level="both")
        self.assertIn("csv_frames", result)
        self.assertTrue(Path(result["csv_frames"]).exists())

    def test_decompose_outputs(self):
        _, result = self._run(decompose=True)
        objs = [json.loads(l) for l in
                Path(result["jsonl"]).read_text(encoding="utf-8").strip().splitlines()]
        roles = {o.get("role") for o in objs}
        self.assertTrue({"instrument", "verb", "target"}.issubset(roles))

    def test_idempotent_files(self):
        d = tempfile.mkdtemp()
        rows = [(i, 7) for i in range(4)]
        path = _csv(d, "triplet_pred.csv", "frame_idx,triplet_id", rows)
        o1, o2 = Path(d) / "o1", Path(d) / "o2"
        convert(in_path=path, outdir=str(o1), fps=2.0, min_duration=0.4)
        convert(in_path=path, outdir=str(o2), fps=2.0, min_duration=0.4)
        for fn in ("triplet_pred_action.jsonl", "triplet_pred_action.srt",
                   "triplet_pred_action.csv"):
            self.assertEqual((o1 / fn).read_text(encoding="utf-8"),
                             (o2 / fn).read_text(encoding="utf-8"))


# ===========================================================================
# clip 入力（SLAM 風）
# ===========================================================================

class TestClipInput(unittest.TestCase):
    def test_clip_actions_segmented(self):
        with tempfile.TemporaryDirectory() as d:
            path = _csv(d, "clip.csv", "clip_idx,action_name", [
                (0, "A"), (1, "A"), (2, "B"),
            ])
            tm = TripletMap.load(None)
            res = build_events(path, tm, clip_len=1.0, min_duration=0.4)
            self.assertEqual(res.unit_kind, "clip")
            by = {e["label"]: e for e in res.events}
            self.assertAlmostEqual(by["A"]["start_sec"], 0.0)
            self.assertAlmostEqual(by["A"]["end_sec"], 2.0)
            self.assertAlmostEqual(by["B"]["start_sec"], 2.0)
            self.assertAlmostEqual(by["B"]["end_sec"], 3.0)


# ===========================================================================
# JSONL → SRT → JSONL 往復
# ===========================================================================

class TestRoundTrip(unittest.TestCase):
    def test_roundtrip_preserves_info(self):
        d = tempfile.mkdtemp()
        # triplet_7 を confidence 付きベクトルで（mean が一定になるよう固定）
        rows = [(i, 0.9) for i in range(6)]
        path = _csv(d, "triplet_pred.csv", "frame_idx,triplet_7", rows)
        result = convert(in_path=path, outdir=str(Path(d) / "out"), fps=2.0,
                         min_duration=0.4)

        restored_path = Path(d) / "restored.jsonl"
        srt_to_jsonl(result["srt"], str(restored_path))
        restored = [json.loads(l) for l in
                    restored_path.read_text(encoding="utf-8").strip().splitlines()]
        original = [json.loads(l) for l in
                    Path(result["jsonl"]).read_text(encoding="utf-8").strip().splitlines()]

        self.assertEqual(len(restored), len(original))
        for orig, rest in zip(original, restored):
            self.assertEqual(rest["type"], orig["type"])
            self.assertEqual(rest["label"], orig["label"])
            self.assertEqual(rest["components"], orig["components"])
            self.assertEqual(rest["triplet_id"], orig["triplet_id"])
            self.assertAlmostEqual(rest["confidence"], orig["confidence"])
            self.assertAlmostEqual(rest["start_sec"], orig["start_sec"], places=2)
            self.assertAlmostEqual(rest["end_sec"], orig["end_sec"], places=2)


# ===========================================================================
# CLI
# ===========================================================================

class TestCLI(unittest.TestCase):
    def test_cli_three_files(self):
        d = tempfile.mkdtemp()
        rows = [(i, 7) for i in range(50)] + [(i, 1) for i in range(50, 100)]
        path = _csv(d, "triplet_pred.csv", "frame_idx,triplet_id", rows)
        outdir = Path(d) / "out"
        rc = main(["--in", path, "--fps", "25", "--outdir", str(outdir),
                   "--min-duration", "0.5", "--decompose", "--level", "both"])
        self.assertEqual(rc, 0)
        stem = "triplet_pred_action"
        self.assertTrue((outdir / f"{stem}.jsonl").exists())
        self.assertTrue((outdir / f"{stem}.srt").exists())
        self.assertTrue((outdir / f"{stem}.csv").exists())
        self.assertTrue((outdir / f"{stem}_frames.csv").exists())


# ===========================================================================
# BaseAnalyzer アダプタ
# ===========================================================================

class TestBaseAnalyzer(unittest.TestCase):
    def test_make_analyzer(self):
        from src.action.action_to_outputs import make_analyzer

        d = tempfile.mkdtemp()
        rows = [(i, 7) for i in range(4)]
        path = _csv(d, "triplet_pred.csv", "frame_idx,triplet_id", rows)
        analyzer = make_analyzer()
        result = analyzer.analyze(video_path=None, in_path=path, fps=2.0,
                                  min_duration=0.4)
        self.assertEqual(result.analyzer_type, "surgical_action")
        self.assertEqual(result.results[0]["type"], "triplet")


if __name__ == "__main__":
    unittest.main()
