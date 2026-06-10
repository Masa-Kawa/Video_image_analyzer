"""
bleed_model_to_outputs.py のユニットテスト

合成 fixture（per-frame 確率系列 CSV、IAE 区間 JSON、region/point 付き）を使い、
ヒステリシス区間化・最小継続長・severity 保持・region/point pass-through・
3出力フォーマット、JSONL→SRT→JSONL の往復（情報保持）、および既存 SRT との
merge_srt 統合を検証する。
"""

import json
import tempfile
import unittest
from pathlib import Path

from src.red.bleed_model_to_outputs import (
    EVENT_TYPE,
    SOURCE,
    FrameScore,
    build_events,
    convert,
    detect_input_kind,
    frames_to_events,
    hysteresis_segments,
    intervals_to_events,
    main,
    parse_interval_json,
    parse_perframe_csv,
    resolve_frame_times,
    resolve_interval_times,
    _pick,
    _region_from,
    _point_from,
)
from src.tools.merge_srt import merge, read_srt
from src.tools.srt_to_jsonl import convert as srt_to_jsonl


def _write(path: Path, text: str) -> str:
    path.write_text(text, encoding="utf-8")
    return str(path)


def _perframe_csv(tmpdir: str, header: str, rows) -> str:
    lines = [header] + [",".join("" if c is None else str(c) for c in r)
                        for r in rows]
    return _write(Path(tmpdir) / "pred.csv", "\n".join(lines) + "\n")


def _json_fixture(tmpdir: str, obj, name="iae.json") -> str:
    return _write(Path(tmpdir) / name, json.dumps(obj, ensure_ascii=False))


# per-frame 確率系列（fps=2 → 0.5秒間隔）。
# 区間1: idx1..3（prob>=0.3 連続、peak=0.7@idx2）→ 0.5..2.0（1.5秒）
# 区間2: idx6 のみ → 3.0..3.5（0.5秒, min_duration=1.0 で除去対象）
_PROB_HEADER = "frame_idx,bleed_prob,x,y,w,h,point_x,point_y"
_PROB_ROWS = [
    (0, 0.10, None, None, None, None, None, None),
    (1, 0.60, None, None, None, None, None, None),
    (2, 0.70, 10, 20, 30, 40, 15, 25),  # peak（region/point 付き）
    (3, 0.40, None, None, None, None, None, None),
    (4, 0.20, None, None, None, None, None, None),
    (5, 0.10, None, None, None, None, None, None),
    (6, 0.90, None, None, None, None, None, None),
    (7, 0.10, None, None, None, None, None, None),
    (8, 0.10, None, None, None, None, None, None),
    (9, 0.10, None, None, None, None, None, None),
]


class TestFormatDetection(unittest.TestCase):
    def test_detect_perframe_csv(self):
        with tempfile.TemporaryDirectory() as d:
            p = _perframe_csv(d, "frame_idx,bleed_prob", [(0, 0.1)])
            self.assertEqual(detect_input_kind(p), "perframe_csv")

    def test_detect_interval_csv(self):
        with tempfile.TemporaryDirectory() as d:
            p = _perframe_csv(d, "start_sec,end_sec,severity",
                              [(1.0, 2.0, 3)])
            self.assertEqual(detect_input_kind(p), "interval_csv")

    def test_detect_interval_json(self):
        with tempfile.TemporaryDirectory() as d:
            p = _json_fixture(d, [{"start_sec": 1.0, "end_sec": 2.0}])
            self.assertEqual(detect_input_kind(p), "interval_json")

    def test_detect_unknown_csv_raises(self):
        with tempfile.TemporaryDirectory() as d:
            p = _perframe_csv(d, "frame_idx,foo", [(0, 1)])
            with self.assertRaises(ValueError):
                detect_input_kind(p)


class TestPerFrameParser(unittest.TestCase):
    def test_parse_prob_region_point(self):
        with tempfile.TemporaryDirectory() as d:
            p = _perframe_csv(d, _PROB_HEADER, _PROB_ROWS)
            frames = parse_perframe_csv(p)
            self.assertEqual(len(frames), 10)
            self.assertAlmostEqual(frames[2].bleed_prob, 0.70)
            self.assertEqual(frames[2].region,
                             {"x": 10.0, "y": 20.0, "w": 30.0, "h": 40.0})
            self.assertEqual(frames[2].point, {"x": 15.0, "y": 25.0})
            self.assertIsNone(frames[0].region)
            self.assertIsNone(frames[0].point)

    def test_row_order_frame_idx(self):
        with tempfile.TemporaryDirectory() as d:
            p = _perframe_csv(d, "bleed_prob", [(0.1,), (0.2,), (0.3,)])
            frames = parse_perframe_csv(p)
            self.assertEqual([f.frame_idx for f in frames], [0, 1, 2])


class TestTimeResolution(unittest.TestCase):
    def test_frame_fps_resolution(self):
        frames = [FrameScore(i, None, 0.1) for i in range(4)]
        resolve_frame_times(frames, video_path=None, fps=2.0)
        self.assertAlmostEqual(frames[0].t_sec, 0.0)
        self.assertAlmostEqual(frames[1].t_sec, 0.5)
        self.assertAlmostEqual(frames[3].t_sec, 1.5)

    def test_existing_timestamp_preserved(self):
        frames = [FrameScore(0, 3.14, 0.1)]
        resolve_frame_times(frames, video_path=None, fps=25.0)
        self.assertAlmostEqual(frames[0].t_sec, 3.14)

    def test_interval_frame_resolution(self):
        records = [{"start_sec": None, "end_sec": None,
                    "start_frame": 50, "end_frame": 100,
                    "severity": None, "label": None, "confidence": None,
                    "region": None, "point": None}]
        resolve_interval_times(records, video_path=None, fps=25.0)
        self.assertAlmostEqual(records[0]["start_sec"], 2.0)
        self.assertAlmostEqual(records[0]["end_sec"], 4.0)


class TestHysteresis(unittest.TestCase):
    def test_hysteresis_on_off(self):
        probs = [0.1, 0.6, 0.7, 0.4, 0.2, 0.1, 0.9, 0.1]
        segs = hysteresis_segments(probs, thr_on=0.5, thr_off=0.3)
        # 区間1: idx1..3（idx4 で OFF）、区間2: idx6
        self.assertEqual(segs, [(1, 4), (6, 7)])

    def test_no_event_below_thr_on(self):
        segs = hysteresis_segments([0.4, 0.45, 0.49], 0.5, 0.3)
        self.assertEqual(segs, [])

    def test_open_ended_event(self):
        segs = hysteresis_segments([0.1, 0.6, 0.7], 0.5, 0.3)
        self.assertEqual(segs, [(1, 3)])


class TestFramesToEvents(unittest.TestCase):
    def _frames(self, fps=2.0):
        with tempfile.TemporaryDirectory() as d:
            p = _perframe_csv(d, _PROB_HEADER, _PROB_ROWS)
            frames = parse_perframe_csv(p)
        resolve_frame_times(frames, None, fps)
        return frames

    def test_min_duration_removes_short(self):
        frames = self._frames()
        events, ids = frames_to_events(frames, 0.5, 0.3, min_duration=1.0,
                                       fps=2.0)
        # 短い区間2（0.5秒）は除去され、区間1のみ
        self.assertEqual(len(events), 1)
        ev = events[0]
        self.assertAlmostEqual(ev["start_sec"], 0.5)
        self.assertAlmostEqual(ev["end_sec"], 2.0)
        self.assertAlmostEqual(ev["duration_sec"], 1.5)
        # frame_event_ids: idx1..3 が区間1、idx6（除去）は None
        self.assertEqual(ids[1], 1)
        self.assertEqual(ids[3], 1)
        self.assertIsNone(ids[6])
        self.assertIsNone(ids[0])

    def test_min_duration_keeps_both(self):
        frames = self._frames()
        events, _ = frames_to_events(frames, 0.5, 0.3, min_duration=0.3,
                                     fps=2.0)
        self.assertEqual(len(events), 2)

    def test_confidence_is_peak(self):
        frames = self._frames()
        events, _ = frames_to_events(frames, 0.5, 0.3, min_duration=1.0,
                                     fps=2.0)
        self.assertAlmostEqual(events[0]["confidence"], 0.70)

    def test_region_point_passthrough(self):
        frames = self._frames()
        events, _ = frames_to_events(frames, 0.5, 0.3, min_duration=1.0,
                                     fps=2.0)
        self.assertEqual(events[0]["region"],
                         {"x": 10.0, "y": 20.0, "w": 30.0, "h": 40.0})
        self.assertEqual(events[0]["point"], {"x": 15.0, "y": 25.0})

    def test_event_schema(self):
        frames = self._frames()
        events, _ = frames_to_events(frames, 0.5, 0.3, 1.0, 2.0)
        ev = events[0]
        for key in ("type", "source", "start_sec", "end_sec",
                    "start_srt", "end_srt", "duration_sec", "confidence"):
            self.assertIn(key, ev)
        self.assertEqual(ev["type"], EVENT_TYPE)
        self.assertEqual(ev["source"], SOURCE)
        # 確率系列には severity が無い
        self.assertNotIn("severity", ev)


# IAE 風区間ラベル JSON（severity / region / point / 非出血ラベルを含む）
_IAE_EVENTS = [
    {"start_sec": 10.0, "end_sec": 15.0, "severity": 3, "type": "Bleeding"},
    {"start_sec": 20.0, "end_sec": 22.0, "severity": 5, "type": "Bleeding",
     "bbox": [1, 2, 3, 4], "point": [5, 6], "confidence": 0.88},
    {"start_sec": 30.0, "end_sec": 31.0, "type": "Mechanical injury"},
]


class TestIntervalEvents(unittest.TestCase):
    def test_parse_and_filter_bleeding(self):
        with tempfile.TemporaryDirectory() as d:
            p = _json_fixture(d, _IAE_EVENTS)
            records = parse_interval_json(p)
            self.assertEqual(len(records), 3)
            events = intervals_to_events(records, min_duration=0.0,
                                         include_all=False)
            # 非出血（Mechanical injury）は除外 → 2件
            self.assertEqual(len(events), 2)
            self.assertEqual(events[0]["severity"], 3)
            self.assertEqual(events[1]["severity"], 5)

    def test_include_all(self):
        with tempfile.TemporaryDirectory() as d:
            p = _json_fixture(d, _IAE_EVENTS)
            records = parse_interval_json(p)
            events = intervals_to_events(records, min_duration=0.0,
                                         include_all=True)
            self.assertEqual(len(events), 3)

    def test_region_point_severity_passthrough(self):
        with tempfile.TemporaryDirectory() as d:
            p = _json_fixture(d, _IAE_EVENTS)
            records = parse_interval_json(p)
            events = intervals_to_events(records, 0.0, include_all=False)
            ev = events[1]
            self.assertEqual(ev["severity"], 5)
            self.assertEqual(ev["region"],
                             {"x": 1.0, "y": 2.0, "w": 3.0, "h": 4.0})
            self.assertEqual(ev["point"], {"x": 5.0, "y": 6.0})
            self.assertAlmostEqual(ev["confidence"], 0.88)

    def test_dict_with_events_key(self):
        with tempfile.TemporaryDirectory() as d:
            p = _json_fixture(d, {"events": _IAE_EVENTS})
            records = parse_interval_json(p)
            self.assertEqual(len(records), 3)

    def test_sorted_by_start(self):
        with tempfile.TemporaryDirectory() as d:
            shuffled = list(reversed(_IAE_EVENTS))
            p = _json_fixture(d, shuffled)
            records = parse_interval_json(p)
            events = intervals_to_events(records, 0.0, include_all=True)
            starts = [e["start_sec"] for e in events]
            self.assertEqual(starts, sorted(starts))


class TestConvertPerFrame(unittest.TestCase):
    def _run(self, min_duration=1.0):
        d = tempfile.mkdtemp()
        p = _perframe_csv(d, _PROB_HEADER, _PROB_ROWS)
        outdir = Path(d) / "out"
        result = convert(in_path=p, outdir=str(outdir), fps=2.0,
                         thr_on=0.5, thr_off=0.3, min_duration=min_duration)
        return outdir, result

    def test_three_files_and_naming(self):
        _, result = self._run()
        for key in ("jsonl", "srt", "csv"):
            self.assertTrue(Path(result[key]).exists())
        self.assertTrue(result["jsonl"].endswith("pred_bleed_model.jsonl"))
        self.assertTrue(result["srt"].endswith("pred_bleed_model.srt"))
        self.assertTrue(result["csv"].endswith("pred_bleed_model.csv"))

    def test_srt_two_line_structure(self):
        _, result = self._run()
        content = Path(result["srt"]).read_text(encoding="utf-8")
        blocks = [b for b in content.strip().split("\n\n") if b.strip()]
        self.assertEqual(len(blocks), 1)
        lines = blocks[0].split("\n")
        self.assertEqual(len(lines), 4)
        self.assertIn("-->", lines[1])
        self.assertEqual(lines[2], "[bleed] bleeding")
        meta = json.loads(lines[3])
        self.assertEqual(meta["type"], EVENT_TYPE)
        self.assertEqual(meta["source"], SOURCE)
        self.assertNotIn("start_sec", meta)
        self.assertNotIn("start_srt", meta)
        # region/point は JSON行に保持
        self.assertIn("region", meta)
        self.assertIn("point", meta)

    def test_frame_csv_columns(self):
        import csv as _csv
        _, result = self._run()
        with open(result["csv"], encoding="utf-8") as f:
            rows = list(_csv.reader(f))
        self.assertEqual(
            rows[0],
            ["frame_idx", "t_sec", "t_srt", "bleed_prob",
             "in_event", "event_id"],
        )
        # 全フレーム分の行（10）
        self.assertEqual(len(rows) - 1, 10)
        # idx2 は in_event=1, event_id=1（行 = ヘッダ + frame_idx）
        self.assertEqual(rows[1 + 2][4], "1")
        self.assertEqual(rows[1 + 2][5], "1")
        # idx6（除去された短区間）は in_event=0
        self.assertEqual(rows[1 + 6][4], "0")

    def test_idempotent(self):
        d = tempfile.mkdtemp()
        p = _perframe_csv(d, _PROB_HEADER, _PROB_ROWS)
        o1 = Path(d) / "o1"
        o2 = Path(d) / "o2"
        convert(in_path=p, outdir=str(o1), fps=2.0, min_duration=1.0)
        convert(in_path=p, outdir=str(o2), fps=2.0, min_duration=1.0)
        for fname in ("pred_bleed_model.jsonl", "pred_bleed_model.srt",
                      "pred_bleed_model.csv"):
            self.assertEqual(
                (o1 / fname).read_text(encoding="utf-8"),
                (o2 / fname).read_text(encoding="utf-8"),
            )


class TestConvertInterval(unittest.TestCase):
    def _run(self):
        d = tempfile.mkdtemp()
        p = _json_fixture(d, _IAE_EVENTS)
        outdir = Path(d) / "out"
        result = convert(in_path=p, outdir=str(outdir), fps=25.0)
        return outdir, result

    def test_three_files(self):
        _, result = self._run()
        for key in ("jsonl", "srt", "csv"):
            self.assertTrue(Path(result[key]).exists())
        self.assertEqual(result["events"], 2)

    def test_srt_severity_tag(self):
        _, result = self._run()
        content = Path(result["srt"]).read_text(encoding="utf-8")
        self.assertIn("[bleed] bleeding(sev=3)", content)
        self.assertIn("[bleed] bleeding(sev=5)", content)

    def test_interval_csv_columns(self):
        _, result = self._run()
        rows = Path(result["csv"]).read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(
            rows[0],
            "event_id,start_sec,start_srt,end_sec,end_srt,"
            "duration_sec,severity,confidence",
        )
        self.assertEqual(len(rows) - 1, 2)


class TestRoundTrip(unittest.TestCase):
    """JSONL → SRT → JSONL（srt_to_jsonl）の情報保持"""

    def test_roundtrip_preserves_payload(self):
        d = tempfile.mkdtemp()
        p = _json_fixture(d, _IAE_EVENTS)
        outdir = Path(d) / "out"
        result = convert(in_path=p, outdir=str(outdir), fps=25.0)

        restored_path = Path(d) / "restored.jsonl"
        srt_to_jsonl(result["srt"], str(restored_path))
        restored = [
            json.loads(l) for l in
            restored_path.read_text(encoding="utf-8").strip().splitlines()
        ]
        original = [
            json.loads(l) for l in
            Path(result["jsonl"]).read_text(encoding="utf-8").strip().splitlines()
        ]

        self.assertEqual(len(restored), len(original))
        for orig, rest in zip(original, restored):
            self.assertEqual(rest["type"], orig["type"])
            self.assertEqual(rest["severity"], orig["severity"])
            self.assertEqual(rest["source"], orig["source"])
            self.assertAlmostEqual(rest["start_sec"], orig["start_sec"], places=2)
            self.assertAlmostEqual(rest["end_sec"], orig["end_sec"], places=2)
        # region/point も保持される（2件目）
        self.assertEqual(restored[1]["region"], original[1]["region"])
        self.assertEqual(restored[1]["point"], original[1]["point"])


class TestMergeIntegration(unittest.TestCase):
    """既存スタイルの _bleed.srt と merge_srt で統合できること"""

    def test_merge_with_existing_bleed_srt(self):
        d = tempfile.mkdtemp()
        # 変換器の出力（区間 10s, 20s）
        p = _json_fixture(d, _IAE_EVENTS)
        outdir = Path(d) / "out"
        result = convert(in_path=p, outdir=str(outdir), fps=25.0)

        # 既存 redlog 風の _bleed.srt（1行構造、5秒に出血候補）
        existing = (Path(d) / "case_bleed.srt")
        existing.write_text(
            "1\n00:00:05,000 --> 00:00:07,000\n"
            "[bleed] delta_over_threshold\n",
            encoding="utf-8",
        )

        merged = Path(d) / "merged.srt"
        n = merge(str(merged), [str(existing), result["srt"]])
        # 既存1 + 変換器2 = 3エントリ
        self.assertEqual(n, 3)

        entries = read_srt(str(merged))
        self.assertEqual(len(entries), 3)
        # 開始時刻でソート済み（5s, 10s, 20s）
        starts = [e.start for e in entries]
        self.assertEqual(starts, sorted(starts))
        self.assertAlmostEqual(entries[0].start, 5.0)
        self.assertAlmostEqual(entries[1].start, 10.0)


class TestCLI(unittest.TestCase):
    def test_cli_perframe(self):
        d = tempfile.mkdtemp()
        p = _perframe_csv(d, _PROB_HEADER, _PROB_ROWS)
        outdir = Path(d) / "out"
        rc = main([
            "--in", p, "--fps", "2", "--outdir", str(outdir),
            "--thr-on", "0.5", "--thr-off", "0.3", "--min-duration", "1.0",
        ])
        self.assertEqual(rc, 0)
        stem = "pred_bleed_model"
        for ext in ("jsonl", "srt", "csv"):
            self.assertTrue((outdir / f"{stem}.{ext}").exists())

    def test_cli_interval_json(self):
        d = tempfile.mkdtemp()
        p = _json_fixture(d, _IAE_EVENTS)
        outdir = Path(d) / "out"
        rc = main(["--in", p, "--fps", "25", "--outdir", str(outdir)])
        self.assertEqual(rc, 0)
        self.assertTrue((outdir / "iae_bleed_model.jsonl").exists())


class TestBaseAnalyzerAdapter(unittest.TestCase):
    def test_make_analyzer(self):
        from src.red.bleed_model_to_outputs import make_analyzer

        d = tempfile.mkdtemp()
        p = _json_fixture(d, _IAE_EVENTS)
        analyzer = make_analyzer()
        result = analyzer.analyze(video_path=None, in_path=p, fps=25.0)
        self.assertEqual(result.analyzer_type, "bleeding")
        self.assertEqual(len(result.results), 2)
        self.assertEqual(result.results[0]["type"], EVENT_TYPE)


class TestBuildEvents(unittest.TestCase):
    def test_build_events_perframe(self):
        with tempfile.TemporaryDirectory() as d:
            p = _perframe_csv(d, _PROB_HEADER, _PROB_ROWS)
            res = build_events(p, fps=2.0, thr_on=0.5, thr_off=0.3,
                               min_duration=1.0)
            self.assertEqual(res.kind, "perframe_csv")
            self.assertEqual(len(res.events), 1)
            self.assertEqual(len(res.frames), 10)

    def test_build_events_interval(self):
        with tempfile.TemporaryDirectory() as d:
            p = _json_fixture(d, _IAE_EVENTS)
            res = build_events(p, fps=25.0)
            self.assertEqual(res.kind, "interval_json")
            self.assertEqual(len(res.events), 2)


class TestBoundaryConditions(unittest.TestCase):
    """境界条件: 空CSV / fps=0 / thr_on<=thr_off / 空イベント / 空JSON。"""

    def test_empty_perframe_csv(self):
        # ヘッダのみ（データ行なし）→ フレームもイベントも空
        with tempfile.TemporaryDirectory() as d:
            p = _perframe_csv(d, "frame_idx,bleed_prob", [])
            res = build_events(p, fps=2.0)
            self.assertEqual(res.kind, "perframe_csv")
            self.assertEqual(res.frames, [])
            self.assertEqual(res.events, [])
            self.assertEqual(res.frame_event_ids, [])

    def test_empty_perframe_csv_convert_writes_files(self):
        # 空入力でも3ファイルを生成し、JSONL/SRT は空、CSV はヘッダのみ
        with tempfile.TemporaryDirectory() as d:
            p = _perframe_csv(d, "frame_idx,bleed_prob", [])
            outdir = Path(d) / "out"
            res = convert(in_path=p, outdir=str(outdir), fps=2.0)
            self.assertEqual(res["events"], 0)
            self.assertEqual(Path(res["jsonl"]).read_text(encoding="utf-8"), "")
            self.assertEqual(Path(res["srt"]).read_text(encoding="utf-8"), "")
            csv_rows = Path(res["csv"]).read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(csv_rows), 1)  # ヘッダのみ

    def test_empty_interval_json(self):
        with tempfile.TemporaryDirectory() as d:
            p = _json_fixture(d, [])
            res = build_events(p, fps=25.0)
            self.assertEqual(res.kind, "interval_json")
            self.assertEqual(res.events, [])

    def test_fps_zero_without_timestamps_raises(self):
        # 時刻列が無く fps=0 → frames_to_seconds が ValueError
        with tempfile.TemporaryDirectory() as d:
            p = _perframe_csv(d, "frame_idx,bleed_prob", [(0, 0.9), (1, 0.9)])
            with self.assertRaises(ValueError):
                build_events(p, fps=0.0)

    def test_fps_zero_with_timestamps_ok(self):
        # 時刻列があれば fps は使われず、fps=0 でも成功する
        with tempfile.TemporaryDirectory() as d:
            p = _perframe_csv(d, "frame_idx,timestamp_sec,bleed_prob",
                              [(0, 0.0, 0.9), (1, 2.0, 0.9)])
            res = build_events(p, fps=0.0, thr_on=0.5, thr_off=0.3,
                               min_duration=0.0)
            self.assertEqual(len(res.events), 1)

    def test_thr_on_le_thr_off_deterministic(self):
        # 境界: thr_on <= thr_off でも例外なく決定的・冪等に変換できる
        with tempfile.TemporaryDirectory() as d:
            p = _perframe_csv(d, "frame_idx,timestamp_sec,bleed_prob",
                              [(0, 0.0, 0.4), (1, 1.0, 0.6), (2, 2.0, 0.4)])
            r1 = build_events(p, fps=1.0, thr_on=0.3, thr_off=0.5,
                              min_duration=0.0)
            r2 = build_events(p, fps=1.0, thr_on=0.3, thr_off=0.5,
                              min_duration=0.0)
            self.assertEqual(
                [e["start_sec"] for e in r1.events],
                [e["start_sec"] for e in r2.events],
            )

    def test_all_below_thr_on_yields_no_events(self):
        with tempfile.TemporaryDirectory() as d:
            p = _perframe_csv(d, "frame_idx,timestamp_sec,bleed_prob",
                              [(0, 0.0, 0.1), (1, 1.0, 0.2), (2, 2.0, 0.1)])
            res = build_events(p, fps=1.0, thr_on=0.5, thr_off=0.3,
                               min_duration=0.0)
            self.assertEqual(res.events, [])
            # フレームは保持されるが、どれもイベントに属さない
            self.assertEqual(len(res.frames), 3)
            self.assertTrue(all(i is None for i in res.frame_event_ids))


class TestPickGuard(unittest.TestCase):
    """_pick / region・point 抽出が非dict入力でも TypeError を出さない。"""

    def test_pick_non_dict_returns_none(self):
        self.assertIsNone(_pick(None, ("x",)))
        self.assertIsNone(_pick("not a dict", ("x",)))
        self.assertIsNone(_pick(123, ("x",)))

    def test_region_point_from_non_dict(self):
        self.assertIsNone(_region_from(None))
        self.assertIsNone(_point_from(None))


class TestCLIValidation(unittest.TestCase):
    """CLI 数値引数のバリデーション（不正値は SystemExit）。"""

    def _args(self, d, **over):
        p = _json_fixture(d, _IAE_EVENTS)
        outdir = str(Path(d) / "out")
        a = ["--in", p, "--outdir", outdir]
        for k, v in over.items():
            a += [f"--{k}", str(v)]
        return a

    def test_fps_zero_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(SystemExit):
                main(self._args(d, fps=0))

    def test_negative_fps_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(SystemExit):
                main(self._args(d, **{"fps": -5}))

    def test_thr_on_le_thr_off_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(SystemExit):
                main(self._args(d, **{"thr-on": 0.3, "thr-off": 0.5}))

    def test_negative_min_duration_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(SystemExit):
                main(self._args(d, **{"min-duration": -1}))

    def test_valid_args_accepted(self):
        with tempfile.TemporaryDirectory() as d:
            rc = main(self._args(d, fps=25, **{"thr-on": 0.6, "thr-off": 0.3}))
            self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
