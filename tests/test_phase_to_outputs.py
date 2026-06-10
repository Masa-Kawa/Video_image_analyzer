"""
phase_to_outputs.py のユニットテスト

合成入力（小さな Cholec80 形式 txt と per-frame CSV）を使い、
入力パース・区間化・最小継続長・スムージング・時刻整合・3出力フォーマット、
および JSONL→SRT→JSONL の往復（情報保持）を検証する。
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from src.phase.phase_to_outputs import (
    PhaseMap,
    FrameRec,
    Segment,
    build_events,
    build_segments,
    convert,
    detect_format,
    enforce_min_duration,
    main,
    parse_cholec80,
    parse_pred_csv,
    resolve_times,
    smooth_frames,
    segments_to_events,
)
from src.tools.srt_to_jsonl import convert as srt_to_jsonl


CHOLEC80_PHASES = [
    "Preparation",
    "CalotTriangleDissection",
    "ClippingCutting",
    "GallbladderDissection",
    "GallbladderPackaging",
    "CleaningCoagulation",
    "GallbladderRetraction",
]


def _write(path: Path, text: str) -> str:
    path.write_text(text, encoding="utf-8")
    return str(path)


def _cholec80_fixture(tmpdir: str, rows) -> str:
    """rows = [(frame_idx, phase_name), ...] から Cholec80 txt を生成。"""
    lines = ["Frame\tPhase"]
    for frame_idx, phase in rows:
        lines.append(f"{frame_idx}\t{phase}")
    return _write(Path(tmpdir) / "video01-phase.txt", "\n".join(lines) + "\n")


def _csv_fixture(tmpdir: str, header: str, rows) -> str:
    lines = [header] + [",".join(str(c) for c in r) for r in rows]
    return _write(Path(tmpdir) / "pred.csv", "\n".join(lines) + "\n")


class TestPhaseMap(unittest.TestCase):
    """同梱フェーズマップの検証"""

    def test_default_map_cholec80(self):
        pm = PhaseMap.load(None)
        self.assertEqual(pm.fps, 25.0)
        self.assertEqual(pm.resolve_name(0), "Preparation")
        self.assertEqual(pm.resolve_name(6), "GallbladderRetraction")
        self.assertEqual(pm.resolve_id("ClippingCutting"), 2)

    def test_case_insensitive_id(self):
        pm = PhaseMap.load(None)
        self.assertEqual(pm.resolve_id("preparation"), 0)

    def test_unknown_id(self):
        pm = PhaseMap.load(None)
        self.assertEqual(pm.resolve_name(99), "Phase_99")
        self.assertIsNone(pm.resolve_id("NoSuchPhase"))


class TestFormatDetection(unittest.TestCase):
    """入力形式の自動判定"""

    def test_detect_cholec80(self):
        with tempfile.TemporaryDirectory() as d:
            path = _cholec80_fixture(d, [(0, "Preparation")])
            self.assertEqual(detect_format(path), "cholec80")

    def test_detect_csv(self):
        with tempfile.TemporaryDirectory() as d:
            path = _csv_fixture(d, "frame_idx,phase_name", [(0, "Preparation")])
            self.assertEqual(detect_format(path), "csv")


class TestParsers(unittest.TestCase):
    """入力パーサの検証"""

    def test_parse_cholec80_skips_header(self):
        with tempfile.TemporaryDirectory() as d:
            path = _cholec80_fixture(d, [
                (0, "Preparation"), (25, "Preparation"),
                (50, "CalotTriangleDissection"),
            ])
            pm = PhaseMap.load(None)
            recs = parse_cholec80(path, pm)
            self.assertEqual(len(recs), 3)
            self.assertEqual(recs[0].frame_idx, 0)
            self.assertEqual(recs[0].phase_name, "Preparation")
            self.assertEqual(recs[0].phase_id, 0)
            self.assertEqual(recs[2].phase_id, 1)
            # 時刻は未解決
            self.assertIsNone(recs[0].t_sec)

    def test_parse_csv_phase_name_and_confidence(self):
        with tempfile.TemporaryDirectory() as d:
            path = _csv_fixture(
                d, "frame_idx,timestamp_sec,phase_name,confidence",
                [(0, 0.0, "Preparation", 0.9), (1, 0.5, "Preparation", 0.8)],
            )
            pm = PhaseMap.load(None)
            recs = parse_pred_csv(path, pm)
            self.assertEqual(len(recs), 2)
            self.assertEqual(recs[0].phase_id, 0)
            self.assertAlmostEqual(recs[0].t_sec, 0.0)
            self.assertAlmostEqual(recs[1].confidence, 0.8)

    def test_parse_csv_phase_id_only(self):
        with tempfile.TemporaryDirectory() as d:
            path = _csv_fixture(
                d, "frame_idx,phase_id", [(0, 2), (1, 2)],
            )
            pm = PhaseMap.load(None)
            recs = parse_pred_csv(path, pm)
            self.assertEqual(recs[0].phase_name, "ClippingCutting")
            self.assertEqual(recs[0].phase_id, 2)

    def test_parse_csv_timestamp_without_frame_idx(self):
        # frame_idx 列が無く t_sec のみ（リポジトリの cholecphaselog 形式）
        with tempfile.TemporaryDirectory() as d:
            path = _csv_fixture(
                d, "t_sec,phase_id,phase_name,confidence",
                [(0.0, 6, "GallbladderRetraction", 0.30),
                 (1.001, 6, "GallbladderRetraction", 0.35),
                 (2.002, 0, "Preparation", 0.48)],
            )
            pm = PhaseMap.load(None)
            recs = parse_pred_csv(path, pm)
            self.assertEqual(len(recs), 3)
            # frame_idx は行順から採番
            self.assertEqual([r.frame_idx for r in recs], [0, 1, 2])
            self.assertAlmostEqual(recs[1].t_sec, 1.001)
            self.assertEqual(recs[2].phase_name, "Preparation")

    def test_parse_csv_missing_columns_raises(self):
        with tempfile.TemporaryDirectory() as d:
            path = _csv_fixture(d, "frame_idx,foo", [(0, 1)])
            pm = PhaseMap.load(None)
            with self.assertRaises(ValueError):
                parse_pred_csv(path, pm)


class TestTimeResolution(unittest.TestCase):
    """時刻解決（fps換算）"""

    def test_fps_resolution(self):
        recs = [FrameRec(i, None, "Preparation", 0) for i in range(4)]
        resolve_times(recs, video_path=None, fps=2.0)
        self.assertAlmostEqual(recs[0].t_sec, 0.0)
        self.assertAlmostEqual(recs[1].t_sec, 0.5)
        self.assertAlmostEqual(recs[3].t_sec, 1.5)

    def test_existing_timestamp_preserved(self):
        recs = [FrameRec(0, 3.14, "Preparation", 0)]
        resolve_times(recs, video_path=None, fps=25.0)
        self.assertAlmostEqual(recs[0].t_sec, 3.14)


class TestSegmentation(unittest.TestCase):
    """区間化・境界の連続性"""

    def test_coalesce_consecutive(self):
        recs = [
            FrameRec(0, 0.0, "Preparation", 0),
            FrameRec(1, 1.0, "Preparation", 0),
            FrameRec(2, 2.0, "CalotTriangleDissection", 1),
            FrameRec(3, 3.0, "CalotTriangleDissection", 1),
        ]
        segs = build_segments(recs, period=1.0)
        self.assertEqual(len(segs), 2)
        # 連続している（seg0.end == seg1.start）
        self.assertAlmostEqual(segs[0].end, segs[1].start)
        self.assertAlmostEqual(segs[0].start, 0.0)
        self.assertAlmostEqual(segs[0].end, 2.0)
        # 最終区間は最後の時刻 + period
        self.assertAlmostEqual(segs[1].end, 4.0)

    def test_mean_confidence(self):
        recs = [
            FrameRec(0, 0.0, "Preparation", 0, 0.6),
            FrameRec(1, 1.0, "Preparation", 0, 0.8),
        ]
        segs = build_segments(recs, period=1.0)
        self.assertAlmostEqual(segs[0].mean_confidence, 0.7)


class TestSmoothing(unittest.TestCase):
    """多数決スムージング"""

    def test_flicker_removed_by_majority(self):
        recs = [
            FrameRec(0, 0.0, "Preparation", 0),
            FrameRec(1, 1.0, "Preparation", 0),
            FrameRec(2, 2.0, "CalotTriangleDissection", 1),  # フリッカ
            FrameRec(3, 3.0, "Preparation", 0),
            FrameRec(4, 4.0, "Preparation", 0),
        ]
        pm = PhaseMap.load(None)
        smoothed = smooth_frames(recs, window=3, phase_map=pm)
        self.assertEqual(smoothed[2].phase_name, "Preparation")
        self.assertEqual(smoothed[2].phase_id, 0)

    def test_window_zero_noop(self):
        recs = [FrameRec(0, 0.0, "Preparation", 0)]
        pm = PhaseMap.load(None)
        self.assertIs(smooth_frames(recs, 0, pm), recs)


class TestMinDuration(unittest.TestCase):
    """最小継続長フィルタ（フリッカ除去）"""

    def test_short_segment_absorbed(self):
        # 1fps: Prep×5, Calot×1（1秒）, Prep×5
        recs = []
        for i in range(5):
            recs.append(FrameRec(i, float(i), "Preparation", 0))
        recs.append(FrameRec(5, 5.0, "CalotTriangleDissection", 1))
        for k, i in enumerate(range(6, 11)):
            recs.append(FrameRec(i, float(i), "Preparation", 0))

        segs = build_segments(recs, period=1.0)
        self.assertEqual(len(segs), 3)

        filtered = enforce_min_duration(segs, min_duration=2.0)
        # 短いCalot区間が吸収され、Prep1区間に統合される
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].phase_name, "Preparation")
        self.assertAlmostEqual(filtered[0].start, 0.0)
        self.assertAlmostEqual(filtered[0].end, 11.0)

    def test_idempotent(self):
        recs = []
        for i in range(5):
            recs.append(FrameRec(i, float(i), "Preparation", 0))
        recs.append(FrameRec(5, 5.0, "CalotTriangleDissection", 1))
        for i in range(6, 11):
            recs.append(FrameRec(i, float(i), "GallbladderRetraction", 6))

        segs = build_segments(recs, period=1.0)
        once = enforce_min_duration(segs, min_duration=2.0)
        twice = enforce_min_duration(once, min_duration=2.0)
        self.assertEqual(
            [(s.phase_name, s.start, s.end) for s in once],
            [(s.phase_name, s.start, s.end) for s in twice],
        )


class TestBuildEventsAndSchema(unittest.TestCase):
    """build_events と JSONL正本スキーマ"""

    def test_event_schema_keys(self):
        with tempfile.TemporaryDirectory() as d:
            path = _csv_fixture(
                d, "frame_idx,phase_name,confidence",
                [(i, "Preparation", 0.9) for i in range(6)],
            )
            pm = PhaseMap.load(None)
            events, recs = build_events(
                path, pm, fps=1.0, min_duration=2.0,
            )
            self.assertEqual(len(events), 1)
            ev = events[0]
            for key in ("type", "phase_id", "phase_name", "label", "source",
                        "start_sec", "end_sec", "start_srt", "end_srt",
                        "duration_sec", "confidence"):
                self.assertIn(key, ev)
            self.assertEqual(ev["type"], "surgical_phase")
            self.assertEqual(ev["label"], ev["phase_name"])
            self.assertEqual(ev["source"], "phase_converter")
            self.assertAlmostEqual(ev["confidence"], 0.9)

    def test_time_consistency(self):
        # fps=2 → frame_idx/2 が時刻
        with tempfile.TemporaryDirectory() as d:
            rows = [(i, "Preparation") for i in range(6)] + \
                   [(i, "CalotTriangleDissection") for i in range(6, 12)]
            path = _cholec80_fixture(d, rows)
            pm = PhaseMap.load(None)
            events, _ = build_events(path, pm, fps=2.0, min_duration=1.0)
            self.assertEqual(len(events), 2)
            self.assertAlmostEqual(events[0]["start_sec"], 0.0)
            # 区間境界は frame6/2 = 3.0
            self.assertAlmostEqual(events[0]["end_sec"], 3.0)
            self.assertAlmostEqual(events[1]["start_sec"], 3.0)


class TestConvertOutputs(unittest.TestCase):
    """convert() の3出力ファイル生成"""

    def _run(self, level="segment"):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        rows = [(i, "Preparation") for i in range(6)] + \
               [(i, "CalotTriangleDissection") for i in range(6, 12)]
        path = _cholec80_fixture(d, rows)
        outdir = Path(d) / "out"
        result = convert(
            in_path=path, outdir=str(outdir), fps=2.0,
            min_duration=1.0, level=level,
        )
        return outdir, result

    def test_three_files_generated(self):
        outdir, result = self._run()
        self.assertTrue(Path(result["jsonl"]).exists())
        self.assertTrue(Path(result["srt"]).exists())
        self.assertTrue(Path(result["csv"]).exists())
        # 命名規約
        self.assertTrue(result["jsonl"].endswith("video01-phase_phase.jsonl"))
        self.assertTrue(result["srt"].endswith("video01-phase_phase.srt"))
        self.assertTrue(result["csv"].endswith("video01-phase_phase.csv"))

    def test_jsonl_valid_lines(self):
        _, result = self._run()
        lines = Path(result["jsonl"]).read_text(encoding="utf-8").strip().split("\n")
        self.assertEqual(len(lines), 2)
        for line in lines:
            obj = json.loads(line)
            self.assertEqual(obj["type"], "surgical_phase")

    def test_srt_two_line_structure(self):
        _, result = self._run()
        content = Path(result["srt"]).read_text(encoding="utf-8")
        blocks = [b for b in content.strip().split("\n\n") if b.strip()]
        self.assertEqual(len(blocks), 2)
        first = blocks[0].split("\n")
        # index / time / tag / json の4行
        self.assertEqual(len(first), 4)
        self.assertIn("-->", first[1])
        self.assertTrue(first[2].startswith("[phase] "))
        meta = json.loads(first[3])
        self.assertEqual(meta["type"], "surgical_phase")
        # JSON行に時刻フィールドを含めない
        self.assertNotIn("start_sec", meta)
        self.assertNotIn("start_srt", meta)

    def test_segment_csv_header(self):
        _, result = self._run(level="segment")
        head = Path(result["csv"]).read_text(encoding="utf-8").splitlines()[0]
        self.assertIn("segment_id", head)
        self.assertIn("phase_name", head)

    def test_frame_level_csv(self):
        _, result = self._run(level="frame")
        head = Path(result["csv"]).read_text(encoding="utf-8").splitlines()[0]
        self.assertIn("frame_idx", head)
        # frameレベルは全フレーム分の行
        rows = Path(result["csv"]).read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(rows) - 1, 12)

    def test_both_level_creates_frames_csv(self):
        _, result = self._run(level="both")
        self.assertIn("csv_frames", result)
        self.assertTrue(Path(result["csv_frames"]).exists())

    def test_idempotent_outputs(self):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        rows = [(i, "Preparation") for i in range(6)] + \
               [(i, "CalotTriangleDissection") for i in range(6, 12)]
        path = _cholec80_fixture(d, rows)
        out1 = Path(d) / "o1"
        out2 = Path(d) / "o2"
        convert(in_path=path, outdir=str(out1), fps=2.0, min_duration=1.0)
        convert(in_path=path, outdir=str(out2), fps=2.0, min_duration=1.0)
        for fname in ("video01-phase_phase.jsonl", "video01-phase_phase.srt",
                      "video01-phase_phase.csv"):
            self.assertEqual(
                (out1 / fname).read_text(encoding="utf-8"),
                (out2 / fname).read_text(encoding="utf-8"),
            )


class TestRoundTrip(unittest.TestCase):
    """JSONL → SRT → JSONL（srt_to_jsonl）の情報保持"""

    def test_roundtrip_preserves_label_and_time(self):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        rows = [(i, "Preparation") for i in range(6)] + \
               [(i, "GallbladderRetraction") for i in range(6, 12)]
        # confidence 付き CSV を使い、confidence も往復確認する
        header = "frame_idx,phase_name,confidence"
        csv_rows = [(i, "Preparation", 0.91) for i in range(6)] + \
                   [(i, "GallbladderRetraction", 0.77) for i in range(6, 12)]
        path = _csv_fixture(d, header, csv_rows)

        outdir = Path(d) / "out"
        result = convert(in_path=path, outdir=str(outdir), fps=2.0,
                         min_duration=1.0)

        # SRT → JSONL で復元
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
            # type / label / phase_name / confidence が保持される
            self.assertEqual(rest["type"], orig["type"])
            self.assertEqual(rest["label"], orig["label"])
            self.assertEqual(rest["phase_name"], orig["phase_name"])
            self.assertAlmostEqual(rest["confidence"], orig["confidence"])
            # 時刻（ミリ秒精度）が保持される
            self.assertAlmostEqual(rest["start_sec"], orig["start_sec"], places=2)
            self.assertAlmostEqual(rest["end_sec"], orig["end_sec"], places=2)


class TestCLI(unittest.TestCase):
    """CLI（main）の検証"""

    def test_cli_generates_three_files(self):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        rows = [(i, "Preparation") for i in range(60)] + \
               [(i, "CalotTriangleDissection") for i in range(60, 120)]
        path = _cholec80_fixture(d, rows)
        outdir = Path(d) / "out"
        rc = main([
            "--in", path, "--fps", "25", "--outdir", str(outdir),
            "--min-duration", "2.0", "--level", "segment",
        ])
        self.assertEqual(rc, 0)
        stem = "video01-phase_phase"
        self.assertTrue((outdir / f"{stem}.jsonl").exists())
        self.assertTrue((outdir / f"{stem}.srt").exists())
        self.assertTrue((outdir / f"{stem}.csv").exists())


class TestBaseAnalyzerAdapter(unittest.TestCase):
    """BaseAnalyzer 互換アダプタ"""

    def test_make_analyzer_results(self):
        from src.phase.phase_to_outputs import make_analyzer

        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        rows = [(i, "Preparation") for i in range(6)] + \
               [(i, "CalotTriangleDissection") for i in range(6, 12)]
        path = _cholec80_fixture(d, rows)

        analyzer = make_analyzer()
        result = analyzer.analyze(
            video_path=None,  # 動画なし（fps換算で時刻解決）
            in_path=path, fps=2.0, min_duration=1.0,
        )
        self.assertEqual(result.analyzer_type, "surgical_phase")
        self.assertEqual(len(result.results), 2)
        self.assertEqual(result.results[0]["type"], "surgical_phase")


if __name__ == "__main__":
    unittest.main()
