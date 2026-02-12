"""
往復変換（ラウンドトリップ）テスト

JSONL → SRT → JSONL の往復変換で情報が保持されることを検証する。
"""

import json
import tempfile
import unittest
from pathlib import Path

from src.tools.jsonl_to_srt import convert as jsonl_to_srt
from src.tools.srt_to_jsonl import convert as srt_to_jsonl


class TestRoundTrip(unittest.TestCase):
    """往復変換テスト"""

    def test_bleed_roundtrip(self):
        """出血候補イベントの往復変換で情報が保持されること"""
        original_event = {
            "type": "bleed_candidate",
            "metric": "red_ratio",
            "thr": 0.03,
            "k_s": 3.0,
            "smooth_s": 5.0,
            "delta_max": 0.045678,
            "start_sec": 135.6,
            "end_sec": 138.8,
            "start_srt": "00:02:15,600",
            "end_srt": "00:02:18,800",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            # Step 1: オリジナルJSONL
            jsonl_path = Path(tmpdir) / "original.jsonl"
            with open(jsonl_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(original_event) + "\n")

            # Step 2: JSONL → SRT
            srt_path = Path(tmpdir) / "converted.srt"
            jsonl_to_srt(str(jsonl_path), str(srt_path))

            # Step 3: SRT → JSONL
            restored_path = Path(tmpdir) / "restored.jsonl"
            srt_to_jsonl(str(srt_path), str(restored_path))

            # Step 4: 復元結果の検証
            content = restored_path.read_text(encoding="utf-8").strip()
            restored_event = json.loads(content)

            # メタデータの保持
            self.assertEqual(restored_event["type"], original_event["type"])
            self.assertEqual(restored_event["metric"], original_event["metric"])
            self.assertAlmostEqual(
                restored_event["thr"], original_event["thr"], places=5
            )
            self.assertAlmostEqual(
                restored_event["k_s"], original_event["k_s"], places=5
            )
            self.assertAlmostEqual(
                restored_event["smooth_s"], original_event["smooth_s"], places=5
            )
            self.assertAlmostEqual(
                restored_event["delta_max"], original_event["delta_max"],
                places=5
            )

            # 時刻情報の保持（SRT時刻精度はミリ秒）
            self.assertAlmostEqual(
                restored_event["start_sec"], original_event["start_sec"],
                places=2
            )
            self.assertAlmostEqual(
                restored_event["end_sec"], original_event["end_sec"],
                places=2
            )

    def test_multiple_events_roundtrip(self):
        """複数イベントの往復変換で順序と情報が保持されること"""
        events = [
            {
                "type": "bleed_candidate",
                "metric": "red_ratio",
                "thr": 0.03,
                "delta_max": 0.05,
                "start_sec": 10.0,
                "end_sec": 15.0,
                "start_srt": "00:00:10,000",
                "end_srt": "00:00:15,000",
            },
            {
                "type": "bleed_candidate",
                "metric": "red_ratio",
                "thr": 0.03,
                "delta_max": 0.08,
                "start_sec": 30.0,
                "end_sec": 35.0,
                "start_srt": "00:00:30,000",
                "end_srt": "00:00:35,000",
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "original.jsonl"
            with open(jsonl_path, "w", encoding="utf-8") as f:
                for ev in events:
                    f.write(json.dumps(ev) + "\n")

            srt_path = Path(tmpdir) / "converted.srt"
            jsonl_to_srt(str(jsonl_path), str(srt_path))

            restored_path = Path(tmpdir) / "restored.jsonl"
            srt_to_jsonl(str(srt_path), str(restored_path))

            # 復元結果確認
            lines = restored_path.read_text(encoding="utf-8").strip().split("\n")
            self.assertEqual(len(lines), 2)

            for i, line in enumerate(lines):
                restored = json.loads(line)
                self.assertEqual(restored["type"], events[i]["type"])
                self.assertAlmostEqual(
                    restored["delta_max"], events[i]["delta_max"], places=5
                )

    def test_time_modification_preserved(self):
        """SRT上で時刻を修正した場合、修正がJSONLに反映されること"""
        original_event = {
            "type": "bleed_candidate",
            "metric": "red_ratio",
            "thr": 0.03,
            "delta_max": 0.05,
            "start_sec": 10.0,
            "end_sec": 15.0,
            "start_srt": "00:00:10,000",
            "end_srt": "00:00:15,000",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            # Step 1: JSONL → SRT
            jsonl_path = Path(tmpdir) / "original.jsonl"
            with open(jsonl_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(original_event) + "\n")

            srt_path = Path(tmpdir) / "converted.srt"
            jsonl_to_srt(str(jsonl_path), str(srt_path))

            # Step 2: SRTの時刻を人手修正（10秒→12秒に変更）
            srt_content = srt_path.read_text(encoding="utf-8")
            modified_content = srt_content.replace(
                "00:00:10,000", "00:00:12,000"
            )
            srt_path.write_text(modified_content, encoding="utf-8")

            # Step 3: 修正SRT → JSONL
            modified_path = Path(tmpdir) / "modified.jsonl"
            srt_to_jsonl(str(srt_path), str(modified_path))

            # Step 4: 修正が反映されていることを確認
            content = modified_path.read_text(encoding="utf-8").strip()
            modified_event = json.loads(content)
            self.assertAlmostEqual(modified_event["start_sec"], 12.0, places=1)
            self.assertEqual(modified_event["start_srt"], "00:00:12,000")
            # 終了時刻は変更なし
            self.assertAlmostEqual(modified_event["end_sec"], 15.0, places=1)


if __name__ == "__main__":
    unittest.main()
