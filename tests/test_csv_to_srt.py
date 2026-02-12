"""
csv_to_srt.py のユニットテスト

CSVの量的データからSRT字幕への変換ロジックを検証する。
"""

import tempfile
import unittest
from pathlib import Path

from src.tools.csv_to_srt import convert, csv_to_srt


class TestCsvToSrt(unittest.TestCase):
    """CSV → SRT 変換テスト"""

    def _create_csv(self, tmpdir: str) -> str:
        """テスト用CSVを生成するヘルパー"""
        csv_path = Path(tmpdir) / "test_redlog.csv"
        csv_path.write_text(
            "t_sec,t_srt,red_ratio,delta,smooth_delta,reader\n"
            "0.000,00:00:00.000,0.012345,0.000000,0.000000,pyav\n"
            "0.200,00:00:00.200,0.015678,0.003333,0.001667,pyav\n"
            "0.400,00:00:00.400,0.020123,0.004445,0.002889,pyav\n",
            encoding="utf-8",
        )
        return str(csv_path)

    def test_basic_conversion(self):
        """基本的なCSV→SRT変換"""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = self._create_csv(tmpdir)
            srt = csv_to_srt(csv_path)
            # デフォルト列 red_ratio + smooth_delta
            self.assertIn("red=", srt)
            self.assertIn("Δs=", srt)
            self.assertIn("1\n", srt)
            self.assertIn("2\n", srt)
            self.assertIn("3\n", srt)
            self.assertIn("-->", srt)

    def test_custom_columns(self):
        """カスタム列の指定"""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = self._create_csv(tmpdir)
            srt = csv_to_srt(csv_path, columns=["delta"])
            self.assertIn("Δ=", srt)
            self.assertNotIn("red=", srt)
            self.assertNotIn("Δs=", srt)

    def test_all_columns(self):
        """全列の表示"""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = self._create_csv(tmpdir)
            srt = csv_to_srt(csv_path, columns=["red_ratio", "delta", "smooth_delta"])
            self.assertIn("red=", srt)
            self.assertIn("Δ=", srt)
            self.assertIn("Δs=", srt)

    def test_time_continuity(self):
        """字幕の時刻が連続すること"""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = self._create_csv(tmpdir)
            srt = csv_to_srt(csv_path)
            # 1番目: 00:00:00,000 --> 00:00:00,200
            self.assertIn("00:00:00,000 --> 00:00:00,200", srt)
            # 2番目: 00:00:00,200 --> 00:00:00,400
            self.assertIn("00:00:00,200 --> 00:00:00,400", srt)

    def test_convert_to_file(self):
        """ファイル出力の動作確認"""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = self._create_csv(tmpdir)
            srt_path = Path(tmpdir) / "output.srt"
            count = convert(csv_path, str(srt_path))
            self.assertEqual(count, 3)
            self.assertTrue(srt_path.exists())
            content = srt_path.read_text(encoding="utf-8")
            self.assertIn("red=", content)

    def test_empty_csv(self):
        """空のCSV（ヘッダのみ）"""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "empty.csv"
            csv_path.write_text(
                "t_sec,t_srt,red_ratio,delta,smooth_delta,reader\n",
                encoding="utf-8",
            )
            srt = csv_to_srt(str(csv_path))
            self.assertEqual(srt, "")


class TestRedlogSplit(unittest.TestCase):
    """redlog.py の分割関数テスト"""

    def test_read_redlog_csv(self):
        """read_redlog_csv がCSVを正しく読み込むこと"""
        from src.red.redlog import read_redlog_csv

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test_redlog.csv"
            csv_path.write_text(
                "t_sec,t_srt,red_ratio,delta,smooth_delta,reader\n"
                "0.000,00:00:00.000,0.012000,0.000000,0.000000,pyav\n"
                "0.200,00:00:00.200,0.015000,0.003000,0.001500,pyav\n"
                "0.400,00:00:00.400,0.020000,0.005000,0.003000,pyav\n",
                encoding="utf-8",
            )
            data = read_redlog_csv(str(csv_path))
            self.assertEqual(len(data["times"]), 3)
            self.assertAlmostEqual(data["times"][0], 0.0)
            self.assertAlmostEqual(data["times"][1], 0.2)
            self.assertAlmostEqual(data["ratios"][1], 0.015)
            self.assertAlmostEqual(data["deltas"][2], 0.005)
            self.assertAlmostEqual(data["smooth_deltas"][2], 0.003)
            self.assertEqual(data["reader"], "pyav")
            self.assertAlmostEqual(data["fps"], 5.0, places=1)

    def test_annotate_bleed_from_csv(self):
        """annotate_bleed がCSVからイベントを正しく抽出すること"""
        from src.red.redlog import annotate_bleed, format_srt_time
        import json

        with tempfile.TemporaryDirectory() as tmpdir:
            # 確実にイベントが発生するCSVデータを作成（thr=0.03を超過するdeltaが3秒以上継続）
            csv_path = Path(tmpdir) / "case_redlog.csv"
            lines = ["t_sec,t_srt,red_ratio,delta,smooth_delta,reader\n"]
            fps = 5.0
            # 20秒分のデータ（5fps = 100サンプル）
            for i in range(100):
                t = i / fps
                # 5〜10秒の区間でsmooth_deltaを0.05に
                if 5.0 <= t <= 10.0:
                    sd = 0.05
                else:
                    sd = 0.0
                lines.append(
                    f"{t:.3f},{format_srt_time(t)},{0.1:.6f},{sd:.6f},{sd:.6f},pyav\n"
                )
            csv_path.write_text("".join(lines), encoding="utf-8")

            result = annotate_bleed(
                csv_path=str(csv_path),
                outdir=tmpdir,
                thr=0.03,
                k_s=3.0,
            )

            self.assertGreater(result["events"], 0)
            self.assertTrue(Path(result["jsonl"]).exists())
            self.assertTrue(Path(result["srt"]).exists())

            # JSONLの中身を確認
            jsonl_content = Path(result["jsonl"]).read_text(encoding="utf-8").strip()
            ev = json.loads(jsonl_content.split("\n")[0])
            self.assertEqual(ev["type"], "bleed_candidate")


if __name__ == "__main__":
    unittest.main()
