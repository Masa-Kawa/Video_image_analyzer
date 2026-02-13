"""
plot_redlog.py のユニットテスト

CSVからPNGグラフが正しく生成されることを検証する。
"""

import tempfile
import unittest
from pathlib import Path

from src.tools.plot_redlog import plot_redlog, plot_bleedlog, plot_auto


class TestPlotRedlog(unittest.TestCase):
    """赤色率ログCSV → PNGグラフのテスト"""

    def _create_csv(self, tmpdir: str, rows: int = 50) -> str:
        """テスト用CSVを生成するヘルパー"""
        csv_path = Path(tmpdir) / "test_redlog.csv"
        lines = ["t_sec,t_srt,red_ratio,delta,smooth_delta,reader\n"]
        for i in range(rows):
            t = i * 0.2
            ratio = 0.01 + 0.005 * (i % 10)
            delta = 0.005 if i % 10 < 5 else -0.003
            sd = delta * 0.5
            h = int(t // 3600)
            m = int((t % 3600) // 60)
            s = int(t % 60)
            ms = int((t % 1) * 1000)
            srt_time = f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
            lines.append(
                f"{t:.3f},{srt_time},{ratio:.6f},{delta:.6f},{sd:.6f},pyav\n"
            )
        csv_path.write_text("".join(lines), encoding="utf-8")
        return str(csv_path)

    def _create_bleedlog_csv(self, tmpdir: str, rows: int = 50) -> str:
        """テスト用bleedlog CSVを生成するヘルパー"""
        csv_path = Path(tmpdir) / "test_bleedlog.csv"
        lines = [
            "t_sec,t_srt,red_ratio,newly_red_ratio,bg_stability,"
            "red_expansion,smooth_expansion,reader\n"
        ]
        for i in range(rows):
            t = i * 0.2
            rr = 0.5 + 0.01 * (i % 10)
            nr = 0.01 if i % 5 == 0 else 0.001
            bs = 0.95 - 0.01 * (i % 5)
            re = nr * bs
            se = re * 0.8
            h = int(t // 3600)
            m = int((t % 3600) // 60)
            s = int(t % 60)
            ms = int((t % 1) * 1000)
            srt_time = f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
            lines.append(
                f"{t:.3f},{srt_time},{rr:.6f},{nr:.6f},{bs:.6f},"
                f"{re:.6f},{se:.6f},pyav\n"
            )
        csv_path.write_text("".join(lines), encoding="utf-8")
        return str(csv_path)

    def test_basic_plot(self):
        """基本的なPNG出力"""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = self._create_csv(tmpdir)
            out_png = str(Path(tmpdir) / "output.png")

            result = plot_redlog(csv_path, out_png)
            self.assertEqual(result, out_png)
            self.assertTrue(Path(out_png).exists())
            # PNGファイルが空でないこと
            self.assertGreater(Path(out_png).stat().st_size, 0)

    def test_with_threshold(self):
        """閾値ライン付きのグラフ出力"""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = self._create_csv(tmpdir)
            out_png = str(Path(tmpdir) / "output_thr.png")

            result = plot_redlog(csv_path, out_png, thr=0.03)
            self.assertTrue(Path(out_png).exists())
            self.assertGreater(Path(out_png).stat().st_size, 0)

    def test_with_custom_title(self):
        """カスタムタイトル指定"""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = self._create_csv(tmpdir)
            out_png = str(Path(tmpdir) / "output_title.png")

            result = plot_redlog(csv_path, out_png, title="テスト:カスタムタイトル")
            self.assertTrue(Path(out_png).exists())

    def test_empty_csv(self):
        """空のCSV（ヘッダのみ）でもエラーにならないこと"""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "empty_redlog.csv"
            csv_path.write_text(
                "t_sec,t_srt,red_ratio,delta,smooth_delta,reader\n",
                encoding="utf-8",
            )
            out_png = str(Path(tmpdir) / "output_empty.png")

            result = plot_redlog(str(csv_path), out_png)
            self.assertTrue(Path(out_png).exists())

    def test_output_directory_creation(self):
        """出力ディレクトリが存在しない場合に自動作成されること"""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = self._create_csv(tmpdir)
            out_png = str(Path(tmpdir) / "subdir" / "nested" / "output.png")

            result = plot_redlog(csv_path, out_png)
            self.assertTrue(Path(out_png).exists())

    def test_bleedlog_plot(self):
        """bleedlog CSVからの3パネルPNG出力"""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = self._create_bleedlog_csv(tmpdir)
            out_png = str(Path(tmpdir) / "output_bleed.png")

            result = plot_bleedlog(csv_path, out_png, thr=0.005)
            self.assertTrue(Path(out_png).exists())
            self.assertGreater(Path(out_png).stat().st_size, 0)

    def test_auto_detect_redlog(self):
        """plot_autoがredlog CSVを正しく判定すること"""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = self._create_csv(tmpdir)
            out_png = str(Path(tmpdir) / "auto_redlog.png")

            result = plot_auto(csv_path, out_png)
            self.assertTrue(Path(out_png).exists())

    def test_auto_detect_bleedlog(self):
        """plot_autoがbleedlog CSVを正しく判定すること"""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = self._create_bleedlog_csv(tmpdir)
            out_png = str(Path(tmpdir) / "auto_bleedlog.png")

            result = plot_auto(csv_path, out_png, thr=0.005)
            self.assertTrue(Path(out_png).exists())


if __name__ == "__main__":
    unittest.main()

