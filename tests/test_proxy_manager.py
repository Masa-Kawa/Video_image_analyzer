"""
proxy_manager.py のユニットテスト

ProxyManagerのパス生成、存在チェックなどを検証する。
（実際のエンコードは行わない）
"""

import tempfile
import unittest
from pathlib import Path

from src.tools.proxy_manager import ProxyManager


class TestProxyManager(unittest.TestCase):
    """ProxyManagerの基本機能テスト"""

    def test_get_proxy_path_default(self):
        """プロキシパス生成（デフォルト: ソースと同じディレクトリ）"""
        pm = ProxyManager()
        result = pm.get_proxy_path("/tmp/case001.mp4")
        self.assertEqual(result, Path("/tmp/case001_720p.mp4"))

    def test_get_proxy_path_custom_dir(self):
        """プロキシパス生成（カスタムディレクトリ）"""
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = ProxyManager(proxy_dir=Path(tmpdir))
            result = pm.get_proxy_path("/tmp/case001.mp4", "480p")
            self.assertEqual(result, Path(tmpdir) / "case001_480p.mp4")

    def test_get_proxy_path_resolutions(self):
        """各解像度のパス生成（完全なファイル名を検証）"""
        pm = ProxyManager()
        for res in ["360p", "480p", "720p", "1080p"]:
            result = pm.get_proxy_path("/tmp/video.mp4", res)
            # 部分一致ではなく厳密なファイル名で検証する
            self.assertEqual(result.name, f"video_{res}.mp4")

    def test_get_proxy_path_no_extension(self):
        """拡張子なしの入力パス → 拡張子なしのプロキシ名"""
        pm = ProxyManager()
        result = pm.get_proxy_path("/tmp/video", "720p")
        self.assertEqual(result.name, "video_720p")

    def test_get_proxy_path_arbitrary_resolution_string(self):
        """解像度文字列はそのまま名前に埋め込まれる（検証は呼び出し側責務）"""
        pm = ProxyManager()
        result = pm.get_proxy_path("/tmp/video.mp4", "bogus")
        self.assertEqual(result.name, "video_bogus.mp4")

    def test_proxy_exists_false(self):
        """存在しないプロキシの判定"""
        pm = ProxyManager()
        self.assertFalse(pm.proxy_exists("/tmp/nonexistent_video.mp4"))

    def test_proxy_exists_true(self):
        """存在するプロキシの判定"""
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = ProxyManager(proxy_dir=Path(tmpdir))
            proxy_path = pm.get_proxy_path("/tmp/test.mp4")
            # ダミーファイルを作成
            proxy_path.write_bytes(b"dummy video content")
            self.assertTrue(pm.proxy_exists("/tmp/test.mp4"))

    def test_proxy_exists_empty_file(self):
        """空ファイルはプロキシとして認めない"""
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = ProxyManager(proxy_dir=Path(tmpdir))
            proxy_path = pm.get_proxy_path("/tmp/test.mp4")
            proxy_path.touch()  # 0バイトファイル
            self.assertFalse(pm.proxy_exists("/tmp/test.mp4"))

    def test_proxy_exists_explicit_resolution(self):
        """明示した解像度のプロキシのみが存在判定されること"""
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = ProxyManager(proxy_dir=Path(tmpdir))
            # 480p のプロキシだけ作成する
            pm.get_proxy_path("/tmp/test.mp4", "480p").write_bytes(b"data")
            self.assertTrue(pm.proxy_exists("/tmp/test.mp4", "480p"))
            # 720p は未作成 → False
            self.assertFalse(pm.proxy_exists("/tmp/test.mp4", "720p"))

    def test_resolutions_dict(self):
        """RESOLUTIONS の構造を検証（特定値ではなく不変条件で検証）。"""
        res = ProxyManager.RESOLUTIONS
        # CLI choices と一致する4解像度を網羅
        self.assertEqual(set(res), {"360p", "480p", "720p", "1080p"})
        for key, value in res.items():
            self.assertIsInstance(key, str)
            self.assertIsInstance(value, tuple)
            self.assertEqual(len(value), 2)
            w, h = value
            self.assertIsInstance(w, int)
            self.assertIsInstance(h, int)
            # 幅 > 高さ（横長）かつ正の値
            self.assertGreater(h, 0)
            self.assertGreater(w, h)
        # 720p の名前と実解像度の対応（リグレッション検出用の代表ケース）
        self.assertEqual(res["720p"], (1280, 720))


if __name__ == "__main__":
    unittest.main()
