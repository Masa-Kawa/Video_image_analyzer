"""
アノテーション編集サーバ（FastAPI）の統合テスト。

TestClient で各エンドポイントの正常系・異常系を検証する:
  - index        : CSRF トークンが埋め込まれ、プレースホルダが漏れない
  - /api/session : ラベル語彙・セグメント・新規作成フラグ
  - /api/save    : CSRF 必須(403)、重複ID拒否(400)、正常保存(200, SRT/JSONL生成)
  - /media/video : プロキシ不在時 404 / 在席時 200 + MIME
  - build_state  : 動画不在(FileNotFoundError) / 未登録術式(KeyError)
"""

import re
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from src.annotate.server import (
    AppState,
    build_state,
    create_app,
    _video_media_type,
)


def _state(tmp: Path, srt: Path = None, proxy: Path = None) -> AppState:
    return AppState(
        video=Path("case001.mp4"),
        srt=srt,
        procedure="cholecystectomy",
        proxy=proxy if proxy is not None else tmp / "proxy.mp4",
        save_target=tmp / "case001_gold.srt",
        pairs_target=tmp / "case001_dpo_pairs.jsonl",
    )


def _token(client: TestClient) -> str:
    html = client.get("/").text
    return re.search(r'name="csrf-token" content="([^"]+)"', html).group(1)


class TestIndexAndSession(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.client = TestClient(create_app(_state(self.tmp)))

    def test_index_injects_csrf_token(self):
        html = self.client.get("/").text
        self.assertNotIn("__CSRF_TOKEN__", html)  # プレースホルダが残っていない
        self.assertRegex(html, r'name="csrf-token" content="[\w\-]{20,}"')

    def test_session_new_mode(self):
        r = self.client.get("/api/session")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["procedure"], "cholecystectomy")
        self.assertTrue(body["is_new"])          # srt=None → 新規作成モード
        self.assertEqual(body["segments"], [])
        self.assertIn("Preparation", body["labels"])

    def test_session_returns_existing_segments(self):
        state = _state(self.tmp)
        state.original_segments = [
            {"id": "s1", "phase_name": "Preparation",
             "start_sec": 0.0, "end_sec": 5.0},
        ]
        client = TestClient(create_app(state))
        body = client.get("/api/session").json()
        self.assertEqual(len(body["segments"]), 1)
        self.assertEqual(body["segments"][0]["id"], "s1")


class TestSave(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.state = _state(self.tmp)
        self.client = TestClient(create_app(self.state))
        self.seg = {"id": "a1", "phase_name": "Preparation",
                    "start_sec": 0.0, "end_sec": 5.0}

    def test_save_requires_csrf_token(self):
        r = self.client.post("/api/save", json={"segments": [self.seg]})
        self.assertEqual(r.status_code, 403)

    def test_save_rejects_bad_csrf_token(self):
        r = self.client.post("/api/save", json={"segments": [self.seg]},
                             headers={"X-CSRF-Token": "wrong"})
        self.assertEqual(r.status_code, 403)

    def test_save_rejects_duplicate_id(self):
        token = _token(self.client)
        dup = [self.seg,
               {"id": "a1", "phase_name": "ClippingCutting",
                "start_sec": 5.0, "end_sec": 10.0}]
        r = self.client.post("/api/save", json={"segments": dup},
                             headers={"X-CSRF-Token": token})
        self.assertEqual(r.status_code, 400)
        self.assertIn("a1", r.json()["detail"])

    def test_save_success_writes_srt_and_pairs(self):
        token = _token(self.client)
        r = self.client.post("/api/save", json={"segments": [self.seg]},
                             headers={"X-CSRF-Token": token})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["saved"], 1)
        self.assertTrue(self.state.save_target.exists())
        self.assertTrue(self.state.pairs_target.exists())
        # 新規作成モードなので全セグメントが inserted ペアになる
        self.assertEqual(body["pairs"], 1)

    def test_save_rejects_malformed_body(self):
        # phase_name 欠落 → Pydantic バリデーションエラー(422)
        token = _token(self.client)
        r = self.client.post("/api/save",
                             json={"segments": [{"id": "x", "start_sec": 0.0}]},
                             headers={"X-CSRF-Token": token})
        self.assertEqual(r.status_code, 422)


class TestMediaVideo(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_media_video_404_when_proxy_missing(self):
        client = TestClient(create_app(_state(self.tmp)))  # proxy 未作成
        r = client.get("/media/video")
        self.assertEqual(r.status_code, 404)

    def test_media_video_200_with_mime_when_present(self):
        proxy = self.tmp / "proxy.mp4"
        proxy.write_bytes(b"\x00\x00\x00\x18ftypmp42")  # ダミーの中身
        client = TestClient(create_app(_state(self.tmp, proxy=proxy)))
        r = client.get("/media/video")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.headers["content-type"], "video/mp4")


class TestMediaType(unittest.TestCase):
    def test_known_and_unknown_suffixes(self):
        self.assertEqual(_video_media_type(Path("a.mp4")), "video/mp4")
        self.assertEqual(_video_media_type(Path("a.MKV")), "video/x-matroska")
        self.assertEqual(_video_media_type(Path("a.mov")), "video/quicktime")
        self.assertEqual(_video_media_type(Path("a.bin")), "video/mp4")  # fallback


class TestBuildState(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_missing_video_raises(self):
        with self.assertRaises(FileNotFoundError):
            build_state(str(self.tmp / "nope.mp4"), None,
                        "cholecystectomy", None)

    def test_unknown_procedure_raises(self):
        # 動画存在チェックを通すためダミーファイルを用意（語彙検証はその後）
        vid = self.tmp / "case.mp4"
        vid.write_bytes(b"0")
        with self.assertRaises(KeyError):
            build_state(str(vid), None, "does_not_exist", None)


if __name__ == "__main__":
    unittest.main()
