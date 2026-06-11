"""
アノテーション編集サーバ（FastAPI）の統合テスト。

TestClient で各エンドポイントの正常系・異常系を検証する:
  - index        : CSRF トークンが埋め込まれ、プレースホルダが漏れない
  - /api/session : ラベル語彙・セグメント・新規作成フラグ
  - /api/save    : CSRF 必須(403)、重複ID拒否(400)、正常保存(200, SRT/JSONL生成)
  - /media/video : プロキシ不在時 404 / 在席時 200 + MIME
  - build_state  : 動画不在(FileNotFoundError) / 未登録術式(KeyError)
"""

import json
import re
import shutil
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
        history_target=tmp / "case001_gold_history.jsonl",
    )


def _token(client: TestClient) -> str:
    html = client.get("/").text
    return re.search(r'name="csrf-token" content="([^"]+)"', html).group(1)


class _TmpCase(unittest.TestCase):
    """self.tmp の一時ディレクトリを tearDown で必ず後始末する基底クラス。

    テスト失敗/例外時も含めて削除されるため、一時ファイルがリークしない。
    """

    def tearDown(self):
        tmp = getattr(self, "tmp", None)
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)


class TestIndexAndSession(_TmpCase):
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


class TestSave(_TmpCase):
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

    def test_save_writes_history_log(self):
        token = _token(self.client)
        r = self.client.post("/api/save", json={"segments": [self.seg]},
                             headers={"X-CSRF-Token": token})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["changes"], 1)  # 新規追加1件
        self.assertTrue(self.state.history_target.exists())
        lines = self.state.history_target.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 1)
        entry = json.loads(lines[0])
        self.assertEqual(entry["n_changes"], 1)
        self.assertEqual(entry["changes"][0]["change"], "inserted")
        self.assertIn("timestamp", entry)

    def test_history_accumulates_per_save(self):
        token = _token(self.client)
        # 1回目: 追加
        self.client.post("/api/save", json={"segments": [self.seg]},
                         headers={"X-CSRF-Token": token})
        # 2回目: 同一IDのラベルを変更 → edited として履歴に追記される
        edited = dict(self.seg, phase_name="ClippingCutting")
        r = self.client.post("/api/save", json={"segments": [edited]},
                             headers={"X-CSRF-Token": token})
        self.assertEqual(r.json()["changes"], 1)
        lines = self.state.history_target.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 2)  # 保存ごとに1エントリ追記
        self.assertEqual(json.loads(lines[1])["changes"][0]["change"], "edited")

    def test_no_change_save_not_logged(self):
        token = _token(self.client)
        self.client.post("/api/save", json={"segments": [self.seg]},
                         headers={"X-CSRF-Token": token})  # 1回目: inserted
        r = self.client.post("/api/save", json={"segments": [self.seg]},
                             headers={"X-CSRF-Token": token})  # 2回目: 変更なし
        self.assertEqual(r.json()["changes"], 0)
        lines = self.state.history_target.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 1)  # 変更0の保存は履歴に残らない

    def test_history_endpoint_requires_csrf(self):
        r = self.client.get("/api/history")  # トークン無し
        self.assertEqual(r.status_code, 403)

    def test_history_endpoint_returns_entries(self):
        token = _token(self.client)
        self.client.post("/api/save", json={"segments": [self.seg]},
                         headers={"X-CSRF-Token": token})
        r = self.client.get("/api/history", headers={"X-CSRF-Token": token})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(len(body["entries"]), 1)
        self.assertEqual(body["history_file"], self.state.history_target.name)

    def test_history_limit_returns_latest(self):
        token = _token(self.client)
        # 3回、内容を変えて保存（毎回 changes>0 で履歴に残る）
        for i in range(3):
            seg = dict(self.seg, end_sec=5.0 + i)
            self.client.post("/api/save", json={"segments": [seg]},
                             headers={"X-CSRF-Token": token})
        r = self.client.get("/api/history?limit=2",
                            headers={"X-CSRF-Token": token})
        body = r.json()
        self.assertEqual(body["limit"], 2)
        self.assertEqual(len(body["entries"]), 2)  # 最新2件のみ


class TestOpen(_TmpCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.client = TestClient(create_app(_state(self.tmp)))
        # base_dir を tmp に限定したサーバ（パストラバーサル検証用）
        self.confined = TestClient(
            create_app(_state(self.tmp), base_dir=self.tmp))

    def test_open_requires_csrf(self):
        r = self.client.post("/api/open", json={"video": "x.mp4"})
        self.assertEqual(r.status_code, 403)

    def test_open_missing_video_returns_400(self):
        token = _token(self.client)
        r = self.client.post("/api/open",
                             json={"video": str(self.tmp / "nope.mp4")},
                             headers={"X-CSRF-Token": token})
        self.assertEqual(r.status_code, 400)

    def _open(self, video):
        token = _token(self.confined)
        return self.confined.post("/api/open", json={"video": video},
                                  headers={"X-CSRF-Token": token})

    def test_open_rejects_path_traversal(self):
        r = self._open("../../../etc/passwd")
        self.assertEqual(r.status_code, 400)
        self.assertIn("許可ディレクトリ外", r.json()["detail"])

    def test_open_rejects_absolute_outside_base(self):
        r = self._open("/etc/passwd")
        self.assertEqual(r.status_code, 400)
        self.assertIn("許可ディレクトリ外", r.json()["detail"])

    def test_open_rejects_empty_path(self):
        r = self._open("")
        self.assertEqual(r.status_code, 400)

    def test_open_rejects_directory_path(self):
        # ベース配下だがファイルでない（ディレクトリ）→ 400
        r = self._open(str(self.tmp))
        self.assertEqual(r.status_code, 400)


class TestMediaVideo(_TmpCase):
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


class TestBuildState(_TmpCase):
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
