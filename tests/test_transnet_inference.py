"""
SceneDetector.predict_video の後処理ロジックの単体テスト。

実モデル・実 ffmpeg を使わず、ffmpeg プローブ/デコードと _process_batch を
モックして、シーン分割・空動画・末尾の不完全フレーム処理といった臨界パスを
検証する。
"""

import unittest
from unittest import mock

import numpy as np

from src.transnet import inference
from src.transnet.inference import SceneDetector

# TransNet V2 の入力サイズ（27x48x3）に対応する1フレームのバイト数
FRAME_NBYTES = 27 * 48 * 3


class _FakeStream:
    """ffmpeg stdout/stderr を模した read/close 可能なストリーム。"""

    def __init__(self, chunks):
        self._chunks = list(chunks)

    def read(self, n):  # n は無視し、用意したチャンクを順に返す
        if self._chunks:
            return self._chunks.pop(0)
        return b""

    def close(self):
        pass


class _FakeProcess:
    def __init__(self, chunks, returncode=0):
        self.stdout = _FakeStream(chunks)
        self.stderr = _FakeStream([])
        self._returncode = returncode

    def wait(self):
        return self._returncode


def _make_detector():
    """__init__（重いモデル生成）を回避して SceneDetector を組み立てる。"""
    det = SceneDetector.__new__(SceneDetector)
    det.device = "cpu"
    det.model = None
    return det


def _patch_ffmpeg(chunks, nb_frames=10, fps=25.0):
    """ffmpeg.probe と ffmpeg.input(...).run_async(...) をモックする。"""
    probe_value = {
        "streams": [{
            "codec_type": "video",
            "width": 48,
            "height": 27,
            "r_frame_rate": f"{int(fps)}/1",
            "nb_frames": str(nb_frames),
            "duration": str(nb_frames / fps),
        }]
    }
    fake_proc = _FakeProcess(chunks)
    chain = mock.MagicMock()
    chain.filter.return_value.output.return_value.run_async.return_value = fake_proc
    return (
        mock.patch.object(inference.ffmpeg, "probe", return_value=probe_value),
        mock.patch.object(inference.ffmpeg, "input", return_value=chain),
    )


class TestPredictVideoPostProcessing(unittest.TestCase):
    def test_scene_split_on_cut(self):
        # 10フレーム、index5 に強いカット（score 0.9）。min_scene_length=5。
        det = _make_detector()
        chunks = [b"\x00" * FRAME_NBYTES for _ in range(10)]
        scores = np.full(10, 0.1, dtype=np.float32)
        scores[5] = 0.9

        def fake_batch(frames, predictions):
            predictions.append(scores)

        p_probe, p_input = _patch_ffmpeg(chunks, nb_frames=10)
        with p_probe, p_input, \
                mock.patch.object(det, "_process_batch", side_effect=fake_batch):
            result = det.predict_video("x.mp4", threshold=0.5, return_scores=True,
                                       min_scene_length=5)
        out_scenes, out_scores = result
        # [0,5) と [5,10) の2シーン
        self.assertEqual(len(out_scenes), 2)
        self.assertEqual(out_scenes[0]["start_frame"], 0)
        self.assertEqual(out_scenes[0]["end_frame"], 5)
        self.assertEqual(out_scenes[1]["start_frame"], 5)
        # 最終シーン終端は実デコード数（10）に一致
        self.assertEqual(out_scenes[1]["end_frame"], 10)
        self.assertEqual(len(out_scores), 10)

    def test_short_cut_below_min_length_ignored(self):
        # index2 にカットがあるが min_scene_length=5 未満なので無視 → 1シーン
        det = _make_detector()
        chunks = [b"\x00" * FRAME_NBYTES for _ in range(10)]
        scores = np.full(10, 0.1, dtype=np.float32)
        scores[2] = 0.9

        def fake_batch(frames, predictions):
            predictions.append(scores)

        p_probe, p_input = _patch_ffmpeg(chunks, nb_frames=10)
        with p_probe, p_input, \
                mock.patch.object(det, "_process_batch", side_effect=fake_batch):
            scenes = det.predict_video("x.mp4", threshold=0.5, min_scene_length=5)
        self.assertEqual(len(scenes), 1)
        self.assertEqual(scenes[0]["start_frame"], 0)
        self.assertEqual(scenes[0]["end_frame"], 10)

    def test_empty_video_returns_empty(self):
        # フレームが1枚もデコードできない → 空結果
        det = _make_detector()
        p_probe, p_input = _patch_ffmpeg([], nb_frames=0)
        with p_probe, p_input, \
                mock.patch.object(det, "_process_batch") as m_batch:
            scenes, scores = det.predict_video("x.mp4", return_scores=True)
        self.assertEqual(scenes, [])
        self.assertEqual(len(scores), 0)
        m_batch.assert_not_called()

    def test_partial_trailing_frame_discarded(self):
        # 2フレーム分の完全データ + 末尾に不完全な端数 → 端数は破棄され2フレーム処理
        det = _make_detector()
        chunks = [b"\x00" * FRAME_NBYTES, b"\x00" * FRAME_NBYTES, b"\x00" * 100]
        captured = {}

        def fake_batch(frames, predictions):
            captured["n"] = len(frames)
            predictions.append(np.full(len(frames), 0.1, dtype=np.float32))

        p_probe, p_input = _patch_ffmpeg(chunks, nb_frames=3)
        with p_probe, p_input, \
                mock.patch.object(det, "_process_batch", side_effect=fake_batch):
            scenes, scores = det.predict_video("x.mp4", return_scores=True)
        # 端数フレームは捨てられ、2フレームのみ処理される
        self.assertEqual(captured["n"], 2)
        self.assertEqual(len(scores), 2)
        # 最終シーンの終端は2（端数を含まない）
        self.assertEqual(scenes[-1]["end_frame"], 2)


if __name__ == "__main__":
    unittest.main()
