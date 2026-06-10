"""
異常検知 学習スクリプト（src/anomaly/train.py）のテスト。

iter_frames / preprocess_frame_anomaly をモックして実 I/O・実動画なしで
以下を検証する:
  - _validate_video_paths : 欠落スキップ・全欠落で FileNotFoundError
  - NormalVideoDataset    : 動画境界を跨がないシーケンス・件数
  - split_indices         : val hold-out・境界ギャップ・train/val 非重複
  - train_model           : 学習→重み保存→resume、val_fraction 経路
  - calibrate_threshold   : 欠落動画で FileNotFoundError
"""

import tempfile
from pathlib import Path

import pytest
import torch

import src.anomaly.train as T
from src.anomaly.train import (
    NormalVideoDataset,
    _validate_video_paths,
    train_model,
    calibrate_threshold,
)

# 小さなテスト用モデル構成（CPU で高速）
TINY = dict(seq_len=2, frame_size=32, feature_dim=8, hidden_dims=[8, 8])


@pytest.fixture
def mock_frames(monkeypatch):
    """iter_frames / preprocess をモックし、動画ごとのフレーム数を制御する。

    動画パスの末尾文字でフレーム数を引く辞書を返す。各フレームは動画タグを
    符号化した定数テンソル（境界跨ぎ検出用）。
    """
    counts = {}

    def fake_iter_frames(vp, fps):
        n = counts.get(Path(vp).stem, 6)
        for i in range(n):
            yield (float(i) / fps, Path(vp).stem, None)

    def fake_preprocess(tag, size):
        # tag（動画stem）をハッシュして定数テンソル化
        val = float(abs(hash(tag)) % 100)
        return torch.full((3, TINY["frame_size"], TINY["frame_size"]), val)

    monkeypatch.setattr(T, "iter_frames", fake_iter_frames)
    monkeypatch.setattr(T, "preprocess_frame_anomaly", fake_preprocess)
    return counts


def _make_video_files(tmp: Path, names) -> list:
    paths = []
    for name in names:
        p = tmp / f"{name}.mp4"
        p.write_bytes(b"0")  # 中身は不要（iter_frames はモック）
        paths.append(str(p))
    return paths


# ---------------------------------------------------------------------------
# _validate_video_paths
# ---------------------------------------------------------------------------

class TestValidatePaths:
    def test_existing_files_pass(self):
        with tempfile.TemporaryDirectory() as d:
            paths = _make_video_files(Path(d), ["a", "b"])
            assert _validate_video_paths(paths) == paths

    def test_missing_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            paths = _make_video_files(Path(d), ["a"])
            mixed = paths + [str(Path(d) / "ghost.mp4")]
            assert _validate_video_paths(mixed) == paths

    def test_all_missing_raises(self):
        with pytest.raises(FileNotFoundError):
            _validate_video_paths(["/no/such/x.mp4", "/no/such/y.mp4"])


# ---------------------------------------------------------------------------
# NormalVideoDataset
# ---------------------------------------------------------------------------

class TestDataset:
    def test_length_and_no_boundary_crossing(self, mock_frames):
        mock_frames.update({"vidA": 5, "vidB": 4})
        with tempfile.TemporaryDirectory() as d:
            paths = _make_video_files(Path(d), ["vidA", "vidB"])
            ds = NormalVideoDataset(paths, seq_len=2, fps=5.0)
            # A: 5-2=3, B: 4-2=2 → 計5
            assert len(ds) == 5
            for i in range(len(ds)):
                seq, target = ds[i]
                vals = torch.cat([seq.reshape(-1), target.reshape(-1)])
                # 全フレームが同一動画タグ（同一定数）であること
                assert torch.unique(vals).numel() == 1

    def test_split_indices_disabled(self, mock_frames):
        mock_frames.update({"vidA": 6})
        with tempfile.TemporaryDirectory() as d:
            paths = _make_video_files(Path(d), ["vidA"])
            ds = NormalVideoDataset(paths, seq_len=2, fps=5.0)
            train, val = ds.split_indices(0.0)
            assert val == []
            assert train == list(range(len(ds)))

    def test_split_indices_holdout_and_gap(self, mock_frames):
        mock_frames.update({"vidA": 12})  # valid 件数 = 12-2 = 10
        with tempfile.TemporaryDirectory() as d:
            paths = _make_video_files(Path(d), ["vidA"])
            ds = NormalVideoDataset(paths, seq_len=2, fps=5.0)
            train, val = ds.split_indices(0.2)
            # n=10, n_val=2, cut=8, gap=seq_len=2 → train=[0..5], val=[8,9]
            assert val == [8, 9]
            assert train == [0, 1, 2, 3, 4, 5]
            # train と val は重複しない & 境界ギャップで隔離
            assert set(train).isdisjoint(val)


# ---------------------------------------------------------------------------
# train_model
# ---------------------------------------------------------------------------

class TestTrainModel:
    def test_train_saves_weights(self, mock_frames):
        mock_frames.update({"vidA": 8})
        with tempfile.TemporaryDirectory() as d:
            paths = _make_video_files(Path(d), ["vidA"])
            out = Path(d) / "models"
            wp = train_model(paths, str(out), epochs=2, batch_size=2,
                             device="cpu", num_workers=0, **TINY)
            assert wp and Path(wp).exists()
            assert (out / "anomaly_convlstm_final.pth").exists()

    def test_resume_loads_existing(self, mock_frames):
        mock_frames.update({"vidA": 8})
        with tempfile.TemporaryDirectory() as d:
            paths = _make_video_files(Path(d), ["vidA"])
            out = Path(d) / "models"
            wp = train_model(paths, str(out), epochs=1, batch_size=2,
                             device="cpu", num_workers=0, **TINY)
            # resume パスを渡して再学習（例外が出ないこと）
            wp2 = train_model(paths, str(out), epochs=1, batch_size=2,
                              device="cpu", num_workers=0, resume_path=wp, **TINY)
            assert Path(wp2).exists()

    def test_val_fraction_path_runs(self, mock_frames):
        mock_frames.update({"vidA": 20})
        with tempfile.TemporaryDirectory() as d:
            paths = _make_video_files(Path(d), ["vidA"])
            out = Path(d) / "models"
            wp = train_model(paths, str(out), epochs=2, batch_size=2,
                             device="cpu", num_workers=0,
                             val_fraction=0.3, patience=1, **TINY)
            assert Path(wp).exists()

    def test_missing_videos_raise(self):
        with pytest.raises(FileNotFoundError):
            train_model(["/no/such.mp4"], "/tmp/out_x", epochs=1, device="cpu")


# ---------------------------------------------------------------------------
# calibrate_threshold
# ---------------------------------------------------------------------------

class TestCalibrate:
    def test_missing_videos_raise(self):
        with pytest.raises(FileNotFoundError):
            calibrate_threshold(["/no/such.mp4"], "/no/weights.pth",
                                device="cpu")
