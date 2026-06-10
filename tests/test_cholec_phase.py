"""
Cholec80 フェーズ認識モジュールのテスト

テスト項目:
  - PhaseRecognitionModel の shape / forward
  - PhaseFeatureDataset のサンプル構造
  - フェーズアノテーション読み込み / 変換
  - CSV 往復テスト
  - フェーズ区間抽出（annotate）
  - SRT / JSONL フォーマット検証
  - タグテンプレート登録
"""

import csv
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from src.cholec_phase import CHOLEC80_PHASES, NUM_PHASES


# ---------------------------------------------------------------------------
# 定数テスト
# ---------------------------------------------------------------------------


class TestConstants:
    def test_phase_count(self):
        assert NUM_PHASES == 7

    def test_phase_names(self):
        assert CHOLEC80_PHASES[0] == "Preparation"
        assert CHOLEC80_PHASES[1] == "CalotTriangleDissection"
        assert CHOLEC80_PHASES[2] == "ClippingCutting"
        assert CHOLEC80_PHASES[3] == "GallbladderDissection"
        assert CHOLEC80_PHASES[4] == "GallbladderPackaging"
        assert CHOLEC80_PHASES[5] == "CleaningCoagulation"
        assert CHOLEC80_PHASES[6] == "GallbladderRetraction"


# ---------------------------------------------------------------------------
# モデルテスト
# ---------------------------------------------------------------------------


class TestPhaseRecognitionModel:
    def test_model_output_shape(self):
        from src.cholec_phase.models import PhaseRecognitionModel

        model = PhaseRecognitionModel(
            feature_dim=2048,
            hidden_dim=64,
            num_layers=1,
            num_classes=7,
            dropout=0.0,
        )
        x = torch.randn(2, 10, 2048)
        logits = model(x)
        assert logits.shape == (2, 10, 7)

    def test_model_with_lengths(self):
        from src.cholec_phase.models import PhaseRecognitionModel

        model = PhaseRecognitionModel(
            feature_dim=2048,
            hidden_dim=64,
            num_layers=1,
            dropout=0.0,
        )
        x = torch.randn(3, 20, 2048)
        lengths = torch.tensor([20, 15, 10])
        logits = model(x, lengths)
        assert logits.shape == (3, 20, 7)

    def test_predict_proba(self):
        from src.cholec_phase.models import PhaseRecognitionModel

        model = PhaseRecognitionModel(
            feature_dim=2048,
            hidden_dim=64,
            num_layers=1,
            dropout=0.0,
        )
        model.eval()
        x = torch.randn(1, 5, 2048)
        probs = model.predict_proba(x)
        assert probs.shape == (1, 5, 7)
        # 確率なので合計が1に近い
        sums = probs.sum(dim=2)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_model_unidirectional(self):
        from src.cholec_phase.models import PhaseRecognitionModel

        model = PhaseRecognitionModel(
            feature_dim=2048,
            hidden_dim=64,
            num_layers=1,
            bidirectional=False,
            dropout=0.0,
        )
        x = torch.randn(1, 10, 2048)
        logits = model(x)
        assert logits.shape == (1, 10, 7)

    def test_model_gradient_flow(self):
        from src.cholec_phase.models import PhaseRecognitionModel

        model = PhaseRecognitionModel(
            feature_dim=2048,
            hidden_dim=64,
            num_layers=1,
            dropout=0.0,
        )
        x = torch.randn(1, 5, 2048)
        labels = torch.randint(0, 7, (1, 5))
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, 7), labels.reshape(-1),
        )
        loss.backward()
        # 内部構造（self.lstm 等の属性名）に依存せず、学習可能パラメータの
        # 少なくとも1つに勾配が流れていることを検証する。
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        assert len(grads) > 0
        assert any(g is not None and torch.any(g != 0) for g in grads)


# ---------------------------------------------------------------------------
# データセットテスト
# ---------------------------------------------------------------------------


class TestPhaseAnnotation:
    def test_read_phase_annotation(self):
        from src.cholec_phase.dataset import read_phase_annotation

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write("Frame\tPhase\n")
            f.write("0\tPreparation\n")
            f.write("25\tPreparation\n")
            f.write("50\tCalotTriangleDissection\n")
            f.write("75\tCalotTriangleDissection\n")
            f.write("100\tClippingCutting\n")
            ann_path = f.name

        try:
            frames, phase_ids = read_phase_annotation(ann_path)
            assert len(frames) == 5
            assert frames == [0, 25, 50, 75, 100]
            assert phase_ids == [0, 0, 1, 1, 2]
        finally:
            os.unlink(ann_path)

    def test_frame_to_second_conversion(self):
        from src.cholec_phase.dataset import frame_annotation_to_second

        frames = [0, 25, 50, 75, 100]
        phase_ids = [0, 0, 1, 1, 2]

        times, sampled = frame_annotation_to_second(
            frames, phase_ids, video_fps=25.0, sample_fps=1.0,
        )
        # At 25fps, 100 frames = 4 seconds → we get t=0,1,2,3,4
        assert len(times) == 5
        assert times[0] == 0.0
        assert times[-1] == 4.0
        # Frame 0 → Preparation (0), Frame 25 → Preparation (0)
        assert sampled[0] == 0
        assert sampled[1] == 0
        # Frame 50 → CalotTriangleDissection (1)
        assert sampled[2] == 1

    def test_phase_to_id_mapping(self):
        from src.cholec_phase.dataset import PHASE_TO_ID

        assert PHASE_TO_ID["Preparation"] == 0
        assert PHASE_TO_ID["GallbladderRetraction"] == 6
        assert len(PHASE_TO_ID) == 7


class TestPhaseFeatureDataset:
    def test_dataset_creation(self):
        from src.cholec_phase.dataset import PhaseFeatureDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            # 合成データ作成
            n_frames = 100
            features = np.random.randn(n_frames, 2048).astype(np.float32)
            labels = np.random.randint(0, 7, n_frames)

            np.save(os.path.join(tmpdir, "video01_features.npy"), features)
            np.save(os.path.join(tmpdir, "video01_labels.npy"), labels)

            ds = PhaseFeatureDataset(
                tmpdir, video_ids=[1], seq_len=30, stride=15,
            )

            assert len(ds) > 0

            feat, lbl, length = ds[0]
            assert feat.shape == (30, 2048)
            assert lbl.shape == (30,)
            assert length <= 30

    def test_dataset_padding(self):
        from src.cholec_phase.dataset import PhaseFeatureDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            n_frames = 20
            features = np.random.randn(n_frames, 2048).astype(np.float32)
            labels = np.random.randint(0, 7, n_frames)

            np.save(os.path.join(tmpdir, "video01_features.npy"), features)
            np.save(os.path.join(tmpdir, "video01_labels.npy"), labels)

            ds = PhaseFeatureDataset(
                tmpdir, video_ids=[1], seq_len=50, stride=50,
            )

            feat, lbl, length = ds[0]
            assert feat.shape == (50, 2048)
            assert lbl.shape == (50,)
            assert length == 20
            # パディング部分は -1
            assert lbl[20].item() == -1


class TestTrainValSplit:
    def test_split_no_overlap(self):
        from src.cholec_phase.dataset import get_train_val_split, get_test_ids

        train_ids, val_ids = get_train_val_split()
        test_ids = get_test_ids()

        # 重複なし
        assert len(set(train_ids) & set(val_ids)) == 0
        assert len(set(train_ids) & set(test_ids)) == 0
        assert len(set(val_ids) & set(test_ids)) == 0

        # train+val は video01-40
        assert all(1 <= i <= 40 for i in train_ids)
        assert all(1 <= i <= 40 for i in val_ids)
        assert set(train_ids) | set(val_ids) == set(range(1, 41))

        # test は video41-80
        assert test_ids == list(range(41, 81))


# ---------------------------------------------------------------------------
# CSV 往復テスト
# ---------------------------------------------------------------------------


class TestCSVRoundtrip:
    def test_csv_write_read(self):
        from src.cholec_phase.detector import read_cholecphaselog_csv
        from src.core.time_utils import format_srt_time

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "test_cholecphaselog.csv")
            prob_headers = [f"prob_{name}" for name in CHOLEC80_PHASES]

            # CSV書き込み
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["t_sec", "t_srt", "phase_id", "phase_name"]
                    + prob_headers
                    + ["confidence", "reader"]
                )
                for i in range(10):
                    t = i * 1.0
                    probs = [0.0] * 7
                    probs[i % 7] = 1.0
                    row = [
                        f"{t:.3f}",
                        format_srt_time(t),
                        str(i % 7),
                        CHOLEC80_PHASES[i % 7],
                    ]
                    for p in probs:
                        row.append(f"{p:.6f}")
                    row.append("1.000000")
                    row.append("pyav")
                    writer.writerow(row)

            # CSV読み込み
            data = read_cholecphaselog_csv(csv_path)
            assert len(data["times"]) == 10
            assert data["times"][0] == 0.0
            assert data["times"][9] == 9.0
            assert data["phase_ids"][0] == 0
            assert data["phase_names"][0] == "Preparation"
            assert data["probs"].shape == (10, 7)
            assert data["reader"] == "pyav"
            # テストデータは整数秒刻み（t=i*1.0）なので fps は厳密に 1.0。
            # 曖昧な閾値比較を避け、浮動小数の丸め誤差のみを許容する。
            assert data["fps"] == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# アノテーションテスト
# ---------------------------------------------------------------------------


class TestAnnotation:
    def _create_test_csv(self, tmpdir: str, n_frames: int = 100) -> str:
        """テスト用CSVを作成する。"""
        from src.core.time_utils import format_srt_time

        csv_path = os.path.join(tmpdir, "test_cholecphaselog.csv")
        prob_headers = [f"prob_{name}" for name in CHOLEC80_PHASES]

        # Preparation(30s) → CalotTriangleDissection(40s) → ClippingCutting(30s)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["t_sec", "t_srt", "phase_id", "phase_name"]
                + prob_headers
                + ["confidence", "reader"]
            )
            for i in range(n_frames):
                t = float(i)
                if i < 30:
                    pid, pname = 0, "Preparation"
                elif i < 70:
                    pid, pname = 1, "CalotTriangleDissection"
                else:
                    pid, pname = 2, "ClippingCutting"

                probs = [0.0] * 7
                probs[pid] = 0.9
                row = [
                    f"{t:.3f}",
                    format_srt_time(t),
                    str(pid),
                    pname,
                ]
                for p in probs:
                    row.append(f"{p:.6f}")
                row.append("0.900000")
                row.append("pyav")
                writer.writerow(row)

        return csv_path

    def test_annotate_phases(self):
        from src.cholec_phase.detector import annotate_phases

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = self._create_test_csv(tmpdir)
            result = annotate_phases(
                csv_path, tmpdir, min_phase_s=5.0, smooth_s=0.0,
            )

            assert result["phases"] == 3
            assert Path(result["jsonl"]).exists()
            assert Path(result["srt"]).exists()

    def test_jsonl_format(self):
        from src.cholec_phase.detector import annotate_phases

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = self._create_test_csv(tmpdir)
            result = annotate_phases(
                csv_path, tmpdir, min_phase_s=5.0, smooth_s=0.0,
            )

            with open(result["jsonl"], "r", encoding="utf-8") as f:
                events = [json.loads(line) for line in f]

            assert len(events) == 3

            # 必須フィールド
            for ev in events:
                assert "type" in ev
                assert ev["type"] == "surgical_phase"
                assert "start_sec" in ev
                assert "end_sec" in ev
                assert "start_srt" in ev
                assert "end_srt" in ev
                assert "phase_id" in ev
                assert "phase_name" in ev

            # フェーズ順序
            assert events[0]["phase_name"] == "Preparation"
            assert events[1]["phase_name"] == "CalotTriangleDissection"
            assert events[2]["phase_name"] == "ClippingCutting"

            # start_sec 昇順
            for i in range(1, len(events)):
                assert events[i]["start_sec"] >= events[i - 1]["start_sec"]

    def test_srt_format(self):
        from src.cholec_phase.detector import annotate_phases

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = self._create_test_csv(tmpdir)
            result = annotate_phases(
                csv_path, tmpdir, min_phase_s=5.0, smooth_s=0.0,
            )

            srt_text = Path(result["srt"]).read_text(encoding="utf-8")
            lines = srt_text.strip().split("\n")

            # 各エントリは 3行 + 空行
            # 最初のエントリ: "1\n00:00:00,000 --> ...\n[phase] Preparation\n"
            assert lines[0] == "1"
            assert "-->" in lines[1]
            assert lines[2].startswith("[phase] ")
            # SRT時刻フォーマット（カンマ区切り）
            assert "," in lines[1]

    def test_short_phase_merge(self):
        """短いフェーズがマージされることを確認"""
        from src.cholec_phase.detector import annotate_phases
        from src.core.time_utils import format_srt_time

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "test_cholecphaselog.csv")
            prob_headers = [f"prob_{name}" for name in CHOLEC80_PHASES]

            # Preparation(50s) → 短いClipping(3s) → Preparation(47s)
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["t_sec", "t_srt", "phase_id", "phase_name"]
                    + prob_headers
                    + ["confidence", "reader"]
                )
                for i in range(100):
                    t = float(i)
                    if 50 <= i < 53:
                        pid, pname = 2, "ClippingCutting"
                    else:
                        pid, pname = 0, "Preparation"
                    probs = [0.0] * 7
                    probs[pid] = 0.9
                    row = [f"{t:.3f}", format_srt_time(t), str(pid), pname]
                    for p in probs:
                        row.append(f"{p:.6f}")
                    row.append("0.900000")
                    row.append("pyav")
                    writer.writerow(row)

            result = annotate_phases(
                csv_path, tmpdir, min_phase_s=10.0, smooth_s=0.0,
            )

            # 3秒のClippingは min_phase_s=10 でマージされる
            assert result["phases"] == 1


# ---------------------------------------------------------------------------
# タグ登録テスト
# ---------------------------------------------------------------------------


class TestTagRegistration:
    def test_jsonl_to_srt_template(self):
        from src.tools.jsonl_to_srt import TAG_TEMPLATES

        assert "surgical_phase" in TAG_TEMPLATES
        template = TAG_TEMPLATES["surgical_phase"]
        assert "[phase]" in template
        assert "{phase_name}" in template

    def test_srt_to_jsonl_pattern(self):
        from src.tools.srt_to_jsonl import TAG_PATTERNS

        found = False
        for pattern, event_type in TAG_PATTERNS.items():
            if event_type == "surgical_phase":
                found = True
                assert pattern.match("[phase] Preparation")
                break
        assert found, "surgical_phase not found in TAG_PATTERNS"

    def test_tag_template_format(self):
        from src.tools.jsonl_to_srt import _build_tag_line

        event = {
            "type": "surgical_phase",
            "phase_name": "CalotTriangleDissection",
        }
        tag = _build_tag_line(event)
        assert tag == "[phase] CalotTriangleDissection"

    def test_srt_to_jsonl_phase_name_extraction(self):
        from src.tools.srt_to_jsonl import read_srt_to_events

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".srt", delete=False, encoding="utf-8"
        ) as f:
            f.write("1\n")
            f.write("00:00:00,000 --> 00:01:00,000\n")
            f.write("[phase] GallbladderDissection\n")
            f.write("\n")
            srt_path = f.name

        try:
            events = read_srt_to_events(srt_path)
            assert len(events) == 1
            assert events[0]["type"] == "surgical_phase"
            assert events[0]["phase_name"] == "GallbladderDissection"
        finally:
            os.unlink(srt_path)


# ---------------------------------------------------------------------------
# フェーズ平滑化テスト
# ---------------------------------------------------------------------------


class TestPhaseSmoothing:
    def test_mode_smooth(self):
        from src.cholec_phase.detector import _mode_smooth_phases

        # 1フレームだけノイズがある場合
        ids = [0] * 10 + [1] + [0] * 10
        smoothed = _mode_smooth_phases(ids, fps=1.0, smooth_s=5.0)
        # ノイズ (1) が最頻値（多数決）で 0 になる
        assert smoothed[10] == 0

    def test_no_smooth(self):
        from src.cholec_phase.detector import _mode_smooth_phases

        ids = [0, 1, 2, 3, 4]
        smoothed = _mode_smooth_phases(ids, fps=1.0, smooth_s=0.0)
        assert smoothed == ids


class TestMergeShortPhases:
    def test_leading_short_merges_forward(self):
        from src.cholec_phase.detector import _merge_short_phases
        # 先頭の短い区間 [0] が次のフェーズ [1] にマージされる
        ids = [0, 1, 1, 1, 1, 1, 1]
        times = [0, 1, 2, 3, 4, 5, 6]
        out = _merge_short_phases(ids, times, min_phase_s=2.0)
        assert out == [1, 1, 1, 1, 1, 1, 1]

    def test_middle_short_merges_backward(self):
        from src.cholec_phase.detector import _merge_short_phases
        ids = [0, 0, 0, 2, 1, 1, 1]
        times = [0, 1, 2, 3, 4, 5, 6]
        out = _merge_short_phases(ids, times, min_phase_s=2.0)
        assert out == [0, 0, 0, 0, 1, 1, 1]

    def test_no_short_intervals_unchanged(self):
        from src.cholec_phase.detector import _merge_short_phases
        ids = [0, 0, 0, 1, 1, 1]
        times = [0, 1, 2, 3, 4, 5]
        out = _merge_short_phases(ids, times, min_phase_s=2.0)
        assert out == ids


class TestReadCsvDefensive:
    def _write(self, tmp, rows, header=None):
        p = Path(tmp) / "x_cholecphaselog.csv"
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if header is None:
                header = (["t_sec", "t_srt", "phase_id", "phase_name",
                           "confidence"]
                          + [f"prob_{n}" for n in CHOLEC80_PHASES])
            w.writerow(header)
            for r in rows:
                w.writerow(r)
        return str(p)

    def _good_row(self):
        return ([0.0, "00:00:00,000", 0, CHOLEC80_PHASES[0], 0.9]
                + [0.1] * NUM_PHASES)

    def _bad_row(self):
        return ["BAD", "x", "", "", ""] + [""] * NUM_PHASES

    def test_skips_invalid_rows(self):
        from src.cholec_phase.detector import read_cholecphaselog_csv
        with tempfile.TemporaryDirectory() as d:
            p = self._write(d, [self._bad_row(), self._good_row()])
            data = read_cholecphaselog_csv(p)
            assert len(data["times"]) == 1

    def test_missing_required_column_raises(self):
        from src.cholec_phase.detector import read_cholecphaselog_csv
        with tempfile.TemporaryDirectory() as d:
            p = self._write(d, [], header=["t_sec", "foo"])
            with pytest.raises(ValueError):
                read_cholecphaselog_csv(p)

    def test_all_invalid_raises(self):
        from src.cholec_phase.detector import read_cholecphaselog_csv
        with tempfile.TemporaryDirectory() as d:
            p = self._write(d, [self._bad_row()])
            with pytest.raises(ValueError):
                read_cholecphaselog_csv(p)


class TestReadPhaseAnnotation:
    def _write(self, tmp, text):
        p = Path(tmp) / "video01-phase.txt"
        p.write_text(text, encoding="utf-8")
        return str(p)

    def test_with_header(self):
        from src.cholec_phase.dataset import read_phase_annotation
        p0, p1 = CHOLEC80_PHASES[0], CHOLEC80_PHASES[1]
        with tempfile.TemporaryDirectory() as d:
            fi, pi = read_phase_annotation(
                self._write(d, f"Frame\tPhase\n0\t{p0}\n25\t{p1}\n"))
            assert fi == [0, 25] and pi == [0, 1]

    def test_without_header_keeps_first_row(self):
        # 1行目が数値（ヘッダー無し）→ 最初の有効行を取りこぼさない
        from src.cholec_phase.dataset import read_phase_annotation
        p0, p1 = CHOLEC80_PHASES[0], CHOLEC80_PHASES[1]
        with tempfile.TemporaryDirectory() as d:
            fi, pi = read_phase_annotation(
                self._write(d, f"0\t{p0}\n25\t{p1}\n"))
            assert fi == [0, 25] and pi == [0, 1]

    def test_unknown_phase_skipped(self):
        from src.cholec_phase.dataset import read_phase_annotation
        p0, p1 = CHOLEC80_PHASES[0], CHOLEC80_PHASES[1]
        with tempfile.TemporaryDirectory() as d:
            fi, pi = read_phase_annotation(
                self._write(d, f"Frame\tPhase\n0\t{p0}\n25\tTypo\n50\t{p1}\n"))
            assert fi == [0, 50] and pi == [0, 1]

    def test_empty_file(self):
        from src.cholec_phase.dataset import read_phase_annotation
        with tempfile.TemporaryDirectory() as d:
            fi, pi = read_phase_annotation(self._write(d, ""))
            assert fi == [] and pi == []


class TestFeatureLabelLengthCheck:
    def test_mismatch_raises(self):
        from src.cholec_phase.dataset import PhaseFeatureDataset
        with tempfile.TemporaryDirectory() as d:
            np.save(Path(d) / "video01_features.npy",
                    np.zeros((10, 2048), np.float32))
            np.save(Path(d) / "video01_labels.npy", np.zeros((8,), np.int64))
            with pytest.raises(ValueError):
                PhaseFeatureDataset(d, [1], seq_len=4)

    def test_matched_ok(self):
        from src.cholec_phase.dataset import PhaseFeatureDataset
        with tempfile.TemporaryDirectory() as d:
            np.save(Path(d) / "video01_features.npy",
                    np.zeros((10, 2048), np.float32))
            np.save(Path(d) / "video01_labels.npy", np.zeros((10,), np.int64))
            ds = PhaseFeatureDataset(d, [1], seq_len=4)
            assert len(ds) > 0

    def test_labels_cached_not_reread(self):
        # __getitem__ がラベルファイルを再読み込みしない（__init__ でキャッシュ）
        from src.cholec_phase import dataset as DS
        with tempfile.TemporaryDirectory() as d:
            np.save(Path(d) / "video01_features.npy",
                    np.arange(10 * 4, dtype=np.float32).reshape(10, 4))
            np.save(Path(d) / "video01_labels.npy", np.arange(10, dtype=np.int64))
            ds = DS.PhaseFeatureDataset(d, [1], seq_len=4, feature_dim=4)
            assert "video01" in ds._labels  # キャッシュ済み
            calls = {"n": 0}
            real_load = DS.np.load

            def counting(path, *a, **k):
                if "labels" in str(path):
                    calls["n"] += 1
                return real_load(path, *a, **k)

            DS.np.load = counting
            try:
                for i in range(len(ds)):
                    feats, labels, length = ds[i]
                    assert labels.shape[0] == ds.seq_len
            finally:
                DS.np.load = real_load
            assert calls["n"] == 0  # ラベルの再読み込みゼロ


# ---------------------------------------------------------------------------
# Collate function テスト
# ---------------------------------------------------------------------------


class TestCollate:
    def test_collate_fn(self):
        from src.cholec_phase.train import collate_fn

        # 2件目は有効長20、残り10フレームは Dataset が -1 でパディング済み
        # （ignore_index=-1 前提）。collate_fn がこのパディングを保持することも検証。
        labels0 = torch.randint(0, 7, (30,))
        labels1 = torch.randint(0, 7, (30,))
        labels1[20:] = -1
        batch = [
            (torch.randn(30, 2048), labels0, 30),
            (torch.randn(30, 2048), labels1, 20),
        ]
        features, labels, lengths = collate_fn(batch)
        assert features.shape == (2, 30, 2048)
        assert labels.shape == (2, 30)
        assert lengths.tolist() == [30, 20]
        # パディング部（-1）が stack 後も保持され、ignore_index=-1 と整合すること
        assert (labels[1, 20:] == -1).all()
        # 有効区間は元のラベルがそのまま保持されること
        assert torch.equal(labels[0], labels0)
        assert torch.equal(labels[1, :20], labels1[:20])
