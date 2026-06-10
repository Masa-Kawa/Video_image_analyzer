"""
Cholec80 データセットローダー

Cholec80データセットのディレクトリ構造:
  cholec80/
    videos/
      video01.mp4 ... video80.mp4
    phase_annotations/
      video01-phase.txt ... video80-phase.txt
    tool_annotations/
      video01-tool.txt ... video80-tool.txt

フェーズアノテーション形式 (TSV):
  Frame\tPhase
  0\tPreparation
  25\tPreparation
  50\tPreparation
  ...
  12500\tCalotTriangleDissection

フレームインデックスは25fps基準（0-indexed、25フレーム刻み）。

このモジュールは2つのモードで動作する:

1. FeatureDataset: 事前抽出済み特徴量(.npy) + アノテーションから学習用データセットを構築
2. Cholec80VideoDataset: 動画ファイル + アノテーションからオンラインで特徴抽出

学習効率のため、モード1（事前抽出）を推奨する。
"""

import bisect
import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.cholec_phase import CHOLEC80_PHASES, NUM_PHASES

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# フェーズアノテーション読み込み
# ---------------------------------------------------------------------------

PHASE_TO_ID: Dict[str, int] = {name: i for i, name in enumerate(CHOLEC80_PHASES)}


def read_phase_annotation(ann_path: str) -> Tuple[List[int], List[int]]:
    """
    Cholec80 フェーズアノテーションファイルを読み込む。

    Args:
        ann_path: アノテーションファイルパス (video01-phase.txt)

    Returns:
        (frame_indices, phase_ids)
        frame_indices: フレーム番号のリスト (0, 25, 50, ...)
        phase_ids: フェーズIDのリスト (0-6)
    """
    frame_indices: List[int] = []
    phase_ids: List[int] = []
    unknown_phases: Dict[str, int] = {}

    with open(ann_path, "r", encoding="utf-8") as f:
        rows = list(csv.reader(f, delimiter="\t"))

    if not rows:
        return [], []

    # ヘッダー判定: 1行目の先頭セルが数値ならヘッダー無し（その行もデータ扱い）。
    # 無条件 skip だとヘッダー無しファイルの最初の有効行を取りこぼすため。
    first_cell = rows[0][0].strip() if rows[0] else ""
    start = 0 if first_cell.lstrip("-").isdigit() else 1

    for row in rows[start:]:
        if len(row) < 2:
            continue
        try:
            frame_idx = int(row[0].strip())
        except ValueError:
            continue  # フレーム番号が数値でない行は無視
        phase_name = row[1].strip()
        phase_id = PHASE_TO_ID.get(phase_name, -1)
        if phase_id < 0:
            # 未知フェーズ名はデータ欠損の原因になるため記録して警告する
            unknown_phases[phase_name] = unknown_phases.get(phase_name, 0) + 1
            continue
        frame_indices.append(frame_idx)
        phase_ids.append(phase_id)

    if unknown_phases:
        logger.warning(
            "未知のフェーズ名を %d 種スキップしました (%s): %s",
            len(unknown_phases),
            ann_path,
            ", ".join(f"{name}×{cnt}" for name, cnt in unknown_phases.items()),
        )

    return frame_indices, phase_ids


def frame_annotation_to_second(
    frame_indices: List[int],
    phase_ids: List[int],
    video_fps: float = 25.0,
    sample_fps: float = 1.0,
) -> Tuple[List[float], List[int]]:
    """
    フレームレベルのアノテーションをサンプリングFPSに変換する。

    Cholec80 は 25fps だが、推論時は 1fps でサンプリングするため、
    各サンプリング時点のフェーズラベルを返す。

    Args:
        frame_indices: フレーム番号リスト
        phase_ids: フェーズIDリスト
        video_fps: 動画のFPS（Cholec80 = 25）
        sample_fps: サンプリングFPS

    Returns:
        (times_sec, sampled_phase_ids)

    Raises:
        ValueError: video_fps / sample_fps が 0 以下の場合（ゼロ除算・無限ループ防止）
    """
    if video_fps <= 0:
        raise ValueError(f"video_fps must be positive, got {video_fps}")
    if sample_fps <= 0:
        raise ValueError(f"sample_fps must be positive, got {sample_fps}")

    if not frame_indices:
        return [], []

    max_frame = frame_indices[-1]
    max_sec = max_frame / video_fps
    sample_interval = 1.0 / sample_fps

    # フレーム→フェーズのルックアップ。frame_indices は昇順前提なので、
    # 最寄り検索は線形走査(O(A))ではなく二分探索(O(log A))で行う（全体 O(T log A)）。
    sorted_frames = sorted(frame_indices)
    frame_to_phase: Dict[int, int] = {fi: pid
                                      for fi, pid in zip(frame_indices, phase_ids)}

    def _nearest_phase(target: int) -> int:
        pos = bisect.bisect_left(sorted_frames, target)
        if pos == 0:
            best = sorted_frames[0]
        elif pos == len(sorted_frames):
            best = sorted_frames[-1]
        else:
            lo, hi = sorted_frames[pos - 1], sorted_frames[pos]
            best = lo if (target - lo) <= (hi - target) else hi
        return frame_to_phase[best]

    times: List[float] = []
    sampled_ids: List[int] = []
    t = 0.0

    while t <= max_sec:
        target_frame = int(round(t * video_fps))
        sampled_ids.append(_nearest_phase(target_frame))
        times.append(round(t, 3))
        t += sample_interval

    return times, sampled_ids


# ---------------------------------------------------------------------------
# 事前抽出済み特徴量を使うデータセット
# ---------------------------------------------------------------------------


class PhaseFeatureDataset(Dataset):
    """
    事前抽出済み特徴量 + フェーズアノテーションのデータセット。

    各サンプルは (feature_sequence, label_sequence) のタプル。
    シーケンスは seq_len フレームに分割される。

    使い方:
        # 特徴量の事前抽出
        python -m src.cholec_phase.train extract-features \\
            --cholec80-dir /path/to/cholec80 --outdir /path/to/features

        # データセット作成
        ds = PhaseFeatureDataset(
            feature_dir="/path/to/features",
            video_ids=[1, 2, ..., 40],
            seq_len=300,
        )
    """

    def __init__(
        self,
        feature_dir: str,
        video_ids: List[int],
        seq_len: int = 300,
        stride: int = 150,
        feature_dim: int = 2048,
    ):
        """
        Args:
            feature_dir: 特徴量ディレクトリ
                video{id:02d}_features.npy, video{id:02d}_labels.npy が必要
            video_ids: 使用する動画ID（1-80）
            seq_len: 1サンプルのシーケンス長（フレーム数）
            stride: スライディングウィンドウのストライド
            feature_dim: 特徴次元
        """
        self.feature_dir = Path(feature_dir)
        self.seq_len = seq_len
        self.feature_dim = feature_dim

        # 各動画をスライディングウィンドウで分割
        self.samples: List[Tuple[str, int, int]] = []  # (video_id_str, start, end)
        # ラベルは小さいので __init__ で一度だけ読み、__getitem__ の毎回 I/O を避ける
        self._labels: Dict[str, np.ndarray] = {}

        for vid in video_ids:
            vid_str = f"video{vid:02d}"
            feat_path = self.feature_dir / f"{vid_str}_features.npy"
            label_path = self.feature_dir / f"{vid_str}_labels.npy"

            if not feat_path.exists() or not label_path.exists():
                continue

            labels = np.load(str(label_path))
            n_frames = len(labels)

            # 特徴量とラベルの長さ一致を検証（mmap でヘッダのみ参照、全読み込み不要）。
            # 不一致のまま __getitem__ でスライス/パディングすると形状崩れで実行時に
            # 落ちるため、構築時に早期エラーで通知する。
            feat_len = int(np.load(str(feat_path), mmap_mode="r").shape[0])
            if feat_len != n_frames:
                raise ValueError(
                    f"{vid_str}: 特徴量({feat_len}フレーム)とラベル"
                    f"({n_frames}フレーム)の長さが一致しません。"
                    f"特徴抽出をやり直してください: {feat_path}"
                )

            self._labels[vid_str] = labels
            start = 0
            while start < n_frames:
                end = min(start + seq_len, n_frames)
                self.samples.append((vid_str, start, end))
                if end == n_frames:
                    break
                start += stride

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Returns:
            features: (seq_len, feature_dim) パディング済み
            labels: (seq_len,) パディング済み（-1でパディング）
            length: 実際のシーケンス長
        """
        vid_str, start, end = self.samples[idx]
        actual_len = end - start

        features = np.load(
            str(self.feature_dir / f"{vid_str}_features.npy"),
            mmap_mode="r",
        )[start:end]
        # ラベルは __init__ でキャッシュ済み（毎回のディスク読み込みを回避）
        labels = self._labels[vid_str][start:end]

        # パディング
        if actual_len < self.seq_len:
            pad_len = self.seq_len - actual_len
            features = np.pad(features, ((0, pad_len), (0, 0)), mode="constant")
            labels = np.pad(labels, (0, pad_len), constant_values=-1)

        return (
            torch.from_numpy(features.copy()).float(),
            torch.from_numpy(labels.copy()).long(),
            actual_len,
        )


# ---------------------------------------------------------------------------
# Cholec80 ディレクトリ探索
# ---------------------------------------------------------------------------


def discover_cholec80(cholec80_dir: str) -> Dict[str, dict]:
    """
    Cholec80 ディレクトリから利用可能なデータを探索する。

    Returns:
        {video_id_str: {"video": path, "phase_ann": path, "tool_ann": path}}
    """
    root = Path(cholec80_dir)
    result: Dict[str, dict] = {}

    # 動画ディレクトリ
    video_dir = root / "videos"
    ann_dir = root / "phase_annotations"

    if not video_dir.exists():
        # フラットな構造も試す
        video_dir = root
        ann_dir = root

    for vid_num in range(1, 81):
        vid_str = f"video{vid_num:02d}"
        info: dict = {}

        # 動画ファイル
        for ext in [".mp4", ".avi", ".mkv"]:
            vpath = video_dir / f"{vid_str}{ext}"
            if vpath.exists():
                info["video"] = str(vpath)
                break

        # フェーズアノテーション
        ann_path = ann_dir / f"{vid_str}-phase.txt"
        if ann_path.exists():
            info["phase_ann"] = str(ann_path)

        # ツールアノテーション
        tool_path = (root / "tool_annotations" / f"{vid_str}-tool.txt")
        if tool_path.exists():
            info["tool_ann"] = str(tool_path)

        if info:
            result[vid_str] = info

    return result


def get_train_val_split(
    n_videos: int = 80,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """
    Cholec80 の標準的な train/val 分割を返す。

    Cholec80 論文の慣例: video01-40 = train, video41-80 = test
    ここではさらに train を train/val に分割する。

    Returns:
        (train_ids, val_ids) 1-indexed
    """
    train_pool = list(range(1, 41))  # video01-40
    rng = np.random.RandomState(seed)
    rng.shuffle(train_pool)

    n_val = max(1, int(len(train_pool) * val_ratio))
    val_ids = sorted(train_pool[:n_val])
    train_ids = sorted(train_pool[n_val:])

    return train_ids, val_ids


def get_test_ids() -> List[int]:
    """Cholec80 標準テストセット（video41-80）"""
    return list(range(41, 81))
