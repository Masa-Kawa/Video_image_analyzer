"""
ConvLSTM Autoencoder 学習スクリプト

正常フレームのみで学習する Self-Supervised 学習。
次フレーム予測の誤差を最小化する。

使用例:
  # 正常動画から学習
  python -m src.anomaly.train \
      --videos normal_video1.mp4 normal_video2.mp4 \
      --outdir models/ \
      --epochs 50

  # 学習済みモデルで追加学習
  python -m src.anomaly.train \
      --videos new_normal.mp4 \
      --outdir models/ \
      --resume models/anomaly_convlstm.pth \
      --epochs 20
"""

import argparse
import random
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

from src.anomaly.models import (
    ConvLSTMAutoencoder,
    ANOMALY_FRAME_SIZE,
    preprocess_frame_anomaly,
)
from src.red.redlog import iter_frames


def _validate_video_paths(video_paths: List[str]) -> List[str]:
    """存在し読み込み可能な動画パスのみを返す。

    欠落/非ファイルは警告して除外し、有効なパスが1つも無ければ
    明確なエラーメッセージとともに FileNotFoundError を送出する。
    （iter_frames に存在しないパスを渡して生のスタックトレースで落ちるのを防ぐ）
    """
    valid: List[str] = []
    for vp in video_paths:
        p = Path(vp).expanduser()
        if p.is_file():
            valid.append(str(p))
        else:
            print(f"警告: 動画が見つからないためスキップ: {vp}", file=sys.stderr)
    if not valid:
        raise FileNotFoundError(
            "有効な動画ファイルがありません（全パスが存在しないか非ファイル）: "
            + ", ".join(video_paths)
        )
    return valid


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class NormalVideoDataset(Dataset):
    """
    正常動画からフレームシーケンスを抽出するデータセット。

    T+1 フレームを取り出し、最初の T フレームを入力、
    最後の 1 フレームをターゲット（次フレーム予測）とする。
    """

    def __init__(
        self,
        video_paths: List[str],
        seq_len: int = 4,
        fps: float = 5.0,
        frame_size: int = ANOMALY_FRAME_SIZE,
        max_frames_per_video: int = 5000,
    ):
        self.seq_len = seq_len
        self.frame_size = frame_size
        self.frames: List[torch.Tensor] = []
        # 各動画の (開始オフセット, 終了オフセット) を記録し、
        # シーケンスが動画境界を跨がないようにする。
        self._spans: List[Tuple[int, int]] = []

        print(f"正常フレームを読み込み中...")
        for vp in video_paths:
            count = 0
            start = len(self.frames)
            print(f"  {vp}")
            for t_sec, bgr, reader in iter_frames(vp, fps):
                tensor = preprocess_frame_anomaly(bgr, frame_size)
                self.frames.append(tensor)
                count += 1
                if count >= max_frames_per_video:
                    break
            self._spans.append((start, len(self.frames)))
            print(f"    {count} フレーム")

        print(f"合計: {len(self.frames)} フレーム")

        # 同一動画内で T+1 フレームが取れる開始インデックスのみを採用。
        # これにより動画Aの末尾とBの先頭が連続シーケンスになる不整合を防ぐ。
        self.valid_indices: List[int] = []
        for start, end in self._spans:
            # idx .. idx+seq_len（計 seq_len+1 フレーム）が end 未満に収まる範囲
            for idx in range(start, end - seq_len):
                self.valid_indices.append(idx)

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            seq: (T, 3, H, W) 入力シーケンス
            target: (3, H, W) 次フレーム（予測対象）
        """
        base = self.valid_indices[idx]
        seq = torch.stack(self.frames[base : base + self.seq_len])
        target = self.frames[base + self.seq_len]
        return seq, target

    def split_indices(self, val_fraction: float) -> Tuple[List[int], List[int]]:
        """train/val のサンプル位置（valid_indices へのインデックス）を返す。

        動画ごとに末尾 ``val_fraction`` を検証用に確保し、訓練側との境界には
        ``seq_len`` 個のギャップを設けてシーケンスの重複（リーク）を防ぐ。
        ``val_fraction <= 0`` の場合は全件を訓練に割り当てる（従来挙動）。
        """
        n_total = len(self.valid_indices)
        if val_fraction <= 0.0:
            return list(range(n_total)), []

        train: List[int] = []
        val: List[int] = []
        pos = 0
        for start, end in self._spans:
            n = max(0, (end - start) - self.seq_len)
            if n == 0:
                continue
            n_val = int(round(n * val_fraction))
            if n_val <= 0:
                train.extend(range(pos, pos + n))
            else:
                cut = n - n_val
                gap_start = max(0, cut - self.seq_len)
                train.extend(range(pos, pos + gap_start))
                val.extend(range(pos + cut, pos + n))
            pos += n
        return train, val


# ---------------------------------------------------------------------------
# 学習ループ
# ---------------------------------------------------------------------------


def train_model(
    video_paths: List[str],
    outdir: str,
    epochs: int = 50,
    batch_size: int = 8,
    lr: float = 1e-3,
    seq_len: int = 4,
    fps: float = 5.0,
    frame_size: int = ANOMALY_FRAME_SIZE,
    feature_dim: int = 64,
    hidden_dims: Optional[List[int]] = None,
    device: str = "cuda",
    resume_path: Optional[str] = None,
    max_frames_per_video: int = 5000,
    num_workers: int = 0,
    val_fraction: float = 0.0,
    patience: int = 10,
) -> str:
    """
    ConvLSTM Autoencoder を正常フレームで学習する。

    Args:
        video_paths: 正常動画ファイルパスのリスト
        outdir: モデル出力ディレクトリ
        epochs: エポック数
        batch_size: バッチサイズ
        lr: 学習率
        seq_len: 入力シーケンス長
        fps: フレームサンプリングFPS
        frame_size: フレームリサイズサイズ
        feature_dim: 空間特徴次元数
        hidden_dims: ConvLSTM隠れ次元リスト
        device: 計算デバイス
        resume_path: 学習再開用の重みパス
        max_frames_per_video: 動画あたり最大フレーム数
        num_workers: DataLoader のワーカー数
        val_fraction: 検証に回す割合（>0 で動画末尾を hold-out、汎化監視＋Early Stopping）
        patience: val_loss が改善しないまま許容するエポック数（Early Stopping）

    Returns:
        保存した重みファイルのパス
    """
    if hidden_dims is None:
        hidden_dims = [64, 64]

    video_paths = _validate_video_paths(video_paths)

    dev = torch.device(
        device if torch.cuda.is_available() and device != "cpu" else "cpu"
    )
    print(f"デバイス: {dev}")

    # データセット
    dataset = NormalVideoDataset(
        video_paths=video_paths,
        seq_len=seq_len,
        fps=fps,
        frame_size=frame_size,
        max_frames_per_video=max_frames_per_video,
    )

    if len(dataset) == 0:
        print("エラー: 学習データが不足しています。", file=sys.stderr)
        return ""

    # 動画単位の hold-out で train/val を分割（汎化性能の監視用）
    train_idx, val_idx = dataset.split_indices(val_fraction)
    train_set = Subset(dataset, train_idx) if val_idx else dataset
    loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(dev.type == "cuda"),
    )
    val_loader = None
    if val_idx:
        val_loader = DataLoader(
            Subset(dataset, val_idx),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(dev.type == "cuda"),
        )
        print(f"train/val 分割: train={len(train_idx)}, val={len(val_idx)}")

    # モデル
    model = ConvLSTMAutoencoder(
        seq_len=seq_len,
        feature_dim=feature_dim,
        hidden_dims=hidden_dims,
        frame_size=frame_size,
    )

    if resume_path and Path(resume_path).exists():
        # weights_only=True: 信頼できないチェックポイント経由の pickle 任意コード
        # 実行（ACE）を防ぐ。保存しているのは state_dict（テンソルのみ）なので互換。
        state = torch.load(resume_path, map_location=dev, weights_only=True)
        model.load_state_dict(state)
        print(f"重み読み込み: {resume_path}")

    model.to(dev)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(dev.type == "cuda"))

    # 出力ディレクトリ
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    weights_path = out_path / "anomaly_convlstm.pth"

    print(f"学習開始: {epochs} エポック, {len(dataset)} サンプル")
    print(f"モデル: ConvLSTM AE (feature={feature_dim}, hidden={hidden_dims})")

    best_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for seq, target in loader:
            seq = seq.to(dev)
            target = target.to(dev)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(dev.type == "cuda")):
                predicted = model(seq)
                loss = nn.functional.mse_loss(predicted, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        avg_loss = epoch_loss / max(1, n_batches)
        elapsed = time.time() - t0
        lr_now = scheduler.get_last_lr()[0]

        # 検証損失（あれば）を評価し、ベスト保存の指標に使う
        val_loss = None
        if val_loader is not None:
            val_loss = _eval_loss(model, val_loader, dev)
        monitored = val_loss if val_loss is not None else avg_loss

        msg = f"  Epoch {epoch:3d}/{epochs}: loss={avg_loss:.6f}"
        if val_loss is not None:
            msg += f"  val_loss={val_loss:.6f}"
        msg += f"  lr={lr_now:.2e}  ({elapsed:.1f}s)"
        print(msg)

        # 監視指標（val があれば val_loss）でベストモデルを保存
        if monitored < best_loss:
            best_loss = monitored
            epochs_no_improve = 0
            torch.save(model.state_dict(), weights_path)
        else:
            epochs_no_improve += 1
            # Early Stopping: val が有効なときのみ作動させる
            if val_loader is not None and epochs_no_improve >= patience:
                print(f"Early Stopping: {patience} エポック val 改善なし "
                      f"(best={best_loss:.6f})")
                break

    # 最終モデルも保存
    final_path = out_path / "anomaly_convlstm_final.pth"
    torch.save(model.state_dict(), final_path)
    print(f"学習完了: best={'val_loss' if val_loader else 'loss'}={best_loss:.6f}")
    print(f"ベストモデル: {weights_path}")
    print(f"最終モデル  : {final_path}")

    return str(weights_path)


def _eval_loss(model: nn.Module, loader: DataLoader, dev: torch.device) -> float:
    """検証セットの平均 MSE を返す（推論モード・勾配なし）。"""
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for seq, target in loader:
            seq = seq.to(dev)
            target = target.to(dev)
            with torch.cuda.amp.autocast(enabled=(dev.type == "cuda")):
                pred = model(seq)
                loss = nn.functional.mse_loss(pred, target)
            total += loss.item()
            n += 1
    return total / max(1, n)


# ---------------------------------------------------------------------------
# 正常度キャリブレーション
# ---------------------------------------------------------------------------


def calibrate_threshold(
    video_paths: List[str],
    weights_path: str,
    percentile: float = 95.0,
    fps: float = 5.0,
    seq_len: int = 4,
    frame_size: int = ANOMALY_FRAME_SIZE,
    feature_dim: int = 64,
    hidden_dims: Optional[List[int]] = None,
    device: str = "cuda",
    max_frames_per_video: int = 2000,
) -> float:
    """
    正常動画の異常スコア分布から閾値を自動決定する。

    指定パーセンタイルの値を閾値として返す。
    これを超えるスコアのフレーム = 異常。

    Args:
        video_paths: 正常動画パスリスト
        weights_path: 学習済み重みパス
        percentile: 閾値パーセンタイル（デフォルト: 95）

    Returns:
        推奨閾値
    """
    if hidden_dims is None:
        hidden_dims = [64, 64]

    video_paths = _validate_video_paths(video_paths)

    from src.anomaly.models import AnomalyModelManager

    mgr = AnomalyModelManager(
        device=device,
        weights_path=weights_path,
        seq_len=seq_len,
        feature_dim=feature_dim,
        hidden_dims=hidden_dims,
        frame_size=frame_size,
    )

    if not mgr.is_ready:
        print("警告: モデルが読み込めません。デフォルト閾値を使用。", file=sys.stderr)
        return 0.01

    # 推論専用。Dropout/BatchNorm を固定し、勾配グラフ構築による GPU メモリ
    # リーク/OOM を防ぐ（predict_and_score 内も no_grad だが念のため明示）。
    if mgr.model is not None:
        mgr.model.eval()

    scores: List[float] = []
    frame_buffer: List[torch.Tensor] = []

    with torch.no_grad():
        for vp in video_paths:
            print(f"キャリブレーション: {vp}")
            frame_count = 0
            frame_buffer.clear()

            for t_sec, bgr, reader in iter_frames(vp, fps):
                tensor = preprocess_frame_anomaly(bgr, frame_size)
                frame_buffer.append(tensor)

                if len(frame_buffer) > seq_len:
                    seq = frame_buffer[-seq_len - 1 : -1]
                    target = frame_buffer[-1]
                    score, _ = mgr.predict_and_score(seq, target)
                    scores.append(score)

                frame_count += 1
                if frame_count >= max_frames_per_video:
                    break

    if not scores:
        print("警告: スコアが計算できません。デフォルト閾値を使用。", file=sys.stderr)
        return 0.01

    threshold = float(np.percentile(scores, percentile))
    print(f"キャリブレーション結果:")
    print(f"  サンプル数: {len(scores)}")
    print(f"  mean={np.mean(scores):.6f}, std={np.std(scores):.6f}")
    print(f"  p50={np.percentile(scores, 50):.6f}, "
          f"p90={np.percentile(scores, 90):.6f}, "
          f"p95={np.percentile(scores, 95):.6f}, "
          f"p99={np.percentile(scores, 99):.6f}")
    print(f"  推奨閾値 (p{percentile:.0f}): {threshold:.6f}")

    return threshold


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Self-Supervised Anomaly Detection - 学習スクリプト",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 正常動画で学習
  python -m src.anomaly.train train \\
      --videos normal1.mp4 normal2.mp4 --outdir models/ --epochs 50

  # 閾値キャリブレーション
  python -m src.anomaly.train calibrate \\
      --videos normal1.mp4 --weights models/anomaly_convlstm.pth
        """,
    )
    subparsers = parser.add_subparsers(dest="command")

    # train/calibrate で一致させるべきモデル構造パラメータ（親パーサーで共有）。
    # 異なるアーキで学習したモデルを誤ってキャリブレーションするのを防ぐため、
    # 両サブコマンドで同一の引数・デフォルトを公開する。
    arch = argparse.ArgumentParser(add_help=False)
    arch.add_argument("--seq-len", type=int, default=4, help="シーケンス長（デフォルト: 4）")
    arch.add_argument("--frame-size", type=int, default=ANOMALY_FRAME_SIZE,
                      help=f"フレームサイズ（デフォルト: {ANOMALY_FRAME_SIZE}）")
    arch.add_argument("--feature-dim", type=int, default=64,
                      help="空間特徴次元数（デフォルト: 64）")
    arch.add_argument("--hidden-dims", type=int, nargs="+", default=[64, 64],
                      help="ConvLSTM隠れ次元リスト（デフォルト: 64 64）")
    arch.add_argument("--fps", type=float, default=5.0, help="サンプリングFPS")
    arch.add_argument("--device", default="cuda", help="計算デバイス")

    # --- train ---
    tr = subparsers.add_parser("train", parents=[arch], help="正常動画で学習")
    tr.add_argument("--videos", nargs="+", required=True, help="正常動画ファイルパス")
    tr.add_argument("--outdir", required=True, help="モデル出力ディレクトリ")
    tr.add_argument("--epochs", type=int, default=50, help="エポック数（デフォルト: 50）")
    tr.add_argument("--batch-size", type=int, default=8, help="バッチサイズ（デフォルト: 8）")
    tr.add_argument("--lr", type=float, default=1e-3, help="学習率（デフォルト: 1e-3）")
    tr.add_argument("--resume", default=None, help="学習再開用の重みパス")
    tr.add_argument("--max-frames", type=int, default=5000,
                     help="動画あたり最大フレーム数（デフォルト: 5000）")
    tr.add_argument("--val-fraction", type=float, default=0.0,
                     help="検証 hold-out 割合（>0 で汎化監視＋Early Stopping、デフォルト: 0）")
    tr.add_argument("--patience", type=int, default=10,
                     help="Early Stopping の許容エポック数（デフォルト: 10）")

    # --- calibrate ---
    cal = subparsers.add_parser("calibrate", parents=[arch],
                                help="正常動画で閾値キャリブレーション")
    cal.add_argument("--videos", nargs="+", required=True, help="正常動画ファイルパス")
    cal.add_argument("--weights", required=True, help="学習済み重みパス")
    cal.add_argument("--percentile", type=float, default=95.0,
                      help="閾値パーセンタイル（デフォルト: 95）")
    cal.add_argument("--max-frames", type=int, default=2000,
                      help="動画あたり最大フレーム数")

    args = parser.parse_args()

    if args.command == "train":
        train_model(
            video_paths=args.videos,
            outdir=args.outdir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seq_len=args.seq_len,
            fps=args.fps,
            frame_size=args.frame_size,
            feature_dim=args.feature_dim,
            hidden_dims=args.hidden_dims,
            device=args.device,
            resume_path=args.resume,
            max_frames_per_video=args.max_frames,
            val_fraction=args.val_fraction,
            patience=args.patience,
        )
    elif args.command == "calibrate":
        calibrate_threshold(
            video_paths=args.videos,
            weights_path=args.weights,
            percentile=args.percentile,
            fps=args.fps,
            seq_len=args.seq_len,
            frame_size=args.frame_size,
            feature_dim=args.feature_dim,
            hidden_dims=args.hidden_dims,
            device=args.device,
            max_frames_per_video=args.max_frames,
        )
    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
