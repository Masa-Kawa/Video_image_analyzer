"""
Cholec80 フェーズ認識モデル 学習パイプライン

2段階の学習プロセス:

Stage 1: 特徴抽出 (extract-features)
  - Cholec80 動画 → ResNet50 (SelfSupSurg DINO) → .npy 特徴量ファイル
  - 各動画ごとに video{id}_features.npy (N, 2048) と video{id}_labels.npy (N,) を保存
  - 1回だけ実行すれば以降は不要

Stage 2: LSTM 学習 (train)
  - 事前抽出済み特徴量 → BiLSTM → フェーズ分類
  - CrossEntropyLoss + CosineAnnealingLR
  - Mixed precision (fp16) 対応
  - video01-40 を train/val に分割

評価:
  - evaluate サブコマンドで test set (video41-80) を評価
  - 指標: Accuracy, Phase-wise Accuracy, Video-level Accuracy

CLI:
  python -m src.cholec_phase.train extract-features \\
      --cholec80-dir /path/to/cholec80 --outdir /path/to/features
  python -m src.cholec_phase.train train \\
      --feature-dir /path/to/features --outdir /path/to/models
  python -m src.cholec_phase.train evaluate \\
      --feature-dir /path/to/features --model /path/to/model.pth
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.cholec_phase import CHOLEC80_PHASES, NUM_PHASES
from src.cholec_phase.models import PhaseFeatureExtractor, PhaseRecognitionModel
from src.cholec_phase.dataset import (
    PhaseFeatureDataset,
    discover_cholec80,
    get_test_ids,
    get_train_val_split,
    read_phase_annotation,
    frame_annotation_to_second,
)
from src.red.redlog import iter_frames


# ---------------------------------------------------------------------------
# Stage 1: 特徴抽出
# ---------------------------------------------------------------------------


def extract_features_for_video(
    video_path: str,
    ann_path: str,
    outdir: str,
    video_id: str,
    extractor: PhaseFeatureExtractor,
    sample_fps: float = 1.0,
    video_fps: float = 25.0,
) -> dict:
    """
    1本の動画から特徴量とラベルを抽出して保存する。

    出力:
      {outdir}/{video_id}_features.npy  (N, 2048)
      {outdir}/{video_id}_labels.npy    (N,)

    Returns:
        {"features": path, "labels": path, "n_frames": int}
    """
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    # アノテーション読み込み
    frame_indices, phase_ids = read_phase_annotation(ann_path)
    times_sec, sampled_labels = frame_annotation_to_second(
        frame_indices, phase_ids, video_fps, sample_fps,
    )

    # アノテーションが空の動画は、全フレームに label=0 を当てると学習データに
    # 大量の誤ラベルノイズが混入するため、警告してスキップする。
    if not times_sec:
        print(f"警告: {video_id} のフェーズアノテーションが空です（{ann_path}）。"
              "誤ラベル混入を避けるためこの動画をスキップします。",
              file=sys.stderr)
        return {}

    # フレーム抽出 + 特徴抽出
    features_list: List[np.ndarray] = []
    labels_list: List[int] = []

    # times_sec は昇順なので np.searchsorted で最寄りを O(log N) で求める
    # （線形探索だと全体 O(N^2) になり長尺動画で著しく遅い）。
    times_arr = np.asarray(times_sec, dtype=float) if times_sec else None

    def _nearest_label(t: float) -> int:
        if times_arr is None or times_arr.size == 0:
            return 0
        pos = int(np.searchsorted(times_arr, t))
        if pos == 0:
            idx = 0
        elif pos >= times_arr.size:
            idx = times_arr.size - 1
        else:
            lo, hi = times_arr[pos - 1], times_arr[pos]
            idx = pos - 1 if (t - lo) <= (hi - t) else pos
        return sampled_labels[idx]

    frame_count = 0
    # ジェネレータを明示的にクローズし、内部のキャプチャ（PyAV/OpenCV）を
    # 例外時・早期終了時にも確実に解放する。
    frame_gen = iter_frames(video_path, sample_fps)
    try:
        for t_sec, bgr, reader in frame_gen:
            feat = extractor.extract(bgr)
            features_list.append(feat)

            # t_sec に最も近いアノテーションラベルを取得
            labels_list.append(_nearest_label(t_sec))

            frame_count += 1
            if frame_count % 60 == 0:
                print(f"  {video_id}: {frame_count} frames ({t_sec:.0f}s)",
                      file=sys.stderr)
    finally:
        frame_gen.close()

    if not features_list:
        return {}

    features = np.array(features_list)
    labels = np.array(labels_list)

    feat_path = out_path / f"{video_id}_features.npy"
    label_path = out_path / f"{video_id}_labels.npy"

    np.save(str(feat_path), features)
    np.save(str(label_path), labels)

    print(f"  {video_id}: {len(features)} frames saved")
    return {
        "features": str(feat_path),
        "labels": str(label_path),
        "n_frames": len(features),
    }


def extract_features(
    cholec80_dir: str,
    outdir: str,
    device: str = "cuda",
    sample_fps: float = 1.0,
    backbone_method: str = "dino",
    video_ids: Optional[List[int]] = None,
) -> dict:
    """
    Cholec80 全動画から特徴量を抽出する（Stage 1）。

    Args:
        cholec80_dir: Cholec80 ルートディレクトリ
        outdir: 出力ディレクトリ
        device: 計算デバイス
        sample_fps: サンプリングFPS
        backbone_method: SelfSupSurg手法
        video_ids: 処理する動画ID（None=全て）

    Returns:
        {"extracted": int, "total": int, "output_dir": str}
    """
    data = discover_cholec80(cholec80_dir)
    if not data:
        print(f"エラー: Cholec80データが見つかりません: {cholec80_dir}",
              file=sys.stderr)
        return {}

    extractor = PhaseFeatureExtractor(
        device=device,
        method=backbone_method,
        auto_download=True,
    )

    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    extracted = 0
    total = 0

    for vid_str, info in sorted(data.items()):
        if "video" not in info or "phase_ann" not in info:
            continue

        vid_num = int(vid_str.replace("video", ""))
        if video_ids and vid_num not in video_ids:
            continue

        total += 1

        # 既に抽出済みならスキップ
        feat_path = out_path / f"{vid_str}_features.npy"
        if feat_path.exists():
            print(f"スキップ（既存）: {vid_str}")
            extracted += 1
            continue

        print(f"特徴抽出中: {vid_str}")
        result = extract_features_for_video(
            video_path=info["video"],
            ann_path=info["phase_ann"],
            outdir=outdir,
            video_id=vid_str,
            extractor=extractor,
            sample_fps=sample_fps,
        )
        if result:
            extracted += 1

    print(f"\n特徴抽出完了: {extracted}/{total} 動画")
    return {"extracted": extracted, "total": total, "output_dir": outdir}


# ---------------------------------------------------------------------------
# Stage 2: LSTM 学習
# ---------------------------------------------------------------------------


def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor, int]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dataset側で seq_len 長にパディング済みの固定長テンソルをバッチ化する。

    PhaseFeatureDataset.__getitem__ が各サンプルを seq_len にパディング済みのため、
    ここでは torch.stack で固定形状のバッチに束ねるだけ（lengths は有効長）。
    """
    features, labels, lengths = zip(*batch)
    return (
        torch.stack(features),
        torch.stack(labels),
        torch.tensor(lengths, dtype=torch.long),
    )


def train_model(
    feature_dir: str,
    outdir: str,
    device: str = "cuda",
    epochs: int = 50,
    batch_size: int = 8,
    lr: float = 1e-3,
    hidden_dim: int = 512,
    num_layers: int = 2,
    seq_len: int = 300,
    stride: int = 150,
    dropout: float = 0.3,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> dict:
    """
    BiLSTM フェーズ認識モデルを学習する（Stage 2）。

    Args:
        feature_dir: 事前抽出済み特徴量ディレクトリ
        outdir: モデル出力ディレクトリ
        device: 計算デバイス
        epochs: エポック数
        batch_size: バッチサイズ
        lr: 学習率
        hidden_dim: LSTM 隠れ次元
        num_layers: LSTM 層数
        seq_len: シーケンス長
        stride: スライディングウィンドウストライド
        dropout: ドロップアウト率
        val_ratio: 検証セット比率
        seed: 乱数シード

    Returns:
        {"model_path": str, "best_val_acc": float, "history": list}
    """
    device_obj = torch.device(
        device if torch.cuda.is_available() and device != "cpu" else "cpu"
    )

    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    # データ分割
    train_ids, val_ids = get_train_val_split(val_ratio=val_ratio, seed=seed)
    print(f"Train: {train_ids}")
    print(f"Val:   {val_ids}")

    # データセット
    train_ds = PhaseFeatureDataset(
        feature_dir, train_ids, seq_len=seq_len, stride=stride,
    )
    val_ds = PhaseFeatureDataset(
        feature_dir, val_ids, seq_len=seq_len, stride=seq_len,
    )

    if len(train_ds) == 0:
        print("エラー: 学習データが見つかりません", file=sys.stderr)
        return {}

    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=2, pin_memory=True,
    )

    # モデル
    model = PhaseRecognitionModel(
        feature_dim=2048,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=NUM_PHASES,
        dropout=dropout,
    ).to(device_obj)

    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler(enabled=device_obj.type == "cuda")

    best_val_acc = 0.0
    best_model_path = out_path / "phase_model_best.pth"
    history: List[dict] = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # --- Train ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for features, labels, lengths in train_loader:
            features = features.to(device_obj)
            labels = labels.to(device_obj)

            optimizer.zero_grad()

            with torch.amp.autocast(device_obj.type, enabled=device_obj.type == "cuda"):
                # lengths を渡すと pad_packed_sequence でT次元が縮む場合があるため
                # 学習時は None を渡して seq_len を保つ
                logits = model(features, None)
                # (B, T, C) → (B*T, C)
                logits_flat = logits.reshape(-1, NUM_PHASES)
                labels_flat = labels.reshape(-1)
                loss = criterion(logits_flat, labels_flat)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * features.size(0)
            mask = labels_flat != -1
            preds = logits_flat.argmax(dim=1)
            train_correct += (preds[mask] == labels_flat[mask]).sum().item()
            train_total += mask.sum().item()

        scheduler.step()

        train_loss /= max(len(train_ds), 1)
        train_acc = train_correct / max(train_total, 1)

        # --- Validate ---
        val_acc = _evaluate_loader(model, val_loader, device_obj)

        elapsed = time.time() - t0
        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_acc": round(val_acc, 4),
            "lr": round(scheduler.get_last_lr()[0], 6),
        })

        print(f"Epoch {epoch:3d}/{epochs} | "
              f"loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_acc={val_acc:.4f} | {elapsed:.1f}s")

        # ベストモデル保存
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
                "config": {
                    "hidden_dim": hidden_dim,
                    "num_layers": num_layers,
                    "dropout": dropout,
                    "seq_len": seq_len,
                },
            }, str(best_model_path))
            print(f"  ★ Best model saved (val_acc={val_acc:.4f})")

    # 最終モデルも保存
    final_path = out_path / "phase_model_final.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "epoch": epochs,
        "val_acc": val_acc,
        "config": {
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "seq_len": seq_len,
        },
    }, str(final_path))

    # 学習履歴を保存
    history_path = out_path / "training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest val_acc: {best_val_acc:.4f}")
    print(f"Best model: {best_model_path}")
    print(f"Final model: {final_path}")

    return {
        "model_path": str(best_model_path),
        "best_val_acc": best_val_acc,
        "history": history,
    }


def _evaluate_loader(
    model: PhaseRecognitionModel,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """データローダー上でフレームレベル精度を計算する。"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for features, labels, lengths in loader:
            features = features.to(device)
            labels = labels.to(device)

            logits = model(features, None)
            logits_flat = logits.reshape(-1, NUM_PHASES)
            labels_flat = labels.reshape(-1)

            mask = labels_flat != -1
            preds = logits_flat.argmax(dim=1)
            correct += (preds[mask] == labels_flat[mask]).sum().item()
            total += mask.sum().item()

    return correct / max(total, 1)


# ---------------------------------------------------------------------------
# 評価
# ---------------------------------------------------------------------------


def evaluate_model(
    feature_dir: str,
    model_path: str,
    device: str = "cuda",
    seq_len: int = 300,
    hidden_dim: int = 512,
    num_layers: int = 2,
    dropout: float = 0.3,
) -> dict:
    """
    テストセット (video41-80) でモデルを評価する。

    Returns:
        {"accuracy": float, "phase_accuracy": dict, "confusion": list}
    """
    device_obj = torch.device(
        device if torch.cuda.is_available() and device != "cpu" else "cpu"
    )

    # モデルロード。weights_only=True で pickle 経由の任意コード実行を防ぐ
    # （保存内容は state_dict + int/float/dict のみで安全）。
    ckpt = torch.load(model_path, map_location=device_obj, weights_only=True)
    config = ckpt.get("config", {})
    # 保存時 config からモデル構造を完全復元する（dropout も含める）
    model = PhaseRecognitionModel(
        feature_dim=2048,
        hidden_dim=config.get("hidden_dim", hidden_dim),
        num_layers=config.get("num_layers", num_layers),
        dropout=config.get("dropout", dropout),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device_obj)
    model.eval()

    test_ids = get_test_ids()
    test_ds = PhaseFeatureDataset(
        feature_dir, test_ids, seq_len=seq_len, stride=seq_len,
    )

    if len(test_ds) == 0:
        print("エラー: テストデータが見つかりません", file=sys.stderr)
        return {}

    test_loader = DataLoader(
        test_ds, batch_size=4, shuffle=False,
        collate_fn=collate_fn, num_workers=2,
    )

    # 全体精度
    all_preds: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for features, labels, lengths in test_loader:
            features = features.to(device_obj)
            logits = model(features, None)
            logits_flat = logits.reshape(-1, NUM_PHASES)
            labels_flat = labels.reshape(-1)

            mask = labels_flat != -1
            preds = logits_flat.argmax(dim=1)
            all_preds.extend(preds[mask].cpu().tolist())
            all_labels.extend(labels_flat[mask].cpu().tolist())

    all_preds_arr = np.array(all_preds)
    all_labels_arr = np.array(all_labels)

    accuracy = float(np.mean(all_preds_arr == all_labels_arr))

    # フェーズごとの精度
    phase_acc: Dict[str, float] = {}
    for i, name in enumerate(CHOLEC80_PHASES):
        mask = all_labels_arr == i
        if mask.sum() > 0:
            phase_acc[name] = float(np.mean(all_preds_arr[mask] == i))
        else:
            phase_acc[name] = 0.0

    # 混同行列
    confusion = np.zeros((NUM_PHASES, NUM_PHASES), dtype=int)
    for p, l in zip(all_preds, all_labels):
        confusion[l][p] += 1

    print(f"\n=== テスト結果 ===")
    print(f"全体精度: {accuracy:.4f}")
    print(f"\nフェーズ別精度:")
    for name, acc in phase_acc.items():
        print(f"  {name:30s}: {acc:.4f}")

    return {
        "accuracy": accuracy,
        "phase_accuracy": phase_acc,
        "confusion": confusion.tolist(),
    }


# ---------------------------------------------------------------------------
# CLI エントリポイント
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cholec80 手術フェーズ認識モデル 学習パイプライン",
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- extract-features ---
    ef = subparsers.add_parser(
        "extract-features",
        help="Stage 1: Cholec80動画から特徴量を抽出",
    )
    ef.add_argument("--cholec80-dir", required=True,
                    help="Cholec80ルートディレクトリ")
    ef.add_argument("--outdir", required=True,
                    help="特徴量出力ディレクトリ")
    ef.add_argument("--device", default="cuda")
    ef.add_argument("--sample-fps", type=float, default=1.0,
                    help="サンプリングFPS（デフォルト: 1.0）")
    ef.add_argument("--backbone-method", default="dino",
                    choices=["dino", "moco_v2", "simclr", "swav"])
    ef.add_argument("--video-ids", type=int, nargs="*", default=None,
                    help="処理する動画ID（例: 1 2 3）")

    # --- train ---
    tr = subparsers.add_parser(
        "train",
        help="Stage 2: BiLSTMフェーズ認識モデルを学習",
    )
    tr.add_argument("--feature-dir", required=True,
                    help="事前抽出済み特徴量ディレクトリ")
    tr.add_argument("--outdir", required=True,
                    help="モデル出力ディレクトリ")
    tr.add_argument("--device", default="cuda")
    tr.add_argument("--epochs", type=int, default=50)
    tr.add_argument("--batch-size", type=int, default=8)
    tr.add_argument("--lr", type=float, default=1e-3)
    tr.add_argument("--hidden-dim", type=int, default=512)
    tr.add_argument("--num-layers", type=int, default=2)
    tr.add_argument("--seq-len", type=int, default=300)
    tr.add_argument("--stride", type=int, default=150)
    tr.add_argument("--dropout", type=float, default=0.3)
    tr.add_argument("--seed", type=int, default=42)

    # --- evaluate ---
    ev = subparsers.add_parser(
        "evaluate",
        help="テストセットでモデルを評価",
    )
    ev.add_argument("--feature-dir", required=True,
                    help="事前抽出済み特徴量ディレクトリ")
    ev.add_argument("--model", required=True,
                    help="学習済みモデルパス")
    ev.add_argument("--device", default="cuda")
    ev.add_argument("--seq-len", type=int, default=300)

    args = parser.parse_args()

    if args.command == "extract-features":
        extract_features(
            cholec80_dir=args.cholec80_dir,
            outdir=args.outdir,
            device=args.device,
            sample_fps=args.sample_fps,
            backbone_method=args.backbone_method,
            video_ids=args.video_ids,
        )
    elif args.command == "train":
        train_model(
            feature_dir=args.feature_dir,
            outdir=args.outdir,
            device=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            seq_len=args.seq_len,
            stride=args.stride,
            dropout=args.dropout,
            seed=args.seed,
        )
    elif args.command == "evaluate":
        evaluate_model(
            feature_dir=args.feature_dir,
            model_path=args.model,
            device=args.device,
            seq_len=args.seq_len,
        )
    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
