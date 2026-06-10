"""
手術フェーズセグメンテーションモジュール（phase_segmenter）

腹腔鏡下胆嚢摘出術の動画を粗いフェーズに分割する。
ラベルなしデータで動作する教師なし手法。

アルゴリズム:
  1. ResNet-18で各フレームの特徴量を抽出（512次元）
  2. 隣接フレーム間のコサイン類似度を計算
  3. 類似度の谷（急激な変化）を変化点として検出
  4. 変化点でセグメントに分割
  5. 各セグメントの平均特徴量をK-meansでクラスタリング
  6. クラスタを視覚的ヒューリスティックでフェーズ名にマッピング

ターゲットフェーズ（粗い分類）:
  1. Preparation / Outside body
  2. Port insertion
  3. Observation
  4. Gallbladder neck handling
  5. Dissection
  6. Irrigation / drain
  7. Extraction / closing

2段階の処理:
  Step 1 - record_timeseries(): 動画→CSV（特徴量・類似度の時系列）
  Step 2 - annotate_phases():   CSV→JSONL/SRT（フェーズ区間アノテーション）

出力: CSV（時系列ログ）、SRT（フェーズ字幕）、JSONL（イベント正本）
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.core.time_utils import format_srt_time
from src.red.redlog import iter_frames, smooth_center


# ---------------------------------------------------------------------------
# フェーズ名定義
# ---------------------------------------------------------------------------

PHASE_NAMES = {
    0: "Preparation",
    1: "Port insertion",
    2: "Observation",
    3: "Neck handling",
    4: "Dissection",
    5: "Irrigation",
    6: "Extraction",
}

# フォールバック: クラスタ数がフェーズ数と異なる場合
def get_phase_name(phase_id: int) -> str:
    return PHASE_NAMES.get(phase_id, f"Phase_{phase_id}")


# ---------------------------------------------------------------------------
# 特徴量抽出（ResNet-18）
# ---------------------------------------------------------------------------

_resnet_model = None
_resnet_transform = None
_resnet_device = None


def _load_resnet(device: str = "cuda"):
    """ResNet-18モデルをロード（遅延初期化・シングルトン）"""
    global _resnet_model, _resnet_transform, _resnet_device

    if _resnet_model is not None and _resnet_device == device:
        return _resnet_model, _resnet_transform

    import torch
    import torchvision.models as models
    import torchvision.transforms as T

    _resnet_device = device
    if device == "cuda" and not torch.cuda.is_available():
        _resnet_device = "cpu"

    # ResNet-18: 軽量で十分な表現力
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # 最終FC層を除去 → 512次元の特徴ベクトル
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()
    model.to(_resnet_device)
    _resnet_model = model

    _resnet_transform = T.Compose([
        T.ToPILImage(),
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    ])

    return _resnet_model, _resnet_transform


def extract_feature(
    frame_bgr: np.ndarray,
    device: str = "cuda",
) -> np.ndarray:
    """
    ResNet-18でフレームから512次元の特徴ベクトルを抽出する。

    Args:
        frame_bgr: BGR画像
        device: 計算デバイス

    Returns:
        特徴ベクトル（512次元, L2正規化済み）
    """
    import torch

    model, transform = _load_resnet(device)

    # BGR → RGB
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tensor = transform(rgb).unsqueeze(0).to(_resnet_device)

    with torch.no_grad():
        feat = model(tensor)

    feat = feat.squeeze().cpu().numpy()
    # L2正規化
    norm = np.linalg.norm(feat)
    if norm > 0:
        feat = feat / norm

    return feat


# ---------------------------------------------------------------------------
# 軽量ヒューリスティック特徴（GPU不要のフォールバック）
# ---------------------------------------------------------------------------

def extract_heuristic_feature(frame_bgr: np.ndarray) -> np.ndarray:
    """
    GPU不要の軽量特徴量: HSV/明度ヒストグラム + 統計量。

    ResNetが使えない場合のフォールバック。
    48次元のベクトルを返す。

    Args:
        frame_bgr: BGR画像

    Returns:
        特徴ベクトル（48次元, L2正規化済み）
    """
    # リサイズ
    small = cv2.resize(frame_bgr, (128, 128))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    # Hue histogram (18 bins)
    h_hist = cv2.calcHist([hsv], [0], None, [18], [0, 180]).flatten()
    h_hist = h_hist / (h_hist.sum() + 1e-10)

    # Saturation histogram (8 bins)
    s_hist = cv2.calcHist([hsv], [1], None, [8], [0, 256]).flatten()
    s_hist = s_hist / (s_hist.sum() + 1e-10)

    # Value histogram (8 bins)
    v_hist = cv2.calcHist([hsv], [2], None, [8], [0, 256]).flatten()
    v_hist = v_hist / (v_hist.sum() + 1e-10)

    # 統計量 (14 values)
    stats = np.array([
        float(np.mean(gray)),
        float(np.std(gray)),
        float(np.mean(hsv[:, :, 0])),
        float(np.std(hsv[:, :, 0])),
        float(np.mean(hsv[:, :, 1])),
        float(np.std(hsv[:, :, 1])),
        float(np.mean(hsv[:, :, 2])),
        float(np.std(hsv[:, :, 2])),
        # 赤色率
        float(np.mean((hsv[:, :, 0] < 10) | (hsv[:, :, 0] > 170))),
        # 暗領域率
        float(np.mean(gray < 50)),
        # 明領域率
        float(np.mean(gray > 200)),
        # エッジ密度
        float(np.mean(cv2.Canny(gray, 50, 150) > 0)),
        # テクスチャ（ラプラシアン分散）
        float(cv2.Laplacian(gray, cv2.CV_64F).var() / 1000.0),
        # コントラスト
        float(gray.max() - gray.min()) / 255.0,
    ])

    feat = np.concatenate([h_hist, s_hist, v_hist, stats])

    norm = np.linalg.norm(feat)
    if norm > 0:
        feat = feat / norm

    return feat


# ---------------------------------------------------------------------------
# 変化点検出
# ---------------------------------------------------------------------------

def detect_change_points(
    similarities: List[float],
    min_segment_length: int = 10,
    sensitivity: float = 1.5,
) -> List[int]:
    """
    コサイン類似度の時系列から変化点を検出する。

    類似度が局所平均から大きく低下する点を変化点とする。
    sensitivity × 標準偏差だけ平均を下回る点を検出。

    Args:
        similarities: コサイン類似度の時系列
        min_segment_length: 変化点間の最小フレーム数
        sensitivity: 検出感度（標準偏差の倍数）

    Returns:
        変化点のインデックスリスト
    """
    if len(similarities) < min_segment_length * 2:
        return []

    arr = np.array(similarities)

    # 局所的な統計量（ウィンドウ = min_segment_length）
    window = min_segment_length
    change_points = []

    for i in range(window, len(arr) - window):
        local_before = arr[max(0, i - window):i]
        local_after = arr[i:min(len(arr), i + window)]
        local_mean = np.mean(np.concatenate([local_before, local_after]))
        local_std = np.std(np.concatenate([local_before, local_after]))

        if local_std < 1e-6:
            continue

        # 類似度が局所平均から大きく低下 → 変化点
        if arr[i] < local_mean - sensitivity * local_std:
            change_points.append(i)

    # 近接する変化点をマージ
    if not change_points:
        return []

    merged = [change_points[0]]
    for cp in change_points[1:]:
        if cp - merged[-1] >= min_segment_length:
            merged.append(cp)
        else:
            # より強い変化点（類似度が低い方）を残す
            if arr[cp] < arr[merged[-1]]:
                merged[-1] = cp

    return merged


# ---------------------------------------------------------------------------
# セグメントクラスタリング
# ---------------------------------------------------------------------------

def cluster_segments(
    features: np.ndarray,
    change_points: List[int],
    n_clusters: int = 7,
) -> List[int]:
    """
    セグメントの平均特徴量をK-meansでクラスタリングする。

    Args:
        features: 全フレームの特徴量行列（N × D）
        change_points: 変化点インデックス
        n_clusters: クラスタ数

    Returns:
        各セグメントのクラスタID
    """
    # セグメント境界
    boundaries = [0] + change_points + [len(features)]

    # 各セグメントの平均特徴量
    segment_features = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        seg_feat = np.mean(features[start:end], axis=0)
        segment_features.append(seg_feat)

    segment_features = np.array(segment_features)

    # クラスタ数の調整
    actual_n = min(n_clusters, len(segment_features))
    if actual_n <= 1:
        return [0] * len(segment_features)

    # K-means（scipy不要の簡易実装）
    labels = _simple_kmeans(segment_features, actual_n)

    return labels


def _simple_kmeans(
    data: np.ndarray,
    k: int,
    max_iter: int = 50,
) -> List[int]:
    """
    簡易K-means実装（外部依存なし）。

    Args:
        data: データ行列（N × D）
        k: クラスタ数
        max_iter: 最大反復回数

    Returns:
        クラスタラベルリスト
    """
    n = len(data)
    if n <= k:
        return list(range(n))

    # 初期化: 時間的に均等な位置からセントロイドを選択
    indices = np.linspace(0, n - 1, k, dtype=int)
    centroids = data[indices].copy()

    labels = np.zeros(n, dtype=int)

    for _ in range(max_iter):
        # 割り当て
        new_labels = np.zeros(n, dtype=int)
        for i in range(n):
            dists = np.linalg.norm(data[i] - centroids, axis=1)
            new_labels[i] = int(np.argmin(dists))

        # 収束判定
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels

        # セントロイド更新
        for c in range(k):
            members = data[labels == c]
            if len(members) > 0:
                centroids[c] = np.mean(members, axis=0)

    return labels.tolist()


# ---------------------------------------------------------------------------
# フェーズ名マッピング（ヒューリスティック）
# ---------------------------------------------------------------------------

def map_clusters_to_phases(
    times: List[float],
    features: np.ndarray,
    change_points: List[int],
    cluster_labels: List[int],
    frame_brightness: List[float],
    frame_tissue_ratio: List[float],
) -> List[str]:
    """
    クラスタIDを手術フェーズ名にマッピングする。

    時間的な位置と視覚的特徴を組み合わせてフェーズ名を推定する。
    - 序盤で明るい → Preparation
    - 序盤で暗くなる → Port insertion
    - 中盤で tissue_ratio が高い → Dissection / Neck handling
    - 終盤 → Extraction / closing

    Args:
        times: タイムスタンプ
        features: 特徴量行列
        change_points: 変化点
        cluster_labels: クラスタラベル
        frame_brightness: 各フレームの平均明度
        frame_tissue_ratio: 各フレームの組織色率

    Returns:
        各セグメントのフェーズ名
    """
    boundaries = [0] + change_points + [len(times)]
    total_duration = times[-1] - times[0] if times else 1.0
    n_segments = len(boundaries) - 1

    phase_names = []
    for seg_idx in range(n_segments):
        start = boundaries[seg_idx]
        end = boundaries[seg_idx + 1]

        # 時間的位置（0.0=開始, 1.0=終了）
        t_center = (times[start] + times[end - 1]) / 2.0
        temporal_pos = (t_center - times[0]) / total_duration if total_duration > 0 else 0.5

        # セグメントの視覚特徴
        seg_brightness = np.mean(frame_brightness[start:end])
        seg_tissue = np.mean(frame_tissue_ratio[start:end])

        # ルールベースのフェーズ推定
        if temporal_pos < 0.10:
            if seg_brightness > 100:
                phase_names.append("Preparation")
            else:
                phase_names.append("Port insertion")
        elif temporal_pos < 0.20:
            if seg_tissue < 0.15:
                phase_names.append("Port insertion")
            else:
                phase_names.append("Observation")
        elif temporal_pos > 0.90:
            phase_names.append("Extraction")
        elif temporal_pos > 0.80:
            if seg_tissue < 0.20:
                phase_names.append("Irrigation")
            else:
                phase_names.append("Extraction")
        else:
            # 中盤: tissue_ratioの高さでDissection/Neck handling/Observationを区別
            if seg_tissue > 0.40:
                phase_names.append("Dissection")
            elif seg_tissue > 0.25:
                phase_names.append("Neck handling")
            else:
                phase_names.append("Observation")

    return phase_names


# ---------------------------------------------------------------------------
# メインパイプライン
# ---------------------------------------------------------------------------

def record_timeseries(
    video_path: str,
    outdir: str,
    fps: float = 1.0,
    device: str = "cuda",
    use_resnet: bool = True,
    smooth_s: float = 3.0,
) -> dict:
    """
    Step 1: 動画をサンプリングして特徴量CSVを出力する。

    Args:
        video_path: 入力動画ファイルパス
        outdir: 出力ディレクトリ
        fps: サンプリングFPS（デフォルト1 — フェーズは秒単位で十分）
        device: 計算デバイス
        use_resnet: True=ResNet-18, False=ヒューリスティック特徴
        smooth_s: 類似度の平滑化窓（秒）

    Returns:
        {"csv": CSVファイルパス}
    """
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    stem = Path(video_path).stem

    times: List[float] = []
    all_features: List[np.ndarray] = []
    brightnesses: List[float] = []
    tissue_ratios: List[float] = []
    reader_name = "opencv"

    print(f"フェーズ解析中: {video_path}", file=sys.stderr)
    print(f"特徴量: {'ResNet-18' if use_resnet else 'heuristic'}, "
          f"FPS: {fps}", file=sys.stderr)

    for t_sec, bgr, reader in iter_frames(video_path, fps):
        reader_name = reader

        # 特徴量抽出
        if use_resnet:
            feat = extract_feature(bgr, device)
        else:
            feat = extract_heuristic_feature(bgr)

        # 明度
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))

        # 組織色率（簡易版）
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        tissue_mask = (
            ((hsv[:, :, 0] <= 35) | (hsv[:, :, 0] >= 165))
            & (hsv[:, :, 1] > 30)
            & (hsv[:, :, 2] > 30)
        )
        tissue_ratio = float(np.mean(tissue_mask))

        times.append(t_sec)
        all_features.append(feat)
        brightnesses.append(brightness)
        tissue_ratios.append(tissue_ratio)

        if len(times) % 30 == 0:
            print(f"  フレーム処理中: {len(times)} ({t_sec:.1f}s)",
                  file=sys.stderr)

    if len(times) < 2:
        print("警告: フレームが不足しています。", file=sys.stderr)
        return {}

    # コサイン類似度
    similarities = [1.0]  # 最初のフレームは類似度1.0
    for i in range(1, len(all_features)):
        sim = float(np.dot(all_features[i - 1], all_features[i]))
        similarities.append(sim)

    # 平滑化
    window_size = max(1, int(round(smooth_s * fps)))
    smooth_sims = smooth_center(similarities, window_size)

    # CSV出力
    csv_path = out_path / f"{stem}_phaselog.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "t_sec", "t_srt",
            "brightness", "tissue_ratio",
            "similarity", "smooth_similarity",
            "reader",
        ])
        for i in range(len(times)):
            writer.writerow([
                f"{times[i]:.3f}",
                format_srt_time(times[i]),
                f"{brightnesses[i]:.2f}",
                f"{tissue_ratios[i]:.6f}",
                f"{similarities[i]:.6f}",
                f"{smooth_sims[i]:.6f}",
                reader_name,
            ])

    # 特徴量をnpyで保存（annotateステップで使用）
    features_path = out_path / f"{stem}_features.npy"
    np.save(str(features_path), np.array(all_features))

    print(f"CSV  : {csv_path}")
    print(f"特徴量: {features_path}")
    return {"csv": str(csv_path), "features_npy": str(features_path)}


# ---------------------------------------------------------------------------
# CSV読み込み
# ---------------------------------------------------------------------------

def read_phaselog_csv(csv_path: str) -> dict:
    """phaselog CSVを読み込む。"""
    times: List[float] = []
    brightnesses: List[float] = []
    tissue_ratios: List[float] = []
    similarities: List[float] = []
    smooth_sims: List[float] = []
    reader = "unknown"

    with open(csv_path, "r", encoding="utf-8") as f:
        reader_obj = csv.DictReader(f)
        for row in reader_obj:
            times.append(float(row["t_sec"]))
            brightnesses.append(float(row["brightness"]))
            tissue_ratios.append(float(row["tissue_ratio"]))
            similarities.append(float(row["similarity"]))
            smooth_sims.append(float(row["smooth_similarity"]))
            reader = row.get("reader", "unknown")

    fps = 1.0 / (times[1] - times[0]) if len(times) >= 2 else 1.0

    return {
        "times": times,
        "brightnesses": brightnesses,
        "tissue_ratios": tissue_ratios,
        "similarities": similarities,
        "smooth_similarities": smooth_sims,
        "reader": reader,
        "fps": fps,
    }


# ---------------------------------------------------------------------------
# フェーズアノテーション
# ---------------------------------------------------------------------------

def annotate_phases(
    csv_path: str,
    outdir: str,
    features_npy: Optional[str] = None,
    n_phases: int = 7,
    sensitivity: float = 1.5,
    min_phase_s: float = 30.0,
) -> dict:
    """
    Step 2: CSVと特徴量から変化点検出・クラスタリングでフェーズを推定する。

    Args:
        csv_path: 入力CSVファイルパス
        outdir: 出力ディレクトリ
        features_npy: 特徴量npyファイルパス（Noneなら自動推定）
        n_phases: 目標フェーズ数
        sensitivity: 変化点検出感度
        min_phase_s: 最小フェーズ長（秒）

    Returns:
        {"jsonl": path, "srt": path, "phases": int}
    """
    data = read_phaselog_csv(csv_path)
    times = data["times"]
    fps = data["fps"]
    brightnesses = data["brightnesses"]
    tissue_ratios = data["tissue_ratios"]
    smooth_sims = data["smooth_similarities"]

    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    stem = Path(csv_path).stem.replace("_phaselog", "")

    # 特徴量の読み込み
    if features_npy is None:
        features_npy = str(Path(csv_path).parent / f"{stem}_features.npy")

    features = np.load(features_npy)

    # 変化点検出
    min_segment = max(3, int(round(min_phase_s * fps)))
    change_points = detect_change_points(
        smooth_sims, min_segment_length=min_segment, sensitivity=sensitivity,
    )

    # クラスタリング
    cluster_labels = cluster_segments(features, change_points, n_clusters=n_phases)

    # フェーズ名マッピング
    phase_names = map_clusters_to_phases(
        times, features, change_points, cluster_labels,
        brightnesses, tissue_ratios,
    )

    # セグメント→イベント変換
    boundaries = [0] + change_points + [len(times)]
    events: List[dict] = []

    for seg_idx in range(len(boundaries) - 1):
        start = boundaries[seg_idx]
        end = boundaries[seg_idx + 1]

        events.append({
            "type": "surgical_phase",
            "phase_id": seg_idx,
            "phase_name": phase_names[seg_idx],
            "cluster_id": cluster_labels[seg_idx],
            "start": times[start],
            "end": times[end - 1],
            "duration": times[end - 1] - times[start],
        })

    # JSONL出力
    jsonl_path = out_path / f"{stem}_phase_events.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for ev in events:
            line = {
                "type": ev["type"],
                "phase_id": ev["phase_id"],
                "phase_name": ev["phase_name"],
                "cluster_id": ev["cluster_id"],
                "start_sec": ev["start"],
                "end_sec": ev["end"],
                "start_srt": format_srt_time(ev["start"]),
                "end_srt": format_srt_time(ev["end"]),
                "duration_sec": round(ev["duration"], 3),
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    # SRT出力
    srt_path = out_path / f"{stem}_phases.srt"
    srt_lines: List[str] = []
    for idx, ev in enumerate(events, start=1):
        start_srt = format_srt_time(ev["start"])
        end_srt = format_srt_time(ev["end"])
        srt_lines.append(f"{idx}")
        srt_lines.append(f"{start_srt} --> {end_srt}")
        srt_lines.append(f"[phase] {ev['phase_name']}")
        srt_lines.append("")

    Path(srt_path).write_text("\n".join(srt_lines), encoding="utf-8")

    print(f"JSONL: {jsonl_path} （正本）")
    print(f"SRT  : {srt_path}")
    print(f"フェーズ数: {len(events)}")

    return {
        "jsonl": str(jsonl_path),
        "srt": str(srt_path),
        "phases": len(events),
    }


# ---------------------------------------------------------------------------
# 一括実行
# ---------------------------------------------------------------------------

def analyze_video(
    video_path: str,
    outdir: str,
    fps: float = 1.0,
    device: str = "cuda",
    use_resnet: bool = True,
    smooth_s: float = 3.0,
    n_phases: int = 7,
    sensitivity: float = 1.5,
    min_phase_s: float = 30.0,
) -> dict:
    """動画を解析し、CSV・SRT・JSONLを出力する（2ステップの一括実行）。"""
    result1 = record_timeseries(
        video_path=video_path,
        outdir=outdir,
        fps=fps,
        device=device,
        use_resnet=use_resnet,
        smooth_s=smooth_s,
    )

    if not result1:
        return {}

    result2 = annotate_phases(
        csv_path=result1["csv"],
        outdir=outdir,
        features_npy=result1.get("features_npy"),
        n_phases=n_phases,
        sensitivity=sensitivity,
        min_phase_s=min_phase_s,
    )

    return {**result1, **result2}


# ---------------------------------------------------------------------------
# CLI エントリポイント
# ---------------------------------------------------------------------------

def main() -> int:
    """コマンドラインエントリポイント（サブコマンド方式）"""
    parser = argparse.ArgumentParser(
        description="手術動画のフェーズセグメンテーション"
    )
    subparsers = parser.add_subparsers(dest="command", help="実行コマンド")

    # --- timeseries ---
    ts_parser = subparsers.add_parser(
        "timeseries", help="Step 1: 特徴量抽出（動画→CSV+npy）"
    )
    ts_parser.add_argument("--video", required=True)
    ts_parser.add_argument("--outdir", required=True)
    ts_parser.add_argument("--fps", type=float, default=1.0)
    ts_parser.add_argument("--device", default="cuda")
    ts_parser.add_argument("--no-resnet", action="store_true",
                           help="ResNetを使わずヒューリスティック特徴を使用")
    ts_parser.add_argument("--smooth-s", type=float, default=3.0)

    # --- annotate ---
    ann_parser = subparsers.add_parser(
        "annotate", help="Step 2: フェーズアノテーション（CSV→JSONL/SRT）"
    )
    ann_parser.add_argument("--csv", required=True)
    ann_parser.add_argument("--outdir", required=True)
    ann_parser.add_argument("--features-npy", default=None)
    ann_parser.add_argument("--n-phases", type=int, default=7)
    ann_parser.add_argument("--sensitivity", type=float, default=1.5)
    ann_parser.add_argument("--min-phase-s", type=float, default=30.0)

    # --- analyze ---
    ana_parser = subparsers.add_parser(
        "analyze", help="一括実行（timeseries + annotate）"
    )
    ana_parser.add_argument("--video", required=True)
    ana_parser.add_argument("--outdir", required=True)
    ana_parser.add_argument("--fps", type=float, default=1.0)
    ana_parser.add_argument("--device", default="cuda")
    ana_parser.add_argument("--no-resnet", action="store_true")
    ana_parser.add_argument("--smooth-s", type=float, default=3.0)
    ana_parser.add_argument("--n-phases", type=int, default=7)
    ana_parser.add_argument("--sensitivity", type=float, default=1.5)
    ana_parser.add_argument("--min-phase-s", type=float, default=30.0)

    args = parser.parse_args()

    if args.command == "timeseries":
        record_timeseries(
            video_path=args.video, outdir=args.outdir,
            fps=args.fps, device=args.device,
            use_resnet=not args.no_resnet, smooth_s=args.smooth_s,
        )
    elif args.command == "annotate":
        annotate_phases(
            csv_path=args.csv, outdir=args.outdir,
            features_npy=args.features_npy,
            n_phases=args.n_phases, sensitivity=args.sensitivity,
            min_phase_s=args.min_phase_s,
        )
    elif args.command == "analyze":
        analyze_video(
            video_path=args.video, outdir=args.outdir,
            fps=args.fps, device=args.device,
            use_resnet=not getattr(args, 'no_resnet', False),
            smooth_s=args.smooth_s,
            n_phases=args.n_phases, sensitivity=args.sensitivity,
            min_phase_s=args.min_phase_s,
        )
    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
