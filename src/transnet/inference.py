import torch
import numpy as np
import ffmpeg
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from transnetv2_pytorch import TransNetV2

logger = logging.getLogger(__name__)

# r_frame_rate が取得できない/0/0 のときに用いるフォールバック fps
DEFAULT_FPS = 25.0


def _parse_frame_rate(stream: Dict[str, Any]) -> float:
    """r_frame_rate（"30000/1001" 等）を fps に変換する。

    r_frame_rate が欠落・"0/0"・不正な場合は avg_frame_rate を試し、
    それも駄目なら DEFAULT_FPS を返す（ZeroDivisionError を防ぐ）。
    """
    for key in ("r_frame_rate", "avg_frame_rate"):
        rate = stream.get(key)
        if not rate or rate == "0/0":
            continue
        try:
            num_str, _, den_str = str(rate).partition("/")
            num = float(num_str)
            den = float(den_str) if den_str else 1.0
            if num > 0 and den > 0:
                return num / den
        except (ValueError, TypeError):
            continue
    logger.warning(f"フレームレートを判定できません。{DEFAULT_FPS} を使用します。")
    return DEFAULT_FPS


class SceneDetector:
    def __init__(self, weights_path: Optional[str] = None, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        self.model = TransNetV2()
        
        if weights_path:
            logger.info(f"Loading weights from {weights_path}")
            # weights_only=True で pickle 経由の任意コード実行（RCE）を防ぐ。
            # 古い PyTorch（<1.13）には引数が無いためフォールバックする。
            try:
                state_dict = torch.load(
                    weights_path, map_location=self.device, weights_only=True)
            except TypeError:
                logger.warning(
                    "torch.load が weights_only 引数に未対応です。"
                    "信頼できる重みファイルのみを使用してください。")
                state_dict = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        else:
            logger.warning("No weights path provided. Using initialized weights (random).")
            
        self.model.to(self.device)
        self.model.eval()

    def predict_video(
        self, video_path: str, threshold: float = 0.5,
        return_scores: bool = False, min_scene_length: int = 5,
    ) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], np.ndarray]]:
        """
        Detect scenes in a video.
        Returns a list of scene dictionaries.
        If return_scores is True, returns (scenes, scores).

        min_scene_length: 隣接カット間がこのフレーム数未満のシーンを除去する。
        """
        # 1. Get Video Info
        # 一部のコンテナ/コーデック（VP8/VP9, VFR, ライブ等）では nb_frames や
        # duration が欠落、r_frame_rate が "0/0" になる。存在確認・ゼロ除算対策を行う。
        try:
            probe = ffmpeg.probe(video_path)
        except ffmpeg.Error as e:
            stderr = e.stderr.decode() if e.stderr else str(e)
            logger.error(f"ffmpeg probe failed: {stderr}")
            raise

        video_stream = next(
            (s for s in probe.get('streams', []) if s.get('codec_type') == 'video'),
            None,
        )
        if not video_stream:
            raise ValueError(f"No video stream found in {video_path}")

        def _safe_int(value, default=0):
            try:
                return int(value)
            except (TypeError, ValueError):
                return default

        width = _safe_int(video_stream.get('width'))
        height = _safe_int(video_stream.get('height'))
        fps = _parse_frame_rate(video_stream)
        # nb_frames は欠落しうる。あれば pbar の total / 概算に使うが、最終的な
        # フレーム数は実デコード数（processed_frames）を正とする。
        num_frames_hint = _safe_int(video_stream.get('nb_frames'), default=0)
        try:
            duration = float(video_stream.get('duration'))
        except (TypeError, ValueError):
            duration = num_frames_hint / fps if num_frames_hint else 0.0
        if num_frames_hint <= 0 and duration > 0:
            num_frames_hint = int(round(duration * fps))

        logger.info(
            f"Video: {width}x{height}, {fps:.2f} fps, "
            f"{num_frames_hint or '?'} frames, {duration:.2f}s")

        # 2. Read Video Frames (Resized to 48x27 for TransNet V2)
        # TransNet V2 expects 48x27 input.
        # We use ffmpeg to resize and output raw rgb24
        
        target_h, target_w = 27, 48
        
        process = (
            ffmpeg
            .input(video_path)
            .filter('scale', target_w, target_h)
            .output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run_async(pipe_stdout=True, pipe_stderr=True)
        )

        # 3. Process in Batches
        batch_size = 100 # TransNet V2 usually works on chunks of frames (e.g. 50 or 100)
        
        predictions = []
        frame_buffer = []

        try:
            from tqdm import tqdm
            pbar = tqdm(total=num_frames_hint or None, unit="frames",
                        desc="Detecting Scenes")
        except ImportError:
            pbar = None
            logger.info("tqdm not installed, progress bar disabled")

        processed_frames = 0
        frame_nbytes = target_h * target_w * 3

        # 例外発生時も ffmpeg サブプロセスを確実に終了させる（プロセス残存防止）。
        try:
            while True:
                in_bytes = process.stdout.read(frame_nbytes)
                if not in_bytes:
                    break

                # ストリーム終了/破損時に1フレーム未満の端数バイトが返ることがある。
                # そのまま reshape すると ValueError になるため、端数は破棄して終了する。
                if len(in_bytes) < frame_nbytes:
                    logger.warning(
                        f"末尾に不完全なフレームデータ ({len(in_bytes)}/{frame_nbytes} "
                        f"bytes) を検出したため破棄します: {video_path}")
                    break

                frame = np.frombuffer(in_bytes, np.uint8).reshape(target_h, target_w, 3)
                frame_buffer.append(frame)
                processed_frames += 1

                # tqdm の __bool__ は total/iterable が共に None だと例外を投げる
                # ため、真偽評価ではなく None 比較で判定する。
                if pbar is not None:
                    pbar.update(1)

                if len(frame_buffer) >= batch_size:
                    self._process_batch(frame_buffer, predictions)
                    frame_buffer = []

            # Process remaining
            if frame_buffer:
                self._process_batch(frame_buffer, predictions)
        finally:
            if pbar is not None:
                pbar.close()
            # stdout/stderr を閉じてプロセスの正常終了 or 強制終了を保証する。
            if process.stdout:
                process.stdout.close()
            ret = process.wait()
            if process.stderr:
                process.stderr.close()
            if ret not in (0, None):
                logger.warning(f"ffmpeg デコードが非ゼロ終了しました (code={ret})")

        # フレームが1枚もデコードできなかった場合は空結果を返す
        if not predictions:
            logger.warning(f"フレームをデコードできませんでした: {video_path}")
            empty = np.array([], dtype=np.float32)
            return ([], empty) if return_scores else []

        # 4. Post-processing (Find Scenes)
        # predictions is a numpy array of shape (N, 1) or (N,) containing scores 0-1
        scores = np.concatenate(predictions)

        # 実デコード数を最終シーンの終端フレームに使う（nb_frames 欠落に頑健）
        num_frames = processed_frames
        
        # Simple thresholding
        # A cut is where score > threshold
        # We need to define scenes as (start, end)
        
        scenes = []
        start_frame = 0
        
        # Find indices where score > threshold
        cut_indices = np.where(scores > threshold)[0]
        
        # Add 0 and last_frame if not present?
        # Actually, TransNet detects transitions.
        # If frame i is a cut, it means scene changes from i to i+1 (or around i).
        # Let's assume frame i is the *start* of a new scene? Or the end of previous?
        # Usually "shot boundary" is between frames.
        # Let's assume the detected frame is the first frame of the new scene.
        
        # We want segments.
        # Segment 1: 0 to cut_1 - 1
        # Segment 2: cut_1 to cut_2 - 1
        # ...
        
        current_start = 0
        scene_id = 1
        
        for cut_frame in cut_indices:
            # 短すぎるシーン（ノイズ的なカット）を除去する。閾値は呼び出し側から
            # min_scene_length で指定可能。
            if cut_frame - current_start < min_scene_length:
                continue
                
            scenes.append({
                "id": scene_id,
                "start_frame": int(current_start),
                "end_frame": int(cut_frame), # Exclusive end for ffmpeg duration calculation?
                "start_time_sec": float(current_start / fps),
                "end_time_sec": float(cut_frame / fps)
            })
            current_start = cut_frame
            scene_id += 1
            
        # Add last scene
        scenes.append({
            "id": scene_id,
            "start_frame": int(current_start),
            "end_frame": int(num_frames),
            "start_time_sec": float(current_start / fps),
            "end_time_sec": float(num_frames / fps)
        })
        
        max_score = np.max(scores) if len(scores) > 0 else 0.0
        logger.info(f"Max scene score detected: {max_score:.4f} (Threshold: {threshold})")
        
        if return_scores:
            return scenes, scores
        return scenes

    def _process_batch(self, frames: List[np.ndarray], predictions: List[np.ndarray]):
        # frames: List of (H, W, C) numpy arrays (uint8)
        # Create batch: (B, T, H, W, C) -> (1, T, 27, 48, 3)
        batch_np = np.array(frames, dtype=np.uint8)
        batch_np = batch_np[np.newaxis, ...] # (1, T, H, W, C)
        
        # Convert to torch tensor (uint8) and move to device.
        # NOTE: TransNetV2.forward() asserts inputs.dtype == torch.uint8 and
        # performs the float cast + /255 normalization internally
        # (permute(...).float().div_(255.)). uint8 入力は仕様であり、
        # ここで .float() に変換してはならない（アサーション違反になる）。
        tensor = torch.from_numpy(batch_np).to(self.device)
        
        with torch.no_grad():
            # Model expects (B, T, H, W, C) uint8
            # Output: (B, T, 1) logits or probs?
            # The source code returns `one_hot` which is `self.cls_layer1(x)`.
            # `cls_layer1` is Linear(..., 1). So it returns logits.
            # We need sigmoid.
            
            logits = self.model(tensor)
            if isinstance(logits, tuple):
                logits = logits[0] # Handle many_hot return if present
                
            probs = torch.sigmoid(logits)
            
        predictions.append(probs.cpu().numpy().flatten())
