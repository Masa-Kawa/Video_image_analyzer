"""
TransNet V2 Scene Detection Analyzer

Wraps the existing SceneDetector class to provide a pluggable analyzer interface.
"""

import logging
import threading
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import numpy as np

from src.analyzers.base import BaseAnalyzer, AnalysisResult
from src.transnet.inference import SceneDetector as TransNetDetector

logger = logging.getLogger(__name__)


class TransNetAnalyzer(BaseAnalyzer):
    """TransNet V2 scene detection analyzer."""
    
    def __init__(self, weights_path: Optional[str] = None, device: str = "cuda"):
        """
        Initialize TransNet analyzer.
        
        Args:
            weights_path: Path to model weights (optional)
            device: Device to use ('cuda' or 'cpu')
        """
        super().__init__(name="transnet_v2", version="1.0")
        
        self.weights_path = weights_path
        self.device = device
        self.detector = None  # Lazy initialization
        # Guards lazy init / teardown so concurrent analyze() calls can't
        # double-initialize the (heavy, GPU-allocating) detector.
        self._init_lock = threading.Lock()

    def _init_detector(self):
        """Initialize the TransNet detector (thread-safe lazy loading)."""
        # Double-checked locking: fast path avoids the lock once initialized.
        if self.detector is None:
            with self._init_lock:
                if self.detector is None:
                    logger.info(f"Initializing TransNet V2 detector on {self.device}")
                    self.detector = TransNetDetector(
                        weights_path=self.weights_path,
                        device=self.device
                    )

    def close(self):
        """Release the detector and free GPU memory.

        Safe to call multiple times. After close(), a subsequent analyze()
        will lazily re-initialize the detector.
        """
        with self._init_lock:
            if self.detector is None:
                return
            self.detector = None
        # Best-effort CUDA cache release (torch is an optional heavy import).
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover - torch absent / no CUDA
            pass

    def __del__(self):
        # __del__ must never raise; guard against partial construction.
        try:
            self.close()
        except Exception:  # pragma: no cover
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
    
    def analyze(
        self,
        video_path: str,
        threshold: float = 0.5,
        min_scene_length: int = 5,
        **params
    ) -> AnalysisResult:
        """
        Detect scenes in a video using TransNet V2.
        
        Args:
            video_path: Path to video file
            threshold: Detection threshold (0.0-1.0)
            min_scene_length: Minimum scene length in frames
            **params: Additional parameters (ignored)
            
        Returns:
            AnalysisResult object with detected scenes
        """
        # 入力検証: 重い初期化の前に動画パスを確認する
        video_file = Path(video_path)
        if not video_file.exists():
            raise FileNotFoundError(f"動画ファイルが見つかりません: {video_path}")
        if not video_file.is_file():
            raise ValueError(f"動画パスがファイルではありません: {video_path}")

        # Initialize detector
        self._init_detector()

        # Get video info
        video_info = self._get_video_info(video_path)

        # Run detection
        logger.info(f"Running TransNet V2 analysis on {video_file.name}")
        logger.info(f"Parameters: threshold={threshold}, min_scene_length={min_scene_length}")

        # predict_video は ffmpeg エラー / GPU OOM / 破損動画等で例外を投げうる。
        # 捕捉してログを残しつつ再送出する（inference 側で ffmpeg プロセスは
        # finally で解放される）。
        try:
            scenes, scores = self.detector.predict_video(
                video_path,
                threshold=threshold,
                return_scores=True,
                min_scene_length=min_scene_length,
            )
        except Exception as e:
            logger.error(f"TransNet 解析に失敗しました ({Path(video_path).name}): {e}")
            raise
        
        # Add confidence scores to scenes
        # Note: The original inference.py doesn't return per-scene confidence
        # We'll compute average score for each scene region
        fps = video_info['fps']
        enriched_scenes = []
        
        for scene in scenes:
            start_frame = scene['start_frame']
            end_frame = scene['end_frame']
            
            # Get scores for this scene range
            if start_frame < len(scores) and end_frame <= len(scores):
                scene_scores = scores[start_frame:end_frame]
                avg_confidence = float(scene_scores.mean()) if len(scene_scores) > 0 else 0.0
                max_confidence = float(scene_scores.max()) if len(scene_scores) > 0 else 0.0
            else:
                avg_confidence = 0.0
                max_confidence = 0.0
            
            enriched_scenes.append({
                "id": scene['id'],
                "start_frame": scene['start_frame'],
                "end_frame": scene['end_frame'],
                "start_time_sec": scene['start_time_sec'],
                "end_time_sec": scene['end_time_sec'],
                "duration_sec": scene['end_time_sec'] - scene['start_time_sec'],
                "num_frames": end_frame - start_frame,
                "avg_confidence": round(avg_confidence, 4),
                "max_confidence": round(max_confidence, 4)
            })
        
        # Create result
        result = AnalysisResult(
            analyzer_type=self.name,
            analyzer_version=self.version,
            parameters={
                "threshold": threshold,
                "min_scene_length": min_scene_length,
                "device": self.device,
                "weights": self.weights_path or "default"
            },
            video_info=video_info,
            results=enriched_scenes,
            metadata={
                "total_scenes": len(enriched_scenes),
                "total_duration": video_info['duration'],
                "max_score_in_video": float(scores.max()) if len(scores) > 0 else 0.0
            }
        )
        
        logger.info(f"Detected {len(enriched_scenes)} scenes")
        
        return result
    
    def get_parameter_info(self) -> Dict[str, Any]:
        """Get information about supported parameters."""
        return {
            "name": self.name,
            "version": self.version,
            "parameters": {
                "threshold": {
                    "type": "float",
                    "default": 0.5,
                    "range": [0.0, 1.0],
                    "description": "Scene detection threshold"
                },
                "min_scene_length": {
                    "type": "int",
                    "default": 5,
                    "range": [1, 1000],
                    "description": "Minimum scene length in frames"
                },
                "device": {
                    "type": "str",
                    "default": "cuda",
                    "choices": ["cuda", "cpu"],
                    "description": "Device for inference"
                }
            }
        }


def main():
    """CLI interface for TransNet analyzer."""
    import argparse
    
    parser = argparse.ArgumentParser(description="TransNet V2 Scene Detection Analyzer")
    parser.add_argument("video", help="Video file to analyze")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection threshold (default: 0.5)")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to model weights (default: none; uses randomly "
                             "initialized weights, consistent with the API default)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use")
    parser.add_argument("--output-json", type=str, help="Output JSON file path")
    parser.add_argument("--output-csv", type=str, help="Output CSV summary file path")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    
    # Create analyzer
    analyzer = TransNetAnalyzer(weights_path=args.weights, device=args.device)
    
    # Determine output paths
    video_path = Path(args.video)
    output_json = args.output_json or str(video_path.stem + "_transnet.json")
    output_csv = args.output_csv or str(video_path.stem + "_transnet_summary.csv")
    
    # Run analysis
    result = analyzer.analyze_and_save(
        str(video_path),
        output_json=output_json,
        output_csv=output_csv,
        threshold=args.threshold
    )
    
    print(f"\n✓ Analysis complete!")
    print(f"  Detected scenes: {len(result.results)}")
    print(f"  JSON output: {output_json}")
    print(f"  CSV summary: {output_csv}")


if __name__ == "__main__":
    main()
