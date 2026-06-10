"""
Motion Detection Analyzer

Detects motion in videos using frame difference analysis.
This serves as an example of an alternative analysis engine.
"""

import logging
from typing import Dict, Any, List
from pathlib import Path
import numpy as np
import cv2

from src.analyzers.base import BaseAnalyzer, AnalysisResult

logger = logging.getLogger(__name__)

# Default fraction of changed pixels (0..1) above which a frame counts as "in motion".
DEFAULT_MOTION_RATIO = 0.01
# Fallback frame rate used when the source video reports an invalid/zero FPS.
DEFAULT_FPS = 30.0


class MotionAnalyzer(BaseAnalyzer):
    """Motion detection analyzer using frame differencing."""
    
    def __init__(self):
        """Initialize Motion analyzer."""
        super().__init__(name="motion_detector", version="1.0")
    
    def analyze(
        self,
        video_path: str,
        threshold: float = 25.0,
        min_motion_frames: int = 5,
        sample_rate: int = 1,
        motion_ratio_threshold: float = DEFAULT_MOTION_RATIO,
        **params
    ) -> AnalysisResult:
        """
        Detect motion events in a video.

        Args:
            video_path: Path to video file
            threshold: Motion detection threshold (pixel difference)
            min_motion_frames: Minimum consecutive frames to detect as motion
            sample_rate: Sample every Nth frame (1 = every frame)
            motion_ratio_threshold: Fraction of changed pixels (0..1) above which
                a frame is considered "in motion" (default: 0.01 = 1%)
            **params: Additional parameters (ignored)
            
        Returns:
            AnalysisResult object with detected motion events
        """
        # Get video info
        video_info = self._get_video_info(video_path)
        
        logger.info(f"Running motion detection on {Path(video_path).name}")
        logger.info(f"Parameters: threshold={threshold}, sample_rate={sample_rate}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        
        # Guard against an invalid/zero FPS, which would cause ZeroDivisionError
        # when converting frame indices to seconds below.
        fps = video_info.get('fps')
        if not fps or fps <= 0:
            probe_fps = cap.get(cv2.CAP_PROP_FPS)
            fps = probe_fps if probe_fps and probe_fps > 0 else DEFAULT_FPS
            logger.warning(
                f"Invalid FPS in video info ({video_info.get('fps')!r}); "
                f"falling back to {fps}"
            )
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Motion detection
        motion_events = []
        prev_gray = None
        motion_start = None
        motion_frames_count = 0
        
        frame_idx = 0
        
        try:
            from tqdm import tqdm
            pbar = tqdm(total=total_frames, desc="Detecting motion", unit="frames")
        except ImportError:
            pbar = None

        # 処理中に例外が起きても VideoCapture / 進捗バーを確実に解放する
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Sample frames
                if frame_idx % sample_rate != 0:
                    frame_idx += 1
                    if pbar:
                        pbar.update(1)
                    continue

                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (21, 21), 0)

                if prev_gray is not None:
                    # Compute frame difference
                    frame_diff = cv2.absdiff(prev_gray, gray)
                    _, thresh = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)

                    # Calculate motion amount (ratio of changed pixels)
                    motion_amount = np.sum(thresh > 0) / thresh.size

                    # Detect motion
                    if motion_amount > motion_ratio_threshold:
                        if motion_start is None:
                            motion_start = frame_idx
                        motion_frames_count += 1
                    else:
                        # Motion ended
                        if motion_start is not None and motion_frames_count >= min_motion_frames:
                            motion_end = frame_idx - 1

                            motion_events.append({
                                "start_frame": motion_start,
                                "end_frame": motion_end,
                                "start_time_sec": motion_start / fps,
                                "end_time_sec": motion_end / fps,
                                "duration_sec": (motion_end - motion_start) / fps,
                                "num_frames": motion_end - motion_start
                            })

                        motion_start = None
                        motion_frames_count = 0

                prev_gray = gray
                frame_idx += 1

                if pbar:
                    pbar.update(1)

            # Handle last motion event
            if motion_start is not None and motion_frames_count >= min_motion_frames:
                motion_end = frame_idx - 1
                motion_events.append({
                    "start_frame": motion_start,
                    "end_frame": motion_end,
                    "start_time_sec": motion_start / fps,
                    "end_time_sec": motion_end / fps,
                    "duration_sec": (motion_end - motion_start) / fps,
                    "num_frames": motion_end - motion_start
                })
        finally:
            if pbar:
                pbar.close()
            cap.release()
        
        # Add IDs to events
        for i, event in enumerate(motion_events, 1):
            event['id'] = i
        
        # Create result
        result = AnalysisResult(
            analyzer_type=self.name,
            analyzer_version=self.version,
            parameters={
                "threshold": threshold,
                "min_motion_frames": min_motion_frames,
                "sample_rate": sample_rate,
                "motion_ratio_threshold": motion_ratio_threshold
            },
            video_info=video_info,
            results=motion_events,
            metadata={
                "total_motion_events": len(motion_events),
                "total_motion_duration": sum(e['duration_sec'] for e in motion_events)
            }
        )
        
        logger.info(f"Detected {len(motion_events)} motion events")
        
        return result
    
    def get_parameter_info(self) -> Dict[str, Any]:
        """Get information about supported parameters."""
        return {
            "name": self.name,
            "version": self.version,
            "parameters": {
                "threshold": {
                    "type": "float",
                    "default": 25.0,
                    "range": [0.0, 255.0],
                    "description": "Pixel difference threshold for motion detection"
                },
                "min_motion_frames": {
                    "type": "int",
                    "default": 5,
                    "range": [1, 1000],
                    "description": "Minimum consecutive frames to count as motion"
                },
                "sample_rate": {
                    "type": "int",
                    "default": 1,
                    "range": [1, 30],
                    "description": "Sample every Nth frame (1 = every frame)"
                },
                "motion_ratio_threshold": {
                    "type": "float",
                    "default": DEFAULT_MOTION_RATIO,
                    "range": [0.0, 1.0],
                    "description": "Fraction of changed pixels (0..1) above which a frame is 'in motion'"
                }
            }
        }


def main():
    """CLI interface for Motion analyzer."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Motion Detection Analyzer")
    parser.add_argument("video", help="Video file to analyze")
    parser.add_argument("--threshold", type=float, default=25.0, help="Motion threshold (default: 25.0)")
    parser.add_argument("--min-frames", type=int, default=5, help="Minimum motion frames (default: 5)")
    parser.add_argument("--sample-rate", type=int, default=1, help="Sample every Nth frame (default: 1)")
    parser.add_argument("--motion-ratio-threshold", type=float, default=DEFAULT_MOTION_RATIO,
                        help=f"Fraction of changed pixels to count as motion (default: {DEFAULT_MOTION_RATIO})")
    parser.add_argument("--output-json", type=str, help="Output JSON file path")
    parser.add_argument("--output-csv", type=str, help="Output CSV summary file path")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    
    # Create analyzer
    analyzer = MotionAnalyzer()
    
    # Determine output paths
    video_path = Path(args.video)
    output_json = args.output_json or str(video_path.stem + "_motion.json")
    output_csv = args.output_csv or str(video_path.stem + "_motion_summary.csv")
    
    # Run analysis
    result = analyzer.analyze_and_save(
        str(video_path),
        output_json=output_json,
        output_csv=output_csv,
        threshold=args.threshold,
        min_motion_frames=args.min_frames,
        sample_rate=args.sample_rate,
        motion_ratio_threshold=args.motion_ratio_threshold
    )
    
    print(f"\n✓ Analysis complete!")
    print(f"  Detected motion events: {len(result.results)}")
    print(f"  Total motion duration: {result.metadata['total_motion_duration']:.2f}s")
    print(f"  JSON output: {output_json}")
    print(f"  CSV summary: {output_csv}")


if __name__ == "__main__":
    main()
