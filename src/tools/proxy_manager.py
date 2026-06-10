"""
Proxy File Manager

This module handles the creation and management of proxy video files.
Supports both single videos and series (multiple video files).
"""

import subprocess
import sys
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Callable
import json
import ffmpeg

logger = logging.getLogger(__name__)


class ProxyManager:
    """Manages proxy file creation and detection."""
    
    RESOLUTIONS = {
        "360p": (640, 360),
        "480p": (854, 480),
        "720p": (1280, 720),
        "1080p": (1920, 1080)
    }
    
    def __init__(self, proxy_dir: Optional[Path] = None):
        """
        Initialize ProxyManager.
        
        Args:
            proxy_dir: Directory to store proxy files. If None, proxies are
                      created in the same directory as the source video.
        """
        self.proxy_dir = proxy_dir
        if self.proxy_dir:
            self.proxy_dir.mkdir(parents=True, exist_ok=True)
    
    def get_proxy_path(self, video_path: str, resolution: str = "720p") -> Path:
        """
        Get the expected path for a proxy file.
        
        Args:
            video_path: Path to original video
            resolution: Target resolution (e.g., "720p")
            
        Returns:
            Path object for the proxy file
        """
        video_path = Path(video_path)
        proxy_name = f"{video_path.stem}_{resolution}{video_path.suffix}"
        
        if self.proxy_dir:
            return self.proxy_dir / proxy_name
        else:
            return video_path.parent / proxy_name
    
    def proxy_exists(self, video_path: str, resolution: str = "720p") -> bool:
        """
        Check if a proxy file already exists.
        
        Args:
            video_path: Path to original video
            resolution: Target resolution
            
        Returns:
            True if proxy exists and is valid, False otherwise
        """
        proxy_path = self.get_proxy_path(video_path, resolution)
        
        if not proxy_path.exists():
            return False
        
        # Basic validation: check if file size > 0
        if proxy_path.stat().st_size == 0:
            logger.warning(f"Proxy file exists but is empty: {proxy_path}")
            return False
        
        return True
    
    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """
        Get video metadata using ffprobe.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video metadata
        """
        try:
            probe = ffmpeg.probe(video_path)
            video_stream = next(
                (s for s in probe['streams'] if s['codec_type'] == 'video'),
                None
            )
            
            if not video_stream:
                raise ValueError("No video stream found")
            
            # Parse frame rate
            fps_parts = video_stream['r_frame_rate'].split('/')
            fps = float(fps_parts[0]) / float(fps_parts[1])
            
            return {
                "path": str(Path(video_path).resolve()),
                "width": int(video_stream['width']),
                "height": int(video_stream['height']),
                "fps": fps,
                "duration": float(video_stream.get('duration', 0)),
                "num_frames": int(video_stream.get('nb_frames', 0)),
                "codec": video_stream.get('codec_name', 'unknown')
            }
        except Exception as e:
            logger.error(f"Failed to get video info: {e}")
            raise
    
    def create_proxy(
        self,
        video_path: str,
        resolution: str = "720p",
        force: bool = False,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> Path:
        """
        Create a proxy file for a video.
        
        Args:
            video_path: Path to original video
            resolution: Target resolution (e.g., "720p")
            force: If True, recreate even if proxy exists
            progress_callback: Optional callback function for progress updates (0-100)
            
        Returns:
            Path to the created proxy file
        """
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        proxy_path = self.get_proxy_path(str(video_path), resolution)
        
        # Skip if exists and not forcing
        if not force and self.proxy_exists(str(video_path), resolution):
            logger.info(f"Proxy already exists, skipping: {proxy_path}")
            return proxy_path
        
        # Get target dimensions
        if resolution not in self.RESOLUTIONS:
            raise ValueError(f"Unsupported resolution: {resolution}")
        
        target_width, target_height = self.RESOLUTIONS[resolution]
        
        logger.info(f"Creating proxy: {proxy_path}")
        logger.info(f"Target resolution: {target_width}x{target_height}")
        
        # Get video info for progress tracking
        try:
            video_info = self.get_video_info(str(video_path))
            total_duration = video_info['duration']
        except Exception:
            total_duration = None
        
        # Build ffmpeg command
        # Use scale filter to maintain aspect ratio
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output
            "-i", str(video_path),
            "-vf", f"scale={target_width}:{target_height}:force_original_aspect_ratio=decrease",
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "128k",
            "-progress", "pipe:1",  # Progress to stdout
            str(proxy_path)
        ]
        
        # Execute with progress tracking
        try:
            if progress_callback and total_duration:
                self._run_with_progress(cmd, total_duration, progress_callback)
            else:
                subprocess.run(cmd, check=True, capture_output=True)
            
            logger.info(f"Proxy created successfully: {proxy_path}")
            return proxy_path
            
        except subprocess.CalledProcessError as e:
            # e.stderr は capture_output 経由では bytes、_run_with_progress
            # (universal_newlines=True) 経由では str になりうる。両対応する。
            stderr = e.stderr
            if isinstance(stderr, bytes):
                stderr = stderr.decode(errors="replace")
            logger.error(f"Failed to create proxy: {stderr}")
            # Clean up partial file
            if proxy_path.exists():
                proxy_path.unlink()
            raise
    
    def create_proxy_series(
        self,
        video_paths: List[str],
        resolution: str = "720p",
        force: bool = False,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> List[Optional[Path]]:
        """
        Create proxy files for multiple videos (series mode).
        
        Args:
            video_paths: List of video file paths
            resolution: Target resolution
            force: Force recreation of existing proxies
            progress_callback: Callback(current, total, filename)
        
        Returns:
            List of created/existing proxy paths (None for failed files)
        """
        logger.info(f"Creating proxies for {len(video_paths)} videos")
        
        proxy_paths = []
        failed_files = []
        total = len(video_paths)
        
        for i, video_path in enumerate(video_paths, 1):
            try:
                logger.info(f"Processing video {i}/{total}: {video_path}")
                if progress_callback:
                    progress_callback(i, total, video_path)
                
                proxy_path = self.create_proxy(video_path, resolution, force)
                proxy_paths.append(proxy_path)
                
            except Exception as e:
                logger.error(f"Failed to create proxy for {Path(video_path).name}: {e}")
                logger.warning(f"Skipping corrupted/invalid file: {video_path}")
                failed_files.append(video_path)
                proxy_paths.append(None)  # Add None to maintain index correspondence
        
        if failed_files:
            logger.warning(f"⚠ {len(failed_files)} file(s) failed:")
            for failed in failed_files:
                logger.warning(f"  - {Path(failed).name}")
            logger.info(f"✓ {len(proxy_paths) - len(failed_files)} proxy files created successfully")
        
        if progress_callback:
            progress_callback(total, total, "Complete")
        
        return proxy_paths
    
    def _run_with_progress(
        self,
        cmd: List[str],
        total_duration: float,
        progress_callback: Callable[[float], None]
    ):
        """
        Run ffmpeg command with progress tracking.
        
        Args:
            cmd: ffmpeg command list
            total_duration: Total video duration in seconds
            progress_callback: Callback function receiving percentage (0-100)
        """
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        current_time = 0.0
        
        for line in process.stdout:
            if line.startswith("out_time_ms="):
                # Parse microseconds
                time_str = line.split("=")[1].strip()
                
                # Handle N/A values
                if time_str == "N/A":
                    continue
                
                try:
                    time_us = int(time_str)
                    current_time = time_us / 1_000_000  # Convert to seconds
                    
                    if total_duration > 0:
                        progress = (current_time / total_duration) * 100
                        progress_callback(min(progress, 100))
                except ValueError:
                    # Skip invalid values
                    continue
        
        process.wait()
        
        if process.returncode != 0:
            stderr = process.stderr.read()
            raise subprocess.CalledProcessError(process.returncode, cmd, stderr=stderr)


def main():
    """CLI interface for proxy manager."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create proxy files for video editing")
    parser.add_argument("videos", nargs="+", help="Video file(s) to create proxies for")
    parser.add_argument("--resolution", default="720p", choices=["360p", "480p", "720p", "1080p"],
                       help="Proxy resolution (default: 720p)")
    parser.add_argument("--force", action="store_true", help="Force recreation of existing proxies")
    parser.add_argument("--proxy-dir", type=str, help="Directory to store proxy files")
    parser.add_argument("--series", action="store_true", help="Treat as series (multiple videos)")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    
    # Create proxy manager
    proxy_dir = Path(args.proxy_dir) if args.proxy_dir else None
    manager = ProxyManager(proxy_dir=proxy_dir)
    
    # series モードでは create_proxy_series が失敗ファイルに None を返すため、
    # 結果リストを受け取り終了時に失敗有無を反映する。
    series_results: Optional[List[Optional[Path]]] = None

    # Progress callback with tqdm if available
    try:
        from tqdm import tqdm

        if args.series:
            pbar = tqdm(total=len(args.videos), desc="Creating proxies", unit="video")

            def series_progress(current, total, filename):
                pbar.n = current
                pbar.set_postfix({"current": Path(filename).name})
                pbar.refresh()

            series_results = manager.create_proxy_series(
                args.videos,
                resolution=args.resolution,
                force=args.force,
                progress_callback=series_progress
            )
            pbar.close()
        else:
            for video in args.videos:
                pbar = tqdm(total=100, desc=f"Creating proxy: {Path(video).name}", unit="%")

                def file_progress(percent):
                    pbar.n = int(percent)
                    pbar.refresh()

                manager.create_proxy(
                    video,
                    resolution=args.resolution,
                    force=args.force,
                    progress_callback=file_progress
                )
                pbar.close()
    except ImportError:
        # No tqdm, simple logging
        if args.series:
            series_results = manager.create_proxy_series(
                args.videos,
                resolution=args.resolution,
                force=args.force
            )
        else:
            for video in args.videos:
                manager.create_proxy(
                    video,
                    resolution=args.resolution,
                    force=args.force
                )

    # series モードでは None（失敗）の件数を集計し、あれば警告して終了コード1。
    if series_results is not None:
        failed = sum(1 for p in series_results if p is None)
        succeeded = len(series_results) - failed
        if failed:
            print(f"⚠ {succeeded} proxy file(s) created, {failed} failed",
                  file=sys.stderr)
            return 1
        print(f"✓ All {succeeded} proxies created successfully")
    else:
        print("✓ All proxies created successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
