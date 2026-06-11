"""
Proxy File Manager

This module handles the creation and management of proxy video files.
Supports both single videos and series (multiple video files).
"""

import subprocess
import sys
import math
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Callable
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

    # h264_nvenc が ffmpeg に存在するかの判定結果をキャッシュ（プロセス内で
    # 一度だけ ffmpeg を起動。series モードでの繰り返し起動を避ける）。
    _nvenc_available: Optional[bool] = None

    @classmethod
    def _check_nvenc(cls) -> bool:
        """Return True if this ffmpeg build exposes the h264_nvenc encoder."""
        if cls._nvenc_available is None:
            try:
                out = subprocess.run(
                    ["ffmpeg", "-hide_banner", "-encoders"],
                    capture_output=True, text=True, check=True
                )
                cls._nvenc_available = "h264_nvenc" in out.stdout
            except (subprocess.CalledProcessError, FileNotFoundError):
                cls._nvenc_available = False
        return cls._nvenc_available

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
        progress_callback: Optional[Callable[[float], None]] = None,
        gpu: bool = False,
        fps: float = 30.0
    ) -> Path:
        """
        Create a proxy file for a video.

        Args:
            video_path: Path to original video
            resolution: Target resolution (e.g., "720p")
            force: If True, recreate even if proxy exists
            progress_callback: Optional callback function for progress updates (0-100)
            gpu: If True, use NVIDIA NVENC (h264_nvenc) hardware encoding
            fps: Output frame rate. Source is downsampled to this rate to keep
                proxies small. Analysis samples at <=5 fps, so 30 (default) is
                ample; use 15 for very long recordings. Pass 0 to keep source fps.

        Returns:
            Path to the created proxy file
        """
        video_path = Path(video_path)

        # 引数検証はファイル存在チェックより前に行う。
        # fps: NaN/Inf/負値はそのまま渡すと不明瞭な ffmpeg エラーになる。
        # 0 は「元の fps を維持」の意味で許容する。
        if not math.isfinite(fps) or fps < 0:
            raise ValueError(f"fps must be a finite, non-negative number, got {fps!r}")

        # GPU 指定時は h264_nvenc が使えるか先に確認し、未対応環境では
        # 分かりやすいメッセージで失敗させる（ffmpeg の難解なエラー回避）。
        if gpu and not self._check_nvenc():
            raise RuntimeError(
                "GPU encoding requested but h264_nvenc is unavailable in this "
                "ffmpeg build / NVIDIA setup. Re-run without --gpu for CPU encoding."
            )

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
        if gpu:
            # NVIDIA NVENC hardware encoding. -cq alone does not constrain
            # bitrate the way libx264's -crf does, so high-motion footage
            # (e.g. surgical video) can balloon larger than the source.
            # Use VBR with a bitrate cap to keep proxies lightweight.
            logger.info("Using GPU encoding (h264_nvenc, capped VBR)")
            video_codec = [
                "-c:v", "h264_nvenc",
                "-preset", "p4",
                "-rc", "vbr",
                "-cq", "27",
                "-b:v", "4M",
                "-maxrate", "6M",
                "-bufsize", "8M",
            ]
        else:
            video_codec = ["-c:v", "libx264", "-preset", "medium", "-crf", "23"]

        # Build the video filter chain: scale, plus optional fps downsample.
        vf = f"scale={target_width}:{target_height}:force_original_aspect_ratio=decrease"
        if fps and fps > 0:
            vf += f",fps={fps}"
            logger.info(f"Downsampling to {fps} fps")

        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output
            "-i", str(video_path),
            "-vf", vf,
            *video_codec,
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
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        gpu: bool = False,
        fps: float = 30.0
    ) -> List[Optional[Path]]:
        """
        Create proxy files for multiple videos (series mode).

        Args:
            video_paths: List of video file paths
            resolution: Target resolution
            force: Force recreation of existing proxies
            progress_callback: Callback(current, total, filename)
            gpu: If True, use NVIDIA NVENC (h264_nvenc) hardware encoding

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
                
                proxy_path = self.create_proxy(video_path, resolution, force, gpu=gpu, fps=fps)
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

    def merge_proxies(
        self,
        proxy_paths: List[Optional[Path]],
        output_path: Path,
        cleanup: bool = False,
    ) -> Path:
        """
        Concatenate multiple proxy files into one using ffmpeg's concat
        demuxer with stream copy (no re-encode: lossless and near-instant).

        All inputs must share identical codec/resolution/params, which is
        guaranteed when they come from this class's create_proxy().

        Args:
            proxy_paths: Proxy files to merge, in order (None entries skipped).
            output_path: Destination merged file.
            cleanup: If True, delete the individual inputs after a successful
                     merge.

        Returns:
            Path to the merged file.
        """
        import tempfile
        import os

        valid = [Path(p) for p in proxy_paths if p is not None]
        if not valid:
            raise ValueError("マージ対象のプロキシがありません")
        if len(valid) == 1:
            logger.info("プロキシが1本のみのためマージ不要です")
            return valid[0]

        output_path = Path(output_path)

        # 入出力衝突チェック: 出力が入力のいずれかと同一だと、ffmpeg の -y で
        # 読み込み中にトランケートされ破損し、cleanup でさらに他の入力も消えて
        # 復元不能なデータ損失になる。事前に検出して拒否する。
        out_resolved = output_path.resolve()
        inputs_resolved = [p.resolve() for p in valid]
        if out_resolved in inputs_resolved:
            raise ValueError(
                f"出力先が入力ファイルと同一です: {output_path}. "
                "別の出力名を指定してください。"
            )

        # concat デマクサのリスト形式は 1 行 1 ファイルで、シングルクォートを
        # '\'' でエスケープする。改行・復帰・NUL はリストの行構造を壊し、
        # 意図しないファイル読み込みを招くため、含むパスは拒否する。
        def _quote(p: Path) -> str:
            s = str(p.resolve())
            if any(c in s for c in ("\n", "\r", "\x00")):
                raise ValueError(
                    f"パスに改行/NUL 文字が含まれており安全に連結できません: {p!r}"
                )
            return s.replace("'", "'\\''")

        with tempfile.NamedTemporaryFile(
            "w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            list_path = f.name
            for p in valid:
                f.write(f"file '{_quote(p)}'\n")

        try:
            cmd = [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-i", list_path,
                "-c", "copy",
                str(output_path),
            ]
            logger.info(f"Merging {len(valid)} proxies -> {output_path}")
            subprocess.run(cmd, check=True, capture_output=True)
        finally:
            os.unlink(list_path)

        logger.info(f"Merged proxy created: {output_path}")

        if cleanup:
            for p in valid:
                if p.resolve() != out_resolved:
                    try:
                        p.unlink()
                        logger.info(f"Removed intermediate proxy: {p}")
                    except OSError as e:
                        logger.warning(f"Could not remove {p}: {e}")

        return output_path


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
    parser.add_argument("--gpu", action="store_true",
                       help="Use NVIDIA NVENC (h264_nvenc) hardware encoding for faster proxies")
    parser.add_argument("--fps", type=float, default=30.0,
                       help="Proxy frame rate (default: 30). Analysis samples at "
                            "<=5 fps, so 30 is ample; use 15 for very long (5-6h) "
                            "recordings to halve size again. Use 0 to keep source fps.")
    parser.add_argument("--merge", action="store_true",
                       help="Concatenate the created proxies into a single file "
                            "(lossless stream copy). Useful for split recordings "
                            "of one continuous session.")
    parser.add_argument("--merge-name", type=str,
                       help="Filename for the merged output (default: "
                            "<first video stem>_merged_<resolution>.MP4)")
    parser.add_argument("--keep-parts", action="store_true",
                       help="With --merge, keep the individual proxy parts "
                            "(default: delete them after a successful merge)")

    args = parser.parse_args()

    # シェルでグロブ(*.mp4 等)が展開されないと、ワイルドカードを含む
    # リテラル文字列がそのまま渡され、分かりにくい FileNotFoundError になる。
    # 早期に検出して原因(大文字小文字の不一致など)を案内する。
    # メタ文字 [ ] { } は実ファイル名にも現れうるため、「グロブ文字を含み
    # かつ実在しない」場合のみ未展開と判定し、誤検知を防ぐ。
    glob_chars = "*?[]{}"
    unexpanded = [
        v for v in args.videos
        if any(c in v for c in glob_chars) and not Path(v).exists()
    ]
    if unexpanded:
        print(
            "✗ ワイルドカードが展開されていません: "
            + ", ".join(unexpanded)
            + "\n  シェルがパターンに一致するファイルを見つけられませんでした。"
            "\n  拡張子の大文字・小文字(例: .mp4 と .MP4)やパスを確認してください。",
            file=sys.stderr,
        )
        return 1
    
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
    # 非 series モードで作成したプロキシのパス（--merge で連結に使う）。
    created_paths: List[Optional[Path]] = []

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
                progress_callback=series_progress,
                gpu=args.gpu,
                fps=args.fps
            )
            pbar.close()
        else:
            for video in args.videos:
                pbar = tqdm(total=100, desc=f"Creating proxy: {Path(video).name}", unit="%")

                def file_progress(percent):
                    pbar.n = int(percent)
                    pbar.refresh()

                created_paths.append(manager.create_proxy(
                    video,
                    resolution=args.resolution,
                    force=args.force,
                    progress_callback=file_progress,
                    gpu=args.gpu,
                    fps=args.fps
                ))
                pbar.close()
    except ImportError:
        # No tqdm, simple logging
        if args.series:
            series_results = manager.create_proxy_series(
                args.videos,
                resolution=args.resolution,
                force=args.force,
                gpu=args.gpu,
                fps=args.fps
            )
        else:
            for video in args.videos:
                created_paths.append(manager.create_proxy(
                    video,
                    resolution=args.resolution,
                    force=args.force,
                    gpu=args.gpu,
                    fps=args.fps
                ))

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

    # --merge: 作成した全プロキシを 1 本に連結する。
    if args.merge:
        proxy_paths = series_results if series_results is not None else created_paths
        valid = [p for p in proxy_paths if p is not None]
        if len(valid) < 2:
            print("ℹ プロキシが1本のみのため、連結はスキップしました。",
                  file=sys.stderr)
        else:
            if args.merge_name:
                # パストラバーサル対策: ユーザー指定名はファイル名部分のみを
                # 採用し、ディレクトリ成分（/, \\, .. 等）を捨てて出力先を
                # プロキシ格納ディレクトリ内に固定する。
                merge_name = Path(args.merge_name).name
                if not merge_name or merge_name in (".", ".."):
                    print(f"✗ 不正な --merge-name です: {args.merge_name!r}",
                          file=sys.stderr)
                    return 1
            else:
                first_stem = Path(args.videos[0]).stem
                merge_name = f"{first_stem}_merged_{args.resolution}.MP4"
            out_dir = valid[0].parent
            output_path = out_dir / merge_name
            try:
                manager.merge_proxies(
                    valid, output_path, cleanup=not args.keep_parts
                )
                print(f"✓ Merged proxy created: {output_path}")
            except Exception as e:
                print(f"✗ プロキシの連結に失敗しました: {e}", file=sys.stderr)
                return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
