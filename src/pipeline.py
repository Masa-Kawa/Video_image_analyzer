"""
Integrated Video Analysis Pipeline

End-to-end workflow:
1. Create proxy files
2. Run analysis with specified engines
3. Generate MLT file with all results
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from src.tools.proxy_manager import ProxyManager
from src.transnet.transnet_analyzer import TransNetAnalyzer
from src.motion.motion_analyzer import MotionAnalyzer
from src.mlt.mlt_generator import MLTGenerator

logger = logging.getLogger(__name__)


ANALYZERS = {
    "transnet": TransNetAnalyzer,
    "motion": MotionAnalyzer
}


def run_pipeline(
    video_paths: List[str],
    analyzers: List[str],
    create_proxy: bool = False,
    proxy_resolution: str = "720p",
    force_proxy: bool = False,
    mlt_output: Optional[str] = None,
    series_mode: bool = False,
    output_dir: Optional[str] = None,
    **analyzer_params
):
    """
    Run the complete video analysis pipeline.
    
    Args:
        video_paths: List of video file paths
        analyzers: List of analyzer names to run
        create_proxy: Whether to create proxy files
        proxy_resolution: Proxy resolution
        force_proxy: Force recreation of existing proxies
        mlt_output: Output MLT file path (optional)
        series_mode: Treat multiple videos as a series
        output_dir: Directory for output files
        **analyzer_params: Parameters for analyzers
    """
    # Determine primary video (first in list)
    primary_video = video_paths[0]
    primary_video_path = Path(primary_video)
    
    # Setup output directory
    if output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = primary_video_path.parent

    # Step 1: Create proxy files if requested
    proxy_paths = []
    if create_proxy:
        logger.info("=" * 60)
        logger.info("STEP 1: Creating Proxy Files")
        logger.info("=" * 60)
        
        proxy_manager = ProxyManager()
        
        if series_mode and len(video_paths) > 1:
            logger.info(f"Creating proxies for {len(video_paths)} videos (series mode)")
            
            def progress(current, total, filename):
                logger.info(f"[{current}/{total}] {Path(filename).name}")
            
            proxy_paths = proxy_manager.create_proxy_series(
                video_paths,
                resolution=proxy_resolution,
                force=force_proxy,
                progress_callback=progress
            )
        else:
            for video in video_paths:
                logger.info(f"Creating proxy for {Path(video).name}")
                
                try:
                    from tqdm import tqdm
                    pbar = tqdm(total=100, desc="Progress", unit="%")
                    
                    def progress(percent):
                        pbar.n = int(percent)
                        pbar.refresh()
                    
                    proxy_path = proxy_manager.create_proxy(
                        video,
                        resolution=proxy_resolution,
                        force=force_proxy,
                        progress_callback=progress
                    )
                    pbar.close()
                except ImportError:
                    proxy_path = proxy_manager.create_proxy(
                        video,
                        resolution=proxy_resolution,
                        force=force_proxy
                    )
                
                proxy_paths.append(proxy_path)
        
        logger.info(f"✓ Proxy creation complete")
    
    # Step 2: Run analysis
    results = {}
    
    if analyzers and len(analyzers) > 0:
        logger.info("=" * 60)
        logger.info("STEP 2: Running Analysis")
        logger.info("=" * 60)
        
        for analyzer_name in analyzers:
            logger.info(f"\nRunning {analyzer_name} analyzer...")
            
            # Initialize analyzer
            if analyzer_name == "transnet":
                weights = analyzer_params.get('transnet_weights', 'transnetv2-pytorch-weights.pth')
                device = analyzer_params.get('device', 'cuda')
                analyzer = TransNetAnalyzer(weights_path=weights, device=device)
                
                threshold = analyzer_params.get('transnet_threshold', 0.5)
                params = {'threshold': threshold}
            elif analyzer_name == "motion":
                analyzer = MotionAnalyzer()
                
                threshold = analyzer_params.get('motion_threshold', 25.0)
                sample_rate = analyzer_params.get('motion_sample_rate', 1)
                params = {
                    'threshold': threshold,
                    'sample_rate': sample_rate
                }
            else:
                logger.warning(f"Unknown analyzer: {analyzer_name}, skipping")
                continue
        
            # Series mode: analyze all videos and merge results
            if series_mode and len(video_paths) > 1:
                logger.info(f"Analyzing {len(video_paths)} videos in series mode...")
            
                all_results = []
                cumulative_duration = 0.0
                failed_videos = []
            
                for idx, video_path in enumerate(video_paths):
                    logger.info(f"  [{idx+1}/{len(video_paths)}] Analyzing {Path(video_path).name}")
                
                    # Skip if proxy creation failed for this file
                    if proxy_paths and idx < len(proxy_paths) and proxy_paths[idx] is None:
                        logger.warning(f"    Skipping (proxy creation failed)")
                        failed_videos.append(video_path)
                        continue
                
                    try:
                        # Use proxy if it exists and was created
                        analysis_video = video_path
                        if proxy_paths and idx < len(proxy_paths) and proxy_paths[idx] is not None:
                            analysis_video = str(proxy_paths[idx])
                            logger.info(f"    Using proxy: {Path(analysis_video).name}")
                    
                        # Analyze this video
                        video_result = analyzer.analyze(analysis_video, **params)
                    
                        # Adjust timestamps for series (add cumulative duration)
                        for detection in video_result.results:
                            detection['start_time_sec'] += cumulative_duration
                            detection['end_time_sec'] += cumulative_duration
                            detection['video_index'] = idx
                            detection['video_name'] = Path(video_path).name
                    
                        all_results.extend(video_result.results)
                        cumulative_duration += video_result.video_info['duration']
                    
                    except Exception as e:
                        logger.error(f"    Failed to analyze {Path(video_path).name}: {e}")
                        logger.warning(f"    Skipping corrupted/invalid video")
                        failed_videos.append(video_path)
                        continue
            
                # Report on processing results
                successful_count = len(video_paths) - len(failed_videos)
                if failed_videos:
                    logger.warning(f"⚠ {len(failed_videos)} video(s) failed during analysis:")
                    for failed in failed_videos:
                        logger.warning(f"  - {Path(failed).name}")
                    logger.info(f"✓ {successful_count} video(s) analyzed successfully")
            
                # Create merged result
                from src.analyzers.base import AnalysisResult
            
                # Use first video's info as base, update with series info
                series_video_info = video_result.video_info.copy()
                series_video_info['series_mode'] = True
                series_video_info['num_videos'] = len(video_paths)
                series_video_info['total_duration'] = cumulative_duration
                series_video_info['video_paths'] = [str(Path(v).resolve()) for v in video_paths]
            
                merged_result = AnalysisResult(
                    analyzer_type=video_result.analyzer_type,
                    analyzer_version=video_result.analyzer_version,
                    parameters=video_result.parameters,
                    video_info=series_video_info,
                    results=all_results,
                    metadata={
                        **video_result.metadata,
                        'series_mode': True,
                        'total_detections': len(all_results)
                    }
                )
            
                # Determine output paths (use series name or first video name)
                if len(video_paths) > 1:
                    series_name = f"series_{len(video_paths)}videos"
                else:
                    series_name = primary_video_path.stem
            
                output_json = out_dir / f"{series_name}_{analyzer_name}.json"
                output_csv = out_dir / f"{series_name}_{analyzer_name}_summary.csv"
            
                # Save results
                merged_result.save_json(str(output_json))
                merged_result.save_csv_summary(str(output_csv))
            
                results[analyzer_name] = {
                    'result': merged_result,
                    'json_path': output_json,
                    'csv_path': output_csv
                }
            
                logger.info(f"✓ {analyzer_name} series analysis complete")
                logger.info(f"  Total detections: {len(all_results)}")
                logger.info(f"  Total duration: {cumulative_duration:.2f}s")
                logger.info(f"  JSON: {output_json}")
                logger.info(f"  CSV: {output_csv}")
        
            else:
                # Single video mode
                output_json = out_dir / f"{primary_video_path.stem}_{analyzer_name}.json"
                output_csv = out_dir / f"{primary_video_path.stem}_{analyzer_name}_summary.csv"
            
                # Use proxy if it exists and was created
                analysis_video = primary_video
                if proxy_paths and len(proxy_paths) > 0:
                    analysis_video = str(proxy_paths[0])
                    logger.info(f"Using proxy for analysis: {Path(analysis_video).name}")
            
                # Run analysis
                result = analyzer.analyze_and_save(
                    analysis_video,
                    output_json=str(output_json),
                    output_csv=str(output_csv),
                    **params
                )
            
                results[analyzer_name] = {
                    'result': result,
                    'json_path': output_json,
                    'csv_path': output_csv
                }
            
                logger.info(f"✓ {analyzer_name} analysis complete")
                logger.info(f"  JSON: {output_json}")
                logger.info(f"  CSV: {output_csv}")
    
    # Step 3: Generate MLT file
    if mlt_output:
        logger.info("=" * 60)
        logger.info("STEP 3: Generating MLT File")
        logger.info("=" * 60)
        
        mlt_path = Path(mlt_output)
        
        # Create MLT generator with series support
        if series_mode and len(video_paths) > 1:
            logger.info(f"Generating MLT for {len(video_paths)} video series")
            generator = MLTGenerator(primary_video, video_paths=video_paths)
        else:
            generator = MLTGenerator(primary_video)
        
        # Add cuts from TransNet if available
        if 'transnet' in results:
            generator.add_cuts(results['transnet']['result'].results)
            logger.info(f"Added {len(results['transnet']['result'].results)} cuts to timeline")
        
        # Add annotations from motion if available
        if 'motion' in results:
            generator.add_annotation_track(
                results['motion']['result'].results,
                track_name="Motion Events"
            )
            logger.info(f"Added {len(results['motion']['result'].results)} motion annotations")
        
        # Generate
        generator.generate(str(mlt_path))
        
        logger.info(f"✓ MLT file generated: {mlt_path}")
    
    # Summary
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    
    if create_proxy:
        logger.info(f"Proxy files: {len(proxy_paths)}")
    
    logger.info(f"Analyses run: {len(results)}")
    for name, data in results.items():
        logger.info(f"  - {name}: {len(data['result'].results)} detections")
    
    if mlt_output:
        logger.info(f"MLT output: {mlt_output}")
    
    return results


def main():
    """CLI interface for the integrated pipeline."""
    parser = argparse.ArgumentParser(
        description="Integrated Video Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic scene detection
  python pipeline.py video.mp4 --analyze transnet --mlt-output project.mlt
  
  # With proxy creation
  python pipeline.py video.mp4 --create-proxy --analyze transnet,motion --mlt-output project.mlt
  
  # Series mode
  python pipeline.py part1.mp4 part2.mp4 part3.mp4 --series --analyze transnet --mlt-output series.mlt
  
  # Custom parameters
  python pipeline.py video.mp4 --analyze transnet --transnet-threshold 0.3 --device cpu
        """
    )
    
    # Input files
    parser.add_argument("videos", nargs="+", help="Video file(s) to process")
    parser.add_argument("--series", action="store_true", help="Treat as series (multiple videos)")
    
    # Proxy options
    parser.add_argument("--create-proxy", action="store_true", help="Create proxy files")
    parser.add_argument("--proxy-resolution", default="720p", choices=["360p", "480p", "720p", "1080p"],
                       help="Proxy resolution (default: 720p)")
    parser.add_argument("--force-proxy", action="store_true", help="Force proxy recreation")
    
    # Analysis options
    parser.add_argument("--analyze", type=str,
                       help="Analyzers to run (comma-separated): transnet, motion")
    
    # Output options
    parser.add_argument("--mlt-output", type=str, help="Output MLT file path")
    parser.add_argument("--output-dir", type=str, help="Directory for output files")
    
    # TransNet parameters
    parser.add_argument("--transnet-threshold", type=float, default=0.5,
                       help="TransNet detection threshold (default: 0.5)")
    parser.add_argument("--transnet-weights", type=str, default="transnetv2-pytorch-weights.pth",
                       help="TransNet weights file")
    
    # Motion parameters
    parser.add_argument("--motion-threshold", type=float, default=25.0,
                       help="Motion detection threshold (default: 25.0)")
    parser.add_argument("--motion-sample-rate", type=int, default=1,
                       help="Motion detection sample rate (default: 1)")
    
    # General parameters
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                       help="Device for analysis (default: cuda)")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Parse analyzers
    if args.analyze:
        analyzers = [a.strip() for a in args.analyze.split(',')]
    elif args.create_proxy:
        analyzers = [] # No analyzers, but proxy creation is requested
    else:
        logger.error("No analyzers specified and --create-proxy not set. Nothing to do.")
        sys.exit(1)
    
    # Validate videos
    for video in args.videos:
        if not Path(video).exists():
            logger.error(f"Video not found: {video}")
            sys.exit(1)
    
    # Run pipeline
    try:
        run_pipeline(
            video_paths=args.videos,
            analyzers=analyzers,
            create_proxy=args.create_proxy,
            proxy_resolution=args.proxy_resolution,
            force_proxy=args.force_proxy,
            mlt_output=args.mlt_output,
            series_mode=args.series,
            output_dir=args.output_dir,
            transnet_threshold=args.transnet_threshold,
            transnet_weights=args.transnet_weights,
            motion_threshold=args.motion_threshold,
            motion_sample_rate=args.motion_sample_rate,
            device=args.device
        )
        
        print("\n" + "=" * 60)
        print("✓ ALL TASKS COMPLETED SUCCESSFULLY")
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
