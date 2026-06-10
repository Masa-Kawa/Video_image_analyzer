"""
Base class for video analyzers.

All analyzer implementations should inherit from BaseAnalyzer.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import json
import csv
import logging

import ffmpeg

logger = logging.getLogger(__name__)


def _safe_number(value: Any, cast, default):
    """Coerce ffprobe field values to a number, tolerating None/'N/A'.

    ffprobe sometimes reports non-numeric placeholders (e.g. 'N/A') for
    fields like duration or nb_frames. Cast defensively and fall back to
    ``default`` when the value is missing or not parseable.
    """
    if value is None:
        return default
    try:
        return cast(value)
    except (TypeError, ValueError):
        return default


@dataclass
class AnalysisResult:
    """Standard format for analysis results."""
    
    analyzer_type: str
    analyzer_version: str
    parameters: Dict[str, Any]
    video_info: Dict[str, Any]
    results: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def save_json(self, output_path: Union[str, Path]):
        """
        Save results as JSON file.
        
        Args:
            output_path: Path to output JSON file
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"Saved JSON results to {output_path}")
    
    def save_csv_summary(self, output_path: Union[str, Path]):
        """
        Save human-readable summary as CSV file.
        
        Args:
            output_path: Path to output CSV file
        """
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header section
            writer.writerow(["# Analysis Summary"])
            writer.writerow([])
            writer.writerow(["Analyzer Type", self.analyzer_type])
            writer.writerow(["Analyzer Version", self.analyzer_version])
            writer.writerow([])
            
            # Parameters section
            writer.writerow(["# Analysis Parameters"])
            for key, value in self.parameters.items():
                writer.writerow([key, value])
            writer.writerow([])
            
            # Video info section
            writer.writerow(["# Video Information"])
            for key, value in self.video_info.items():
                writer.writerow([key, value])
            writer.writerow([])
            
            # Results section
            writer.writerow(["# Detection Results"])
            writer.writerow([])
            
            if self.results:
                # Build header as the union of all keys across results so that
                # entries with non-uniform schemas are not silently dropped.
                # First-seen order is preserved for readability.
                headers: List[str] = []
                seen = set()
                for result in self.results:
                    for key in result.keys():
                        if key not in seen:
                            seen.add(key)
                            headers.append(key)
                writer.writerow(headers)

                for result in self.results:
                    row = [result.get(key, '') for key in headers]
                    writer.writerow(row)
            else:
                writer.writerow(["No results found"])
        
        logger.info(f"Saved CSV summary to {output_path}")


class BaseAnalyzer(ABC):
    """Abstract base class for video analyzers."""
    
    def __init__(self, name: str, version: str):
        """
        Initialize analyzer.
        
        Args:
            name: Analyzer name
            version: Analyzer version
        """
        self.name = name
        self.version = version
    
    @abstractmethod
    def analyze(
        self,
        video_path: str,
        **params
    ) -> AnalysisResult:
        """
        Analyze a video file.
        
        Args:
            video_path: Path to video file
            **params: Analyzer-specific parameters
            
        Returns:
            AnalysisResult object containing detection results
        """
        pass
    
    def get_parameter_info(self) -> Dict[str, Any]:
        """
        Get information about supported parameters.
        
        Returns:
            Dictionary describing supported parameters
        """
        return {
            "name": self.name,
            "version": self.version,
            "parameters": {}
        }
    
    def _get_video_info(self, video_path: str) -> Dict[str, Any]:
        """
        Helper method to get video metadata.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video metadata
        """
        # Validate the input path before handing it to ffmpeg, which builds a
        # subprocess command line from it. Resolve to an absolute path and
        # confirm it points at an existing regular file so that malformed or
        # crafted paths cannot trigger unintended file access.
        path = Path(video_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        try:
            probe = ffmpeg.probe(str(path))
            video_stream = next(
                (s for s in probe['streams'] if s['codec_type'] == 'video'),
                None
            )
            
            if not video_stream:
                raise ValueError("No video stream found")
            
            # Parse frame rate. r_frame_rate is a "num/den" string; corrupt
            # files or special streams may report a zero denominator (e.g.
            # "0/0"), so guard against division by zero and fall back to 0.
            fps = 0.0
            fps_parts = video_stream.get('r_frame_rate', '0/0').split('/')
            if len(fps_parts) == 2:
                num, den = float(fps_parts[0]), float(fps_parts[1])
                fps = num / den if den != 0 else 0.0
            
            # ffprobe may report non-numeric placeholders such as 'N/A' for
            # duration/nb_frames; coerce defensively so a single bad field
            # does not crash the whole probe.
            return {
                "path": str(path),
                "width": int(video_stream['width']),
                "height": int(video_stream['height']),
                "fps": fps,
                "duration": _safe_number(video_stream.get('duration'), float, 0.0),
                "total_frames": _safe_number(video_stream.get('nb_frames'), int, 0),
                "codec": video_stream.get('codec_name', 'unknown')
            }
        except Exception as e:
            logger.error(f"Failed to get video info: {e}")
            raise
    
    def analyze_and_save(
        self,
        video_path: Union[str, Path],
        output_json: Optional[Union[str, Path]] = None,
        output_csv: Optional[Union[str, Path]] = None,
        **params
    ) -> AnalysisResult:
        """
        Analyze video and save results.
        
        Args:
            video_path: Path to video file
            output_json: Path to output JSON (optional)
            output_csv: Path to output CSV summary (optional)
            **params: Analyzer-specific parameters
            
        Returns:
            AnalysisResult object
        """
        logger.info(f"Analyzing video with {self.name} v{self.version}")
        
        # Run analysis
        result = self.analyze(video_path, **params)
        
        # Save outputs
        if output_json:
            result.save_json(output_json)
        
        if output_csv:
            result.save_csv_summary(output_csv)
        
        return result
