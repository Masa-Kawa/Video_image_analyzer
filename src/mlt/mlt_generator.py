"""
MLT (Media Lovin' Toolkit) Generator

Generates Shotcut-compatible MLT files with support for:
- Multiple JSON inputs with different usage modes
- Cuts/scenes on timeline
- Markers
- Annotation tracks
- Text/subtitle tracks
"""

import logging
from fractions import Fraction
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import xml.etree.ElementTree as ET
import json

from src.core.time_utils import format_mlt_time, seconds_to_frames

logger = logging.getLogger(__name__)


class MLTGenerator:
    """Generate MLT files for Shotcut video editor."""
    
    def __init__(self, video_path: str, fps: Optional[float] = None, video_paths: Optional[List[str]] = None):
        """
        Initialize MLT generator.
        
        Args:
            video_path: Path to the primary video file (or first in series)
            fps: Video frame rate (will be auto-detected if None)
            video_paths: List of all video paths for series mode (optional)
        """
        self.video_path = str(Path(video_path).resolve())
        self.video_paths = video_paths if video_paths else [self.video_path]
        self.fps = fps
        # 正確なフレームレート分数 (例: 30000/1001)。int 切り捨てによるドリフトを防ぐ。
        self.fps_num: int = 30
        self.fps_den: int = 1

        if self.fps is None:
            self.fps = self._detect_fps()  # self.fps_num/den も設定する
        else:
            frac = Fraction(self.fps).limit_denominator(1001)
            self.fps_num, self.fps_den = frac.numerator, frac.denominator

        # Storage for different elements
        self.cuts: List[Dict] = []
        self.markers: List[Dict] = []
        self.annotations: List[Dict] = []
        self.texts: List[Dict] = []
        
        logger.info(f"Initialized MLT generator for {Path(video_path).name}")
        if len(self.video_paths) > 1:
            logger.info(f"Series mode: {len(self.video_paths)} videos")
        logger.info(f"FPS: {self.fps}")
    
    def _detect_fps(self) -> float:
        """Detect video FPS using ffprobe. r_frame_rate の分数(num/den)を保持する。"""
        import ffmpeg

        try:
            probe = ffmpeg.probe(self.video_path)
            video_stream = next(
                (s for s in probe['streams'] if s['codec_type'] == 'video'),
                None
            )

            if video_stream:
                num_str, _, den_str = video_stream['r_frame_rate'].partition('/')
                num = int(num_str)
                den = int(den_str) if den_str else 1
                if den == 0 or num == 0:
                    logger.warning("Invalid frame rate, using default 30.0")
                    self.fps_num, self.fps_den = 30, 1
                    return 30.0
                self.fps_num, self.fps_den = num, den
                return num / den
            else:
                logger.warning("Could not detect FPS, using default 30.0")
                self.fps_num, self.fps_den = 30, 1
                return 30.0
        except Exception as e:
            logger.warning(f"Failed to detect FPS: {e}, using default 30.0")
            self.fps_num, self.fps_den = 30, 1
            return 30.0
    
    def add_cuts(self, scenes: List[Dict]):
        """
        Add timeline cuts from scene detection.
        
        Args:
            scenes: List of scene dictionaries with start_time_sec and end_time_sec
        """
        self.cuts = scenes
        logger.info(f"Added {len(scenes)} cuts to timeline")
    
    def add_markers(self, markers: List[Dict]):
        """
        Add markers to timeline.
        
        Args:
            markers: List of marker dictionaries with time_sec and optional label
        """
        self.markers = markers
        logger.info(f"Added {len(markers)} markers")
    
    def add_annotation_track(self, annotations: List[Dict], track_name: str = "Annotations"):
        """
        Add annotation track with time ranges.
        
        Args:
            annotations: List of annotation dictionaries with start_time_sec, end_time_sec, and label
            track_name: Name for the annotation track
        """
        # Copy each annotation before tagging it so we don't mutate the
        # caller's dictionaries (avoid surprising side effects).
        tagged = [
            {**ann, 'track_name': track_name}
            for ann in annotations
            if isinstance(ann, dict)
        ]
        self.annotations.extend(tagged)
        logger.info(f"Added {len(tagged)} annotations to track '{track_name}'")
    
    def add_text_track(self, texts: List[Dict]):
        """
        Add text/subtitle track.
        
        Args:
            texts: List of text dictionaries with start_time_sec, end_time_sec, and text
        """
        self.texts = texts
        logger.info(f"Added {len(texts)} text overlays")
    
    def seconds_to_clock(self, seconds: float) -> str:
        """Convert seconds to HH:MM:SS.mmm format for MLT (see core.time_utils)."""
        return format_mlt_time(seconds)

    def seconds_to_frames(self, seconds: float) -> int:
        """Convert seconds to frame number (see core.time_utils)."""
        return seconds_to_frames(seconds, self.fps)

    def _estimate_timeline_duration(self, scene_groups: List[List[Dict]]) -> float:
        """
        Estimate the total timeline duration in seconds.

        Probes the source video(s); if that fails, falls back to the largest
        end_time_sec found across the supplied scene groups (cuts/annotations/
        texts). Used to size the black background producer so it never truncates
        a long timeline.
        """
        total = 0.0

        # Sum durations of all source videos.
        try:
            import ffmpeg
            for path in self.video_paths:
                probe = ffmpeg.probe(path)
                dur = probe.get('format', {}).get('duration')
                if dur:
                    total += float(dur)
        except Exception as e:  # ffmpeg missing or probe failed
            logger.warning(f"Could not probe video duration: {e}")

        # Fall back to / combine with the furthest end time on the timeline.
        for scenes in scene_groups:
            for sc in scenes:
                end = sc.get('end_time_sec')
                if isinstance(end, (int, float)):
                    total = max(total, float(end))

        # Guarantee a sane minimum so the background always covers the timeline.
        return max(total, 600.0)

    @staticmethod
    def _valid_scenes(items: Any, context: str = "") -> List[Dict]:
        """start_time_sec / end_time_sec を持つ dict のみを返す（欠落は警告スキップ）。"""
        if not isinstance(items, (list, tuple)):
            if items:
                logger.warning(f"{context}: list を期待しましたが "
                               f"{type(items).__name__} を受け取りました; 無視します")
            return []
        valid: List[Dict] = []
        for i, sc in enumerate(items):
            if not isinstance(sc, dict):
                logger.warning(f"{context}: 要素 {i} が dict ではないためスキップ")
                continue
            if 'start_time_sec' not in sc or 'end_time_sec' not in sc:
                logger.warning(f"{context}: 要素 {i} に start/end_time_sec が"
                               "無いためスキップ")
                continue
            valid.append(sc)
        return valid

    @staticmethod
    def _valid_markers(items: Any) -> List[Dict]:
        """time_sec を持つ dict のみを返す（欠落は警告スキップ）。"""
        if not isinstance(items, (list, tuple)):
            return []
        valid: List[Dict] = []
        for i, mk in enumerate(items):
            if isinstance(mk, dict) and 'time_sec' in mk:
                valid.append(mk)
            else:
                logger.warning(f"markers: 要素 {i} に time_sec が無いためスキップ")
        return valid
    
    def generate(self, output_path: str):
        """
        Generate MLT file with full Shotcut compatibility.
        
        Args:
            output_path: Path to output .mlt file
        """
        logger.info(f"Generating MLT file: {output_path}")

        # Validate all timeline inputs up front (missing start/end_time_sec are
        # warned and skipped rather than raising KeyError mid-generation).
        cut_scenes = self._valid_scenes(self.cuts, "cuts")
        ann_scenes = self._valid_scenes(self.annotations, "annotations")
        text_scenes = self._valid_scenes(self.texts, "texts")

        # Create root MLT element with Shotcut attributes
        mlt = ET.Element("mlt", 
                        LC_NUMERIC="C",
                        version="7.33.0",
                        title="Shotcut version 25.10.31",
                        producer="main_bin")
        
        # Add profile
        ET.SubElement(mlt, "profile",
                     description="PAL 4:3 DV or DVD",
                     width="1920",
                     height="1080",
                     progressive="1",
                     sample_aspect_num="1",
                     sample_aspect_den="1",
                     display_aspect_num="16",
                     display_aspect_den="9",
                     frame_rate_num=str(self.fps_num),
                     frame_rate_den=str(self.fps_den),
                     colorspace="709")
        
        # Create producers for videos
        if len(self.video_paths) > 1:
            # Series mode: create producers for all videos
            for idx, video_path in enumerate(self.video_paths):
                producer = ET.SubElement(mlt, "producer", id=f"producer{idx}")
                ET.SubElement(producer, "property", name="resource").text = str(Path(video_path).resolve())
                ET.SubElement(producer, "property", name="mlt_service").text = "avformat"
                ET.SubElement(producer, "property", name="seekable").text = "1"
            
            # Create concat playlist
            concat_playlist = ET.SubElement(mlt, "playlist", id="concat_playlist")
            for idx in range(len(self.video_paths)):
                ET.SubElement(concat_playlist, "entry", producer=f"producer{idx}")
            
            main_producer_id = "concat_playlist"
        else:
            # Single video: create producer0
            producer = ET.SubElement(mlt, "producer", id="producer0")
            ET.SubElement(producer, "property", name="resource").text = self.video_path
            ET.SubElement(producer, "property", name="mlt_service").text = "avformat"
            ET.SubElement(producer, "property", name="seekable").text = "1"
            main_producer_id = "producer0"
        
        # Create main_bin playlist (clip collection)
        main_bin = ET.SubElement(mlt, "playlist", id="main_bin")
        ET.SubElement(main_bin, "property", name="shotcut:skipConvert").text = "0"
        ET.SubElement(main_bin, "property", name="xml_retain").text = "1"
        
        if cut_scenes:
            for scene in cut_scenes:
                entry = ET.SubElement(main_bin, "entry", producer=main_producer_id)
                entry.set("in", self.seconds_to_clock(scene['start_time_sec']))
                entry.set("out", self.seconds_to_clock(scene['end_time_sec']))
        else:
            entry = ET.SubElement(main_bin, "entry", producer=main_producer_id)

        # Create black background producer. Size its length to the full timeline
        # so it never truncates long videos in Shotcut's preview/render.
        black_length = self.seconds_to_clock(
            self._estimate_timeline_duration([cut_scenes, ann_scenes, text_scenes])
        )
        black = ET.SubElement(mlt, "producer", id="black")
        ET.SubElement(black, "property", name="length").text = black_length
        ET.SubElement(black, "property", name="eof").text = "pause"
        ET.SubElement(black, "property", name="resource").text = "0"
        ET.SubElement(black, "property", name="aspect_ratio").text = "1"
        ET.SubElement(black, "property", name="mlt_service").text = "color"
        ET.SubElement(black, "property", name="mlt_image_format").text = "rgba"
        ET.SubElement(black, "property", name="set.test_audio").text = "0"
        
        # Create background playlist
        background = ET.SubElement(mlt, "playlist", id="background")
        ET.SubElement(background, "entry", producer="black")
        
        # Create timeline video track (V1)
        playlist0 = ET.SubElement(mlt, "playlist", id="playlist0")
        ET.SubElement(playlist0, "property", name="shotcut:video").text = "1"
        ET.SubElement(playlist0, "property", name="shotcut:name").text = "V1"
        
        if cut_scenes:
            for scene in cut_scenes:
                entry = ET.SubElement(playlist0, "entry", producer=main_producer_id)
                entry.set("in", self.seconds_to_clock(scene['start_time_sec']))
                entry.set("out", self.seconds_to_clock(scene['end_time_sec']))
        else:
            entry = ET.SubElement(playlist0, "entry", producer=main_producer_id)
        
        # Create audio track (A1)
        playlist1 = ET.SubElement(mlt, "playlist", id="playlist1")
        ET.SubElement(playlist1, "property", name="shotcut:audio").text = "1"
        ET.SubElement(playlist1, "property", name="shotcut:name").text = "A1"
        ET.SubElement(playlist1, "blank", length="00:00:00.040")
        
        # Create additional video track (V2) — アノテーション（時間範囲）を配置する
        playlist2 = ET.SubElement(mlt, "playlist", id="playlist2")
        ET.SubElement(playlist2, "property", name="shotcut:video").text = "1"
        ET.SubElement(playlist2, "property", name="shotcut:name").text = "Annotations"
        if ann_scenes:
            for ann in ann_scenes:
                entry = ET.SubElement(playlist2, "entry", producer=main_producer_id)
                entry.set("in", self.seconds_to_clock(ann['start_time_sec']))
                entry.set("out", self.seconds_to_clock(ann['end_time_sec']))
        else:
            ET.SubElement(playlist2, "blank", length="00:00:00.040")

        # Create text/subtitle track (V3) only when text overlays were added
        playlist3 = None
        if text_scenes:
            playlist3 = ET.SubElement(mlt, "playlist", id="playlist3")
            ET.SubElement(playlist3, "property", name="shotcut:video").text = "1"
            ET.SubElement(playlist3, "property", name="shotcut:name").text = "Text"
            for txt in text_scenes:
                entry = ET.SubElement(playlist3, "entry", producer=main_producer_id)
                entry.set("in", self.seconds_to_clock(txt['start_time_sec']))
                entry.set("out", self.seconds_to_clock(txt['end_time_sec']))

        # Create tractor (timeline composition)
        tractor = ET.SubElement(mlt, "tractor", id="tractor0",
                               title="Shotcut version 25.10.31")
        ET.SubElement(tractor, "property", name="shotcut").text = "1"
        ET.SubElement(tractor, "property", name="shotcut:projectAudioChannels").text = "2"
        ET.SubElement(tractor, "property", name="shotcut:projectFolder").text = "0"
        
        # Add markers (Shotcut はトラクタ上のプロパティとしてマーカーを保持する。
        # 想定外スキーマでも MLT は未知プロパティを無視するため読み込みは壊れない)。
        for i, marker in enumerate(self._valid_markers(self.markers)):
            ET.SubElement(tractor, "property",
                          name=f"shotcut:marker.{i}:time").text = \
                self.seconds_to_clock(marker['time_sec'])
            ET.SubElement(tractor, "property",
                          name=f"shotcut:marker.{i}:text").text = \
                str(marker.get('label', f"Marker {i + 1}"))

        # Add tracks to tractor
        ET.SubElement(tractor, "track", producer="background")
        ET.SubElement(tractor, "track", producer="playlist0")
        track_audio = ET.SubElement(tractor, "track", producer="playlist1")
        track_audio.set("hide", "video")
        ET.SubElement(tractor, "track", producer="playlist2")
        if playlist3 is not None:
            ET.SubElement(tractor, "track", producer="playlist3")

        # Add transitions
        # Background to V1
        trans0 = ET.SubElement(tractor, "transition", id="transition0")
        ET.SubElement(trans0, "property", name="a_track").text = "0"
        ET.SubElement(trans0, "property", name="b_track").text = "1"
        ET.SubElement(trans0, "property", name="mlt_service").text = "mix"
        ET.SubElement(trans0, "property", name="always_active").text = "1"
        ET.SubElement(trans0, "property", name="sum").text = "1"
        
        # Overlay transition
        trans1 = ET.SubElement(tractor, "transition", id="transition1")
        ET.SubElement(trans1, "property", name="a_track").text = "0"
        ET.SubElement(trans1, "property", name="b_track").text = "1"
        ET.SubElement(trans1, "property", name="mlt_service").text = "movit.overlay"
        ET.SubElement(trans1, "property", name="disable").text = "1"
        
        # Audio mix
        trans2 = ET.SubElement(tractor, "transition", id="transition2")
        ET.SubElement(trans2, "property", name="a_track").text = "0"
        ET.SubElement(trans2, "property", name="b_track").text = "2"
        ET.SubElement(trans2, "property", name="mlt_service").text = "mix"
        ET.SubElement(trans2, "property", name="always_active").text = "1"
        ET.SubElement(trans2, "property", name="sum").text = "1"
        
        # V2 mix
        trans3 = ET.SubElement(tractor, "transition", id="transition3")
        ET.SubElement(trans3, "property", name="a_track").text = "0"
        ET.SubElement(trans3, "property", name="b_track").text = "3"
        ET.SubElement(trans3, "property", name="mlt_service").text = "mix"
        ET.SubElement(trans3, "property", name="always_active").text = "1"
        ET.SubElement(trans3, "property", name="sum").text = "1"
        
        # V2 overlay
        trans4 = ET.SubElement(tractor, "transition", id="transition4")
        ET.SubElement(trans4, "property", name="a_track").text = "1"
        ET.SubElement(trans4, "property", name="b_track").text = "3"
        ET.SubElement(trans4, "property", name="mlt_service").text = "movit.overlay"
        ET.SubElement(trans4, "property", name="disable").text = "0"

        # V3 (text track) overlay — only when a text track was added (tractor index 4)
        if playlist3 is not None:
            trans5 = ET.SubElement(tractor, "transition", id="transition5")
            ET.SubElement(trans5, "property", name="a_track").text = "1"
            ET.SubElement(trans5, "property", name="b_track").text = "4"
            ET.SubElement(trans5, "property", name="mlt_service").text = "movit.overlay"
            ET.SubElement(trans5, "property", name="disable").text = "0"

        # Write to file. ET.indent() で整形（minidom 再パースの往復を回避）。
        ET.indent(mlt, space="  ")
        ET.ElementTree(mlt).write(output_path, encoding="utf-8", xml_declaration=True)

        logger.info(f"✓ MLT file generated: {output_path}")


def load_json_results(json_path: str) -> List[Dict]:
    """Load analysis results from JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle both old format (list) and new format (dict with 'results' key)
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and 'results' in data:
        return data['results']
    else:
        raise ValueError(f"Unknown JSON format in {json_path}")


def main():
    """CLI interface for MLT generator."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Shotcut MLT files from analysis JSON")
    parser.add_argument("video", help="Original video file (primary / first in series)")
    parser.add_argument("--video-series", type=str, nargs='+',
                        help="Additional video files for series mode (concatenated after the primary video)")
    parser.add_argument("--json-cuts", type=str, help="JSON file for timeline cuts")
    parser.add_argument("--json-markers", type=str, help="JSON file for markers")
    parser.add_argument("--json-annotations", type=str, nargs='+', help="JSON file(s) for annotations")
    parser.add_argument("--json-text", type=str, help="JSON file for text overlays")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output MLT file path")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    
    # Create generator (series mode when extra videos are supplied)
    if args.video_series:
        video_paths = [args.video] + list(args.video_series)
        logger.info(f"Series mode: {len(video_paths)} videos")
        generator = MLTGenerator(args.video, video_paths=video_paths)
    else:
        generator = MLTGenerator(args.video)
    
    # Load and add cuts
    if args.json_cuts:
        logger.info(f"Loading cuts from {args.json_cuts}")
        cuts = load_json_results(args.json_cuts)
        generator.add_cuts(cuts)
    
    # Load and add markers
    if args.json_markers:
        logger.info(f"Loading markers from {args.json_markers}")
        markers = load_json_results(args.json_markers)
        generator.add_markers(markers)
    
    # Load and add annotations
    if args.json_annotations:
        for i, ann_file in enumerate(args.json_annotations):
            logger.info(f"Loading annotations from {ann_file}")
            annotations = load_json_results(ann_file)
            track_name = f"Track_{i+1}_{Path(ann_file).stem}"
            generator.add_annotation_track(annotations, track_name)
    
    # Load and add text
    if args.json_text:
        logger.info(f"Loading text from {args.json_text}")
        texts = load_json_results(args.json_text)
        generator.add_text_track(texts)
    
    # Generate MLT
    generator.generate(args.output)
    
    print(f"\n✓ MLT file created: {args.output}")
    print(f"  Open this file in Shotcut to edit your video.")


if __name__ == "__main__":
    main()
