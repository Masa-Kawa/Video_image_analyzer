"""
Flexible MLT Builder

Construct MLT files with configurable track layout and analysis result mapping.
"""

import xml.etree.ElementTree as ET
from fractions import Fraction
from pathlib import Path
from typing import List, Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)


class MLTBuilder:
    """
    Flexible MLT file builder with configuration-based result mapping.
    """
    
    def __init__(self, project_name: str, videos: List[str], fps: float = 30.0):
        """
        Initialize MLT builder.
        
        Args:
            project_name: Project name
            videos: List of source video paths
            fps: Frame rate (default: 30.0)
        """
        self.project_name = project_name
        self.videos = [str(Path(v).resolve()) for v in videos]
        self.fps = fps
        
        # Track configuration
        self.tracks = []

        # Analysis result storage
        self.cuts = []
        self.annotations_by_track = {}
        # NOTE: マーカー/テキストオーバーレイの出力は本ビルダーでは未対応。
        # それらが必要な場合は src/mlt/mlt_generator.py（実装済み）を使うこと。

        logger.info(f"Initialized MLT builder: {project_name}")
        logger.info(f"Videos: {len(self.videos)}")
    
    def add_track(self, track_id: str, track_type: str, track_name: str):
        """
        Add a custom track.
        
        Args:
            track_id: Track ID (e.g., "V1", "A1")
            track_type: "video" or "audio"
            track_name: Display name
        """
        self.tracks.append({
            'id': track_id,
            'type': track_type,
            'name': track_name
        })
        logger.info(f"Added track: {track_id} ({track_type}) - {track_name}")
    
    @staticmethod
    def _valid_scenes(analysis_data: Any, context: str = "") -> List[Dict]:
        """analysis_data を検証し、有効なシーン辞書のリストを返す。

        - dict なら 'results' を、無ければ単一シーン辞書として扱う
        - list/tuple ならそのまま走査
        - 各要素は dict かつ start_time_sec / end_time_sec を持つもののみ採用し、
          欠落・型不正は警告してスキップする（generate() での KeyError/TypeError 防止）。
        """
        if isinstance(analysis_data, dict):
            if 'results' in analysis_data:
                results = analysis_data['results']
            elif {'start_time_sec', 'end_time_sec'} <= set(analysis_data.keys()):
                results = [analysis_data]
            else:
                results = []
        elif isinstance(analysis_data, (list, tuple)):
            results = analysis_data
        else:
            logger.warning(
                f"{context}: analysis_data は dict/list である必要があります "
                f"(got {type(analysis_data).__name__}); 無視します")
            return []

        valid: List[Dict] = []
        for i, scene in enumerate(results):
            if not isinstance(scene, dict):
                logger.warning(f"{context}: 要素 {i} が dict ではないためスキップ")
                continue
            if 'start_time_sec' not in scene or 'end_time_sec' not in scene:
                logger.warning(
                    f"{context}: 要素 {i} に start_time_sec/end_time_sec が"
                    "無いためスキップ")
                continue
            valid.append(scene)
        return valid

    def apply_cuts(self, analysis_data: Dict):
        """
        Apply cuts from analysis results.

        Args:
            analysis_data: Analysis result dictionary with 'results' key
        """
        self.cuts = self._valid_scenes(analysis_data, context="apply_cuts")
        logger.info(f"Applied {len(self.cuts)} cuts")

    def add_annotations(self, analysis_data: Dict, track: str = "V2",
                       track_name: str = "Annotations"):
        """
        Add annotations to a specific track.

        Args:
            analysis_data: Analysis result dictionary
            track: Track ID (default: "V2")
            track_name: Track display name
        """
        if track not in self.annotations_by_track:
            self.annotations_by_track[track] = []

        results = self._valid_scenes(analysis_data, context=f"add_annotations[{track}]")
        self.annotations_by_track[track].extend(results)

        logger.info(f"Added {len(results)} annotations to track {track}")
    
    def _seconds_to_clock(self, seconds: float) -> str:
        """Convert seconds to HH:MM:SS.mmm format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
    
    def _seconds_to_frames(self, seconds: float) -> int:
        """Convert seconds to frame number."""
        return int(seconds * self.fps)
    
    def generate(self, output_path: str):
        """
        Generate complete MLT file.
        
        Args:
            output_path: Output .mlt file path
        """
        logger.info(f"Generating MLT file: {output_path}")

        # fps を分数(num/den)に変換し、29.97(30000/1001) 等の非整数フレームレートで
        # int 切り捨てによるタイムラインのフレームズレを防ぐ。
        # limit_denominator(1001) で NTSC 系（den=1001）まで正確に表現する。
        fps_frac = Fraction(self.fps).limit_denominator(1001)
        fps_num, fps_den = fps_frac.numerator, fps_frac.denominator

        # Create root MLT element
        mlt = ET.Element("mlt",
                        LC_NUMERIC="C",
                        version="7.33.0",
                        title=f"Shotcut version 25.10.31",
                        producer="main_bin")
        
        # Add profile
        ET.SubElement(mlt, "profile",
                     description="automatic",
                     width="1920",
                     height="1080",
                     progressive="1",
                     sample_aspect_num="1",
                     sample_aspect_den="1",
                     display_aspect_num="16",
                     display_aspect_den="9",
                     frame_rate_num=str(fps_num),
                     frame_rate_den=str(fps_den),
                     colorspace="709")
        
        # Create producers for videos
        if len(self.videos) > 1:
            # Series mode: multiple producers
            for idx, video_path in enumerate(self.videos):
                producer = ET.SubElement(mlt, "producer", id=f"producer{idx}")
                ET.SubElement(producer, "property", name="resource").text = video_path
                ET.SubElement(producer, "property", name="mlt_service").text = "avformat"
                ET.SubElement(producer, "property", name="seekable").text = "1"
            
            # Concat playlist
            concat_playlist = ET.SubElement(mlt, "playlist", id="concat_playlist")
            for idx in range(len(self.videos)):
                ET.SubElement(concat_playlist, "entry", producer=f"producer{idx}")
            
            main_producer_id = "concat_playlist"
        else:
            # Single video
            producer = ET.SubElement(mlt, "producer", id="producer0")
            ET.SubElement(producer, "property", name="resource").text = self.videos[0]
            ET.SubElement(producer, "property", name="mlt_service").text = "avformat"
            ET.SubElement(producer, "property", name="seekable").text = "1"
            main_producer_id = "producer0"
        
        # Create main_bin playlist
        main_bin = ET.SubElement(mlt, "playlist", id="main_bin")
        ET.SubElement(main_bin, "property", name="shotcut:skipConvert").text = "0"
        ET.SubElement(main_bin, "property", name="xml_retain").text = "1"
        
        if self.cuts:
            for scene in self.cuts:
                entry = ET.SubElement(main_bin, "entry", producer=main_producer_id)
                entry.set("in", self._seconds_to_clock(scene['start_time_sec']))
                entry.set("out", self._seconds_to_clock(scene['end_time_sec']))
        else:
            ET.SubElement(main_bin, "entry", producer=main_producer_id)
        
        # Create black background
        black = ET.SubElement(mlt, "producer", id="black")
        ET.SubElement(black, "property", name="length").text = "00:10:00.000"
        ET.SubElement(black, "property", name="eof").text = "pause"
        ET.SubElement(black, "property", name="resource").text = "0"
        ET.SubElement(black, "property", name="aspect_ratio").text = "1"
        ET.SubElement(black, "property", name="mlt_service").text = "color"
        ET.SubElement(black, "property", name="mlt_image_format").text = "rgba"
        ET.SubElement(black, "property", name="set.test_audio").text = "0"
        
        # Background playlist
        background = ET.SubElement(mlt, "playlist", id="background")
        ET.SubElement(background, "entry", producer="black")
        
        # Create main video track (V1)
        playlist0 = ET.SubElement(mlt, "playlist", id="playlist0")
        ET.SubElement(playlist0, "property", name="shotcut:video").text = "1"
        ET.SubElement(playlist0, "property", name="shotcut:name").text = "V1"
        
        if self.cuts:
            for scene in self.cuts:
                entry = ET.SubElement(playlist0, "entry", producer=main_producer_id)
                entry.set("in", self._seconds_to_clock(scene['start_time_sec']))
                entry.set("out", self._seconds_to_clock(scene['end_time_sec']))
        else:
            ET.SubElement(playlist0, "entry", producer=main_producer_id)
        
        # Create audio track (A1)
        playlist1 = ET.SubElement(mlt, "playlist", id="playlist1")
        ET.SubElement(playlist1, "property", name="shotcut:audio").text = "1"
        ET.SubElement(playlist1, "property", name="shotcut:name").text = "A1"
        ET.SubElement(playlist1, "blank", length="00:00:00.040")
        
        # Create additional tracks from configuration
        track_playlists = {}
        track_index = 2
        
        for track_config in self.tracks:
            if track_config['id'] in ['V1', 'A1']:
                continue  # Skip default tracks
            
            playlist = ET.SubElement(mlt, "playlist", id=f"playlist{track_index}")
            if track_config['type'] == 'video':
                ET.SubElement(playlist, "property", name="shotcut:video").text = "1"
            else:
                ET.SubElement(playlist, "property", name="shotcut:audio").text = "1"
            ET.SubElement(playlist, "property", name="shotcut:name").text = track_config['name']
            
            # Add annotations if present
            track_id = track_config['id']
            if track_id in self.annotations_by_track:
                for ann in self.annotations_by_track[track_id]:
                    entry = ET.SubElement(playlist, "entry", producer=main_producer_id)
                    entry.set("in", self._seconds_to_clock(ann['start_time_sec']))
                    entry.set("out", self._seconds_to_clock(ann['end_time_sec']))
            else:
                ET.SubElement(playlist, "blank", length="00:00:00.040")
            
            track_playlists[track_id] = f"playlist{track_index}"
            track_index += 1
        
        # Create tractor
        tractor = ET.SubElement(mlt, "tractor", id="tractor0",
                               title=f"Shotcut version 25.10.31")
        ET.SubElement(tractor, "property", name="shotcut").text = "1"
        ET.SubElement(tractor, "property", name="shotcut:projectAudioChannels").text = "2"
        ET.SubElement(tractor, "property", name="shotcut:projectFolder").text = "0"
        
        # Add tracks to tractor
        ET.SubElement(tractor, "track", producer="background")
        ET.SubElement(tractor, "track", producer="playlist0")
        track_audio = ET.SubElement(tractor, "track", producer="playlist1")
        track_audio.set("hide", "video")
        
        n_extra_tracks = 0
        for track_config in self.tracks:
            if track_config['id'] in ['V1', 'A1']:
                continue
            ET.SubElement(tractor, "track", producer=track_playlists[track_config['id']])
            n_extra_tracks += 1

        # Add transitions (standard Shotcut transitions).
        # tractor のトラックは [0:background, 1:V1, 2:A1, 3..:追加トラック]。
        # V1/A1 は self.tracks 内でスキップされるため、len(self.tracks) ではなく
        # 実際に追加したトラック数を使い、存在しないトラックへの transition 生成を防ぐ。
        self._add_standard_transitions(tractor, 2 + n_extra_tracks)
        
        # Write to file. ET.indent() でインデント整形（Python 3.9+）。
        # minidom による再パース往復を避け、大規模ツリーでの無駄なメモリ複製と
        # CPU 負荷を削減する。
        ET.indent(mlt, space="  ")
        tree = ET.ElementTree(mlt)
        tree.write(output_path, encoding="utf-8", xml_declaration=True)

        logger.info(f"✓ MLT file generated: {output_path}")
    
    def _add_standard_transitions(self, tractor, num_tracks):
        """Add standard Shotcut transitions."""
        trans_id = 0
        
        # Background to V1
        trans = ET.SubElement(tractor, "transition", id=f"transition{trans_id}")
        ET.SubElement(trans, "property", name="a_track").text = "0"
        ET.SubElement(trans, "property", name="b_track").text = "1"
        ET.SubElement(trans, "property", name="mlt_service").text = "mix"
        ET.SubElement(trans, "property", name="always_active").text = "1"
        ET.SubElement(trans, "property", name="sum").text = "1"
        trans_id += 1
        
        # Overlay
        trans = ET.SubElement(tractor, "transition", id=f"transition{trans_id}")
        ET.SubElement(trans, "property", name="a_track").text = "0"
        ET.SubElement(trans, "property", name="b_track").text = "1"
        ET.SubElement(trans, "property", name="mlt_service").text = "movit.overlay"
        ET.SubElement(trans, "property", name="disable").text = "1"
        trans_id += 1
        
        # Audio mix
        trans = ET.SubElement(tractor, "transition", id=f"transition{trans_id}")
        ET.SubElement(trans, "property", name="a_track").text = "0"
        ET.SubElement(trans, "property", name="b_track").text = "2"
        ET.SubElement(trans, "property", name="mlt_service").text = "mix"
        ET.SubElement(trans, "property", name="always_active").text = "1"
        ET.SubElement(trans, "property", name="sum").text = "1"
        trans_id += 1
        
        # Additional track transitions
        for i in range(3, num_tracks + 1):
            trans = ET.SubElement(tractor, "transition", id=f"transition{trans_id}")
            ET.SubElement(trans, "property", name="a_track").text = "0"
            ET.SubElement(trans, "property", name="b_track").text = str(i)
            ET.SubElement(trans, "property", name="mlt_service").text = "mix"
            ET.SubElement(trans, "property", name="always_active").text = "1"
            ET.SubElement(trans, "property", name="sum").text = "1"
            trans_id += 1
