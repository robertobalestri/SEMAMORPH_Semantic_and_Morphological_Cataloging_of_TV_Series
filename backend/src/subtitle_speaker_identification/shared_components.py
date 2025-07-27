"""
Shared components for speaker identification pipelines.
"""

from .audio_processor import AudioProcessor
from .character_mapper import CharacterMapper
from .subtitle_face_extractor import SubtitleFaceExtractor
from .subtitle_face_embedder import SubtitleFaceEmbedder
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class SubtitleEntry:
    """Represents a single subtitle entry."""
    index: int
    start_time: str
    end_time: str
    text: str
    start_seconds: float
    end_seconds: float

def parse_srt_time_to_seconds(time_str: str) -> float:
    """Convert SRT time format (HH:MM:SS,mmm) to seconds."""
    time_str = time_str.replace(',', '.')
    parts = time_str.split(':')
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])
    return hours * 3600 + minutes * 60 + seconds

def parse_srt_file(srt_path: str) -> List[SubtitleEntry]:
    """Parse an SRT subtitle file and return list of subtitle entries."""
    # Assuming logger is available from the calling context or imported globally
    # For now, using a placeholder logger or assuming it's handled upstream
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Parsing SRT file: {srt_path}")
    
    try:
        with open(srt_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        # Try with different encoding
        with open(srt_path, 'r', encoding='latin-1') as f:
            content = f.read()
    
    # Split into subtitle blocks
    blocks = re.split(r'\n\s*\n', content.strip())
    subtitles = []
    
    for block in blocks:
        if not block.strip():
            continue
            
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue
            
        try:
            # Parse index
            index = int(lines[0])
            
            # Parse time range
            time_line = lines[1]
            time_match = re.match(r'(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})', time_line)
            if not time_match:
                continue
                
            start_time = time_match.group(1)
            end_time = time_match.group(2)
            
            # Parse subtitle text (may span multiple lines)
            text = '\n'.join(lines[2:]).strip()
            
            # Convert times to seconds
            start_seconds = parse_srt_time_to_seconds(start_time)
            end_seconds = parse_srt_time_to_seconds(end_time)
            
            subtitles.append(SubtitleEntry(
                index=index,
                start_time=start_time,
                end_time=end_time,
                text=text,
                start_seconds=start_seconds,
                end_seconds=end_seconds
            ))
            
        except (ValueError, IndexError) as e:
            logger.warning(f"Failed to parse subtitle block: {block[:50]}... Error: {e}")
            continue
    
    logger.info(f"Parsed {len(subtitles)} subtitle entries")
    return subtitles

class FaceProcessor:
    def __init__(self, path_handler):
        self.face_extractor = SubtitleFaceExtractor(path_handler)
        self.face_embedder = SubtitleFaceEmbedder(path_handler)

    def extract_faces_from_video(self, video_path, dialogue_lines):
        return self.face_extractor.extract_faces_from_subtitles(dialogue_lines)

    def generate_embeddings(self, face_data):
        return self.face_embedder.generate_embeddings(face_data)

    def cluster_faces(self, embeddings):
        # This is a placeholder. The actual implementation is in the face_clustering_system.py file.
        return {"clusters": []}

    def assign_speakers_from_clusters(self, dialogue_lines, clusters):
        # This is a placeholder. The actual implementation is in the speaker_cluster_associator.py file.
        return dialogue_lines

class ConfidenceScorer:
    def calculate_audio_confidence(self, dialogue):
        return True

    def calculate_face_confidence(self, dialogue):
        return True

    def calculate_hybrid_confidence(self, dialogue):
        return True

class SpeakerIdentificationComponents:
    """Container for shared components."""

    def __init__(self, path_handler, config, llm):
        self.audio_processor = AudioProcessor(config, path_handler)
        self.face_processor = FaceProcessor(path_handler)
        self.character_mapper = CharacterMapper()
        self.confidence_scorer = ConfidenceScorer()
