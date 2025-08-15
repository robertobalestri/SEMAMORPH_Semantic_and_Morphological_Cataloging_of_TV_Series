"""
Simple data models for recap generation.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class Event:
    """Represents a selected narrative event."""
    id: str
    content: str
    series: str
    season: str
    episode: str
    start_time: str  # HH:MM:SS,mmm format
    end_time: str    # HH:MM:SS,mmm format
    narrative_arc_id: str
    arc_title: str
    relevance_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "content": self.content,
            "series": self.series,
            "season": self.season,
            "episode": self.episode,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "narrative_arc_id": self.narrative_arc_id,
            "arc_title": self.arc_title,
            "relevance_score": self.relevance_score
        }


@dataclass
class VideoClip:
    """Represents an extracted video clip."""
    event_id: str
    file_path: str
    start_seconds: float
    end_seconds: float
    duration: float
    subtitle_lines: List[str]
    arc_title: str


@dataclass
class RecapResult:
    """Final recap generation result."""
    video_path: str
    events: List[Event]
    clips: List[VideoClip]
    total_duration: float
    success: bool
    error_message: Optional[str] = None
