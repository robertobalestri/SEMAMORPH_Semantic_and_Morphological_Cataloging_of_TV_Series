"""
Recap generation data models.

This module defines Pydantic models for recap generation functionality,
including recap events, clips, configuration, and metadata.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime
import uuid


class RecapEvent(BaseModel):
    """Model representing an event selected for recap inclusion."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str = Field(description="Event description/content")
    series: str
    season: str 
    episode: str
    start_timestamp: str = Field(description="Start time in HH:MM:SS or HH:MM:SS,mmm format")
    end_timestamp: str = Field(description="End time in HH:MM:SS or HH:MM:SS,mmm format") 
    relevance_score: float = Field(ge=0.0, le=1.0, description="How relevant this event is to current episode")
    narrative_arc_id: Optional[str] = Field(None, description="ID of the narrative arc this event belongs to")
    arc_title: Optional[str] = Field(None, description="Title of the narrative arc")
    main_characters: List[str] = Field(default=[], description="Main characters involved in this event")
    # Legacy field support
    characters: Optional[List[str]] = Field(None, description="Alias for main_characters")
    narrative_arcs: Optional[List[str]] = Field(None, description="Legacy support for arc list")
    selection_reasoning: Optional[str] = Field(None, description="LLM reasoning for why this event was selected")
    vector_similarity_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    ordinal_position: int = Field(default=1, description="Position within the narrative arc")
    
    @validator('start_timestamp', 'end_timestamp')
    def validate_timestamp_format(cls, v):
        """Validate timestamp format (HH:MM:SS or HH:MM:SS,mmm)."""
        if not v:
            return v
        
        # Accept both formats: HH:MM:SS and HH:MM:SS,mmm
        if ',' in v:
            # Format: HH:MM:SS,mmm
            parts = v.split(',')
            if len(parts) != 2:
                raise ValueError('Timestamp must be in HH:MM:SS or HH:MM:SS,mmm format')
            time_part, ms_part = parts
            if len(time_part.split(':')) != 3 or len(ms_part) != 3:
                raise ValueError('Timestamp must be in HH:MM:SS,mmm format')
        else:
            # Format: HH:MM:SS
            if len(v.split(':')) != 3:
                raise ValueError('Timestamp must be in HH:MM:SS or HH:MM:SS,mmm format')
        
        return v
    
    def model_post_init(self, __context):
        """Handle legacy field mapping after initialization."""
        # Map characters to main_characters if provided
        if self.characters and not self.main_characters:
            self.main_characters = self.characters
        
        # Map narrative_arcs to arc_title if provided and arc_title not set
        if self.narrative_arcs and not self.arc_title and len(self.narrative_arcs) > 0:
            self.arc_title = self.narrative_arcs[0]
    
    @property
    def duration(self) -> float:
        """Calculate duration in seconds between start and end timestamps."""
        try:
            from ..utils.validation_utils import ValidationUtils
            return ValidationUtils.calculate_duration(self.start_timestamp, self.end_timestamp)
        except:
            return 0.0


class RecapClip(BaseModel):
    """Model representing a video clip to be included in the recap."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_id: str = Field(description="ID of the associated recap event")
    start_time: float = Field(description="Start time in seconds")
    end_time: float = Field(description="End time in seconds")
    duration: float = Field(description="Clip duration in seconds")
    subtitle_lines: List[str] = Field(default=[], description="Selected subtitle lines for this clip")
    video_quality_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Video quality assessment")
    audio_quality_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Audio quality assessment")
    has_dialogue: bool = Field(default=True, description="Whether clip contains dialogue")
    main_speakers: List[str] = Field(default=[], description="Primary speakers in this clip")
    file_path: Optional[str] = Field(None, description="Path to extracted clip file")
    extraction_method: str = Field(default="ffmpeg", description="Method used to extract clip")
    
    @validator('end_time')
    def validate_end_after_start(cls, v, values):
        """Ensure end time is after start time."""
        if 'start_time' in values and v <= values['start_time']:
            raise ValueError('End time must be after start time')
        return v
    
    @validator('duration')
    def validate_duration_matches(cls, v, values):
        """Ensure duration matches start/end times."""
        if 'start_time' in values and 'end_time' in values:
            calculated_duration = values['end_time'] - values['start_time']
            if abs(v - calculated_duration) > 0.1:  # Allow small floating point differences
                raise ValueError('Duration must match end_time - start_time')
        return v


class RecapConfiguration(BaseModel):
    """Configuration settings for recap generation."""
    
    target_duration_seconds: int = Field(default=60, ge=30, le=120, description="Target recap duration")
    max_events: int = Field(default=8, ge=3, le=15, description="Maximum number of events to include")
    min_event_duration: float = Field(default=3.0, ge=1.0, le=10.0, description="Minimum duration per event clip")
    max_event_duration: float = Field(default=12.0, ge=5.0, le=20.0, description="Maximum duration per event clip")
    subtitle_lines_per_event: int = Field(default=5, ge=2, le=10, description="Target subtitle lines per event")
    quality_threshold: float = Field(default=0.6, ge=0.0, le=1.0, description="Minimum quality threshold for clips")
    relevance_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Minimum relevance threshold for events")
    video_format: str = Field(default="mp4", description="Output video format")
    video_codec: str = Field(default="libx264", description="Video codec for output")
    audio_codec: str = Field(default="aac", description="Audio codec for output")
    enable_transitions: bool = Field(default=True, description="Whether to add fade transitions between clips")
    enable_subtitles: bool = Field(default=True, description="Whether to burn subtitles into clips")
    ffmpeg_preset: str = Field(default="medium", description="FFmpeg encoding preset")
    
    @validator('max_event_duration')
    def validate_max_greater_than_min(cls, v, values):
        """Ensure max duration is greater than min duration."""
        if 'min_event_duration' in values and v <= values['min_event_duration']:
            raise ValueError('max_event_duration must be greater than min_event_duration')
        return v


class RecapMetadata(BaseModel):
    """Metadata for a generated recap."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    series: str
    season: str
    episode: str
    generated_at: datetime = Field(default_factory=datetime.now)
    configuration: RecapConfiguration
    events: List[RecapEvent]
    clips: List[RecapClip]
    total_duration: float = Field(description="Total recap duration in seconds")
    processing_time_seconds: Optional[float] = Field(None, description="Time taken to generate recap")
    llm_queries_count: int = Field(default=0, description="Number of LLM queries made")
    vector_search_count: int = Field(default=0, description="Number of vector searches performed")
    ffmpeg_operations_count: int = Field(default=0, description="Number of FFmpeg operations performed")
    success: bool = Field(default=True, description="Whether recap generation was successful")
    error_message: Optional[str] = Field(None, description="Error message if generation failed")
    quality_metrics: Dict[str, float] = Field(default={}, description="Quality assessment metrics")
    file_paths: Dict[str, str] = Field(default={}, description="Generated file paths")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    @validator('clips')
    def validate_clips_match_events(cls, v, values):
        """Ensure clips correspond to events."""
        if 'events' in values:
            event_ids = {event.id for event in values['events']}
            clip_event_ids = {clip.event_id for clip in v}
            if not clip_event_ids.issubset(event_ids):
                raise ValueError('All clips must reference valid events')
        return v
    
    @validator('total_duration')
    def validate_total_duration_reasonable(cls, v, values):
        """Ensure total duration is reasonable."""
        if 'configuration' in values:
            target = values['configuration'].target_duration_seconds
            if v > target * 1.5:  # Allow 50% over target
                raise ValueError(f'Total duration {v}s exceeds target {target}s by too much')
        return v
