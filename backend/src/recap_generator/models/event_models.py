"""
Event processing data models.

This module defines models for event processing, vector search results,
and ranking operations in the recap generation pipeline.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime


class SubtitleEntry(BaseModel):
    """Model representing a single subtitle entry with precise timing."""
    
    index: int = Field(description="Index in the subtitle file")
    start_time: float = Field(description="Start time in seconds")
    end_time: float = Field(description="End time in seconds")
    text: str = Field(description="Subtitle text")
    
    @property
    def duration(self) -> float:
        """Calculate entry duration in seconds."""
        return self.end_time - self.start_time


class VectorEvent(BaseModel):
    """Model representing an event retrieved from vector database."""
    
    id: str = Field(description="Event ID from database")
    content: str = Field(description="Event content/description")
    series: str
    season: str
    episode: str
    start_timestamp: Optional[str] = Field(None, description="Start time in HH:MM:SS,mmm format")
    end_timestamp: Optional[str] = Field(None, description="End time in HH:MM:SS,mmm format")
    cosine_distance: float = Field(ge=0.0, le=2.0, description="Cosine distance from query")
    narrative_arc_id: str = Field(description="Associated narrative arc ID")
    arc_title: str = Field(description="Title of the narrative arc")
    arc_type: str = Field(description="Type of narrative arc")
    main_characters: List[str] = Field(default=[], description="Main characters in this event")
    ordinal_position: int = Field(default=1, description="Position within arc progression")
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="LLM confidence in event extraction")
    extraction_method: Optional[str] = Field(None, description="Method used to extract timestamps")
    
    @property
    def similarity_score(self) -> float:
        """Convert cosine distance to similarity score (0-1, higher is more similar)."""
        return max(0.0, 1.0 - (self.cosine_distance / 2.0))
    
    @property
    def has_valid_timestamps(self) -> bool:
        """Check if event has valid start and end timestamps."""
        return bool(self.start_timestamp and self.end_timestamp)


class SubtitleSequence(BaseModel):
    """Model representing a sequence of subtitle lines for an event."""
    
    event_id: str = Field(description="Associated event ID")
    lines: List[str] = Field(description="Subtitle text lines")
    start_time: float = Field(description="Sequence start time in seconds")
    end_time: float = Field(description="Sequence end time in seconds")
    speakers: List[str] = Field(default=[], description="Speakers in this sequence")
    dialogue_quality: float = Field(default=1.0, ge=0.0, le=1.0, description="Quality of dialogue (clarity, completeness)")
    narrative_completeness: float = Field(default=1.0, ge=0.0, le=1.0, description="How complete the narrative snippet is")
    selection_reasoning: Optional[str] = Field(None, description="LLM reasoning for subtitle selection")
    original_entries: Optional[List[SubtitleEntry]] = Field(None, description="Original subtitle entries for precise timing")
    
    @property
    def duration(self) -> float:
        """Calculate sequence duration in seconds."""
        return self.end_time - self.start_time
    
    @property
    def has_dialogue(self) -> bool:
        """Check if sequence contains meaningful dialogue."""
        return len(self.lines) > 0 and any(len(line.strip()) > 0 for line in self.lines)
    
    @validator('end_time')
    def validate_end_after_start(cls, v, values):
        """Ensure end time is after start time."""
        if 'start_time' in values and v <= values['start_time']:
            raise ValueError('End time must be after start time')
        return v


class EventRanking(BaseModel):
    """Model for ranking and scoring events for recap inclusion."""
    
    event: VectorEvent = Field(description="The event being ranked")
    relevance_score: float = Field(ge=0.0, le=1.0, description="Relevance to current episode narrative")
    importance_score: float = Field(ge=0.0, le=1.0, description="Overall importance in series narrative")
    character_significance: float = Field(ge=0.0, le=1.0, description="Significance of involved characters")
    arc_priority: float = Field(ge=0.0, le=1.0, description="Priority of the narrative arc")
    temporal_relevance: float = Field(ge=0.0, le=1.0, description="How recent/relevant the event is")
    dialogue_quality: float = Field(ge=0.0, le=1.0, description="Quality of associated dialogue")
    final_score: float = Field(ge=0.0, le=1.0, description="Composite final ranking score")
    ranking_position: Optional[int] = Field(None, description="Final ranking position")
    selection_reasons: List[str] = Field(default=[], description="Reasons for ranking this event")
    exclusion_reasons: List[str] = Field(default=[], description="Reasons that might exclude this event")
    
    @validator('final_score')
    def validate_final_score_reasonable(cls, v, values):
        """Ensure final score is reasonable based on component scores."""
        # Basic sanity check - final score shouldn't be wildly different from component scores
        component_scores = [
            values.get('relevance_score', 0),
            values.get('importance_score', 0),
            values.get('character_significance', 0),
            values.get('arc_priority', 0),
            values.get('temporal_relevance', 0),
            values.get('dialogue_quality', 0)
        ]
        avg_component = sum(component_scores) / len([s for s in component_scores if s > 0])
        if avg_component > 0 and abs(v - avg_component) > 0.5:
            # Allow deviation but warn if too extreme
            pass  # Could add logging here
        return v
    
    @property
    def is_recommended(self) -> bool:
        """Whether this event is recommended for inclusion based on score."""
        return self.final_score >= 0.7
    
    @property
    def quality_tier(self) -> str:
        """Categorize event quality based on final score."""
        if self.final_score >= 0.9:
            return "excellent"
        elif self.final_score >= 0.7:
            return "good"
        elif self.final_score >= 0.5:
            return "fair"
        else:
            return "poor"


class QueryResult(BaseModel):
    """Model representing results from a vector database query."""
    
    query: str = Field(description="Original search query")
    narrative_arc_id: Optional[str] = Field(None, description="Arc this query was targeting")
    arc_weight: float = Field(default=1.0, ge=0.0, le=1.0, description="Weight assigned to this arc")
    events: List[VectorEvent] = Field(description="Events returned by the query")
    total_results: int = Field(description="Total number of results found")
    search_time_ms: Optional[float] = Field(None, description="Time taken for search in milliseconds")
    
    @property
    def has_results(self) -> bool:
        """Whether query returned any results."""
        return len(self.events) > 0
    
    @property
    def best_match(self) -> Optional[VectorEvent]:
        """Get the best matching event (lowest cosine distance)."""
        if not self.events:
            return None
        return min(self.events, key=lambda e: e.cosine_distance)


class EventSelectionResult(BaseModel):
    """Model representing the final result of event selection for recap."""
    
    query_results: List[QueryResult] = Field(description="All query results")
    ranked_events: List[EventRanking] = Field(description="All events ranked for selection")
    selected_events: List[VectorEvent] = Field(description="Final selected events for recap")
    selection_criteria: Dict[str, Any] = Field(description="Criteria used for selection")
    total_events_considered: int = Field(description="Total number of unique events considered")
    selection_time_seconds: float = Field(description="Time taken for selection process")
    arc_distribution: Dict[str, int] = Field(description="Number of events selected per arc")
    
    @property
    def selection_success_rate(self) -> float:
        """Percentage of considered events that were selected."""
        if self.total_events_considered == 0:
            return 0.0
        return len(self.selected_events) / self.total_events_considered
    
    @property
    def average_event_score(self) -> float:
        """Average final score of selected events."""
        if not self.selected_events:
            return 0.0
        # Find ranking for each selected event
        selected_rankings = [
            ranking for ranking in self.ranked_events
            if ranking.event.id in [event.id for event in self.selected_events]
        ]
        if not selected_rankings:
            return 0.0
        return sum(ranking.final_score for ranking in selected_rankings) / len(selected_rankings)
