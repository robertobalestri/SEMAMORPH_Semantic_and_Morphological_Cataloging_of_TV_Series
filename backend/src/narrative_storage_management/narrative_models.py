# narrative_models.py

from __future__ import annotations
from typing import List, Optional
from sqlmodel import SQLModel, Field, Relationship
from sqlalchemy.orm import relationship
import uuid
from sqlalchemy import UniqueConstraint, func

from ..utils.logger_utils import setup_logging
logger = setup_logging(__name__)

class DialogueLine:
    """Model representing a dialogue line from subtitles."""
    
    def __init__(self, index: int, start_time: float, end_time: float, text: str):
        self.index = index
        self.start_time = start_time
        self.end_time = end_time
        self.text = text
        self.speaker: Optional[str] = None  # Final speaker name (after all processing) - kept for backward compatibility
        self.characters: Optional[List[str]] = None  # List of character names (multiple speakers)
        self.is_llm_confident: Optional[bool] = None  # Boolean confidence from LLM
        self.scene_number: Optional[int] = None
        self.face_image_paths: Optional[List[str]] = None
        self.frame_image_paths: Optional[List[str]] = None
        # New fields for tracking speaker assignment history
        self.original_llm_speaker: Optional[str] = None  # Original LLM assignment
        self.original_llm_is_confident: Optional[bool] = None  # Original LLM boolean confidence
        self.resolution_method: Optional[str] = None  # How final speaker was determined
        # Multi-face processing fields
        self.candidate_speakers: Optional[List[str]] = None  # Multiple speaker candidates from faces (qualified)
        self.face_similarities: Optional[List[float]] = None  # Similarities for candidate speakers
        self.face_cluster_ids: Optional[List[int]] = None  # Cluster IDs for candidate speakers
        
        # All detected faces (for enhanced SRT display, including low-confidence)
        self.all_candidate_speakers: Optional[List[str]] = None  # All speaker candidates from faces
        self.all_face_similarities: Optional[List[float]] = None  # All similarities 
        self.all_face_cluster_ids: Optional[List[int]] = None  # All cluster IDs
        self.audio_cluster_assignments: Optional[List[Dict]] = None # New field for audio cluster assignments (N-to-N)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "index": self.index,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "text": self.text,
            "speaker": self.speaker,
            "characters": self.characters,
            "is_llm_confident": self.is_llm_confident,
            "scene_number": self.scene_number,
            "face_image_paths": self.face_image_paths,
            "frame_image_paths": self.frame_image_paths,
            "original_llm_speaker": self.original_llm_speaker,
            "original_llm_is_confident": self.original_llm_is_confident,
            "resolution_method": self.resolution_method,
            "candidate_speakers": self.candidate_speakers,
            "face_similarities": self.face_similarities,
            "face_cluster_ids": self.face_cluster_ids,
            "all_candidate_speakers": self.all_candidate_speakers,
            "all_face_similarities": self.all_face_similarities,
            "all_face_cluster_ids": self.all_face_cluster_ids,
            "audio_cluster_assignments": self.audio_cluster_assignments
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "DialogueLine":
        """Create from dictionary."""
        line = cls(data["index"], data["start_time"], data["end_time"], data["text"])
        line.speaker = data.get("speaker")
        line.characters = data.get("characters")
        line.is_llm_confident = data.get("is_llm_confident")
        line.scene_number = data.get("scene_number")
        line.face_image_paths = data.get("face_image_paths")
        line.frame_image_paths = data.get("frame_image_paths")
        line.original_llm_speaker = data.get("original_llm_speaker")
        line.original_llm_is_confident = data.get("original_llm_is_confident")
        line.resolution_method = data.get("resolution_method")
        line.candidate_speakers = data.get("candidate_speakers")
        line.face_similarities = data.get("face_similarities")
        line.face_cluster_ids = data.get("face_cluster_ids")
        line.all_candidate_speakers = data.get("all_candidate_speakers")
        line.all_face_similarities = data.get("all_face_similarities")
        line.all_face_cluster_ids = data.get("all_face_cluster_ids")
        line.audio_cluster_assignments = data.get("audio_cluster_assignments")
        return line

class ArcMainCharacterLink(SQLModel, table=True):
    """Junction table for main characters in narrative arcs."""
    __tablename__ = "arc_main_characters"

    arc_id: str = Field(foreign_key="narrativearc.id", primary_key=True)
    character_id: str = Field(foreign_key="character.entity_name", primary_key=True)

class ProgressionInterferingCharacterLink(SQLModel, table=True):
    """Junction table for interfering characters in arc progressions."""
    __tablename__ = "progression_interfering_characters"

    progression_id: str = Field(foreign_key="arcprogression.id", primary_key=True)
    character_id: str = Field(foreign_key="character.entity_name", primary_key=True)

class CharacterAppellation(SQLModel, table=True):
    """Model representing a character's appellation."""
    __tablename__ = "character_appellation"

    appellation: str = Field(primary_key=True)
    character_id: str = Field(foreign_key="character.entity_name")
    character: Optional["Character"] = Relationship(back_populates="appellations",sa_relationship=relationship(
        "Character",
        back_populates="appellations",
        lazy="selectin"
    ))

class Character(SQLModel, table=True):
    """Model representing a character in the series."""
    __tablename__ = "character"

    entity_name: str = Field(primary_key=True)
    best_appellation: str
    series: str = Field(index=True)
    biological_sex: Optional[str] = Field(default=None, index=True)  # NEW: 'M', 'F', or None for unknown

    # Relationships
    appellations: List["CharacterAppellation"] = Relationship(
        back_populates="character",
        sa_relationship=relationship(
            "CharacterAppellation",
            back_populates="character",
            cascade="all, delete-orphan"
        )
    )

    main_narrative_arcs: List["NarrativeArc"] = Relationship(
        back_populates="main_characters",
        sa_relationship=relationship(
            "NarrativeArc",
            secondary="arc_main_characters",
            back_populates="main_characters",
            lazy="selectin"
        )
    )

    interfering_progressions: List["ArcProgression"] = Relationship(
        back_populates="interfering_characters",
        sa_relationship=relationship(
            "ArcProgression",
            secondary="progression_interfering_characters",
            back_populates="interfering_characters",
            lazy="selectin"
        )
    )

class NarrativeArc(SQLModel, table=True):
    """Model representing a narrative arc."""
    __tablename__ = "narrativearc"

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    title: str
    arc_type: str
    description: str
    series: str

    # Relationships
    main_characters: List["Character"] = Relationship(
        back_populates="main_narrative_arcs",
        sa_relationship=relationship(
            "Character",
            secondary="arc_main_characters",
            back_populates="main_narrative_arcs",
            lazy="selectin"
        )
    )

    progressions: List["ArcProgression"] = Relationship(
        back_populates="narrative_arc",
        sa_relationship=relationship(
            "ArcProgression",
            back_populates="narrative_arc",
            cascade="all, delete-orphan",
            lazy="selectin"
        )
    )

class ArcProgression(SQLModel, table=True):
    """Model representing a progression within a narrative arc."""
    __tablename__ = "arcprogression"

    id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    main_arc_id: str = Field(foreign_key="narrativearc.id")
    content: str
    series: str
    season: str
    episode: str
    ordinal_position: int = Field(default=1, nullable=False)

    # Relationships
    interfering_characters: List["Character"] = Relationship(
        back_populates="interfering_progressions",
        sa_relationship=relationship(
            "Character",
            secondary="progression_interfering_characters",
            back_populates="interfering_progressions",
            lazy="selectin"
        )
    )

    narrative_arc: Optional["NarrativeArc"] = Relationship(
        back_populates="progressions",
        sa_relationship=relationship(
            "NarrativeArc",
            back_populates="progressions",
            lazy="selectin"
        )
    )
