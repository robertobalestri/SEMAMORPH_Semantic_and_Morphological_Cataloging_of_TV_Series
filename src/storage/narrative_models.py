from __future__ import annotations
from typing import List, Optional
from sqlmodel import SQLModel, Field, Relationship, Session
from sqlalchemy.types import String, TypeDecorator
from sqlalchemy import Column, Table, ForeignKey, select
from sqlalchemy.orm import relationship, selectinload
import uuid
import json

# Custom JSON List type
class StringList(TypeDecorator):
    """Custom type for storing lists as JSON strings"""
    impl = String
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return '[]'
        return json.dumps(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return []
        return json.loads(value)

# Junction tables for many-to-many relationships
arc_characters = Table(
    "arc_characters",
    SQLModel.metadata,
    Column("arc_id", String, ForeignKey("narrativearc.id"), primary_key=True),
    Column("character_id", String, ForeignKey("character.id"), primary_key=True)
)

progression_characters = Table(
    "progression_characters",
    SQLModel.metadata,
    Column("progression_id", String, ForeignKey("arcprogression.id"), primary_key=True),
    Column("character_id", String, ForeignKey("character.id"), primary_key=True)
)

class Character(SQLModel, table=True):
    """Model representing a character in the series."""
    __tablename__ = "character"

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    entity_name: str = Field(index=True)
    best_appellation: str
    appellations_data: str = Field(
        default='[]',
        sa_column=Column('appellations', StringList)
    )
    series: str = Field(index=True)

    # Using string literals for forward references
    narrative_arcs: List["NarrativeArc"] = Relationship(sa_relationship=relationship(
        "NarrativeArc",
        secondary=arc_characters,
        back_populates="characters",
        lazy="selectin"
    ))
    
    progressions: List["ArcProgression"] = Relationship(sa_relationship=relationship(
        "ArcProgression",
        secondary=progression_characters,
        back_populates="characters",
        lazy="selectin"
    ))

    def __init__(self, **data):
        appellations = data.pop('appellations', [])
        if isinstance(appellations, str):
            appellations = [a.strip() for a in appellations.split(",")]
        data['appellations_data'] = json.dumps(appellations)
        super().__init__(**data)

    @property
    def appellations(self) -> List[str]:
        """Get appellations as a list."""
        try:
            return json.loads(self.appellations_data)
        except json.JSONDecodeError:
            return []

    @appellations.setter
    def appellations(self, value: List[str]):
        """Set appellations from a list."""
        self.appellations_data = json.dumps(value)

class ArcProgression(SQLModel, table=True):
    __tablename__ = "arcprogression"

    id: Optional[str] = Field(default=None, primary_key=True)
    main_arc_id: str = Field(foreign_key="narrativearc.id")
    content: str
    series: str
    season: str
    episode: str
    ordinal_position: int = Field(default=0, nullable=False)
    
    interfering_episode_characters_data: str = Field(
        default='[]',
        sa_column=Column('interfering_episode_characters', StringList)
    )
    
    # Using string literals for forward references
    characters: List["Character"] = Relationship(sa_relationship=relationship(
        "Character",
        secondary=progression_characters,
        back_populates="progressions",
        lazy="selectin"
    ))
    narrative_arc: Optional["NarrativeArc"] = Relationship(sa_relationship=relationship(
        "NarrativeArc",
        back_populates="progressions",
        lazy="selectin"
    ))

    def __init__(self, **data):
        interfering_chars = data.pop('interfering_episode_characters', [])
        if isinstance(interfering_chars, str):
            interfering_chars = [c.strip() for c in interfering_chars.split(",")]
        data['interfering_episode_characters_data'] = json.dumps(interfering_chars)
        super().__init__(**data)

    @property
    def interfering_episode_characters(self) -> List[str]:
        """Get interfering characters as a list."""
        try:
            return json.loads(self.interfering_episode_characters_data)
        except json.JSONDecodeError:
            return []

    @interfering_episode_characters.setter
    def interfering_episode_characters(self, value: List[str]):
        """Set interfering characters from a list."""
        self.interfering_episode_characters_data = json.dumps(value)
    
    def get_title(self, session: Optional[Session] = None) -> str:
        result = session.exec(
            select(NarrativeArc)
            .where(NarrativeArc.id == self.main_arc_id)
        ).first()
        main_arc = result._data[0] if result else None
        return main_arc.title if main_arc else "Unknown Title"

class NarrativeArc(SQLModel, table=True):
    """Model representing a narrative arc."""
    __tablename__ = "narrativearc"

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    title: str
    arc_type: str
    description: str
    episodic: bool
    series: str
    main_characters_data: str = Field(
        default='[]',
        sa_column=Column('main_characters', StringList)
    )
    
    # Using string literals for forward references
    characters: List["Character"] = Relationship(sa_relationship=relationship(
        "Character",
        secondary=arc_characters,
        back_populates="narrative_arcs",
        lazy="selectin"
    ))
    progressions: List["ArcProgression"] = Relationship(sa_relationship=relationship(
        "ArcProgression",
        back_populates="narrative_arc",
        cascade="all, delete-orphan",
        lazy="selectin"
    ))

    def __init__(self, **data):
        if 'id' not in data or data['id'] is None:
            data['id'] = str(uuid.uuid4())
        main_chars = data.pop('main_characters', [])
        if isinstance(main_chars, str):
            main_chars = [c.strip() for c in main_chars.split(",")]
        data['main_characters_data'] = json.dumps(main_chars)
        super().__init__(**data)

    @property
    def main_characters(self) -> List[str]:
        """Get main characters as a list."""
        try:
            return json.loads(self.main_characters_data)
        except json.JSONDecodeError:
            return []

    @main_characters.setter
    def main_characters(self, value: List[str]):
        """Set main characters from a list."""
        self.main_characters_data = json.dumps(value)

    def __eq__(self, other):
        if not isinstance(other, NarrativeArc):
            return False
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)
